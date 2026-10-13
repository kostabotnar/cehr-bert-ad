"""Feature importance step 02 gives the same attributions in padded batches as one patient at a time."""

from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("cehrbert")

VOCAB = ["[PAD]", "[MASK]", "[UNUSED]", "[OOV]", "[CLS]", "[VS]", "[VE]"] + [f"c{i}" for i in range(20)]
CLS_ID = VOCAB.index("[CLS]")
VS_ID, VE_ID = VOCAB.index("[VS]"), VOCAB.index("[VE]")
MAX_LENGTH = 32
TOL = 1e-5


class FakeTokenizer:
    """The subset of CehrBertTokenizer that the collator and step 02 use."""

    pad_token_index = 0
    mask_token_index = 1
    oov_token_index = 3
    cls_token_index = CLS_ID
    vocab_size = len(VOCAB)

    def convert_token_to_id(self, token):
        return VOCAB.index(token) if token in VOCAB else self.oov_token_index

    def convert_id_to_token(self, index):
        return VOCAB[index]


@pytest.fixture
def mod(step):
    return step("feature_importance/02_attention_analysis.py")


@pytest.fixture
def model():
    from cehrbert.models.hf_models.config import CehrBertConfig
    from cehrbert.models.hf_models.hf_cehrbert import CehrBertForClassification

    torch.manual_seed(0)
    config = CehrBertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        n_time_embd=8,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=MAX_LENGTH,
        cls_token_id=CLS_ID,
    )
    config._attn_implementation = "eager"
    m = CehrBertForClassification(config)
    assert m.config._attn_implementation == "eager"
    # Non-trivial running stats, so batch statistics (the failure mode) would visibly differ.
    m.age_batch_norm.running_mean.fill_(70.0)
    m.age_batch_norm.running_var.fill_(25.0)
    return m.eval()


@pytest.fixture
def collator(mod):
    return mod._build_collator(FakeTokenizer(), MAX_LENGTH)


def _record(person_id, tokens, seed):
    rng = random.Random(seed)
    n = len(tokens)
    return {
        "input_ids": [VOCAB.index(t) for t in tokens],
        "ages": [60 + i // 4 for i in range(n)],
        "dates": [2000 + 3 * i for i in range(n)],
        "visit_concept_orders": [1 + i // 4 for i in range(n)],
        "concept_values": [rng.random() for _ in range(n)],
        "concept_value_masks": [rng.randint(0, 1) for _ in range(n)],
        "visit_segments": [1 + (i // 4) % 2 for i in range(n)],
        "age_at_index": float(55 + 4 * person_id),
        "classifier_label": person_id % 2,
        "person_id": person_id,
    }


def _concepts(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        out += ["[VS]"] + [f"c{rng.randrange(20)}" for _ in range(rng.randint(1, 3))] + ["[VE]"]
    return out[:n]


@pytest.fixture
def records():
    lengths = [7, 2, 19, 31, 45, 12, 4]  # 45 exceeds MAX_LENGTH - 1 and is tail-truncated
    return [_record(i, _concepts(n, seed=i), seed=i) for i, n in enumerate(lengths)]


def test_padded_batch_scores_match_one_at_a_time(mod, model, collator, records):
    singles = [mod._attribute_batch(model, collator([r]), "cpu")[0] for r in records]
    # A batch-size-1 forward leaves age_batch_norm in train mode; batching must not inherit that.
    assert model.age_batch_norm.training
    batched = mod._attribute_batch(model, collator(records), "cpu")

    assert len(batched) == len(records)
    widths = {length for _, _, length in batched}
    assert len(widths) > 1, "the batch must mix sequence lengths to exercise padding"
    for (s_scores, s_prob, s_len), (b_scores, b_prob, b_len) in zip(singles, batched):
        assert s_len == b_len
        assert b_prob == pytest.approx(s_prob, abs=TOL)
        for method in ("raw", "rollout", "gradxatt"):
            assert b_scores[method].shape == (s_len,)
            torch.testing.assert_close(b_scores[method], s_scores[method], atol=TOL, rtol=0)


def test_batched_real_length_is_cls_plus_truncated_tokens(mod, model, collator, records):
    batched = mod._attribute_batch(model, collator(records), "cpu")
    for record, (_, _, length) in zip(records, batched):
        # Tail truncation may start later than MAX_LENGTH - 1 from the end, at a [VS] boundary.
        assert length == 1 + len(collator.generate_start_end_index(record)["input_ids"]) <= MAX_LENGTH


def test_run_attribution_is_batch_size_invariant_and_skips_empty_patients(mod, model, collator, records):
    dataset = list(records)
    dataset.insert(2, _record(90, [], seed=90))  # no tokens at all: filtered before collating
    dataset.insert(5, _record(91, ["[VS]", "[VE]"], seed=91))  # only excluded tokens
    tok = FakeTokenizer()

    ref = mod._run_attribution(model, collator, dataset, len(dataset), 1, "cpu", tok)
    for batch_size in (3, 32):
        got = mod._run_attribution(model, collator, dataset, len(dataset), batch_size, "cpu", tok)
        acc, present, counts, n_attr, n_skip = got
        assert (n_attr, n_skip) == (ref[3], ref[4]) == (len(records), 2)
        assert dict(counts) == dict(ref[2])
        assert {g: dict(v) for g, v in present.items()} == {g: dict(v) for g, v in ref[1].items()}
        for group in acc:
            for method in acc[group]:
                assert acc[group][method].keys() == ref[0][group][method].keys()
                for cid, value in acc[group][method].items():
                    assert value == pytest.approx(ref[0][group][method][cid], abs=TOL)


def test_run_attribution_caps_at_n_run(mod, model, collator, records):
    _, _, counts, n_attr, n_skip = mod._run_attribution(model, collator, records, 3, 2, "cpu", FakeTokenizer())
    assert n_attr + n_skip == 3 and sum(counts.values()) == n_attr


def test_token_lookup_caches_tokenizer_calls(mod):
    calls = []

    class Counting(FakeTokenizer):
        def convert_id_to_token(self, index):
            calls.append(index)
            return super().convert_id_to_token(index)

    lookup = mod._TokenLookup(Counting())
    assert lookup([CLS_ID, VS_ID, 8, 8, VE_ID]) == ["[CLS]", "[VS]", "c1", "c1", "[VE]"]
    assert lookup([8, CLS_ID]) == ["c1", "[CLS]"]
    assert sorted(calls) == sorted({CLS_ID, VS_ID, 8, VE_ID})
