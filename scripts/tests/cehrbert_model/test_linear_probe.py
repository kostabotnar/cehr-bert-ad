"""Tests for the frozen CEHR-BERT linear probe (steps 13 and 14, pipeline/linear_probe.py).

Everything runs on CPU with a tiny randomly initialized CehrBertForPreTraining (hidden size 16,
one layer) and synthetic data.
"""

from __future__ import annotations

import datetime as dt
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
pytest.importorskip("cehrbert")

from cehrbert.models.hf_models.config import CehrBertConfig  # noqa: E402
from cehrbert.models.hf_models.hf_cehrbert import CehrBertForPreTraining  # noqa: E402
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import WhitespaceSplit  # noqa: E402

from pipeline import evaluation as ev  # noqa: E402
from pipeline import linear_probe as lp  # noqa: E402
from pipeline import paths  # noqa: E402

HIDDEN = 16
SPECIAL = ["[PAD]", "[OOV]", "[MASK]", "[CLS]", "[UNUSED]", "[VS]", "[VE]"]
CONCEPTS = [str(i) for i in range(100, 130)]
VOCAB = {tok: i for i, tok in enumerate(SPECIAL + CONCEPTS)}
VS_ID, VE_ID = VOCAB["[VS]"], VOCAB["[VE]"]


def make_tokenizer() -> CehrBertTokenizer:
    tok = Tokenizer(WordLevel(VOCAB, unk_token="[OOV]"))
    tok.pre_tokenizer = WhitespaceSplit()
    return CehrBertTokenizer(tok, [], {})


def make_model(seed: int = 0) -> CehrBertForPreTraining:
    torch.manual_seed(seed)
    config = CehrBertConfig(
        vocab_size=len(VOCAB),
        hidden_size=HIDDEN,
        n_time_embd=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        cls_token_id=VOCAB["[CLS]"],
        include_value_prediction=True,
    )
    return CehrBertForPreTraining(config).eval()


def make_record(rng: np.random.Generator, person_id: int, n_tokens: int, label: int = 0) -> dict:
    """One post-extract_cohort_sequences style record: visits of [VS] c c c [VE] with visit orders."""
    ids, orders = [], []
    visit = 0
    while len(ids) < n_tokens:
        visit += 1
        body = rng.integers(len(SPECIAL), len(VOCAB), size=3).tolist()
        ids.extend([VS_ID] + body + [VE_ID])
        orders.extend([visit] * 5)
    ids, orders = ids[:n_tokens], orders[:n_tokens]
    n = len(ids)
    return {
        "person_id": person_id,
        "input_ids": ids,
        "ages": [50 + o for o in orders],
        "dates": [100 * o for o in orders],
        "visit_concept_orders": orders,
        "concept_values": [0.0] * n,
        "concept_value_masks": [0] * n,
        "visit_segments": [1 + (o % 2) for o in orders],
        "classifier_label": label,
        "index_date": 1.6e9,
        "age_at_index": 60,
    }


def add_prompt(record: dict) -> dict:
    """Prepend the four demographic prompt tokens (age -1) as extract_cohort_sequences keeps them."""
    n_events = len(record["input_ids"])
    out = dict(record)
    out["input_ids"] = [VOCAB["100"], VOCAB["101"], VOCAB["102"], VOCAB["103"]] + record["input_ids"]
    out["concept_ids"] = ["year:2015", "age:76", "8507", "8516"] + ["c"] * n_events
    out["ages"] = [-1] * 4 + record["ages"]
    out["dates"] = [0] * 4 + record["dates"]
    out["visit_concept_orders"] = [0] * 4 + record["visit_concept_orders"]
    out["concept_values"] = [0.0] * 4 + record["concept_values"]
    out["concept_value_masks"] = [0] * 4 + record["concept_value_masks"]
    out["visit_segments"] = [0] * 4 + record["visit_segments"]
    return out


def make_dataset(lengths, seed: int = 1, prompt: bool = False):
    """Records with ``lengths`` event tokens; with ``prompt`` four prompt tokens are added in front."""
    rng = np.random.default_rng(seed)
    records = [make_record(rng, 1000 + i, n, label=i % 2) for i, n in enumerate(lengths)]
    if prompt:
        records = [add_prompt(r) for r in records]
    else:
        for r in records:
            r["concept_ids"] = ["c"] * len(r["input_ids"])
    return datasets.Dataset.from_list(records)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def test_features_do_not_depend_on_batch_size():
    model, tokenizer = make_model(), make_tokenizer()
    ds = make_dataset([6, 9, 14, 20, 30, 11, 5])
    kwargs = dict(max_length=16, device="cpu", amp_dtype=None)
    one = lp.extract_features(model, ds, tokenizer, batch_size=1, **kwargs)
    four = lp.extract_features(model, ds, tokenizer, batch_size=4, **kwargs)
    for kind in lp.FEATURE_KINDS:
        a, b = getattr(one, kind), getattr(four, kind)
        assert a.shape == (7, HIDDEN) and a.dtype == np.float32
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
    assert not np.allclose(one.cls, one.mean)
    np.testing.assert_array_equal(one.person_id, np.arange(1000, 1007))


def test_truncation_bookkeeping():
    model, tokenizer = make_model(), make_tokenizer()
    ds = make_dataset([5, 15, 16, 40])
    feats = lp.extract_features(model, ds, tokenizer, max_length=16, batch_size=2, device="cpu", amp_dtype=None)
    # The collator keeps max_length - 1 record tokens plus [CLS].
    assert feats.n_tokens_before_truncation.tolist() == [5, 15, 16, 40]
    assert feats.truncated.tolist() == [False, False, True, True]
    assert feats.label.tolist() == [0, 1, 0, 1]


def test_empty_windows_keep_their_rows_and_do_not_break_batches():
    """Regression: an empty record made torch.tensor([]) float32 and the batch's input_ids float."""
    model, tokenizer = make_model(), make_tokenizer()
    lengths = [0, 7, 0, 12, 9, 0]
    ds = make_dataset(lengths)
    kwargs = dict(max_length=16, device="cpu", amp_dtype=None)
    mixed = lp.extract_features(model, ds, tokenizer, batch_size=6, **kwargs)
    one = lp.extract_features(model, ds, tokenizer, batch_size=1, **kwargs)
    all_empty = lp.extract_features(model, make_dataset([0, 0]), tokenizer, batch_size=2, **kwargs)

    np.testing.assert_array_equal(mixed.person_id, np.arange(1000, 1006))
    assert mixed.n_tokens_before_truncation.tolist() == lengths
    assert mixed.truncated.tolist() == [False] * 6
    for kind in lp.FEATURE_KINDS:
        got = getattr(mixed, kind)
        assert got.shape == (6, HIDDEN) and np.isfinite(got).all()
        np.testing.assert_allclose(got, getattr(one, kind), atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(got[0], got[2], atol=1e-6)
        np.testing.assert_allclose(getattr(all_empty, kind)[0], got[0], atol=1e-5)
    # A CLS-only sequence has one real position, so the mean equals the CLS state.
    np.testing.assert_allclose(mixed.cls[0], mixed.mean[0], atol=1e-6)


def test_prompt_survives_truncation_and_is_not_counted_as_events():
    model, tokenizer = make_model(), make_tokenizer()
    ds = make_dataset([5, 40], prompt=True)
    kwargs = dict(max_length=16, device="cpu", amp_dtype=None)
    feats = lp.extract_features(model, ds, tokenizer, batch_size=2, keep_demographic_prompt=True, **kwargs)
    assert feats.n_tokens_before_truncation.tolist() == [9, 44]
    assert feats.n_event_tokens.tolist() == [5, 40]
    assert feats.truncated.tolist() == [False, True]
    assert feats.demographic_prompt_tokens.tolist() == [4, 4]

    plain = lp.extract_features(model, ds, tokenizer, batch_size=2, **kwargs)
    assert plain.demographic_prompt_tokens.tolist() == [0, 0]
    assert plain.n_event_tokens.tolist() == plain.n_tokens_before_truncation.tolist()
    # Without the option the collator drops the prompt of the long record.
    assert not np.allclose(plain.cls[1], feats.cls[1])


def test_prompt_only_rows_are_empty_windows_but_go_through_the_collator():
    model, tokenizer = make_model(), make_tokenizer()
    ds = make_dataset([0, 7, 0, 12], prompt=True)
    kwargs = dict(max_length=16, device="cpu", amp_dtype=None, keep_demographic_prompt=True)
    feats = lp.extract_features(model, ds, tokenizer, batch_size=4, **kwargs)
    assert feats.n_event_tokens.tolist() == [0, 7, 0, 12]
    assert feats.n_tokens_before_truncation.tolist() == [4, 11, 4, 16]
    assert (feats.demographic_prompt_tokens == 4).all()
    # CLS plus four prompt tokens is not the CLS-only encoding.
    cls_only = lp.extract_features(model, make_dataset([0]), tokenizer, batch_size=1, **kwargs)
    assert not np.allclose(feats.cls[0], cls_only.cls[0])
    np.testing.assert_allclose(feats.cls[0], feats.cls[2], atol=1e-6)


def test_prompt_features_do_not_depend_on_batch_size():
    model, tokenizer = make_model(), make_tokenizer()
    ds = make_dataset([0, 9, 14, 20, 30, 0, 5], prompt=True)
    kwargs = dict(max_length=16, device="cpu", amp_dtype=None, keep_demographic_prompt=True)
    one = lp.extract_features(model, ds, tokenizer, batch_size=1, **kwargs)
    four = lp.extract_features(model, ds, tokenizer, batch_size=4, **kwargs)
    for kind in lp.FEATURE_KINDS:
        np.testing.assert_allclose(getattr(one, kind), getattr(four, kind), atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(one.demographic_prompt_tokens, four.demographic_prompt_tokens)


def test_mixed_batch_with_truly_empty_rows_and_prompts():
    """A row without any token (prompt dropped, no events) still uses the CLS-only path."""
    model, tokenizer = make_model(), make_tokenizer()
    rng = np.random.default_rng(1)
    with_prompt = add_prompt(make_record(rng, 1, 6))
    empty = make_record(rng, 2, 0)
    empty["concept_ids"] = []
    ds = datasets.Dataset.from_list([with_prompt, empty])
    feats = lp.extract_features(
        model, ds, tokenizer, max_length=16, batch_size=2, device="cpu", amp_dtype=None, keep_demographic_prompt=True
    )
    assert feats.n_tokens_before_truncation.tolist() == [10, 0]
    assert feats.n_event_tokens.tolist() == [6, 0]
    assert feats.demographic_prompt_tokens.tolist() == [4, 0]
    assert np.isfinite(feats.cls).all()


def test_two_token_prompt_counts_and_survives_truncation():
    model, tokenizer = make_model(), make_tokenizer()
    rng = np.random.default_rng(2)
    rows = []
    for pid, n in enumerate([4, 40]):
        r = add_prompt(make_record(rng, pid, n))
        # Keep only the gender and race prompt tokens, as the mapping does after the index date.
        for k, v in r.items():
            if isinstance(v, list) and len(v) == n + 4:
                r[k] = v[2:]
        rows.append(r)
    ds = datasets.Dataset.from_list(rows)
    feats = lp.extract_features(
        model, ds, tokenizer, max_length=16, batch_size=2, device="cpu", amp_dtype=None, keep_demographic_prompt=True
    )
    assert feats.demographic_prompt_tokens.tolist() == [2, 2]
    assert feats.n_event_tokens.tolist() == [4, 40]
    assert feats.truncated.tolist() == [False, True]


def test_cls_only_batch_matches_collator_dtypes():
    from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator

    tokenizer = make_tokenizer()
    collator = CehrBertDataCollator(tokenizer, max_length=16, is_pretraining=False)
    real = collator([make_record(np.random.default_rng(0), 1, 6)])
    synth = lp.cls_only_batch(2, tokenizer.cls_token_index)
    for key, tensor in synth.items():
        assert tensor.dtype == real[key].dtype, key
    assert synth["input_ids"].tolist() == [[tokenizer.cls_token_index]] * 2


def test_mean_pooling_ignores_padding():
    hidden = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float)
    ids = torch.full((2, 5), 7)
    ids[:, 0] = 3
    _, mean = lp.pool_features(hidden, mask, ids, cls_token_index=3)
    poisoned = hidden.clone()
    poisoned[0, 3:] = 1e6
    _, mean_poisoned = lp.pool_features(poisoned, mask, ids, cls_token_index=3)
    torch.testing.assert_close(mean, mean_poisoned)
    torch.testing.assert_close(mean[0], hidden[0, :3].mean(dim=0))
    torch.testing.assert_close(mean[1], hidden[1].mean(dim=0))


def test_cls_feature_is_position_zero():
    hidden = torch.randn(3, 4, 6)
    ids = torch.full((3, 4), 9)
    ids[:, 0] = 3
    cls, _ = lp.pool_features(hidden, torch.ones(3, 4), ids, cls_token_index=3)
    torch.testing.assert_close(cls, hidden[:, 0, :])


def test_cls_assertion_fires_when_position_zero_is_not_cls():
    hidden = torch.randn(2, 4, 6)
    ids = torch.full((2, 4), 9)
    ids[0, 0] = 3
    with pytest.raises(AssertionError):
        lp.pool_features(hidden, torch.ones(2, 4), ids, cls_token_index=3)


def test_load_frozen_model_uses_sdpa_and_eval(tmp_path):
    make_model().save_pretrained(tmp_path)
    model = lp.load_frozen_model(tmp_path, device="cpu")
    assert not model.training
    assert model.config._attn_implementation == "sdpa"


# --------------------------------------------------------------------------
# Step 13 end to end on a synthetic tokenized dataset
# --------------------------------------------------------------------------
# Days before the index date of the synthetic events (0 = on the index date).
EVENT_OFFSET_DAYS = [0, 1, 50, 365, 366, 500]


def _tokenized_dataset_dict(people: dict[int, dt.date], prompt_offset_days=None, offsets_by_person=None):
    """Synthetic tokenized dataset; with ``prompt_offset_days`` each sequence starts with the four
    demographic prompt tokens dated that many days before the index date (age -1)."""
    rng = np.random.default_rng(3)
    rows = []
    for pid, index_date in people.items():
        index_ts = dt.datetime(index_date.year, index_date.month, index_date.day, tzinfo=dt.timezone.utc).timestamp()
        offsets = sorted((offsets_by_person or {}).get(pid, EVENT_OFFSET_DAYS), reverse=True)
        epoch = [index_ts - d * 86400 for d in offsets]
        n = len(epoch)
        orders = list(range(1, n + 1))
        row = {
            "person_id": pid,
            # concept_ids carries the offset so the test can see which events survived.
            "concept_ids": [str(d) for d in offsets],
            "input_ids": rng.integers(len(SPECIAL), len(VOCAB), size=n).tolist(),
            "epoch_times": epoch,
            "ages": [50 + o for o in orders],
            "dates": [100 * o for o in orders],
            "visit_concept_orders": orders,
            "concept_values": [0.0] * n,
            "concept_value_masks": [0] * n,
            "visit_segments": [1] * n,
        }
        if prompt_offset_days is not None:
            prompt = {
                "concept_ids": ["year:2015", "age:76", "8507", "8516"],
                "input_ids": [VOCAB["100"], VOCAB["101"], VOCAB["102"], VOCAB["103"]],
                "epoch_times": [index_ts - prompt_offset_days * 86400] * 4,
                "ages": [-1] * 4,
                "dates": [0] * 4,
                "visit_concept_orders": [0] * 4,
                "concept_values": [0.0] * 4,
                "concept_value_masks": [0] * 4,
                "visit_segments": [0] * 4,
            }
            row.update({k: prompt[k] + row[k] for k in prompt})
        rows.append(row)
    half = len(rows) // 2
    return datasets.DatasetDict(
        {"train": datasets.Dataset.from_list(rows[:half]), "validation": datasets.Dataset.from_list(rows[half:])}
    )


def test_shift_index_dates_for_exclusive_end():
    df = pl.DataFrame({"person_id": [1], "index_date": [dt.date(2020, 3, 1)], "label": [0]})
    out = lp.shift_index_dates_for_exclusive_end(df)
    assert out["index_date"].dtype == pl.Datetime("us")
    assert out["index_date"][0] == dt.datetime(2020, 2, 29, 23, 59, 59)
    assert out["person_id"].to_list() == [1] and out["label"].to_list() == [0]


@pytest.mark.parametrize("window_days, kept", [(365, [365, 50, 1]), (-1, [500, 366, 365, 50, 1])])
def test_window_excludes_index_day_and_matches_step11_semantics(tmp_path, window_days, kept):
    index = dt.date(2020, 1, 1)
    base = tmp_path / "pretrain_prepared" / "hash"
    _tokenized_dataset_dict({1: index, 2: index}).save_to_disk(str(base))
    split = pl.DataFrame(
        {"person_id": [1, 2], "index_date": [index] * 2, "label": [0, 1]},
        schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64},
    )
    config = tmp_path / "ft.yaml"
    config.write_text(
        yaml.safe_dump({"data_folder": "x", "dataset_prepared_path": "x", "preprocessing_num_workers": 1}),
        encoding="utf-8",
    )
    ds = lp.build_split_dataset(split, tmp_path / "work", config, base, window_days)
    # classical_ml step 01 window: [index - window, index), i.e. offsets 1..window days (all history: > 0).
    expected = sorted(kept, reverse=True)
    for row in ds:
        assert row["concept_ids"] == [str(d) for d in expected]
    assert len(ds) == 2


def test_step15_end_to_end(step, tmp_path):
    index = dt.date(2020, 1, 1)
    splits = pl.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "index_date": [index] * 6,
            "label": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "val", "val", "test", "test"],
        },
        schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64, "split": pl.Utf8},
    )
    splits_file = tmp_path / "splits.parquet"
    splits.write_parquet(splits_file)

    tokenized = tmp_path / "pretrain_prepared" / "hash"
    _tokenized_dataset_dict({p: index for p in range(1, 7)}).save_to_disk(str(tokenized))

    model_dir = tmp_path / "model"
    make_model().save_pretrained(model_dir)
    make_tokenizer().save_pretrained(model_dir)

    config = tmp_path / "finetune.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data_folder": "unused",
                "dataset_prepared_path": "unused",
                "preprocessing_num_workers": 1,
                "preprocessing_batch_size": 4,
                "observation_window": 365,
                "is_data_in_meds": False,
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "features"
    mod = step("cehrbert_model/13_linear_probe_features.py")
    rc = mod.main(
        [
            "--splits-file", str(splits_file),
            "--tokenized-dataset", str(tmp_path / "pretrain_prepared"),
            "--finetune-config", str(config),
            "--model-dir", str(model_dir),
            "--output-dir", str(out),
            "--report-dir", str(tmp_path / "report"),
            "--ctx", "8",
            "--batch-size", "2",
            "--device", "cpu",
        ]
    )  # fmt: skip
    assert rc == 0
    status = json.loads((tmp_path / "report" / "status.json").read_text())
    assert status["status"] == "pass"

    for split, pids in {"train": [1, 2], "val": [3, 4], "test": [5, 6]}.items():
        meta = pd.read_parquet(out / split / "meta.parquet")
        assert sorted(meta["person_id"]) == pids
        assert list(meta.columns) == [
            "person_id",
            "index_date",
            "label",
            "n_tokens_before_truncation",
            "truncated",
            "n_event_tokens",
            "demographic_prompt_tokens",
        ]
        assert meta["n_event_tokens"].tolist() == [3, 3]
        assert (meta["demographic_prompt_tokens"] == 0).all()
        # Events at 1, 50 and 365 days before the index date are in the window; day 0 and 366 are not.
        assert meta["n_tokens_before_truncation"].tolist() == [3, 3]
        assert not meta["truncated"].any()
        assert (meta["index_date"] == pd.Timestamp(index)).all()
        for kind in lp.FEATURE_KINDS:
            X = np.load(out / split / f"features_{kind}.npy")
            assert X.shape == (2, HIDDEN) and X.dtype == np.float32
    assert not (out / "_work").exists()


def _run_step15_with_prompt(step, tmp_path, offsets_by_person, ctx, batch_size, keep=True, prompt_offset_days=2000):
    index = dt.date(2020, 1, 1)
    splits = pl.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "index_date": [index] * 6,
            "label": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "val", "val", "test", "test"],
        },
        schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64, "split": pl.Utf8},
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    splits_file = tmp_path / "splits.parquet"
    splits.write_parquet(splits_file)
    tokenized = tmp_path / "pretrain_prepared" / "hash"
    # By default the prompt is dated 2000 days before the index date, far outside the 365 day window.
    _tokenized_dataset_dict(
        {p: index for p in range(1, 7)}, prompt_offset_days=prompt_offset_days, offsets_by_person=offsets_by_person
    ).save_to_disk(str(tokenized))
    model_dir = tmp_path / "model"
    make_model().save_pretrained(model_dir)
    make_tokenizer().save_pretrained(model_dir)
    config = tmp_path / "finetune.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data_folder": "unused",
                "dataset_prepared_path": "unused",
                "preprocessing_num_workers": 1,
                "preprocessing_batch_size": 4,
                "observation_window": 365,
                "is_data_in_meds": False,
                "keep_demographic_prompt": keep,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "features"
    mod = step("cehrbert_model/13_linear_probe_features.py")
    rc = mod.main(
        [
            "--splits-file", str(splits_file),
            "--tokenized-dataset", str(tmp_path / "pretrain_prepared"),
            "--finetune-config", str(config),
            "--model-dir", str(model_dir),
            "--output-dir", str(out),
            "--report-dir", str(tmp_path / "report"),
            "--ctx", str(ctx),
            "--batch-size", str(batch_size),
            "--device", "cpu",
        ]
    )  # fmt: skip
    assert rc == 0
    return out


# Person 3 has no event inside the 365 day window; the others have events at 1, 50 and 365 days.
PROMPT_OFFSETS = {3: [366, 500, 0]}


def test_step15_keeps_demographic_prompt_and_counts_empty_windows_without_it(step, tmp_path):
    out = _run_step15_with_prompt(step, tmp_path, PROMPT_OFFSETS, ctx=9, batch_size=2)
    status = json.loads((tmp_path / "report" / "status.json").read_text())
    assert status["status"] == "pass"
    val = pd.read_parquet(out / "val" / "meta.parquet").set_index("person_id")
    assert val.loc[3, "n_event_tokens"] == 0 and val.loc[3, "n_tokens_before_truncation"] == 4
    assert val.loc[4, "n_event_tokens"] == 3 and val.loc[4, "n_tokens_before_truncation"] == 7
    assert val["demographic_prompt_tokens"].tolist() == [4, 4]
    assert not val["truncated"].any()
    assert (val["n_event_tokens"] == 0).sum() == 1


def test_step15_prompt_features_are_batch_size_invariant(step, tmp_path):
    outs = {bs: _run_step15_with_prompt(step, tmp_path / f"bs{bs}", PROMPT_OFFSETS, 9, bs) for bs in (1, 6)}
    for split in ("train", "val", "test"):
        for kind in lp.FEATURE_KINDS:
            a = np.load(outs[1] / split / f"features_{kind}.npy")
            b = np.load(outs[6] / split / f"features_{kind}.npy")
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


def test_step15_prompt_survives_truncation(step, tmp_path):
    # ctx 6 keeps 5 record tokens: the prompt plus the first [VS]-aligned tail token or none.
    out = _run_step15_with_prompt(step, tmp_path, PROMPT_OFFSETS, ctx=6, batch_size=2)
    val = pd.read_parquet(out / "val" / "meta.parquet").set_index("person_id")
    assert val.loc[4, "truncated"] and val.loc[4, "demographic_prompt_tokens"] == 4


def test_step15_post_index_prompt_keeps_only_gender_and_race(step, tmp_path):
    # The prompt is dated 30 days after the index date: year and age are dropped, gender and race kept.
    out = _run_step15_with_prompt(step, tmp_path, PROMPT_OFFSETS, ctx=9, batch_size=2, prompt_offset_days=-30)
    status = json.loads((tmp_path / "report" / "status.json").read_text())
    assert status["status"] == "pass"
    assert status["metrics"]["val_prompt_tokens_2"] == 2
    assert status["metrics"]["val_prompt_tokens_0"] == 0 and status["metrics"]["val_prompt_tokens_4"] == 0
    val = pd.read_parquet(out / "val" / "meta.parquet").set_index("person_id")
    assert val.loc[3, "n_event_tokens"] == 0 and val.loc[3, "n_tokens_before_truncation"] == 2
    assert val.loc[4, "n_event_tokens"] == 3 and val.loc[4, "n_tokens_before_truncation"] == 5
    assert val["demographic_prompt_tokens"].tolist() == [2, 2]
    # Truncation keeps the two prompt tokens as well.
    out = _run_step15_with_prompt(step, tmp_path / "t", PROMPT_OFFSETS, ctx=5, batch_size=2, prompt_offset_days=-30)
    val = pd.read_parquet(out / "val" / "meta.parquet").set_index("person_id")
    assert val.loc[4, "truncated"] and val.loc[4, "demographic_prompt_tokens"] == 2


def test_step15_post_index_prompt_is_batch_size_invariant(step, tmp_path):
    outs = {
        bs: _run_step15_with_prompt(step, tmp_path / f"bs{bs}", PROMPT_OFFSETS, 9, bs, prompt_offset_days=-30)
        for bs in (1, 6)
    }
    for split in ("train", "val", "test"):
        for kind in lp.FEATURE_KINDS:
            a = np.load(outs[1] / split / f"features_{kind}.npy")
            b = np.load(outs[6] / split / f"features_{kind}.npy")
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


def test_step15_without_flag_drops_the_prompt(step, tmp_path):
    out = _run_step15_with_prompt(step, tmp_path, PROMPT_OFFSETS, ctx=9, batch_size=2, keep=False)
    val = pd.read_parquet(out / "val" / "meta.parquet").set_index("person_id")
    assert val.loc[4, "n_tokens_before_truncation"] == 3
    assert val.loc[3, "n_tokens_before_truncation"] == 0
    assert (val["demographic_prompt_tokens"] == 0).all()


# --------------------------------------------------------------------------
# Step 14
# --------------------------------------------------------------------------
def _write_features(root, n_train=120, n_val=40, n_test=40, dim=6, seed=0, shuffle_test_labels=False):
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        y = np.arange(n) % 3 == 0
        X = rng.normal(size=(n, dim)).astype(np.float32)
        X[:, 0] += 1.2 * y
        # The mean kind is pure noise so the two kinds differ in validation AUROC.
        noise = rng.normal(size=(n, dim)).astype(np.float32)
        labels = y.astype(np.int64)
        if split == "test" and shuffle_test_labels:
            labels = np.random.default_rng(99).permutation(labels)
        meta = pd.DataFrame(
            {
                "person_id": np.arange(n) + {"train": 0, "val": 10_000, "test": 20_000}[split],
                "index_date": pd.Timestamp("2020-01-01"),
                "label": labels,
                "n_tokens_before_truncation": 10,
                "truncated": False,
            }
        )
        d = root / split
        d.mkdir(parents=True)
        np.save(d / "features_cls.npy", X)
        np.save(d / "features_mean.npy", noise)
        meta.to_parquet(d / "meta.parquet", index=False)


def _run_step16(step, features, out, report):
    mod = step("cehrbert_model/14_train_linear_probe.py")
    rc = mod.main(
        [
            "--features-dir", str(features),
            "--output-dir", str(out),
            "--report-dir", str(report),
            "--cv-folds", "2",
            "--n-jobs", "1",
            "--skip-evaluate",
        ]
    )  # fmt: skip
    assert rc == 0


def test_step16_smoke_outputs_and_schema(step, tmp_path):
    features = tmp_path / "features"
    _write_features(features)
    out = tmp_path / "probe"
    _run_step16(step, features, out, tmp_path / "report")

    summary = json.loads((out / "summary.json").read_text())
    assert summary["primary_kind"] == "cls"
    assert summary["secondary_kind"] == "mean"
    assert summary["kinds"]["cls"]["val_auroc"] > summary["kinds"]["mean"]["val_auroc"]

    for kind in lp.FEATURE_KINDS:
        assert (out / kind / "model.joblib").is_file()
        params = json.loads((out / kind / "best_params.json").read_text())
        assert set(params["best_params"]) == {"clf__C", "clf__class_weight"}
        for sub, n in (("test_predictions", 40), ("validation_predictions", 40)):
            df = pd.read_parquet(out / kind / sub)
            assert list(df.columns) == [
                "subject_id",
                "prediction_time",
                "predicted_boolean_probability",
                "predicted_boolean_value",
                "boolean_value",
            ]
            assert len(df) == n and df["subject_id"].is_unique
            assert df["predicted_boolean_value"].isna().all()
            assert df["boolean_value"].dtype == bool
            assert df["predicted_boolean_probability"].between(0, 1).all()
            y_true, y_prob = ev.load_predictions(out / kind / sub)
            assert len(y_true) == n and len(y_prob) == n
    test_meta = pd.read_parquet(features / "test" / "meta.parquet")
    pred = pd.read_parquet(out / "cls" / "test_predictions")
    assert sorted(pred["subject_id"]) == sorted(test_meta["person_id"])


def test_step16_tuning_never_reads_test(step, tmp_path):
    runs = {}
    for name, shuffled in (("plain", False), ("shuffled", True)):
        features = tmp_path / name / "features"
        _write_features(features, shuffle_test_labels=shuffled)
        out = tmp_path / name / "probe"
        _run_step16(step, features, out, tmp_path / name / "report")
        runs[name] = out
    for kind in lp.FEATURE_KINDS:
        a = json.loads((runs["plain"] / kind / "best_params.json").read_text())
        b = json.loads((runs["shuffled"] / kind / "best_params.json").read_text())
        assert a == b
        val_a = pd.read_parquet(runs["plain"] / kind / "validation_predictions")
        val_b = pd.read_parquet(runs["shuffled"] / kind / "validation_predictions")
        pd.testing.assert_frame_equal(val_a, val_b)
    summary_a = json.loads((runs["plain"] / "summary.json").read_text())
    summary_b = json.loads((runs["shuffled"] / "summary.json").read_text())
    assert summary_a["primary_kind"] == summary_b["primary_kind"]


def test_step16_fails_without_features(step, tmp_path):
    mod = step("cehrbert_model/14_train_linear_probe.py")
    rc = mod.main(
        [
            "--features-dir", str(tmp_path / "missing"),
            "--output-dir", str(tmp_path / "out"),
            "--report-dir", str(tmp_path / "report"),
            "--skip-evaluate",
        ]
    )  # fmt: skip
    assert rc != 0


def test_choose_primary_kind_prefers_higher_validation_auroc():
    assert lp.choose_primary_kind({"cls": 0.7, "mean": 0.8}) == "mean"
    assert lp.choose_primary_kind({"cls": 0.8, "mean": 0.7}) == "cls"
    assert lp.choose_primary_kind({"cls": 0.7, "mean": 0.7}) == "cls"


def test_search_grid_matches_plan():
    search = lp.build_search(cv_folds=5, seed=42, n_jobs=1)
    np.testing.assert_allclose(search.param_grid["clf__C"], np.logspace(-4, 2, 7))
    assert search.param_grid["clf__class_weight"] == [None, "balanced"]
    assert search.scoring == "roc_auc"
    assert search.cv.n_splits == 5 and search.cv.shuffle and search.cv.random_state == 42


def test_evaluate_command_pins_arguments(tmp_path):
    cmd = lp.evaluate_command(tmp_path / "cls", tmp_path / "rep")
    assert cmd[cmd.index("--n-bootstrap") + 1] == "2000"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert cmd[cmd.index("--cal-bins") + 1] == "10"
    assert cmd[cmd.index("--val-predictions") + 1].endswith("validation_predictions")


def test_probe_paths_use_run_label():
    label = paths.run_label(365, 512)
    assert paths.linear_probe_features_dir(label).as_posix().endswith("build/w_365_ctx512/linear_probe/features")
