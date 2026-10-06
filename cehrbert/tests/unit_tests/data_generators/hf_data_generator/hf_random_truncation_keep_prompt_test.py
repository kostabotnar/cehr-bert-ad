import random
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import TruncationType

VS, VE, CLS, PAD, MASK = 2, 3, 4, 0, 1
PROMPT = [100, 101, 102, 103]
RANDOM_TYPES = (
    TruncationType.RANDOM_RIGHT_TRUNCATION,
    TruncationType.RANDOM_TRUNCATION,
    TruncationType.RANDOM_COMPLETE,
)


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.convert_token_to_id.side_effect = lambda token: {"VS": VS, "VE": VE}.get(token, 99)
    tokenizer.oov_token_index = 99
    tokenizer.pad_token_index = PAD
    tokenizer.cls_token_index = CLS
    tokenizer.mask_token_index = MASK
    tokenizer.vocab_size = 1000
    return tokenizer


def _record(n_visits=30, prompt_length=4):
    """Prompt tokens followed by visits [VS] a b [VE]; every column encodes the source position."""
    events = []
    for v in range(n_visits):
        events += [VS, 200 + 2 * v, 201 + 2 * v, VE]
    input_ids = PROMPT[4 - prompt_length :] + events
    n = len(input_ids)
    return {
        "input_ids": input_ids,
        "ages": [-1] * prompt_length + [70] * (n - prompt_length),
        "dates": list(range(n)),
        "visit_concept_orders": list(range(n)),
        "concept_values": [float(i) for i in range(n)],
        "concept_value_masks": [i % 2 for i in range(n)],
        "visit_segments": [i % 3 for i in range(n)],
        "labels": [1000 + i for i in range(n)],
        "person_id": 7,
    }


def _old_random_right_truncation(record, new_max_length, vs_token_id):
    """The truncation of RANDOM_RIGHT_TRUNCATION before keep_demographic_prompt applied to it."""
    seq_length = len(record["input_ids"])
    starting_points = [i for i in range(seq_length - new_max_length) if record["input_ids"][i] == vs_token_id]
    if starting_points:
        start_index = random.choice(starting_points)
    else:
        start_index = next(
            (i for i in range(seq_length - new_max_length, seq_length) if record["input_ids"][i] == vs_token_id),
            seq_length - new_max_length,
        )
    return start_index, min(start_index + new_max_length, seq_length)


class TestRandomTruncationKeepsPrompt(unittest.TestCase):
    max_length = 21

    def _collator(self, truncate_type, keep):
        return CehrBertDataCollator(
            tokenizer=_tokenizer(),
            max_length=self.max_length,
            is_pretraining=True,
            truncate_type=truncate_type,
            keep_demographic_prompt=keep,
        )

    def _assert_prompt_and_contiguous_slice(self, record, result, prompt_length, truncate_type):
        ids = result["input_ids"]
        self.assertLessEqual(len(ids), self.max_length - 1)
        self.assertEqual(ids[:prompt_length], record["input_ids"][:prompt_length])
        # The positions of the kept tokens in the source sequence: the prompt, then one contiguous run.
        positions = result["dates"]
        self.assertEqual(positions[:prompt_length], list(range(prompt_length)))
        body = positions[prompt_length:]
        self.assertGreater(len(body), 0)
        self.assertGreaterEqual(body[0], prompt_length)
        self.assertEqual(body, list(range(body[0], body[0] + len(body))))
        for column in ("input_ids", "ages", "visit_concept_orders", "concept_values", "labels"):
            self.assertEqual(result[column], [record[column][p] for p in positions], column)
        self.assertEqual(result["person_id"], 7)
        if truncate_type != TruncationType.RANDOM_TRUNCATION:
            self.assertEqual(ids[prompt_length], VS)

    def test_prompt_survives_every_random_truncation_type(self):
        record = _record()
        for truncate_type in RANDOM_TYPES:
            collator = self._collator(truncate_type, keep=True)
            starts = set()
            for seed in range(40):
                random.seed(seed)
                result = collator.generate_start_end_index(dict(record))
                self._assert_prompt_and_contiguous_slice(record, result, 4, truncate_type)
                starts.add(result["dates"][4])
            # The slice really is random, not pinned to the start or the end.
            self.assertGreater(len(starts), 3, truncate_type)

    def test_slice_uses_the_remaining_budget(self):
        record = _record()
        random.seed(0)
        result = self._collator(TruncationType.RANDOM_TRUNCATION, keep=True).generate_start_end_index(record)
        self.assertEqual(len(result["input_ids"]), self.max_length - 1)

    def test_two_token_prompt(self):
        record = _record(prompt_length=2)
        collator = self._collator(TruncationType.RANDOM_RIGHT_TRUNCATION, keep=True)
        for seed in range(20):
            random.seed(seed)
            result = collator.generate_start_end_index(dict(record))
            self._assert_prompt_and_contiguous_slice(record, result, 2, TruncationType.RANDOM_RIGHT_TRUNCATION)
            self.assertEqual(result["ages"][2], 70)

    def test_numpy_and_tensor_columns_stay_aligned(self):
        record = _record()
        collator = self._collator(TruncationType.RANDOM_RIGHT_TRUNCATION, keep=True)
        for convert in (np.asarray, torch.as_tensor):
            converted = {k: (convert(v) if isinstance(v, list) else v) for k, v in record.items()}
            random.seed(3)
            result = collator.generate_start_end_index(converted)
            self.assertEqual([int(x) for x in result["input_ids"][:4]], PROMPT)
            self.assertEqual(len(result["ages"]), len(result["input_ids"]))
            self.assertEqual(len(result["dates"]), len(result["input_ids"]))
            positions = [int(p) for p in result["dates"]]
            self.assertEqual([int(x) for x in result["input_ids"]], [record["input_ids"][p] for p in positions])

    def test_short_sequence_unchanged(self):
        record = _record(n_visits=3)
        result = self._collator(TruncationType.RANDOM_RIGHT_TRUNCATION, keep=True).generate_start_end_index(record)
        self.assertIs(result, record)

    def test_flag_off_matches_the_old_behavior(self):
        record = _record()
        new_max_length = self.max_length - 1
        collator = self._collator(TruncationType.RANDOM_RIGHT_TRUNCATION, keep=False)
        dropped_prompt = 0
        for seed in range(40):
            random.seed(seed)
            start, end = _old_random_right_truncation(record, new_max_length, VS)
            random.seed(seed)
            result = collator.generate_start_end_index(dict(record))
            self.assertEqual(result["dates"], list(range(start, end)))
            self.assertEqual(result["input_ids"], record["input_ids"][start:end])
            dropped_prompt += int(result["input_ids"][:4] != PROMPT)
        self.assertGreater(dropped_prompt, 0)

    def test_flag_off_random_truncation_draws_the_same_start(self):
        record = _record()
        new_max_length = self.max_length - 1
        collator = self._collator(TruncationType.RANDOM_TRUNCATION, keep=False)
        for seed in range(20):
            random.seed(seed)
            start = random.randint(0, len(record["input_ids"]) - new_max_length)
            random.seed(seed)
            result = collator.generate_start_end_index(dict(record))
            self.assertEqual(result["dates"], list(range(start, start + new_max_length)))

    def test_batch_has_prompt_right_after_cls(self):
        records = [_record(), _record(n_visits=12), _record(prompt_length=2)]
        collator = self._collator(TruncationType.RANDOM_RIGHT_TRUNCATION, keep=True)
        torch.manual_seed(0)
        random.seed(0)
        batch = collator(records)
        self.assertTrue(bool((batch["input_ids"][:, 0] == CLS).all()))
        self.assertLessEqual(batch["input_ids"].shape[1], self.max_length)
        # MLM masking may change input_ids, but ages and dates are untouched.
        self.assertTrue(bool((batch["ages"][:2, 1:5] == -1).all()))
        self.assertTrue(bool((batch["ages"][2, 1:3] == -1).all()))
        self.assertEqual(int(batch["ages"][2, 3]), 70)
        self.assertEqual(batch["dates"][0, 1:5].tolist(), [0, 1, 2, 3])
        # Every unmasked label position still matches its date-encoded source position.
        for row, record in enumerate(records):
            n = int(batch["attention_mask"][row].sum().item()) - 1
            positions = batch["dates"][row, 1 : n + 1].tolist()
            labels = batch["labels"][row, 1 : n + 1].tolist()
            for p, label in zip(positions, labels):
                self.assertIn(label, (-100, record["labels"][p]))


if __name__ == "__main__":
    unittest.main()
