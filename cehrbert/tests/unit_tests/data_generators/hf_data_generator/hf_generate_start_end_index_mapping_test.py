import random
import unittest
from unittest.mock import MagicMock

from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator

# Seed the random number generator for reproducibility in tests
random.seed(42)


class TestGenerateStartEndIndexMapping(unittest.TestCase):
    def setUp(self):
        # Initialize with a fixed sequence length for consistent testing
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab_size = 100
        self.mock_tokenizer.mask_token_index = 1
        self.mock_tokenizer.unused_token_index = 99
        self.mock_tokenizer.encode.return_value = [10, 20, 30]  # Example token IDs
        self.mock_tokenizer.convert_token_to_id.side_effect = [2, 3, 17, 18]
        self.mock_tokenizer.convert_id_to_token.side_effect = [
            "year:2000",
            "age:20-30",
        ]
        self.data_collator = CehrBertDataCollator(tokenizer=self.mock_tokenizer, max_length=10)

    def test_long_sequence(self):
        # Test with a sequence longer than max_sequence_length
        record = {"input_ids": [2, 4, 3, 5, 2, 6, 7, 8, 9, 10, 3, 11, 2, 12, 3, 13, 2, 14, 3]}
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result["input_ids"], [2, 4, 3, 5, 2, 6, 7, 8, 9])

    def test_short_sequence(self):
        # Test with a sequence shorter than max_sequence_length
        record = {"input_ids": list(range(5))}  # Shorter than max_sequence_length
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result["input_ids"], list(range(5)))

    def test_edge_case_sequence_length_equal_to_max(self):
        # Test with a sequence exactly equal to max_sequence_length
        record = {"input_ids": list(range(9))}  # Exactly max_sequence_length - 1
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result["input_ids"], list(range(9)))

    def test_tail_case_sequence_length_equal_to_max(self):
        from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import TruncationType

        # Test with a sequence exactly equal to max_sequence_length
        default_val = self.data_collator.truncate_type
        self.data_collator.truncate_type = TruncationType.TAIL
        record = {
            "input_ids": [13, 14, 15, 16] + list(range(2, 8)),  # Exactly max_sequence_length - 1,
            "dates": [0, 0, 0, 0] + list(range(2052, 2058)),
        }
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result["input_ids"], [2, 3, 4, 5, 6, 7])
        self.assertListEqual(result["dates"], [2052, 2053, 2054, 2055, 2056, 2057])
        self.data_collator.truncate_type = default_val


class TestKeepDemographicPromptTruncation(unittest.TestCase):
    def setUp(self):
        from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import TruncationType

        tokenizer = MagicMock()
        tokenizer.convert_token_to_id.side_effect = [2, 3]  # [VS], [VE]
        self.collator = CehrBertDataCollator(
            tokenizer=tokenizer, max_length=10, is_pretraining=False, keep_demographic_prompt=True
        )
        self.assertEqual(self.collator.truncate_type, TruncationType.TAIL)
        # Four prompt tokens, then ten visits of ([VS], event); ids are unique per position.
        self.record = {
            "input_ids": list(range(100, 104)) + [2 if i % 2 == 0 else 50 + i for i in range(20)],
            "ages": [-1] * 4 + [70] * 20,
        }

    def test_prompt_survives_tail_truncation(self):
        result = self.collator.generate_start_end_index(self.record)
        # max_length 10 leaves 9 tokens after [CLS]: the 4 prompt tokens plus the most recent 5,
        # starting at a [VS] token.
        self.assertEqual(result["input_ids"][:4], [100, 101, 102, 103])
        self.assertEqual(result["input_ids"][4], 2)
        self.assertEqual(result["input_ids"][-1], self.record["input_ids"][-1])
        self.assertLessEqual(len(result["input_ids"]), 9)
        self.assertEqual(result["ages"][:4], [-1] * 4)
        self.assertEqual(len(result["ages"]), len(result["input_ids"]))

    def test_two_token_prompt_survives_tail_truncation(self):
        record = {"input_ids": self.record["input_ids"][2:], "ages": self.record["ages"][2:]}
        result = self.collator.generate_start_end_index(record)
        # 9 tokens after [CLS]: 2 prompt tokens plus at most 7 recent ones, from a [VS] token.
        self.assertEqual(result["input_ids"][:2], [102, 103])
        self.assertEqual(result["input_ids"][2], 2)
        self.assertEqual(result["input_ids"][-1], record["input_ids"][-1])
        self.assertLessEqual(len(result["input_ids"]), 9)
        self.assertEqual(result["ages"][:3], [-1, -1, 70])

    def test_without_option_prompt_is_dropped(self):
        self.collator.keep_demographic_prompt = False
        result = self.collator.generate_start_end_index(self.record)
        self.assertNotIn(100, result["input_ids"])

    def test_short_sequence_is_returned_unchanged(self):
        record = {"input_ids": list(range(100, 106)), "ages": [-1] * 4 + [70] * 2}
        self.assertEqual(self.collator.generate_start_end_index(record)["input_ids"], record["input_ids"])

    def test_sequence_without_prompt_is_truncated_as_before(self):
        record = {"input_ids": self.record["input_ids"], "ages": [70] * 24}
        result = self.collator.generate_start_end_index(record)
        self.assertNotIn(100, result["input_ids"])

    def test_numpy_and_tensor_columns(self):
        import numpy as np
        import torch

        for convert in (np.asarray, torch.as_tensor):
            record = {k: convert(v) for k, v in self.record.items()}
            result = self.collator.generate_start_end_index(record)
            self.assertEqual([int(x) for x in result["input_ids"][:4]], [100, 101, 102, 103])
            self.assertEqual(len(result["ages"]), len(result["input_ids"]))


if __name__ == "__main__":
    unittest.main()
