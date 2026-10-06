import unittest
from unittest.mock import MagicMock

from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import SamplePackingCehrBertDataCollator
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import TruncationType
from cehrbert.runners.hf_cehrbert_pretrain_runner import create_pretraining_data_collator
from cehrbert.runners.hf_runner_argument_dataclass import CehrBertArguments, DataTrainingArguments, ModelArguments


class CreatePretrainingDataCollatorTest(unittest.TestCase):
    def _make(self, keep, sample_packing=False):
        tokenizer = MagicMock()
        tokenizer.convert_token_to_id.return_value = 2
        data_args = DataTrainingArguments(data_folder="x", dataset_prepared_path="y", keep_demographic_prompt=keep)
        model_args = ModelArguments(model_name_or_path="m", tokenizer_name_or_path="m", max_position_embeddings=64)
        cehrbert_args = CehrBertArguments(sample_packing=sample_packing, max_tokens_per_batch=256)
        return create_pretraining_data_collator(cehrbert_args, data_args, model_args, tokenizer, 0.15)

    def test_flag_is_passed_to_the_collator(self):
        collator = self._make(keep=True)
        self.assertTrue(collator.keep_demographic_prompt)
        self.assertTrue(collator.is_pretraining)
        self.assertEqual(collator.truncate_type, TruncationType.RANDOM_RIGHT_TRUNCATION)
        self.assertEqual(collator.max_length, 64)
        self.assertEqual(collator.mlm_probability, 0.15)

    def test_default_keeps_old_behavior(self):
        self.assertFalse(self._make(keep=False).keep_demographic_prompt)

    def test_sample_packing(self):
        collator = self._make(keep=True, sample_packing=True)
        self.assertIsInstance(collator, SamplePackingCehrBertDataCollator)
        self.assertEqual(collator.max_length, 256)
        self.assertEqual(collator.max_position_embeddings, 64)
        self.assertTrue(collator.keep_demographic_prompt)


if __name__ == "__main__":
    unittest.main()
