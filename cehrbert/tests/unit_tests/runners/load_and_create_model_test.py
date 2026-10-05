import tempfile
import unittest
from types import SimpleNamespace

from cehrbert.models.hf_models.config import CehrBertConfig
from cehrbert.runners.hf_cehrbert_pretrain_runner import load_and_create_model
from cehrbert.runners.hf_runner_argument_dataclass import CehrBertArguments, ModelArguments


class LoadAndCreateModelTest(unittest.TestCase):
    def _args(self, path):
        model_args = ModelArguments(
            model_name_or_path=path,
            tokenizer_name_or_path=path,
            hidden_size=16,
            num_hidden_layers=1,
            max_position_embeddings=32,
        )
        return CehrBertArguments(), model_args

    def test_directory_without_config_uses_tokenizer_vocab_size(self):
        tokenizer = SimpleNamespace(vocab_size=31802, lab_token_ids=[], cls_token_index=0)
        with tempfile.TemporaryDirectory() as tmp:
            cehrbert_args, model_args = self._args(tmp)
            model = load_and_create_model(cehrbert_args, model_args, tokenizer)
        self.assertEqual(model.config.vocab_size, 31802)

    def test_existing_config_is_loaded(self):
        tokenizer = SimpleNamespace(vocab_size=31802, lab_token_ids=[], cls_token_index=0)
        with tempfile.TemporaryDirectory() as tmp:
            CehrBertConfig(vocab_size=1234, hidden_size=16, num_hidden_layers=1).save_pretrained(tmp)
            cehrbert_args, model_args = self._args(tmp)
            model = load_and_create_model(cehrbert_args, model_args, tokenizer)
        self.assertEqual(model.config.vocab_size, 1234)


if __name__ == "__main__":
    unittest.main()
