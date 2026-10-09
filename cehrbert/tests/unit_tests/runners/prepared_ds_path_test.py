import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cehrbert.runners.runner_util import generate_prepared_ds_path


def make_args(prepared_dir):
    data_args = SimpleNamespace(
        data_folder="unused",
        validation_split_percentage=None,
        test_eval_ratio=0.1,
        split_by_patient=True,
        chronological_split=False,
        keep_demographic_prompt=False,
        dataset_prepared_path=str(prepared_dir),
    )
    return data_args, SimpleNamespace(tokenizer_name_or_path="tok")


class TestPreparedDsPathCohortContents(unittest.TestCase):
    def test_path_changes_with_cohort_parquet_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp) / "cohort"
            cohort.mkdir()
            data_args, model_args = make_args(Path(tmp) / "prepared")
            (cohort / "cohort.parquet").write_bytes(b"version one")
            first = generate_prepared_ds_path(data_args, model_args, data_folder=str(cohort))
            again = generate_prepared_ds_path(data_args, model_args, data_folder=str(cohort))
            (cohort / "cohort.parquet").write_bytes(b"version two")
            second = generate_prepared_ds_path(data_args, model_args, data_folder=str(cohort))
            self.assertEqual(first, again)
            self.assertNotEqual(first, second)

    def test_path_without_override_ignores_folder_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_args, model_args = make_args(Path(tmp) / "prepared")
            data_args.data_folder = str(Path(tmp) / "pretrain")
            Path(data_args.data_folder).mkdir()
            before = generate_prepared_ds_path(data_args, model_args)
            (Path(data_args.data_folder) / "x.parquet").write_bytes(b"data")
            self.assertEqual(before, generate_prepared_ds_path(data_args, model_args))


if __name__ == "__main__":
    unittest.main()
