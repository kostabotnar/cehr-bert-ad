import importlib
import sys
import unittest
from unittest import mock

MODULES = (
    "cehrbert.data_generators.hf_data_generator.meds_utils",
    "cehrbert.data_generators.hf_data_generator.patient_block",
)


class MedsReaderOptionalTest(unittest.TestCase):
    def tearDown(self):
        for name in MODULES:
            sys.modules.pop(name, None)

    def test_modules_import_without_meds_reader(self):
        with mock.patch.dict(sys.modules, {"meds_reader": None}):
            for name in MODULES:
                sys.modules.pop(name, None)
                importlib.import_module(name)

    def test_meds_entry_points_raise_helpful_import_error(self):
        with mock.patch.dict(sys.modules, {"meds_reader": None}):
            sys.modules.pop(MODULES[0], None)
            meds_utils = importlib.import_module(MODULES[0])
            with self.assertRaisesRegex(ImportError, "is_data_in_meds: true"):
                meds_utils.get_subject_split("/nonexistent")
            with self.assertRaisesRegex(ImportError, "is_data_in_meds: true"):
                meds_utils.create_dataset_from_meds_reader(mock.MagicMock())


if __name__ == "__main__":
    unittest.main()
