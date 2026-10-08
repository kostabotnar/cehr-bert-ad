"""Tests for pipeline/paths.py: part folders, report locations and the project root."""

from __future__ import annotations

import pytest

from pipeline import paths


def test_project_root_is_the_parent_of_scripts():
    assert paths.SCRIPTS_DIR.name == "scripts"
    assert paths.PROJECT_ROOT == paths.SCRIPTS_DIR.parent
    assert (paths.SCRIPTS_DIR / "pipeline" / "paths.py").is_file()


def test_every_part_has_a_folder_and_a_constant():
    constants = {
        "data_verification": paths.DATA_VERIFICATION_DIR,
        "cohort": paths.COHORT_PART_DIR,
        "data_qc": paths.DATA_QC_DIR,
        "cehrbert_model": paths.CEHRBERT_MODEL_DIR,
        "classical_ml": paths.CLASSICAL_ML_DIR,
        "evaluation": paths.EVALUATION_DIR,
        "feature_importance": paths.FEATURE_IMPORTANCE_DIR,
    }
    assert tuple(constants) == paths.PARTS
    for name, folder in constants.items():
        assert folder == paths.SCRIPTS_DIR / name
        assert folder.is_dir()


def test_config_paths_point_at_the_part_configs():
    assert paths.PRETRAIN_CONFIG.is_file()
    assert paths.FINETUNE_CONFIG.is_file()
    assert paths.OMOP_PROFILE_CONCEPTS.is_file()
    assert paths.PRETRAIN_CONFIG.parent == paths.CEHRBERT_MODEL_DIR / "configs"
    assert paths.OMOP_PROFILE_CONCEPTS.parent == paths.DATA_QC_DIR / "configs"


def test_report_dir_for_nests_under_the_part():
    got = paths.report_dir_for("cohort", "01_build_cohort")
    assert got == paths.SCRIPTS_DIR / "cohort" / "reports" / "01_build_cohort"


@pytest.mark.parametrize("part", paths.PARTS)
def test_report_dir_for_accepts_every_part(part):
    assert paths.report_dir_for(part, "00_check_environment").parts[-3:] == (part, "reports", "00_check_environment")


def test_report_dir_for_rejects_an_unknown_part():
    with pytest.raises(ValueError, match="unknown part"):
        paths.report_dir_for("ad_pipeline", "01_validate_dataset")


def test_run_label_report_location_is_unchanged():
    label = paths.run_label(365, 512)
    assert label == "w_365_ctx512"
    assert paths.build_report_dir_for("01_evaluate", label) == paths.BUILD_DIR / label / "reports" / "01_evaluate"
