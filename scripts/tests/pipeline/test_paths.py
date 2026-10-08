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


def test_labels():
    assert paths.window_label(365) == "w_365"
    assert paths.window_label(-1) == paths.window_label(None) == "w_all"
    assert paths.run_label(365, 512) == "w_365_ctx512"
    assert paths.run_label(None, 4096) == "w_all_ctx4096"


def test_each_part_writes_under_its_own_build_root():
    label, wlabel = paths.run_label(365, 512), paths.window_label(365)
    b = paths.BUILD_DIR
    assert paths.OMOP_PROFILE_DIR == b / "data_qc" / "omop_profile"
    assert paths.HISTORY_LENGTH_DIR == b / "data_qc" / "history_length"
    assert (
        paths.cehrbert_report_dir("12_token_stats", label)
        == b / "cehrbert_model" / label / "reports" / "12_token_stats"
    )
    assert paths.cehrbert_linear_probe_features_dir(label) == b / "cehrbert_model" / label / "linear_probe" / "features"
    assert paths.cehrbert_manifest_path(label) == b / "cehrbert_model" / label / "predictions_manifest.json"
    assert paths.classical_features_dir(wlabel) == b / "classical_ml" / "w_365" / "features"
    assert paths.classical_manifest_path(wlabel) == b / "classical_ml" / "w_365" / "predictions_manifest.json"
    assert paths.evaluation_model_dir("cmp", "lr") == b / "evaluation" / "cmp" / "models" / "lr"
    assert (
        paths.evaluation_report_dir("02_compare_models", "cmp")
        == b / "evaluation" / "cmp" / "reports" / "02_compare_models"
    )
    assert paths.feature_importance_report_dir("03_x", label) == b / "feature_importance" / label / "reports" / "03_x"
