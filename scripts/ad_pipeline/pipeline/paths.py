"""Canonical filesystem locations for the AD pipeline.

All paths are derived from ``PROJECT_ROOT`` so the scripts work regardless of the
current working directory. ``PROJECT_ROOT`` is the repository root
(``.../cehr-bert``), two levels above ``scripts/ad_pipeline``.
"""

from __future__ import annotations

from pathlib import Path

# .../cehr-bert/scripts/ad_pipeline/pipeline/paths.py
#   parents[0] = pipeline
#   parents[1] = ad_pipeline
#   parents[2] = scripts
#   parents[3] = cehr-bert  (PROJECT_ROOT)
AD_PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --- pipeline assets -------------------------------------------------------
CONFIG_DIR = AD_PIPELINE_DIR / "configs"
PRETRAIN_CONFIG = CONFIG_DIR / "ad_pretrain_config.yaml"
FINETUNE_CONFIG = CONFIG_DIR / "ad_finetune_config.yaml"
REPORTS_DIR = AD_PIPELINE_DIR / "reports"

# --- data tree -------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
OMOP_DIR = DATA_DIR / "omop_data"
CEHRBERT_DATA_DIR = DATA_DIR / "cehrbert_data"
PATIENT_SEQUENCE_DIR = CEHRBERT_DATA_DIR / "patient_sequence"

PRETRAIN_PREPARED_DIR = DATA_DIR / "pretrain_prepared"
PRETRAIN_RESULTS_DIR = DATA_DIR / "pretrain_results"

FINETUNE_PREPARED_DIR = DATA_DIR / "finetune_prepared"
FINETUNE_RESULTS_DIR = DATA_DIR / "finetune_results"
PREDICT_PREPARED_DIR = DATA_DIR / "predict_prepared"

COHORT_DIR = OMOP_DIR / "ad_cohort"
# The labeled cohort is split up front (step 6) into a fine-tuning set and a held-out
# test set, each in its own folder (cohort_folder reads *.parquet from a single folder).
COHORT_FINETUNE_DIR = COHORT_DIR / "finetune"
COHORT_TEST_DIR = COHORT_DIR / "test"
COHORT_FINETUNE_FILE = COHORT_FINETUNE_DIR / "cohort.parquet"
COHORT_TEST_FILE = COHORT_TEST_DIR / "cohort.parquet"


def report_dir_for(step_name: str) -> Path:
    """Default report directory for a step, e.g. ``reports/01_validate_dataset``."""
    return REPORTS_DIR / step_name


def resolve_under_root(path: str | Path) -> Path:
    """Resolve a (possibly relative) config path against ``PROJECT_ROOT``.

    The cehrbert runners are invoked with the project root as the working
    directory, so config paths like ``data/pretrain_results`` are relative to it.
    """
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
