"""Canonical filesystem locations for the AD pipeline.

All paths are derived from ``PROJECT_ROOT`` so the scripts work regardless of the
current working directory. ``PROJECT_ROOT`` is the repository root
(``.../cehr-bert``), two levels above ``scripts/ad_pipeline``.

Evaluation outputs (steps 10 / 10a) are nested per experiment under
``build/<run_label>/``, where the run label encodes the experiment parameters
(look-back window in days and CEHR-BERT token cap). The label format is
``w_<window>_ctx<ctx>`` (e.g. ``w_all_ctx4096``, ``w_365_ctx512``), with
``window`` being ``"all"`` for an unbounded look-back. Use ``run_label`` to build
the label and the label-aware ``*_dir`` functions below to resolve concrete
output locations under a given experiment folder.
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

# Prediction parquet folders (written by the finetune runner's do_predict).
TEST_PREDICTIONS_DIR = FINETUNE_RESULTS_DIR / "test_predictions"
VAL_PREDICTIONS_DIR = FINETUNE_RESULTS_DIR / "validation_predictions"

# Evaluation outputs (steps 10 / 10a) live under build/, nested per experiment in
# build/<run_label>/ (see run_label / the label-aware *_dir functions below).
BUILD_DIR = PROJECT_ROOT / "build"

# Defaults for the experiment parameters that make up a run label.
DEFAULT_WINDOW_DAYS = -1  # -1 or None => "all history" (unbounded look-back)
DEFAULT_CTX = 4096


def run_label(window_days, ctx) -> str:
    """Build a run label, e.g. ``'w_all_ctx4096'`` or ``'w_365_ctx512'``.

    ``window_days`` of ``None`` or negative encodes an unbounded ("all") look-back.
    """
    w = "all" if window_days is None or int(window_days) < 0 else str(int(window_days))
    return f"w_{w}_ctx{int(ctx)}"


def run_dir(label: str) -> Path:
    """Per-experiment output folder, e.g. ``build/w_all_ctx4096``."""
    return BUILD_DIR / label


def eval_results_dir(label: str) -> Path:
    """Evaluation results folder for a run (metrics.json / stats json live here)."""
    return run_dir(label)


def figures_dir(label: str) -> Path:
    """Figures folder for a run, e.g. ``build/w_all_ctx4096/figures``."""
    return run_dir(label) / "figures"


def build_reports_dir(label: str) -> Path:
    """Reports folder for a run, e.g. ``build/w_all_ctx4096/reports``."""
    return run_dir(label) / "reports"


def build_report_dir_for(step_name: str, label: str) -> Path:
    """Report directory for a step within a run, e.g. build/<label>/reports/10_evaluate."""
    return build_reports_dir(label) / step_name


def baselines_dir(label: str) -> Path:
    """Baselines folder for a run, e.g. ``build/w_all_ctx4096/baselines``."""
    return run_dir(label) / "baselines"


def features_dir(label: str) -> Path:
    """Baseline features folder for a run, e.g. ``build/w_all_ctx4096/baselines/features``."""
    return baselines_dir(label) / "features"

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
