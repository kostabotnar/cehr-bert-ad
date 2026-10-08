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
# Read-only view of the OMOP table folders (symlinks). cehrbert_data writes processed_measurement and reads
# qualified_concept_list inside its input folder, so step 02 points it here and never at OMOP_DIR.
CEHRBERT_OMOP_INPUT_DIR = CEHRBERT_DATA_DIR / "omop_input"
PATIENT_SEQUENCE_DIR = CEHRBERT_DATA_DIR / "patient_sequence"
# patient_sequence without the step 6 test patients; the only input of pretraining (step 3b).
PATIENT_SEQUENCE_PRETRAIN_DIR = CEHRBERT_DATA_DIR / "patient_sequence_pretrain"

PRETRAIN_PREPARED_DIR = DATA_DIR / "pretrain_prepared"
PRETRAIN_RESULTS_DIR = DATA_DIR / "pretrain_results"
# Every patient of patient_sequence tokenized with the pretrained tokenizer (step 5b); read by
# steps 8, 9, 10a and 15 through tokenized_full_dataset_path.
FULL_TOKENIZED_DIR = DATA_DIR / "full_tokenized"

FINETUNE_PREPARED_DIR = DATA_DIR / "finetune_prepared"
FINETUNE_RESULTS_DIR = DATA_DIR / "finetune_results"
PREDICT_PREPARED_DIR = DATA_DIR / "predict_prepared"

# Prediction parquet folders (written by the finetune runner's do_predict).
TEST_PREDICTIONS_DIR = FINETUNE_RESULTS_DIR / "test_predictions"
VAL_PREDICTIONS_DIR = FINETUNE_RESULTS_DIR / "validation_predictions"

# Evaluation outputs (steps 10 / 10a) live under build/, nested per experiment in
# build/<run_label>/ (see run_label / the label-aware *_dir functions below).
BUILD_DIR = PROJECT_ROOT / "build"

# Raw OMOP profile (step 18): tables, figures and exclusion flow of AD cases versus controls.
OMOP_PROFILE_DIR = BUILD_DIR / "omop_profile"

# Defaults for the experiment parameters that make up a run label.
# These match ad_finetune_config.yaml (observation_window) and the pretrained model's
# max_position_embeddings, so the default label is "w_365_ctx512".
DEFAULT_WINDOW_DAYS = 365
DEFAULT_CTX = 512


def run_label(window_days, ctx) -> str:
    """Build a run label, e.g. ``'w_all_ctx4096'`` or ``'w_365_ctx512'``.

    ``window_days`` of ``None`` or negative encodes an unbounded ("all") look-back.
    """
    w = "all" if window_days is None or int(window_days) < 0 else str(int(window_days))
    return f"w_{w}_ctx{int(ctx)}"


def run_dir(label: str) -> Path:
    """Per-experiment output folder, e.g. ``build/w_365_ctx512``."""
    return BUILD_DIR / label


def eval_results_dir(label: str) -> Path:
    """Evaluation results folder for a run (metrics.json / stats json live here)."""
    return run_dir(label)


def figures_dir(label: str) -> Path:
    """Figures folder for a run, e.g. ``build/w_365_ctx512/figures``."""
    return run_dir(label) / "figures"


def build_reports_dir(label: str) -> Path:
    """Reports folder for a run, e.g. ``build/w_365_ctx512/reports``."""
    return run_dir(label) / "reports"


def build_report_dir_for(step_name: str, label: str) -> Path:
    """Report directory for a step within a run, e.g. build/<label>/reports/10_evaluate."""
    return build_reports_dir(label) / step_name


def baselines_dir(label: str) -> Path:
    """Baselines folder for a run, e.g. ``build/w_365_ctx512/baselines``."""
    return run_dir(label) / "baselines"


def features_dir(label: str) -> Path:
    """Baseline features folder for a run, e.g. ``build/w_365_ctx512/baselines/features``."""
    return baselines_dir(label) / "features"


def linear_probe_dir(label: str) -> Path:
    """Frozen-CEHR-BERT linear probe folder for a run, e.g. ``build/w_365_ctx512/linear_probe``."""
    return run_dir(label) / "linear_probe"


def linear_probe_features_dir(label: str) -> Path:
    """Per-split frozen-model features, e.g. ``build/w_365_ctx512/linear_probe/features``."""
    return linear_probe_dir(label) / "features"


# Labeled cohort (step 6), built from the OMOP tables only: one row per patient with
# person_id, index_date, label and split in {train, val, test}.
COHORT_DIR = DATA_DIR / "cohort"
COHORT_SPLITS_FILE = COHORT_DIR / "splits.parquet"

# Cohort folders read by the CEHR-BERT finetune runner (cohort_folder reads *.parquet from one
# folder); derived from the splits file by step 02b.
CEHRBERT_COHORT_DIR = CEHRBERT_DATA_DIR / "cohort"
CEHRBERT_COHORT_FINETUNE_DIR = CEHRBERT_COHORT_DIR / "finetune"
CEHRBERT_COHORT_TEST_DIR = CEHRBERT_COHORT_DIR / "test"


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
