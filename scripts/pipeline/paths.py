"""Canonical filesystem locations for the AD pipeline parts.

All paths are derived from ``PROJECT_ROOT`` so the scripts work regardless of the
current working directory. ``PROJECT_ROOT`` is the repository root (``.../cehr-bert``),
the parent of ``SCRIPTS_DIR`` (``scripts/``), which holds this package and one folder per part.

Each part writes under its own build root (``build/<part>/``):

  * data_qc             : ``omop_profile/`` and ``history_length/``
  * cehrbert_model      : ``<run_label>/`` with the run label ``w_<window>_ctx<ctx>``
                          (e.g. ``w_all_ctx4096``, ``w_365_ctx512``; ``all`` means an unbounded look-back)
  * classical_ml        : ``<window_label>/`` with the window-only label ``w_<window>`` (no token context)
  * evaluation          : ``<comparison name>/``
  * feature_importance  : ``<run_label>/``

The model parts describe their predictions in a ``predictions_manifest.json`` inside their label folder;
the evaluation part reads only those manifests.
"""

from __future__ import annotations

from pathlib import Path

# .../cehr-bert/scripts/pipeline/paths.py
#   parents[0] = pipeline
#   parents[1] = scripts  (SCRIPTS_DIR)
#   parents[2] = cehr-bert  (PROJECT_ROOT)
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- part folders ----------------------------------------------------------
DATA_VERIFICATION_DIR = SCRIPTS_DIR / "data_verification"
COHORT_PART_DIR = SCRIPTS_DIR / "cohort"
DATA_QC_DIR = SCRIPTS_DIR / "data_qc"
CEHRBERT_MODEL_DIR = SCRIPTS_DIR / "cehrbert_model"
CLASSICAL_ML_DIR = SCRIPTS_DIR / "classical_ml"
EVALUATION_DIR = SCRIPTS_DIR / "evaluation"
FEATURE_IMPORTANCE_DIR = SCRIPTS_DIR / "feature_importance"
# Folder names in the order run_all.sh runs them.
PARTS = (
    "data_verification",
    "cohort",
    "data_qc",
    "cehrbert_model",
    "classical_ml",
    "evaluation",
    "feature_importance",
)

# --- part assets -----------------------------------------------------------
PRETRAIN_CONFIG = CEHRBERT_MODEL_DIR / "configs" / "ad_pretrain_config.yaml"
FINETUNE_CONFIG = CEHRBERT_MODEL_DIR / "configs" / "ad_finetune_config.yaml"
OMOP_PROFILE_CONCEPTS = DATA_QC_DIR / "configs" / "omop_profile_concepts.yaml"

# --- data tree -------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
OMOP_DIR = DATA_DIR / "omop_data"
CEHRBERT_DATA_DIR = DATA_DIR / "cehrbert_data"
# Read-only view of the OMOP table folders (symlinks). cehrbert_data writes processed_measurement and reads
# qualified_concept_list inside its input folder, so cehrbert_model step 01 points it here and never at OMOP_DIR.
CEHRBERT_OMOP_INPUT_DIR = CEHRBERT_DATA_DIR / "omop_input"
PATIENT_SEQUENCE_DIR = CEHRBERT_DATA_DIR / "patient_sequence"
# patient_sequence without the cohort step 01 test patients; the only input of pretraining (cehrbert_model step 03).
PATIENT_SEQUENCE_PRETRAIN_DIR = CEHRBERT_DATA_DIR / "patient_sequence_pretrain"

PRETRAIN_PREPARED_DIR = DATA_DIR / "pretrain_prepared"
PRETRAIN_RESULTS_DIR = DATA_DIR / "pretrain_results"
# Every patient of patient_sequence tokenized with the pretrained tokenizer (cehrbert_model step 07); read by
# cehrbert_model steps 09, 10, 11 and 13 through tokenized_full_dataset_path.
FULL_TOKENIZED_DIR = DATA_DIR / "full_tokenized"

FINETUNE_PREPARED_DIR = DATA_DIR / "finetune_prepared"
FINETUNE_RESULTS_DIR = DATA_DIR / "finetune_results"
PREDICT_PREPARED_DIR = DATA_DIR / "predict_prepared"

# Prediction parquet folders (written by the finetune runner's do_predict).
TEST_PREDICTIONS_DIR = FINETUNE_RESULTS_DIR / "test_predictions"
VAL_PREDICTIONS_DIR = FINETUNE_RESULTS_DIR / "validation_predictions"

# Every part writes under its own build root, nested per experiment label (see the helpers below).
BUILD_DIR = PROJECT_ROOT / "build"
DATA_QC_BUILD_DIR = BUILD_DIR / "data_qc"
CEHRBERT_MODEL_BUILD_DIR = BUILD_DIR / "cehrbert_model"
CLASSICAL_ML_BUILD_DIR = BUILD_DIR / "classical_ml"
EVALUATION_BUILD_DIR = BUILD_DIR / "evaluation"
FEATURE_IMPORTANCE_BUILD_DIR = BUILD_DIR / "feature_importance"

# Raw OMOP profile (data_qc step 01): tables, figures and exclusion flow of AD cases versus controls.
OMOP_PROFILE_DIR = DATA_QC_BUILD_DIR / "omop_profile"
# Patient history length statistics (data_qc step 02).
HISTORY_LENGTH_DIR = DATA_QC_BUILD_DIR / "history_length"

MANIFEST_NAME = "predictions_manifest.json"

# Defaults for the experiment parameters that make up a run label.
# These match ad_finetune_config.yaml (observation_window) and the pretrained model's
# max_position_embeddings, so the default label is "w_365_ctx512".
DEFAULT_WINDOW_DAYS = 365
DEFAULT_CTX = 512


def window_label(window_days) -> str:
    """Window-only label, e.g. ``'w_365'`` or ``'w_all'`` (``None`` or negative = unbounded)."""
    w = "all" if window_days is None or int(window_days) < 0 else str(int(window_days))
    return f"w_{w}"


def run_label(window_days, ctx) -> str:
    """Build a run label, e.g. ``'w_all_ctx4096'`` or ``'w_365_ctx512'``."""
    return f"{window_label(window_days)}_ctx{int(ctx)}"


# --- cehrbert_model: build/cehrbert_model/<run_label>/ ---------------------
def cehrbert_model_dir(label: str) -> Path:
    return CEHRBERT_MODEL_BUILD_DIR / label


def cehrbert_figures_dir(label: str) -> Path:
    return cehrbert_model_dir(label) / "figures"


def cehrbert_report_dir(step_name: str, label: str) -> Path:
    return cehrbert_model_dir(label) / "reports" / step_name


def cehrbert_manifest_path(label: str) -> Path:
    return cehrbert_model_dir(label) / MANIFEST_NAME


def cehrbert_linear_probe_dir(label: str) -> Path:
    return cehrbert_model_dir(label) / "linear_probe"


def cehrbert_linear_probe_features_dir(label: str) -> Path:
    return cehrbert_linear_probe_dir(label) / "features"


# --- classical_ml: build/classical_ml/<window_label>/ ----------------------
def classical_ml_dir(wlabel: str) -> Path:
    return CLASSICAL_ML_BUILD_DIR / wlabel


def classical_features_dir(wlabel: str) -> Path:
    return classical_ml_dir(wlabel) / "features"


def classical_report_dir(step_name: str, wlabel: str) -> Path:
    return classical_ml_dir(wlabel) / "reports" / step_name


def classical_manifest_path(wlabel: str) -> Path:
    return classical_ml_dir(wlabel) / MANIFEST_NAME


# --- evaluation: build/evaluation/<comparison name>/ -----------------------
def evaluation_dir(name: str) -> Path:
    return EVALUATION_BUILD_DIR / name


def evaluation_model_dir(name: str, model: str) -> Path:
    """Per-model scoring outputs: ``metrics.json``, ``figures/``, ``reports/01_evaluate/``."""
    return evaluation_dir(name) / "models" / model


def evaluation_figures_dir(name: str) -> Path:
    return evaluation_dir(name) / "figures"


def evaluation_report_dir(step_name: str, name: str) -> Path:
    return evaluation_dir(name) / "reports" / step_name


# --- feature_importance: build/feature_importance/<run_label>/ -------------
def feature_importance_dir(label: str) -> Path:
    return FEATURE_IMPORTANCE_BUILD_DIR / label


def feature_importance_figures_dir(label: str) -> Path:
    return feature_importance_dir(label) / "figures"


def feature_importance_report_dir(step_name: str, label: str) -> Path:
    return feature_importance_dir(label) / "reports" / step_name


# Labeled cohort (cohort step 01), built from the OMOP tables only: one row per patient with
# person_id, index_date, label and split in {train, val, test}.
COHORT_DIR = DATA_DIR / "cohort"
COHORT_SPLITS_FILE = COHORT_DIR / "splits.parquet"

# Cohort folders read by the CEHR-BERT finetune runner (cohort_folder reads *.parquet from one
# folder); derived from the splits file by cehrbert_model step 02.
CEHRBERT_COHORT_DIR = CEHRBERT_DATA_DIR / "cohort"
CEHRBERT_COHORT_FINETUNE_DIR = CEHRBERT_COHORT_DIR / "finetune"
CEHRBERT_COHORT_TEST_DIR = CEHRBERT_COHORT_DIR / "test"


def report_dir_for(part: str, step_name: str) -> Path:
    """Default report directory for a step, e.g. ``scripts/cohort/reports/01_build_cohort``."""
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; expected one of {', '.join(PARTS)}")
    return SCRIPTS_DIR / part / "reports" / step_name


def resolve_under_root(path: str | Path) -> Path:
    """Resolve a (possibly relative) config path against ``PROJECT_ROOT``.

    The cehrbert runners are invoked with the project root as the working
    directory, so config paths like ``data/pretrain_results`` are relative to it.
    """
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
