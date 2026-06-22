"""Validate (dry-run) the pretrain/finetune YAML configs before launching a runner.

Two layers of validation:
  1. Structural: when the real cehrbert/transformers dataclasses are importable, parse
     the YAML with the very same ``HfArgumentParser`` the runner uses. This catches
     unknown keys, typos and type errors exactly as the runner would. When those deps
     are absent (e.g. a lightweight validation host), this layer is skipped and recorded
     as a warning instead of failing.
  2. Semantic: value ranges, required flags, and referenced-path existence. These run
     against the raw YAML dict and never need the heavy deps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from . import model_checks
from .dataset_checks import resolve_prepared_path
from .reporting import StepReport

# cehrbert-specific config keys (from hf_runner_argument_dataclass). Used only for the
# fallback "looks-like-a-typo" heuristic when the real parser is unavailable; HF
# TrainingArguments keys are intentionally not enumerated, so unknown-key detection is
# only authoritative when the real dataclasses can be imported.
_CEHRBERT_KNOWN_KEYS = frozenset(
    {
        # CehrBertArguments
        "tokenized_full_dataset_path", "use_early_stopping", "early_stopping_threshold",
        "sample_packing", "max_tokens_per_batch", "average_over_sequence", "retrain_with_full",
        "hyperparameter_tuning", "hyperparameter_tuning_is_grid", "hyperparameter_tuning_percentage",
        "n_trials", "hyperparameter_batch_sizes", "hyperparameter_num_train_epochs",
        "hyperparameter_learning_rates", "hyperparameter_weight_decays",
        # DataTrainingArguments
        "data_folder", "dataset_prepared_path", "test_data_folder", "cohort_folder",
        "chronological_split", "split_by_patient", "validation_split_percentage",
        "validation_split_num", "test_eval_ratio", "preprocessing_num_workers",
        "preprocessing_batch_size", "att_function_type", "is_data_in_meds",
        "inpatient_att_function_type", "meds_exclude_tables", "meds_to_cehrbert_conversion_type",
        "include_auxiliary_token", "include_demographic_prompt", "disconnect_problem_list_events",
        "observation_window", "streaming", "vocab_size", "min_frequency", "min_num_tokens",
        "shuffle_records", "offline_stats_capacity", "value_outlier_std",
        # ModelArguments
        "model_name_or_path", "tokenizer_name_or_path", "early_stopping_patience", "cache_dir",
        "use_auth_token", "trust_remote_code", "torch_dtype", "hidden_size", "num_hidden_layers",
        "n_head", "max_position_embeddings", "finetune_model_type", "use_lora", "lora_rank",
        "lora_alpha", "target_modules", "lora_dropout", "exclude_position_ids", "include_values",
        "use_sub_time_tokenization", "include_value_prediction", "include_ttv_prediction",
        "time_token_loss_weight", "time_to_visit_loss_weight", "attn_implementation",
    }
)


def load_yaml(config_path: Path) -> Dict[str, Any]:
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def write_config(cfg: Dict[str, Any], dst_path: Path) -> Path:
    """Write a config dict to ``dst_path`` as YAML (used to derive the step-9 predict config)."""
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    Path(dst_path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Path(dst_path)


def _try_real_parse(config_path: Path) -> Tuple[str, str]:
    """Parse the YAML with the runner's own HfArgumentParser when available.

    Returns (outcome, detail) where outcome is "ok", "error" or "skipped".
    """
    try:
        from transformers import HfArgumentParser, TrainingArguments

        from cehrbert.runners.hf_runner_argument_dataclass import (
            CehrBertArguments,
            DataTrainingArguments,
            ModelArguments,
        )
    except Exception as exc:
        return "skipped", f"cehrbert/transformers not importable ({type(exc).__name__})"

    try:
        parser = HfArgumentParser(
            (CehrBertArguments, DataTrainingArguments, ModelArguments, TrainingArguments)
        )
        parser.parse_yaml_file(yaml_file=str(config_path))
        return "ok", "parsed with HfArgumentParser"
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


def _add_structural_check(report: StepReport, config_path: Path, cfg: Dict[str, Any]) -> None:
    outcome, detail = _try_real_parse(config_path)
    if outcome == "ok":
        report.add_check("config parses against runner dataclasses", True, detail=detail)
    elif outcome == "error":
        report.add_check("config parses against runner dataclasses", False, detail=detail)
    else:
        report.warn("config parses against runner dataclasses", ok=False,
                    detail=detail + "; structural validation skipped")
        # Fallback heuristic: flag cehrbert-namespace typos only.
        suspicious = [
            k for k in cfg
            if k not in _CEHRBERT_KNOWN_KEYS and _looks_like_cehrbert_typo(k)
        ]
        report.warn("no obvious cehrbert key typos", ok=not suspicious,
                    detail="ok" if not suspicious else f"unrecognized: {', '.join(suspicious)}")


def _looks_like_cehrbert_typo(key: str) -> bool:
    """Heuristic: a key close to a known cehrbert key but not an exact/HF match."""
    # Only flag keys sharing a prefix with a known key (cheap, no edit-distance dep).
    for known in _CEHRBERT_KNOWN_KEYS:
        if key != known and (key.startswith(known[:6]) or known.startswith(key[:6])) and abs(len(key) - len(known)) <= 3:
            return True
    return False


# --- shared numeric/flag checks -------------------------------------------
def _check_ranges(report: StepReport, cfg: Dict[str, Any]) -> None:
    def num(key):
        return cfg.get(key)

    checks = [
        ("num_train_epochs", lambda v: v is None or v > 0, "must be > 0"),
        ("learning_rate", lambda v: v is None or 0 < v < 1, "must be in (0, 1)"),
        ("per_device_train_batch_size", lambda v: v is None or v > 0, "must be > 0"),
        ("vocab_size", lambda v: v is None or v > 0, "must be > 0"),
        ("max_position_embeddings", lambda v: v is None or v > 0, "must be > 0"),
        ("num_hidden_layers", lambda v: v is None or v > 0, "must be > 0"),
        ("weight_decay", lambda v: v is None or v >= 0, "must be >= 0"),
    ]
    for key, ok_fn, msg in checks:
        if key in cfg:
            report.add_check(f"{key} valid", ok_fn(num(key)), detail=f"{key}={num(key)} ({msg})")

    # Resource-heavy settings: informative warnings, not failures.
    if isinstance(num("per_device_train_batch_size"), (int, float)) and num("per_device_train_batch_size") > 128:
        report.warn("train batch size is modest", ok=False, detail=f"{num('per_device_train_batch_size')} may OOM")
    if isinstance(num("num_hidden_layers"), (int, float)) and num("num_hidden_layers") > 12:
        report.warn("model depth is modest", ok=False, detail=f"{num('num_hidden_layers')} layers is large")


def _check_path(report: StepReport, label: str, raw_value: Any, resolve, must: str, level: str = "error") -> None:
    """Generic path-existence check. ``resolve`` maps a raw config value to a Path."""
    if raw_value is None:
        report.add_check(f"{label} configured", False, detail="not set in config", level=level)
        return
    path = resolve(raw_value)
    exists = Path(path).exists()
    report.add_check(f"{label} exists", exists, detail=f"{must}: {path}", level=level)


# --- public entrypoints ----------------------------------------------------
def add_pretrain_config_checks(report: StepReport, config_path: Path, resolve_under_root) -> StepReport:
    """Validate the pretraining config. ``resolve_under_root`` maps relative -> abs path."""
    config_path = Path(config_path)
    if not report.add_check("config file exists", config_path.is_file(), detail=str(config_path)):
        return report
    try:
        cfg = load_yaml(config_path)
    except Exception as exc:
        report.add_check("config is valid YAML", False, detail=str(exc))
        return report
    report.add_check("config is valid YAML", True)

    _add_structural_check(report, config_path, cfg)
    _check_ranges(report, cfg)

    # data_folder must point at generated patient sequences.
    _check_path(report, "data_folder (patient sequences)", cfg.get("data_folder"), resolve_under_root,
                must="patient_sequence dir")
    if cfg.get("data_folder"):
        seq_dir = Path(resolve_under_root(cfg["data_folder"]))
        if seq_dir.exists():
            has_parquet = any(seq_dir.glob("*.parquet"))
            report.add_check("data_folder contains parquet", has_parquet, detail=str(seq_dir))

    # Output dirs: parents should exist / be creatable (warning).
    for key in ("output_dir", "dataset_prepared_path"):
        if cfg.get(key):
            parent = Path(resolve_under_root(cfg[key])).parent
            report.warn(f"{key} parent exists", ok=parent.exists(), detail=str(parent))

    report.warn("do_train enabled", ok=bool(cfg.get("do_train")), detail=f"do_train={cfg.get('do_train')}")
    return report


def add_finetune_config_checks(report: StepReport, config_path: Path, resolve_under_root) -> StepReport:
    """Validate the finetuning config and that its upstream dependencies exist."""
    config_path = Path(config_path)
    if not report.add_check("config file exists", config_path.is_file(), detail=str(config_path)):
        return report
    try:
        cfg = load_yaml(config_path)
    except Exception as exc:
        report.add_check("config is valid YAML", False, detail=str(exc))
        return report
    report.add_check("config is valid YAML", True)

    _add_structural_check(report, config_path, cfg)
    _check_ranges(report, cfg)

    # Pretrained model + tokenizer must exist (these come from step 4/5).
    if cfg.get("model_name_or_path"):
        model_dir = Path(resolve_under_root(cfg["model_name_or_path"]))
        model_checks.add_model_checks(report, model_dir, require_tokenizer=False)
    else:
        report.add_check("model_name_or_path configured", False, detail="not set")

    if cfg.get("tokenizer_name_or_path"):
        tok_dir = Path(resolve_under_root(cfg["tokenizer_name_or_path"]))
        report.add_check("tokenizer path exists", tok_dir.exists(), detail=str(tok_dir))

    # Tokenized full dataset (from pretraining) must resolve to a loadable dir.
    if cfg.get("tokenized_full_dataset_path"):
        base = Path(resolve_under_root(cfg["tokenized_full_dataset_path"]))
        report.add_check("tokenized_full_dataset_path exists", base.exists(), detail=str(base))
        if base.exists():
            resolved = resolve_prepared_path(base)
            report.add_check("tokenized dataset resolves to a single DatasetDict",
                             resolved is not None,
                             detail=resolved or "none/ambiguous under base path")

    # Cohort folder + cohort schema.
    if cfg.get("cohort_folder"):
        cohort_dir = Path(resolve_under_root(cfg["cohort_folder"]))
        report.add_check("cohort_folder exists", cohort_dir.exists(), detail=str(cohort_dir))
        if cohort_dir.exists():
            _check_cohort_schema(report, cohort_dir, is_meds=bool(cfg.get("is_data_in_meds")))

    # Observation window.
    ow = cfg.get("observation_window")
    report.add_check("observation_window > 0", isinstance(ow, (int, float)) and ow > 0, detail=f"observation_window={ow}")

    # Fine-tuning trains + evaluates; prediction is a separate step (step 9) on the test cohort.
    report.warn("do_train enabled", ok=bool(cfg.get("do_train")), detail=f"do_train={cfg.get('do_train')}")
    report.warn("do_eval enabled", ok=bool(cfg.get("do_eval")), detail=f"do_eval={cfg.get('do_eval')}")
    report.warn("do_predict disabled for fine-tuning", ok=not bool(cfg.get("do_predict")),
                detail=f"do_predict={cfg.get('do_predict')}; prediction runs in step 9 on the test cohort")

    val = cfg.get("validation_split_percentage")
    report.add_check("validation_split_percentage in (0, 1)",
                     isinstance(val, (int, float)) and 0 < val < 1,
                     detail=f"validation_split_percentage={val}")
    return report


def _check_cohort_schema(report: StepReport, cohort_dir: Path, is_meds: bool) -> None:
    """The cohort parquet must expose person_id/index_date/label (or MEDS equivalents)."""
    import pyarrow.dataset as pads

    files = sorted(cohort_dir.glob("*.parquet"))
    if not report.add_check("cohort parquet present", bool(files), detail=str(cohort_dir)):
        return
    try:
        cols = {f.name for f in pads.dataset([str(f) for f in files], format="parquet").schema}
    except Exception as exc:
        report.add_check("cohort parquet readable", False, detail=str(exc))
        return
    expected = ["subject_id", "prediction_time"] if is_meds else ["person_id", "index_date", "label"]
    missing = [c for c in expected if c not in cols]
    report.add_check("cohort has required columns", not missing,
                     detail="ok" if not missing else f"missing: {', '.join(missing)} (have: {', '.join(sorted(cols))})")
