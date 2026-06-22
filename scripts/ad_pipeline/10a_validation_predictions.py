#!/usr/bin/env python
"""Step 10a: Per-sample predictions on the fine-tuning VALIDATION split.

This step scores the *validation* split that the fine-tuning runner created internally
(train/validation split of ``ad_cohort/finetune``), using the model fine-tuned in step 8.
Downstream evaluation (step 10) uses these per-sample validation probabilities to select
an UNBIASED operating threshold -- one chosen on data the model was tuned/early-stopped on
but never scored at the patient level, and crucially NOT on the held-out test set. Picking a
threshold on the test set and then reporting test metrics at that same threshold would be
optimistically biased; this step exists to avoid exactly that leakage.

Why we KEEP ``cohort_folder`` and ``dataset_prepared_path`` from the fine-tuning config
(rather than pointing at the test cohort like step 9 does):
    ``generate_prepared_ds_path`` hashes (data_folder/cohort_folder + tokenizer path +
    validation_split_percentage + test_eval_ratio + split_by_patient + chronological_split)
    to derive ``dataset_prepared_path/<hash>``. By reproducing those exact inputs, the hash
    resolves to the SAME ``data/finetune_prepared/<hash>`` directory that step 8 already
    populated -- the one that physically contains the ``train`` and ``validation`` splits.
    We then load that DatasetDict and score its ``validation`` split. We do NOT regenerate
    any data and we do NOT touch the test cohort or ``predict_prepared``.

Overrides relative to the fine-tuning config:
    model_name_or_path -> data/finetune_results   (the FINE-TUNED model)
    output_dir         -> data/finetune_results   (outputs land beside test_predictions)
    tokenizer_name_or_path -> kept (data/pretrain_results, where tokenizer.json lives)
    cohort_folder / dataset_prepared_path -> KEPT (hash reproduction; see above)
    do_train=False, do_eval=False, do_predict=True

Outputs (under output_dir):
    validation_predictions/<i>.parquet  columns:
        subject_id, prediction_time, predicted_boolean_probability,
        predicted_boolean_value (None), boolean_value
    validation_results.json  {roc_auc, pr_auc, val_loss}

This is an in-process orchestrator: it imports ``cehrbert`` directly and runs inference
itself (no bash wrapper / subprocess). It MUST run in the environment that has torch,
transformers, and the ``cehrbert`` package installed.

Usage:
    python scripts/ad_pipeline/10a_validation_predictions.py
    python scripts/ad_pipeline/10a_validation_predictions.py --finetune-config configs/ad_finetune_config.yaml

After this completes, re-run ``python scripts/ad_pipeline/10_evaluate.py`` to pick up the
validation threshold automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline.config_validation import load_yaml, write_config  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402
from pipeline.predictions_checks import _read_json  # noqa: E402


def build_validation_predict_config(finetune_cfg: dict) -> dict:
    """Derive a predict-only runner config that reuses the existing fine-tuning prepared dataset.

    Unlike step 9, ``cohort_folder`` and ``dataset_prepared_path`` are KEPT exactly as in the
    fine-tuning config so ``generate_prepared_ds_path`` hashes to the same prepared directory
    (``data/finetune_prepared/<hash>``) that already holds the train+validation splits. Only the
    model/output dirs are repointed at the fine-tuned model; the tokenizer path is left as the
    fine-tuning value (the pretraining dir, where tokenizer.json actually lives).
    """
    cfg = dict(finetune_cfg)
    finetuned_model = str(paths.FINETUNE_RESULTS_DIR)
    cfg.update(
        {
            # cohort_folder: KEEP (hash reproduction -> reuse existing validation split).
            # dataset_prepared_path: KEEP (same hashed prepared dir as fine-tuning).
            "model_name_or_path": finetuned_model,
            # tokenizer_name_or_path: keep the fine-tuning value (data/pretrain_results).
            "output_dir": finetuned_model,  # predictions land in data/finetune_results/validation_predictions
            "do_train": False,
            "do_eval": False,
            "do_predict": True,
        }
    )
    # Safety net: only if the source config never set a tokenizer (it does), default it.
    cfg.setdefault("tokenizer_name_or_path", finetuned_model)
    return cfg


def _resolve_prepared_ds_path(data_args, model_args, report: StepReport):
    """Resolve the prepared dataset dir, preferring the runner's own hash function.

    Falls back to globbing ``dataset_prepared_path`` for a single subdir that contains a
    ``validation`` subdir. Returns a ``Path`` or ``None`` (after recording a failed check).
    """
    from cehrbert.runners.runner_util import generate_prepared_ds_path

    try:
        prepared_ds_path = generate_prepared_ds_path(data_args, model_args, data_folder=data_args.cohort_folder)
    except Exception as exc:  # pragma: no cover - defensive; signature is stable
        report.note(f"generate_prepared_ds_path raised {type(exc).__name__}: {exc}; falling back to globbing.")
        prepared_ds_path = None

    if prepared_ds_path is not None and (prepared_ds_path / "validation").is_dir():
        report.add_check("prepared dataset path resolved (hash)", True, detail=str(prepared_ds_path))
        return prepared_ds_path

    # Fallback: find a single subdir under dataset_prepared_path that has a validation split.
    base = paths.resolve_under_root(data_args.dataset_prepared_path)
    candidates = [p for p in sorted(Path(base).glob("*")) if (p / "validation").is_dir()]
    if len(candidates) == 1:
        chosen = candidates[0]
        report.note(
            f"Hash path did not resolve; using single prepared dir with a validation split: {chosen}"
        )
        report.add_check("prepared dataset path resolved (glob fallback)", True, detail=str(chosen))
        return chosen

    detail = (
        f"hash path: {prepared_ds_path}; "
        f"glob under {base} found {len(candidates)} candidate(s) with a validation split"
    )
    report.add_check("prepared dataset path with validation split found", False, detail=detail)
    return None


def _preflight(report: StepReport, prepared_ds_path: Path, model_args, training_args) -> bool:
    """Preflight: prepared validation split, fine-tuned model config.json, tokenizer.json."""
    has_val = prepared_ds_path is not None and (prepared_ds_path / "validation").is_dir()
    ok_val = report.add_check(
        "prepared validation split exists",
        bool(has_val),
        detail=str((prepared_ds_path / "validation") if prepared_ds_path else "unresolved"),
    )

    model_dir = paths.resolve_under_root(training_args.output_dir)
    ok_model = report.add_check(
        "fine-tuned model exists",
        Path(model_dir).is_dir() and (Path(model_dir) / "config.json").is_file(),
        detail=str(model_dir),
    )

    tok_dir = paths.resolve_under_root(model_args.tokenizer_name_or_path)
    ok_tok = report.add_check(
        "tokenizer exists",
        (Path(tok_dir) / "tokenizer.json").is_file(),
        detail=f"{tok_dir}/tokenizer.json",
    )
    return ok_val and ok_model and ok_tok


def run_validation_predictions(
    prepared_ds_path: Path,
    cehrbert_args,
    data_args,
    model_args,
    training_args,
    report: StepReport,
) -> Path:
    """Score the validation split, mirroring the runner's ``do_predict`` body.

    Builds the data collator + DataLoader exactly like the runner's ``do_predict`` block does
    for the test split (both sample-packing and normal paths), runs the same forward call and
    ``sigmoid(logits)``, writes the identical parquet schema to ``validation_predictions/<i>.parquet``,
    then computes metrics with the runner's ``compute_metrics`` and writes ``validation_results.json``.

    Returns the path to the ``validation_predictions`` folder.
    """
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_from_disk
    from scipy.special import expit as sigmoid
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers.utils import logging as hf_logging

    from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import (
        CehrBertDataCollator,
        SamplePackingCehrBertDataCollator,
    )
    from cehrbert.data_generators.hf_data_generator.sample_packing_sampler import SamplePackingBatchSampler
    from cehrbert.models.hf_models.config import CehrBertConfig
    from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
    from cehrbert.runners.hf_cehrbert_finetune_runner import (
        compute_metrics,
        load_finetuned_model,
        load_lora_model,
    )

    log = hf_logging.get_logger("transformers")

    # --- load tokenizer + the existing prepared dataset (reuse the fine-tuning splits) ----
    tokenizer = CehrBertTokenizer.from_pretrained(str(paths.resolve_under_root(model_args.tokenizer_name_or_path)))
    processed_dataset = load_from_disk(str(prepared_ds_path))
    validation_dataset = processed_dataset["validation"]

    # Match the runner: format as torch tensors unless streaming / sample packing.
    if not data_args.streaming and not cehrbert_args.sample_packing:
        validation_dataset.set_format("pt")

    config = CehrBertConfig.from_pretrained(str(paths.resolve_under_root(model_args.model_name_or_path)))
    # Persist this in case sample packing overrides it (mirrors the runner).
    per_device_eval_batch_size = training_args.per_device_eval_batch_size

    # --- build the collator exactly like the runner ---------------------------------------
    if cehrbert_args.sample_packing:
        data_collator_fn = partial(
            SamplePackingCehrBertDataCollator,
            cehrbert_args.max_tokens_per_batch,
            config.max_position_embeddings,
        )
    else:
        data_collator_fn = CehrBertDataCollator

    data_collator = data_collator_fn(
        tokenizer=tokenizer,
        max_length=(
            cehrbert_args.max_tokens_per_batch if cehrbert_args.sample_packing else config.max_position_embeddings
        ),
        is_pretraining=False,
        mlm_probability=config.mlm_probability,
    )

    # --- build the DataLoader for the validation split (mirrors do_predict block) ----------
    if cehrbert_args.sample_packing:
        batch_sampler = SamplePackingBatchSampler(
            lengths=validation_dataset["num_of_concepts"],
            max_tokens_per_batch=cehrbert_args.max_tokens_per_batch,
            max_position_embeddings=config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
        per_device_eval_batch_size = 1
    else:
        batch_sampler = None

    val_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=batch_sampler,
    )

    # --- inference loop = a copy of do_predict's body, writing to validation_predictions ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = (
        load_finetuned_model(model_args, training_args.output_dir)
        if not model_args.use_lora
        else load_lora_model(model_args, training_args)
    )
    model = model.to(device).eval()

    val_prediction_folder = Path(training_args.output_dir) / "validation_predictions"
    val_prediction_folder.mkdir(parents=True, exist_ok=True)
    log.info("Generating predictions for validation set at %s", val_prediction_folder)

    val_losses: list[float] = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(val_dataloader, desc="Predicting (validation)")):
            person_ids = batch.pop("person_id").numpy().squeeze().astype(int)
            # Extract and process index_dates
            index_dates = None
            if "index_date" in batch:
                try:
                    timestamps = batch.pop("index_date").numpy().squeeze().tolist()
                    # Handle potential NaN or invalid timestamps
                    index_dates = [datetime.fromtimestamp(ts) if not np.isnan(ts) else None for ts in timestamps]
                except (ValueError, OverflowError, TypeError):
                    index_dates = [None] * len(timestamps)
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass
            output = model(**batch, output_attentions=False, output_hidden_states=False)
            val_losses.append(output.loss.item())

            # Collect logits and labels for prediction
            logits = output.logits.cpu().numpy().squeeze()
            labels = batch["classifier_label"].cpu().numpy().squeeze().astype(bool)
            probabilities = sigmoid(logits)
            # Save predictions to parquet file (IDENTICAL schema to do_predict)
            val_prediction_pd = pd.DataFrame(
                {
                    "subject_id": person_ids,
                    "prediction_time": index_dates,
                    "predicted_boolean_probability": probabilities,
                    "predicted_boolean_value": None,
                    "boolean_value": labels,
                }
            )
            val_prediction_pd.to_parquet(val_prediction_folder / f"{index}.parquet")

    log.info("Computing metrics using the validation set predictions at %s", val_prediction_folder)
    val_prediction_pd = pd.read_parquet(val_prediction_folder)
    metrics = compute_metrics(
        references=val_prediction_pd.boolean_value,
        probs=val_prediction_pd.predicted_boolean_probability,
    )
    metrics["val_loss"] = float(np.mean(val_losses)) if val_losses else None

    val_results_path = Path(training_args.output_dir) / "validation_results.json"
    with open(val_results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info("Validation results: %s", metrics)

    report.add_artifact(val_results_path)
    for key in ("roc_auc", "pr_auc", "val_loss"):
        report.add_metric(key, metrics.get(key))
    return val_prediction_folder


def _add_output_checks(report: StepReport, val_prediction_folder: Path, results_dir: Path) -> None:
    """Verify validation_predictions parquet (non-empty, prob range, both classes) + results json."""
    import pandas as pd

    if not report.add_check(
        "validation_predictions folder exists", val_prediction_folder.is_dir(), detail=str(val_prediction_folder)
    ):
        return
    parquet_files = sorted(val_prediction_folder.glob("*.parquet"))
    if not report.add_check(
        "validation prediction parquet files present", bool(parquet_files), detail=f"{len(parquet_files)} files"
    ):
        return

    try:
        df = pd.read_parquet(val_prediction_folder)
    except Exception as exc:
        report.add_check("validation predictions readable", False, detail=str(exc))
        return

    report.add_metric("n_predictions", int(len(df)))
    report.add_check("validation predictions non-empty", len(df) > 0)

    if "predicted_boolean_probability" in df.columns and len(df):
        probs = pd.to_numeric(df["predicted_boolean_probability"], errors="coerce")
        in_range = bool(((probs >= 0) & (probs <= 1)).all()) and not probs.isna().any()
        report.add_check(
            "probabilities in [0, 1]",
            in_range,
            detail=f"min={probs.min():.4f}, max={probs.max():.4f}" if len(probs) else "no values",
        )

    if "boolean_value" in df.columns and len(df):
        n_pos = int(pd.Series(df["boolean_value"]).astype("boolean").fillna(False).sum())
        report.add_metric("n_positive_labels", n_pos)
        report.add_check(
            "validation set has both classes", 0 < n_pos < len(df), detail=f"{n_pos} positive of {len(df)}"
        )

    results_path = Path(results_dir) / "validation_results.json"
    metrics = _read_json(results_path)
    report.add_check("validation_results.json present & valid", metrics is not None, detail=str(results_path))


def main(argv: list[str] | None = None) -> int:
    # Parse THIS script's args with a dedicated parser FIRST, before we touch sys.argv for
    # cehrbert's own (single-yaml-path) parser.
    parser = argparse.ArgumentParser(
        description="Generate per-sample predictions on the fine-tuning validation split"
    )
    parser.add_argument("--finetune-config", default=str(paths.FINETUNE_CONFIG))
    parser.add_argument(
        "--results-dir",
        default=str(paths.FINETUNE_RESULTS_DIR),
        help="Where predictions/metrics are written (must match the derived output_dir)",
    )
    parser.add_argument("--report-dir", default=str(paths.build_report_dir_for("10a_validation_predictions")))
    args = parser.parse_args(argv)

    report = StepReport(
        "10a_validation_predictions",
        title="Step 10a - Validation-split predictions for unbiased thresholding",
    )
    report.add_metric("finetune_config", args.finetune_config)

    # 1. Load fine-tuning config and derive the validation predict config.
    finetune_cfg = load_yaml(Path(args.finetune_config))
    predict_cfg = build_validation_predict_config(finetune_cfg)
    derived_config_path = write_config(predict_cfg, Path(args.report_dir) / "validation_predict_config.yaml")
    report.add_artifact(derived_config_path)

    # 2. Parse the derived config with cehrbert's parser. parse_runner_args reads a single
    #    .yaml path from sys.argv, so swap argv (we already parsed our own args above).
    try:
        from cehrbert.runners.runner_util import parse_runner_args
    except Exception as exc:
        report.add_check("cehrbert importable", False, detail=f"{type(exc).__name__}: {exc}")
        return _finalize(report, args.report_dir)
    report.add_check("cehrbert importable", True)

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], str(derived_config_path)]
        cehrbert_args, data_args, model_args, training_args = parse_runner_args()
    finally:
        sys.argv = saved_argv

    # 3. Resolve the prepared dataset path (hash reproduction; glob fallback).
    prepared_ds_path = _resolve_prepared_ds_path(data_args, model_args, report)

    # 4. Preflight checks.
    if not _preflight(report, prepared_ds_path, model_args, training_args):
        report.note("Preflight failed; not running validation prediction. Run step 8 (fine-tuning) first.")
        return _finalize(report, args.report_dir)

    # 5-7. Run the inference loop (mirrors do_predict) and write outputs.
    try:
        val_prediction_folder = run_validation_predictions(
            prepared_ds_path, cehrbert_args, data_args, model_args, training_args, report
        )
    except Exception as exc:
        report.add_check("validation prediction ran", False, detail=f"{type(exc).__name__}: {exc}")
        return _finalize(report, args.report_dir)
    report.add_check("validation prediction ran", True)

    # 8. Verify outputs.
    report.add_metric("results_dir", args.results_dir)
    _add_output_checks(report, val_prediction_folder, Path(args.results_dir))

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Validation predictions in data/finetune_results/validation_predictions/")
        print("Next: re-run `python scripts/ad_pipeline/10_evaluate.py` to pick up the validation threshold.")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
