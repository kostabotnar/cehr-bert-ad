#!/usr/bin/env python
"""Step 9: Predict on the held-out test cohort and verify the results.

Runs prediction as a standalone step on the test cohort that step 6 held out
(``ad_cohort/test``), using the model fine-tuned in step 8. A predict config is derived
from the fine-tuning config with these overrides:

    cohort_folder        -> the held-out test cohort
    model/tokenizer/output_dir -> the fine-tuned model (data/finetune_results)
    do_train, do_eval    -> false
    do_predict           -> true

The runner, in predict-only mode (do_predict without do_train), treats the whole test
cohort as the ``test`` split (see the vendored runner edit), writes
``test_predictions/*.parquet`` and ``test_results.json`` under output_dir, and this step
then verifies those outputs (columns, probability range, ROC-AUC / PR-AUC).

Usage:
    python 09_predict.py
    python 09_predict.py --finetune-config configs/ad_finetune_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import launch, paths  # noqa: E402
from pipeline.config_validation import load_yaml, write_config  # noqa: E402
from pipeline.predictions_checks import add_predictions_checks, add_test_results_checks  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

WRAPPER = Path(__file__).resolve().parent / "09_predict.sh"

ALL_HISTORY_DAYS = 36500  # sentinel look-back used when a negative window is requested


def _apply_overrides(config: dict, max_position_embeddings: int | None, observation_window: int | None) -> None:
    """Override token context and observation window in-place from CLI values.

    ``max_position_embeddings`` also mirrors into ``sample_packing_max_positions`` when that
    key is present. A negative ``observation_window`` is treated as the all-history sentinel
    (``ALL_HISTORY_DAYS``). ``None`` for either preserves the YAML value.
    """
    if max_position_embeddings is not None:
        config["max_position_embeddings"] = int(max_position_embeddings)
        if "sample_packing_max_positions" in config:
            config["sample_packing_max_positions"] = int(max_position_embeddings)
    if observation_window is not None:
        ow = int(observation_window)
        config["observation_window"] = ALL_HISTORY_DAYS if ow < 0 else ow


def build_predict_config(finetune_cfg: dict) -> dict:
    """Derive a predict-only runner config from the fine-tuning config.

    The fine-tuned *model* is in data/finetune_results, but the *tokenizer* is not saved
    there (the finetune runner's save_model does not persist it). So model_name_or_path and
    output_dir point at finetune_results, while tokenizer_name_or_path is left as the
    fine-tuning config's value (the pretraining dir, where tokenizer.json actually lives).
    """
    cfg = dict(finetune_cfg)
    finetuned_model = str(paths.FINETUNE_RESULTS_DIR)
    cfg.update(
        {
            "cohort_folder": str(paths.COHORT_TEST_DIR),
            "model_name_or_path": finetuned_model,
            # tokenizer_name_or_path: keep the fine-tuning value (data/pretrain_results).
            "output_dir": finetuned_model,  # predictions land in data/finetune_results/test_predictions
            "dataset_prepared_path": str(paths.PREDICT_PREPARED_DIR),
            "do_train": False,
            "do_eval": False,
            "do_predict": True,
        }
    )
    cfg.setdefault("tokenizer_name_or_path", finetuned_model)
    return cfg


def _preflight(report: StepReport, predict_cfg: dict) -> bool:
    """Check that the test cohort, fine-tuned model, and tokenizer exist before launching."""
    cohort_dir = Path(paths.resolve_under_root(predict_cfg["cohort_folder"]))
    ok_cohort = report.add_check("test cohort exists", cohort_dir.is_dir() and any(cohort_dir.glob("*.parquet")),
                                 detail=str(cohort_dir))
    model_dir = Path(paths.resolve_under_root(predict_cfg["model_name_or_path"]))
    ok_model = report.add_check("fine-tuned model exists", model_dir.is_dir() and (model_dir / "config.json").is_file(),
                                detail=str(model_dir))
    tok_dir = Path(paths.resolve_under_root(predict_cfg["tokenizer_name_or_path"]))
    ok_tok = report.add_check("tokenizer exists", (tok_dir / "tokenizer.json").is_file(),
                              detail=f"{tok_dir}/tokenizer.json")
    return ok_cohort and ok_model and ok_tok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict on the held-out test cohort and verify results")
    parser.add_argument("--finetune-config", default=str(paths.FINETUNE_CONFIG))
    parser.add_argument("--results-dir", default=str(paths.FINETUNE_RESULTS_DIR),
                        help="Where predictions/metrics are written (must match the predict output_dir)")
    parser.add_argument("--skip-launch", action="store_true", help="Verify existing predictions only; do not run inference")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("09_predict")))
    parser.add_argument(
        "--max-position-embeddings", "--ctx", type=int, default=None, dest="max_position_embeddings",
        help="Override the token context (max_position_embeddings / sample_packing_max_positions); "
             "default uses the YAML value.")
    parser.add_argument(
        "--observation-window", type=int, default=None, dest="observation_window",
        help="Override the look-back window in days; a negative value means all history. "
             "Default uses the YAML value.")
    args = parser.parse_args(argv)

    report = StepReport("09_predict", title="Step 9 — Predict on held-out test & verify")
    report.add_metric("finetune_config", args.finetune_config)

    finetune_cfg = load_yaml(Path(args.finetune_config))
    _apply_overrides(finetune_cfg, args.max_position_embeddings, args.observation_window)
    predict_cfg = build_predict_config(finetune_cfg)
    predict_config_path = write_config(predict_cfg, Path(args.report_dir) / "predict_config.yaml")
    report.add_artifact(predict_config_path)

    if not args.skip_launch:
        if not _preflight(report, predict_cfg):
            report.note("Preflight failed; not running prediction. Run steps 6 and 8 first.")
            return _finalize(report, args.report_dir)
        Path(paths.PREDICT_PREPARED_DIR).mkdir(parents=True, exist_ok=True)

        # The runner predicts on processed_dataset["test"]. Its extract_cohort_sequences
        # route builds train/validation, so we collapse the prepared dataset into a single
        # 'test' split from the pipeline side. This works whether or not the vendored runner
        # carries the predict-only patch. Done before launch (covers re-runs) and, on a first
        # run where the dataset is created during the run, after a failed launch + re-launch.
        from pipeline.dataset_checks import collapse_to_test_split, datasets_available

        if datasets_available():
            collapse_to_test_split(paths.PREDICT_PREPARED_DIR)
        rc = launch.run_command(["bash", str(WRAPPER), str(predict_config_path)], cwd=paths.PROJECT_ROOT)
        if rc != 0 and datasets_available():
            collapsed = collapse_to_test_split(paths.PREDICT_PREPARED_DIR)
            if collapsed:
                report.note("Collapsed the prepared dataset into a 'test' split; re-launching prediction.")
                rc = launch.run_command(["bash", str(WRAPPER), str(predict_config_path)], cwd=paths.PROJECT_ROOT)

        report.add_metric("launch_returncode", rc)
        if not report.add_check("prediction process succeeded", rc == 0, detail=f"exit code {rc}"):
            return _finalize(report, args.report_dir)

    # --- verify predictions + metrics -------------------------------------
    results_dir = Path(args.results_dir)
    report.add_metric("results_dir", str(results_dir))
    add_predictions_checks(report, results_dir)
    add_test_results_checks(report, results_dir)

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Pipeline complete. Predictions in data/finetune_results/test_predictions/")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
