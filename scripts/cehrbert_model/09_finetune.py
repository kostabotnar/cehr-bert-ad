#!/usr/bin/env python
"""Step 09: Fine-tune CEHR-BERT for Alzheimer's Disease prediction.

Fine-tunes on the fine-tuning cohort (``data/cehrbert_data/cohort/finetune`` from step 02). With
``tokenized_full_dataset_path`` set (``data/full_tokenized``, step 07) the runner keeps the cohort
members of that dataset, applies ``observation_window`` and uses its splits as they are: train is the
canonical train split and validation the canonical val split. ``validation_split_percentage`` does
not split the data in this flow; it must still lie in (0, 1) for the config check and it enters the
hash of the prepared dataset directory (``data/finetune_prepared/<hash>``).

Prediction is a separate step (step 10) on the held-out test cohort, so this step runs with
``do_predict: false`` (a true value only produces a warning). No test split is needed here.

Usage:
    python cehrbert_model/09_finetune.py
    python cehrbert_model/09_finetune.py --config cehrbert_model/configs/ad_finetune_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import launch, paths  # noqa: E402
from pipeline.config_validation import add_finetune_config_checks, load_yaml, write_config  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

WRAPPER = Path(__file__).resolve().parent / "09_finetune.sh"

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune CEHR-BERT (preflight + launch)")
    parser.add_argument("--config", default=str(paths.FINETUNE_CONFIG))
    parser.add_argument("--skip-launch", action="store_true", help="Run preflight only; do not launch training")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("cehrbert_model", "09_finetune")))
    parser.add_argument(
        "--max-position-embeddings",
        "--ctx",
        type=int,
        default=None,
        dest="max_position_embeddings",
        help="Override the token context (max_position_embeddings / sample_packing_max_positions); "
        "default uses the YAML value.",
    )
    parser.add_argument(
        "--observation-window",
        type=int,
        default=None,
        dest="observation_window",
        help="Override the look-back window in days; a negative value means all history. "
        "Default uses the YAML value.",
    )
    args = parser.parse_args(argv)

    report = StepReport("09_finetune", title="Step 09 - Fine-tune CEHR-BERT")
    config_path = Path(args.config)
    report.add_metric("config", args.config)

    # --- preflight ---------------------------------------------------------
    add_finetune_config_checks(report, config_path, paths.resolve_under_root)
    if not report.passed:
        report.note("Config preflight failed; not launching training (see step 08).")
        return _finalize(report, args.report_dir)

    config = load_yaml(config_path)
    if bool(config.get("do_predict")):
        report.warn(
            "do_predict is off for fine-tuning",
            ok=False,
            detail="do_predict=true; prediction is a separate step (step 10) on the test cohort",
        )

    paths.FINETUNE_PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    paths.FINETUNE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_launch:
        report.note("Launch skipped (--skip-launch).")
        return _finalize(report, args.report_dir)

    # CLI overrides: when provided, write an effective config and launch with that;
    # otherwise launch with the original YAML unchanged (preserves current behavior).
    launch_config_path = config_path
    if args.max_position_embeddings is not None or args.observation_window is not None:
        _apply_overrides(config, args.max_position_embeddings, args.observation_window)
        report.add_metric("max_position_embeddings", config["max_position_embeddings"])
        report.add_metric("observation_window", config.get("observation_window"))
        launch_config_path = write_config(config, Path(args.report_dir) / "effective_finetune_config.yaml")
        report.add_artifact(launch_config_path)

    # --- launch ------------------------------------------------------------
    rc = launch.run_command(["bash", str(WRAPPER), str(launch_config_path)], cwd=paths.PROJECT_ROOT)
    report.add_metric("launch_returncode", rc)
    if not report.add_check("fine-tuning process succeeded", rc == 0, detail=f"exit code {rc}"):
        return _finalize(report, args.report_dir)

    report.add_check(
        "results directory produced", paths.FINETUNE_RESULTS_DIR.is_dir(), detail=str(paths.FINETUNE_RESULTS_DIR)
    )
    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python cehrbert_model/10_predict.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
