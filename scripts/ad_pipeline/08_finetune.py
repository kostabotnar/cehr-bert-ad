#!/usr/bin/env python
"""Step 8: Fine-tune CEHR-BERT for Alzheimer's Disease prediction.

Fine-tunes on the fine-tuning cohort (``ad_cohort/finetune`` from step 6) with an internal
train/validation split (``validation_split_percentage``). Prediction is a separate step
(step 9) on the held-out test cohort, so this step runs with ``do_predict: false`` — no
test split is needed here, and the run completes in one launch.

Usage:
    python 08_finetune.py
    python 08_finetune.py --config configs/ad_finetune_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import launch, paths  # noqa: E402
from pipeline.config_validation import add_finetune_config_checks, load_yaml  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

WRAPPER = Path(__file__).resolve().parent / "08_finetune.sh"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune CEHR-BERT (preflight + launch)")
    parser.add_argument("--config", default=str(paths.FINETUNE_CONFIG))
    parser.add_argument("--skip-launch", action="store_true", help="Run preflight only; do not launch training")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("08_finetune")))
    args = parser.parse_args(argv)

    report = StepReport("08_finetune", title="Step 8 — Fine-tune CEHR-BERT")
    config_path = Path(args.config)
    report.add_metric("config", args.config)

    # --- preflight ---------------------------------------------------------
    add_finetune_config_checks(report, config_path, paths.resolve_under_root)
    if not report.passed:
        report.note("Config preflight failed; not launching training (see step 7).")
        return _finalize(report, args.report_dir)

    config = load_yaml(config_path)
    if bool(config.get("do_predict")):
        report.warn("do_predict is off for fine-tuning", ok=False,
                    detail="do_predict=true; prediction is a separate step (step 9) on the test cohort")

    paths.FINETUNE_PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    paths.FINETUNE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_launch:
        report.note("Launch skipped (--skip-launch).")
        return _finalize(report, args.report_dir)

    # --- launch ------------------------------------------------------------
    rc = launch.run_command(["bash", str(WRAPPER), str(config_path)], cwd=paths.PROJECT_ROOT)
    report.add_metric("launch_returncode", rc)
    if not report.add_check("fine-tuning process succeeded", rc == 0, detail=f"exit code {rc}"):
        return _finalize(report, args.report_dir)

    report.add_check("results directory produced", paths.FINETUNE_RESULTS_DIR.is_dir(),
                     detail=str(paths.FINETUNE_RESULTS_DIR))
    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python 09_predict.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
