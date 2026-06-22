#!/usr/bin/env python
"""Step 5: Verify pretraining results.

Confirms that step 4 produced a usable pretrained model and a tokenized dataset:
  * data/pretrain_results/ has config.json, model weights, tokenizer, train_results.json
  * data/pretrain_prepared/ holds a loadable DatasetDict with train/validation splits
    and a person_id column (which step 6/8 rely on).

Usage:
    python 05_verify_pretraining.py
    python 05_verify_pretraining.py --results-dir data/pretrain_results --prepared-dir data/pretrain_prepared
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline.dataset_checks import add_prepared_dataset_checks  # noqa: E402
from pipeline.model_checks import add_model_checks, add_train_results_checks  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify pretraining results")
    parser.add_argument("--results-dir", default=str(paths.PRETRAIN_RESULTS_DIR))
    parser.add_argument("--prepared-dir", default=str(paths.PRETRAIN_PREPARED_DIR))
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("05_verify_pretraining")))
    args = parser.parse_args(argv)

    report = StepReport("05_verify_pretraining", title="Step 5 — Verify pretraining results")
    report.add_metric("results_dir", args.results_dir)
    report.add_metric("prepared_dir", args.prepared_dir)

    add_model_checks(report, Path(args.results_dir), require_tokenizer=True)
    add_train_results_checks(report, Path(args.results_dir))
    add_prepared_dataset_checks(
        report,
        Path(args.prepared_dir),
        expected_splits=["train", "validation"],
        require_person_id=True,
    )

    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    if report.passed:
        print("Next: python 06_prepare_finetuning_data.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
