#!/usr/bin/env python
"""Step 06: Verify pretraining results.

Confirms that step 05 produced a usable pretrained model and a tokenized dataset:
  * data/pretrain_results/ has config.json, model weights, tokenizer, train_results.json
  * the prepared dataset the runner used (data/pretrain_prepared/<name>_<hash>, derived from
    the pretrain config) is a loadable DatasetDict with train/validation splits and person_id
  * no person of the cohort step 01 test split is in its train or validation split

Usage:
    python cehrbert_model/06_verify_pretraining.py
    python cehrbert_model/06_verify_pretraining.py --results-dir data/pretrain_results --prepared-dir data/pretrain_prepared/<dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import paths  # noqa: E402
from pipeline import pretrain_data as pd_  # noqa: E402
from pipeline.dataset_checks import add_prepared_dataset_checks, datasets_available, resolve_prepared_path  # noqa: E402
from pipeline.model_checks import add_model_checks, add_train_results_checks  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify pretraining results")
    parser.add_argument("--results-dir", default=str(paths.PRETRAIN_RESULTS_DIR))
    parser.add_argument("--config", default=str(paths.PRETRAIN_CONFIG), help="Pretrain config (locates the dataset)")
    parser.add_argument(
        "--prepared-dir",
        default=None,
        help="Prepared dataset (or a folder holding exactly one); default: the hashed folder of --config",
    )
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE))
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("cehrbert_model", "06_verify_pretraining")))
    args = parser.parse_args(argv)

    report = StepReport("06_verify_pretraining", title="Step 06 - Verify pretraining results")
    prepared_dir = args.prepared_dir or str(pd_.pretrain_prepared_path(args.config, paths.resolve_under_root))
    report.add_metric("results_dir", args.results_dir)
    report.add_metric("prepared_dir", prepared_dir)

    add_model_checks(report, Path(args.results_dir), require_tokenizer=True)
    add_train_results_checks(report, Path(args.results_dir))
    add_prepared_dataset_checks(
        report,
        Path(prepared_dir),
        expected_splits=["train", "validation"],
        require_person_id=True,
    )
    resolved = resolve_prepared_path(prepared_dir)
    if resolved is not None and datasets_available():
        pd_.add_test_exclusion_check(report, resolved, args.splits_file)
    else:
        report.add_check("no test person in the pretraining dataset", False, detail="dataset not loadable")

    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    if report.passed:
        print("Next: python cehrbert_model/07_tokenize_full_dataset.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
