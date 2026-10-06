#!/usr/bin/env python
"""Step 3b: Build the pretraining input without the held-out test patients.

Writes ``data/cehrbert_data/patient_sequence_pretrain``: ``patient_sequence`` (step 2) minus every
person in the ``test`` split of ``ad_cohort/splits.parquet`` (step 6). Pretraining (step 4,
``data_folder`` in ``ad_pretrain_config.yaml``) reads only this folder, so the test patients are
in neither the pretraining train nor validation split. The Spark layout is kept: one file per
source file, same names, schema and snappy compression (a file left with no rows is skipped).

The step reports the sequence counts (total, removed, remaining) and fails if any test person
is still present in the output.

Run after step 6 and before steps 3 and 4:
    python 03b_build_pretraining_input.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline import pretrain_data as pd_  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write patient_sequence without the test-split patients")
    parser.add_argument("--patient-sequence-dir", default=str(paths.PATIENT_SEQUENCE_DIR))
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE))
    parser.add_argument("--output-dir", default=str(paths.PATIENT_SEQUENCE_PRETRAIN_DIR))
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("03b_build_pretraining_input")))
    args = parser.parse_args(argv)

    report = StepReport("03b_build_pretraining_input", title="Step 3b: Build the pretraining input (no test patients)")
    src, dst = Path(args.patient_sequence_dir), Path(args.output_dir)
    for name, value in (("patient_sequence_dir", src), ("splits_file", args.splits_file), ("output_dir", dst)):
        report.add_metric(name, str(value))

    if not report.add_check("patient_sequence has parquet files", bool(pd_.sequence_files(src)), detail=str(src)):
        report.note("Run step 2 first.")
        return _finalize(report, args.report_dir)
    if not report.add_check("splits file exists", Path(args.splits_file).is_file(), detail=args.splits_file):
        report.note("Run step 6 first.")
        return _finalize(report, args.report_dir)
    if src.resolve() == dst.resolve():
        report.add_check("output differs from the source folder", False, detail=str(dst))
        return _finalize(report, args.report_dir)

    test_ids = pd_.split_person_ids(args.splits_file, "test")
    report.add_metric("test_persons", len(test_ids))
    report.add_check("test split is non-empty", bool(test_ids), detail=f"{len(test_ids)} persons")

    counts = pd_.write_sequences_without(src, dst, test_ids)
    report.add_metric("files", counts["files"])
    report.add_metric("files_written", counts["files_written"])
    report.add_metric("sequences_total", counts["rows_total"])
    report.add_metric("sequences_removed", counts["rows_removed"])
    report.add_metric("sequences_remaining", counts["rows_remaining"])
    report.add_artifact(dst)

    remaining_ids = pd_.sequence_person_ids(dst, require_concepts=False)
    leaked = remaining_ids & test_ids
    report.add_check(
        "no test person remains in the pretraining input",
        not leaked,
        detail="ok" if not leaked else f"{len(leaked)} test persons present",
    )
    report.add_check(
        "removed + remaining = total",
        counts["rows_removed"] + counts["rows_remaining"] == counts["rows_total"],
        detail=f"{counts['rows_removed']} + {counts['rows_remaining']} vs {counts['rows_total']}",
    )
    # Step 6 keeps only persons with a sequence, so a shortfall means it ran on other sequences.
    report.warn(
        "every test person had a sequence",
        ok=counts["rows_removed"] == len(test_ids),
        detail=f"removed {counts['rows_removed']} sequences for {len(test_ids)} test persons",
    )
    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python 03_evaluate_pretrain_config.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
