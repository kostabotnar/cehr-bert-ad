#!/usr/bin/env python
"""Step 01: Prepare data for pretraining - generate CEHR-BERT patient sequences.

Preflight-validates the OMOP tables the Spark job needs, stages them as symlinks in
``data/cehrbert_data/omop_input``, then invokes the thin ``cehrbert_model/01_generate_sequences.sh``
wrapper (cehrbert_data Spark apps) on that folder, and finally verifies that the
patient_sequence output was produced. cehrbert_data writes ``processed_measurement`` and
reads ``qualified_concept_list`` inside its input folder, so the OMOP folder is never passed to
it and nothing is written there. All validation/reporting is here; the bash wrapper only sets
Spark env and launches the apps.

Usage:
    python cehrbert_model/01_generate_sequences.py
    python cehrbert_model/01_generate_sequences.py --spark-driver-memory 24g
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import launch, paths  # noqa: E402
from pipeline import pretrain_data as pd_  # noqa: E402
from pipeline.omop_common import MIN_CONCEPT_PATIENTS  # noqa: E402
from pipeline.omop_validation import (  # noqa: E402
    add_dataset_checks,
    add_visit_datetime_check,
    count_rows,
    table_glob,
)
from pipeline.reporting import StepReport  # noqa: E402

WRAPPER = Path(__file__).resolve().parent / "01_generate_sequences.sh"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate pretraining data (preflight + launch + verify)")
    parser.add_argument("--input-folder", default=str(paths.OMOP_DIR), help="OMOP data folder (read only)")
    parser.add_argument(
        "--staging-dir",
        default=str(paths.CEHRBERT_OMOP_INPUT_DIR),
        help="Folder of symlinks to the OMOP tables that cehrbert_data uses as its input folder",
    )
    parser.add_argument("--output-folder", default=str(paths.CEHRBERT_DATA_DIR))
    parser.add_argument("--start-date", default="1952-01-01")
    parser.add_argument("--spark-driver-memory", default="96g")
    parser.add_argument("--spark-executor-memory", default="96g")
    parser.add_argument(
        "--skip-launch", action="store_true", help="Run preflight + verification only; do not launch the Spark job"
    )
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("cehrbert_model", "01_generate_sequences")))
    args = parser.parse_args(argv)

    report = StepReport("01_generate_sequences", title="Step 01 - Prepare pretraining data")
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    report.add_metric("input_folder", str(input_folder))
    report.add_metric("output_folder", str(output_folder))
    report.add_metric("min_concept_patients", MIN_CONCEPT_PATIENTS)

    # --- preflight: the Spark job reads these OMOP tables ------------------
    add_dataset_checks(report, input_folder, require_concept=True)
    add_visit_datetime_check(report, input_folder)
    if not report.passed:
        report.note(
            "Preflight failed; not launching Spark job. Fix dataset issues (see data_verification steps 01 and 02)."
        )
        return _finalize(report, args.report_dir)

    # --- stage the OMOP tables as cehrbert_data's input folder ----------------
    staging_dir = Path(args.staging_dir)
    try:
        tables = pd_.stage_omop_input(input_folder, staging_dir)
    except FileExistsError as exc:
        report.add_check("OMOP tables staged", False, detail=str(exc))
        return _finalize(report, args.report_dir)
    report.add_metric("staging_dir", str(staging_dir))
    report.add_check("OMOP tables staged", bool(tables), detail=f"{len(tables)} tables linked in {staging_dir}")

    # --- launch ------------------------------------------------------------
    if args.skip_launch:
        report.note("Launch skipped (--skip-launch).")
    else:
        rc = launch.run_command(
            ["bash", str(WRAPPER)],
            cwd=paths.PROJECT_ROOT,
            env={
                "PYTHON": sys.executable,
                "INPUT_FOLDER": str(staging_dir),
                "OUTPUT_FOLDER": str(output_folder),
                "START_DATE": args.start_date,
                "MIN_CONCEPT_PATIENTS": str(MIN_CONCEPT_PATIENTS),
                "SPARK_DRIVER_MEMORY": args.spark_driver_memory,
                "SPARK_EXECUTOR_MEMORY": args.spark_executor_memory,
            },
        )
        report.add_metric("launch_returncode", rc)
        if not report.add_check("sequence generation process succeeded", rc == 0, detail=f"exit code {rc}"):
            return _finalize(report, args.report_dir)

    # --- verify output -----------------------------------------------------
    seq_dir = output_folder / "patient_sequence"
    if report.add_check("patient_sequence dir created", seq_dir.is_dir(), detail=str(seq_dir)):
        glob = table_glob(seq_dir.parent, "patient_sequence")
        has_parquet = glob is not None
        report.add_check("patient_sequence has parquet", has_parquet, detail=str(seq_dir))
        if has_parquet:
            n = count_rows(glob)
            report.add_metric("n_sequences", n)
            report.add_check("patient_sequence non-empty", n > 0, detail=f"{n} sequences")

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python cehrbert_model/04_validate_pretrain_config.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
