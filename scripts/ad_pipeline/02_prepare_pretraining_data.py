#!/usr/bin/env python
"""Step 2: Prepare data for pretraining — generate CEHR-BERT patient sequences.

Preflight-validates the OMOP tables the Spark job needs, then invokes the thin
``02_generate_pretraining_data.sh`` wrapper (cehrbert_data Spark apps), and finally
verifies that the patient_sequence output was produced. All validation/reporting is
here; the bash wrapper only sets Spark env and launches the apps.

Usage:
    python 02_prepare_pretraining_data.py
    python 02_prepare_pretraining_data.py --spark-driver-memory 24g
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import launch, paths  # noqa: E402
from pipeline.omop_validation import add_dataset_checks, count_rows, table_glob  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

WRAPPER = Path(__file__).resolve().parent / "02_generate_pretraining_data.sh"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate pretraining data (preflight + launch + verify)")
    parser.add_argument("--input-folder", default=str(paths.OMOP_DIR))
    parser.add_argument("--output-folder", default=str(paths.CEHRBERT_DATA_DIR))
    parser.add_argument("--start-date", default="1952-01-01")
    parser.add_argument("--spark-driver-memory", default="16g")
    parser.add_argument("--spark-executor-memory", default="16g")
    parser.add_argument("--skip-launch", action="store_true",
                        help="Run preflight + verification only; do not launch the Spark job")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("02_prepare_pretraining_data")))
    args = parser.parse_args(argv)

    report = StepReport("02_prepare_pretraining_data", title="Step 2 — Prepare pretraining data")
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    report.add_metric("input_folder", str(input_folder))
    report.add_metric("output_folder", str(output_folder))

    # --- preflight: the Spark job reads these OMOP tables ------------------
    add_dataset_checks(report, input_folder, require_concept=True)
    if not report.passed:
        report.note("Preflight failed; not launching Spark job. Fix dataset issues (see step 1).")
        return _finalize(report, args.report_dir)

    # --- launch ------------------------------------------------------------
    if args.skip_launch:
        report.note("Launch skipped (--skip-launch).")
    else:
        rc = launch.run_command(
            ["bash", str(WRAPPER)],
            cwd=paths.PROJECT_ROOT,
            env={
                "INPUT_FOLDER": str(input_folder),
                "OUTPUT_FOLDER": str(output_folder),
                "START_DATE": args.start_date,
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
        print("Next: python 03_evaluate_pretrain_config.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
