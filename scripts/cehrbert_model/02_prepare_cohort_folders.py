#!/usr/bin/env python
"""Step 02: Write the CEHR-BERT cohort folders from the canonical splits.

The finetune runner reads a cohort folder (``cohort_folder``, every ``*.parquet`` in it). From
``data/cohort/splits.parquet`` (cohort step 01) this step writes

    data/cehrbert_data/cohort/finetune/cohort.parquet   train + val rows
    data/cehrbert_data/cohort/test/cohort.parquet       test rows

with the columns person_id (int64), index_date (date), label (int64) and age_at_index (int64).
age_at_index is ``year(index_date) - person.year_of_birth`` from the OMOP person table, the same
definition the cohort and the baselines use. The CEHR-BERT classifier reads it instead of the age token
of the first event of the record.

Gate: every cohort person must have a non-empty sequence in ``patient_sequence`` (step 01). A
person without one cannot be tokenized and would make step 09 fail; the step fails and lists up to 20
person_ids. cohort step 01 builds the cohort from OMOP tables only, with an event rule that mirrors how the
sequences are built, so a failure here points at a mismatch between the two (for example a concept
threshold that no longer matches ``omop_common.MIN_CONCEPT_PATIENTS``).

Gate: every cohort row must have a non-null age_at_index. Cohort step 01 already excludes persons
without year_of_birth, so a null points at a mismatch between the OMOP data and the splits; the step fails
and lists up to 20 person_ids.

Run after step 01 (and cohort step 01):
    python cehrbert_model/02_prepare_cohort_folders.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import omop_common, paths  # noqa: E402
from pipeline import pretrain_data as pd_  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

MAX_LISTED = 20
SCHEMA = {"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64, "age_at_index": pl.Int64}


def add_age_at_index(splits: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    """Add ``age_at_index`` = index year minus OMOP ``year_of_birth``; null when the person has none."""
    person = omop_common.scan_table(data_dir, "person", ["person_id", "year_of_birth"])
    if person is None:
        raise FileNotFoundError(f"no OMOP person table under {data_dir}")
    birth = (
        person.select(pl.col("person_id").cast(pl.Int64), pl.col("year_of_birth").cast(pl.Int64).alias("_yob"))
        .unique("person_id")
        .collect()
    )
    return (
        splits.join(birth, on="person_id", how="left")
        .with_columns((pl.col("index_date").dt.year().cast(pl.Int64) - pl.col("_yob")).alias("age_at_index"))
        .drop("_yob")
    )


def write_cohort_file(df: pl.DataFrame, folder: Path) -> Path:
    """Write ``df`` as the only parquet file of ``folder``; stale parquet files would be read by the runner."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "cohort.parquet"
    for stale in folder.glob("*.parquet"):
        if stale != target:
            stale.unlink()
    df.select([pl.col(c).cast(t) for c, t in SCHEMA.items()]).sort("person_id").write_parquet(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the CEHR-BERT finetune/test cohort folders from the splits")
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE))
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR), help="OMOP directory with the person table")
    parser.add_argument("--patient-sequence-dir", default=str(paths.PATIENT_SEQUENCE_DIR))
    parser.add_argument("--finetune-dir", default=str(paths.CEHRBERT_COHORT_FINETUNE_DIR))
    parser.add_argument("--test-dir", default=str(paths.CEHRBERT_COHORT_TEST_DIR))
    parser.add_argument(
        "--report-dir", default=str(paths.report_dir_for("cehrbert_model", "02_prepare_cohort_folders"))
    )
    args = parser.parse_args(argv)

    report = StepReport("02_prepare_cohort_folders", title="Step 02: Prepare the CEHR-BERT cohort folders")
    splits_file, seq_dir = Path(args.splits_file), Path(args.patient_sequence_dir)
    report.add_metric("splits_file", str(splits_file))
    report.add_metric("patient_sequence_dir", str(seq_dir))

    if not report.add_check("splits file exists", splits_file.is_file(), detail=str(splits_file)):
        report.note("Run cohort step 01 first.")
        return _finalize(report, args.report_dir)
    if not report.add_check(
        "patient_sequence has parquet files", bool(pd_.sequence_files(seq_dir)), detail=str(seq_dir)
    ):
        report.note("Run step 01 first.")
        return _finalize(report, args.report_dir)

    splits = pl.read_parquet(splits_file)
    report.add_metric("cohort_size", splits.height)
    report.add_check("splits file is non-empty", splits.height > 0, detail=f"{splits.height} rows")

    with_sequence = pd_.sequence_person_ids(seq_dir, require_concepts=True)
    report.add_metric("sequence_patients", len(with_sequence))
    missing = sorted(set(splits["person_id"].to_list()) - with_sequence)
    report.add_metric("cohort_without_sequence", len(missing))
    listed = ", ".join(str(p) for p in missing[:MAX_LISTED])
    if not report.add_check(
        "every cohort person has a non-empty sequence",
        not missing,
        detail="ok" if not missing else f"{len(missing)} without one; first {min(len(missing), MAX_LISTED)}: {listed}",
    ):
        return _finalize(report, args.report_dir)

    try:
        splits = add_age_at_index(splits, Path(args.data_dir))
    except FileNotFoundError as exc:
        report.add_check("OMOP person table exists", False, detail=str(exc))
        return _finalize(report, args.report_dir)
    no_age = sorted(splits.filter(pl.col("age_at_index").is_null())["person_id"].unique().to_list())
    report.add_metric("cohort_without_age", len(no_age))
    listed_age = ", ".join(str(p) for p in no_age[:MAX_LISTED])
    if not report.add_check(
        "every cohort row has an age_at_index",
        not no_age,
        detail="ok" if not no_age else f"{len(no_age)} without one; first {min(len(no_age), MAX_LISTED)}: {listed_age}",
    ):
        return _finalize(report, args.report_dir)

    finetune = splits.filter(pl.col("split").is_in(["train", "val"]))
    test = splits.filter(pl.col("split") == "test")
    report.add_metric("finetune_rows", finetune.height)
    report.add_metric("test_rows", test.height)
    report.add_check(
        "finetune and test are non-empty",
        finetune.height > 0 and test.height > 0,
        detail=f"{finetune.height}, {test.height}",
    )
    report.add_check(
        "finetune and test are patient-disjoint",
        set(finetune["person_id"].to_list()).isdisjoint(test["person_id"].to_list()),
    )
    if not report.passed:
        return _finalize(report, args.report_dir)

    for df, folder in ((finetune, Path(args.finetune_dir)), (test, Path(args.test_dir))):
        report.add_artifact(write_cohort_file(df, folder))
    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python cehrbert_model/03_build_pretraining_input.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
