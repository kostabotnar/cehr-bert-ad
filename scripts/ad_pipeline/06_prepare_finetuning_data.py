#!/usr/bin/env python
"""Step 6: Prepare the finetuning dataset — build the labeled AD cohort.

Produces ``data/omop_data/ad_cohort/cohort.parquet`` with the columns the finetune
runner expects:

    person_id  (int)   patient identifier
    index_date (date)  prediction time point (chart is "frozen" here)
    label      (int)   1 = develops Alzheimer's after index_date, 0 = control

Task framing ("predict AD anytime in the future"):
  * Cases (label=1): patients with >=1 Alzheimer's diagnosis. index_date is set GAP days
    BEFORE the first AD diagnosis, giving the model a real lead time.
  * Controls (label=0): no AD diagnosis. index_date = last visit minus GAP days.
  * Inclusion: every patient needs >= MIN_HISTORY days of record (and a visit) before
    index_date, matching observation_window in ad_finetune_config.yaml.

AD is identified from source ICD codes because this dataset's concept table is
de-identified (concept_code holds strings like "ICD-10-CM:G30.9").

A final step intersects the cohort with the person_ids present in the tokenized
pretraining dataset (data/pretrain_prepared); patients without a usable sequence would
otherwise make step 8 fail. Skip with --skip-tokenized-check (e.g. before pretraining).

Usage:
    python 06_prepare_finetuning_data.py
    python 06_prepare_finetuning_data.py --gap-days 730 --skip-tokenized-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline.omop_validation import table_glob  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

# Source codes for Alzheimer's disease. Matched as a prefix against concept_code, so
# "ICD-10-CM:G30" captures G30.0/G30.1/G30.8/G30.9.
AD_CODE_PATTERNS = [r"ICD-9-CM:331\.0", r"ICD-10-CM:G30"]


def _require_glob(data_dir: Path, table: str) -> str:
    glob = table_glob(data_dir, table)
    if glob is None:
        raise FileNotFoundError(f"No parquet found for table '{table}' under {data_dir}")
    return glob


def build_cohort(
    data_dir: Path,
    gap_days: int,
    min_history_days: int,
    report: StepReport,
) -> pl.DataFrame:
    """Build the (pre-tokenized-filter) cohort dataframe and record stats on ``report``."""
    gap = pl.duration(days=gap_days)

    concept = pl.scan_parquet(_require_glob(data_dir, "concept")).select("concept_id", "concept_code")
    ad_pat = "|".join(AD_CODE_PATTERNS)
    ad_ids = (
        concept.filter(pl.col("concept_code").str.contains(ad_pat))
        .select("concept_id")
        .collect()
        .get_column("concept_id")
        .to_list()
    )
    report.add_check("AD concept ids matched", bool(ad_ids), detail=f"{len(ad_ids)} ids; patterns={AD_CODE_PATTERNS}")
    if not ad_ids:
        return pl.DataFrame(schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64})

    visits = (
        pl.scan_parquet(_require_glob(data_dir, "visit_occurrence"))
        .select("person_id", pl.col("visit_start_date").cast(pl.Date))
        .group_by("person_id")
        .agg(first_visit=pl.col("visit_start_date").min(), last_visit=pl.col("visit_start_date").max())
    )
    first_ad = (
        pl.scan_parquet(_require_glob(data_dir, "condition_occurrence"))
        .filter(pl.col("condition_concept_id").is_in(ad_ids))
        .select("person_id", pl.col("condition_start_date").cast(pl.Date))
        .group_by("person_id")
        .agg(first_ad_date=pl.col("condition_start_date").min())
    )
    persons = pl.scan_parquet(_require_glob(data_dir, "person")).select("person_id")

    df = (
        persons.join(visits, on="person_id", how="left")
        .join(first_ad, on="person_id", how="left")
        .with_columns(is_case=pl.col("first_ad_date").is_not_null())
        .with_columns(
            index_date=pl.when(pl.col("is_case"))
            .then(pl.col("first_ad_date") - gap)
            .otherwise(pl.col("last_visit") - gap),
            label=pl.col("is_case").cast(pl.Int64),
        )
        .with_columns(history_days=(pl.col("index_date") - pl.col("first_visit")).dt.total_days())
        .collect()
    )
    report.add_metric("total_persons", df.height)

    valid = df.filter(
        pl.col("index_date").is_not_null()
        & pl.col("first_visit").is_not_null()
        & (pl.col("first_visit") < pl.col("index_date"))
        & (pl.col("history_days") >= min_history_days)
    )
    report.add_metric("excluded_by_filters", df.height - valid.height)
    return valid.select("person_id", "index_date", "label").sort("person_id")


def restrict_to_tokenized(cohort: pl.DataFrame, prepared_dir: Path, report: StepReport) -> pl.DataFrame:
    """Keep only patients present in the tokenized pretraining dataset."""
    from pipeline.dataset_checks import count_person_ids, datasets_available, resolve_prepared_path

    if not prepared_dir.exists() or resolve_prepared_path(prepared_dir) is None:
        report.warn("tokenized dataset available for verification", ok=False,
                    detail=f"{prepared_dir} not found; run pretraining first or use --skip-tokenized-check")
        return cohort
    if not datasets_available():
        report.warn("tokenized dataset verification", ok=False,
                    detail="`datasets` not installed; cannot intersect cohort with tokenized ids")
        return cohort

    tokenized_ids = count_person_ids(prepared_dir)
    before = cohort.height
    cohort = cohort.filter(pl.col("person_id").is_in(list(tokenized_ids)))
    report.add_metric("tokenized_patients", len(tokenized_ids))
    report.add_metric("dropped_no_sequence", before - cohort.height)
    report.add_check("cohort patients remain after tokenized intersect", cohort.height > 0,
                     detail=f"{cohort.height} remain of {before}")
    return cohort


def split_finetune_test(cohort: pl.DataFrame, test_holdout: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Patient-level split into (finetune, test). No person_id appears in both.

    Deterministic: patients are ordered by a stable hash of (seed, person_id), and the
    first ``test_holdout`` fraction become the test set.
    """
    import hashlib

    def _bucket(pid: int) -> int:
        # Stable per-patient hash in [0, 1_000_000); independent of row order / polars version.
        h = hashlib.md5(f"{seed}:{pid}".encode(), usedforsecurity=False).hexdigest()
        return int(h[:8], 16) % 1_000_000

    threshold = int(test_holdout * 1_000_000)
    with_bucket = cohort.with_columns(
        pl.col("person_id").map_elements(_bucket, return_dtype=pl.Int64).alias("_bucket")
    )
    test = with_bucket.filter(pl.col("_bucket") < threshold).drop("_bucket").sort("person_id")
    finetune = with_bucket.filter(pl.col("_bucket") >= threshold).drop("_bucket").sort("person_id")
    return finetune, test


def _split_stats(report: StepReport, name: str, df: pl.DataFrame) -> None:
    n_case = int(df.filter(pl.col("label") == 1).height)
    n_ctrl = int(df.filter(pl.col("label") == 0).height)
    report.add_metric(f"{name}_size", df.height)
    report.add_metric(f"{name}_cases", n_case)
    report.add_metric(f"{name}_controls", n_ctrl)
    report.add_metric(f"{name}_prevalence", round(n_case / max(df.height, 1), 4))
    report.warn(f"{name} cohort has both classes", ok=n_case > 0 and n_ctrl > 0,
                detail=f"{n_case} cases, {n_ctrl} controls")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build labeled AD cohort (finetuning dataset)")
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR))
    parser.add_argument("--finetune-output", default=str(paths.COHORT_FINETUNE_FILE),
                        help="Output parquet for the fine-tuning cohort")
    parser.add_argument("--test-output", default=str(paths.COHORT_TEST_FILE),
                        help="Output parquet for the held-out test cohort (used by step 9 prediction)")
    parser.add_argument("--test-holdout", type=float, default=0.1,
                        help="Fraction of patients held out for the test set (default: 0.1)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the patient-level finetune/test split (default: 42)")
    parser.add_argument("--gap-days", type=int, default=365, help="Lead time between index_date and first AD dx")
    parser.add_argument("--min-history-days", type=int, default=182, help="Required history before index_date")
    parser.add_argument("--tokenized-dataset-path", default=str(paths.PRETRAIN_PREPARED_DIR))
    parser.add_argument("--skip-tokenized-check", action="store_true",
                        help="Do not restrict to person_ids present in the tokenized dataset")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("06_prepare_finetuning_data")))
    args = parser.parse_args(argv)

    report = StepReport("06_prepare_finetuning_data", title="Step 6 — Prepare finetuning data (AD cohort)")
    data_dir = Path(args.data_dir).resolve()
    report.add_metric("data_dir", str(data_dir))
    report.add_metric("gap_days", args.gap_days)
    report.add_metric("min_history_days", args.min_history_days)

    try:
        cohort = build_cohort(data_dir, args.gap_days, args.min_history_days, report)
    except FileNotFoundError as exc:
        report.add_check("required OMOP tables present", False, detail=str(exc))
        return _finalize(report, args.report_dir)

    if args.skip_tokenized_check:
        report.note("tokenized-data verification skipped (--skip-tokenized-check)")
    else:
        cohort = restrict_to_tokenized(cohort, Path(args.tokenized_dataset_path), report)

    n_case = int(cohort.filter(pl.col("label") == 1).height)
    n_ctrl = int(cohort.filter(pl.col("label") == 0).height)
    report.add_metric("cohort_size", cohort.height)
    report.add_metric("cases", n_case)
    report.add_metric("controls", n_ctrl)
    report.add_metric("prevalence", round(n_case / max(cohort.height, 1), 4))
    if cohort.height:
        report.add_metric("index_date_min", str(cohort["index_date"].min()))
        report.add_metric("index_date_max", str(cohort["index_date"].max()))
    report.add_check("cohort is non-empty", cohort.height > 0, detail=f"{cohort.height} patients")
    report.warn("cohort has both classes", ok=n_case > 0 and n_ctrl > 0, detail=f"{n_case} cases, {n_ctrl} controls")

    if not (0 < args.test_holdout < 1):
        report.add_check("test-holdout in (0, 1)", False, detail=f"test_holdout={args.test_holdout}")
        return _finalize(report, args.report_dir)

    # --- patient-level finetune/test split -----------------------------------
    finetune_path = Path(args.finetune_output)
    test_path = Path(args.test_output)
    report.add_metric("test_holdout", args.test_holdout)
    report.add_metric("split_seed", args.split_seed)

    if cohort.height > 0:
        finetune_df, test_df = split_finetune_test(cohort, args.test_holdout, args.split_seed)

        # No patient leaks across the split.
        overlap = set(finetune_df["person_id"].to_list()) & set(test_df["person_id"].to_list())
        report.add_check("finetune/test patient-disjoint", not overlap,
                         detail="ok" if not overlap else f"{len(overlap)} shared person_ids")

        _split_stats(report, "finetune", finetune_df)
        _split_stats(report, "test", test_df)
        report.add_check("both splits non-empty", finetune_df.height > 0 and test_df.height > 0,
                         detail=f"finetune={finetune_df.height}, test={test_df.height}")

        for path, df in ((finetune_path, finetune_df), (test_path, test_df)):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
            report.add_artifact(path)
        report.ok("finetune & test cohorts written",
                  detail=f"{finetune_path}  |  {test_path}")

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python 07_validate_finetune_config.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
