#!/usr/bin/env python
"""Step 6: Prepare the finetuning dataset — build the labeled AD cohort.

Produces ``data/omop_data/ad_cohort/cohort.parquet`` with the columns the finetune
runner expects:

    person_id  (int)   patient identifier
    index_date (date)  prediction time point (chart is "frozen" here)
    label      (int)   1 = develops Alzheimer's after index_date, 0 = control

Task framing ("predict AD anytime in the future"):
  * Cases (label=1): patients with >=1 Alzheimer's diagnosis. Anchor = first AD diagnosis,
    index_date = anchor minus GAP days, giving the model a real lead time.
  * Controls (label=0): no AD diagnosis and >=1 visit. Anchor = one of the control's own visit
    dates, with >= --control-followup-days of observation after it. The anchor year is drawn so
    control anchor years follow the case anchor years (capped at cutoff minus follow-up).
    index_date = anchor minus GAP days.
  * Inclusion, identical for both groups at index_date: age >= --min-age, index_date >=
    --index-floor, first visit <= index_date minus --window-days, and >= 1 condition, drug,
    procedure or measurement in [index_date - window, index_date).
  See pipeline/ad_cohort.py for the exact definitions.

AD is identified from condition_occurrence.condition_source_value, which holds strings
like "ICD-10-CM:G30.9". condition_concept_id holds standard SNOMED ids, and the concept
table uses plain codes (vocabulary ICD10CM, concept_code "G30.9"), so neither can be
matched against the "ICD-10-CM:" strings.

A final step keeps only patients with a usable sequence; the others would make step 8 fail.
``--intersect-with`` picks the source of the person_ids:
  * ``sequences`` (default): the raw sequences of step 2 (data/cehrbert_data/patient_sequence)
    with at least one concept, the ones tokenization keeps. Works before pretraining, which is
    where this step now runs: steps 3b and 4 need its test split to exclude test patients.
  * ``tokenized``: the full tokenized dataset of step 5b (data/full_tokenized).
  * ``none``: no intersection (same as --skip-tokenized-check).

Usage:
    python 06_prepare_finetuning_data.py
    python 06_prepare_finetuning_data.py --gap-days 730 --intersect-with none
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import ad_cohort, paths  # noqa: E402
from pipeline.omop_common import AD_CODE_PATTERNS, is_ad_source_value  # noqa: E402
from pipeline.omop_validation import table_glob  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

def _require_glob(data_dir: Path, table: str) -> str:
    glob = table_glob(data_dir, table)
    if glob is None:
        raise FileNotFoundError(f"No parquet found for table '{table}' under {data_dir}")
    return glob


EMPTY_COHORT = {"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64}
SMD_WARN = 0.1


def _add_counts(report: StepReport, prefix: str, counts: dict[str, int]) -> None:
    for name, value in counts.items():
        report.add_metric(f"{prefix}_{name}", value)


def build_cohort(
    data_dir: Path,
    gap_days: int,
    report: StepReport,
    window_days: int = 365,
    min_age: int = 50,
    index_floor: dt.date = dt.date(2000, 1, 1),
    followup_days: int = 365,
    seed: int = 42,
) -> pl.DataFrame:
    """Build the (pre-intersection) cohort dataframe and record stats on ``report``.

    See ``pipeline/ad_cohort.py`` for the anchor, index and inclusion definitions.
    """
    ad_rows = (
        pl.scan_parquet(_require_glob(data_dir, "condition_occurrence"))
        .filter(is_ad_source_value())
        .select("person_id", pl.col("condition_start_date").cast(pl.Date))
    )
    n_rows, n_patients = ad_rows.select(pl.len(), pl.col("person_id").n_unique()).collect().row(0)
    report.add_metric("ad_condition_rows", n_rows)
    report.add_metric("ad_condition_patients", n_patients)
    report.add_check(
        "AD condition rows found",
        n_rows > 0,
        detail=f"{n_rows} rows, {n_patients} patients; patterns={AD_CODE_PATTERNS}",
    )
    if n_rows == 0:
        return pl.DataFrame(schema=EMPTY_COHORT)

    persons = ad_cohort.read_persons(data_dir)
    first_ad = (
        ad_rows.group_by("person_id")
        .agg(anchor_date=pl.col("condition_start_date").min())
        .collect(engine="streaming")
    )
    visit_dates = (
        pl.scan_parquet(_require_glob(data_dir, "visit_occurrence"))
        .select("person_id", pl.col("visit_start_date").cast(pl.Date).alias("visit_date"))
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
    )
    vsum = visit_dates.group_by("person_id").agg(
        first_visit=pl.col("visit_date").min(), last_visit=pl.col("visit_date").max()
    )
    report.add_metric("total_persons", persons.height)

    release = ad_cohort.read_release_date(data_dir)
    cutoff = release if release is not None else visit_dates["visit_date"].max()
    cap_date = cutoff - dt.timedelta(days=followup_days)
    report.add_metric("data_cutoff", str(cutoff))
    report.add_metric(
        "data_cutoff_source", "cdm_source.source_release_date" if release is not None else "max visit date"
    )
    report.add_metric("target_cap_date", str(cap_date))

    # Cases are not capped by the cutoff: recent cases stay in the cohort and only the control target is capped.
    cases = persons.join(first_ad, on="person_id").join(vsum, on="person_id", how="left")
    cases = ad_cohort.add_rule_flags(cases, gap_days, window_days, min_age, index_floor)
    cases, case_counts = ad_cohort.apply_rules(cases, data_dir, window_days)
    _add_counts(report, "cases", case_counts)

    capped = cases.filter(pl.col("anchor_date") <= cap_date)["anchor_date"]
    target = ad_cohort.year_counts(capped)
    report.add_metric("cases_beyond_cap", cases.height - capped.len())
    report.add_metric("target_anchor_year_counts", {str(y): n for y, n in target.items()})

    controls = persons.join(first_ad.select("person_id"), on="person_id", how="anti")
    controls_with_visits = controls.join(vsum, on="person_id")
    report.add_metric("controls_no_ad_persons", controls.height)
    report.add_metric("controls_no_visits", controls.height - controls_with_visits.height)
    candidates, cand_counts = ad_cohort.control_candidates(
        visit_dates.join(controls_with_visits.select("person_id"), on="person_id"),
        controls_with_visits,
        gap_days,
        window_days,
        min_age,
        index_floor,
        followup_days,
    )
    _add_counts(report, "control_candidate_dates", cand_counts)
    target_df = pl.DataFrame(
        {"year": list(target), "n_case": list(target.values())}, schema={"year": pl.Int64, "n_case": pl.Int64}
    )
    picked, draw_info = ad_cohort.draw_control_anchors(candidates, target_df, seed)
    report.add_metric("controls_no_eligible_candidate", controls_with_visits.height - draw_info["n_with_candidates"])
    report.add_metric("controls_zero_target_weight_only", draw_info["n_zero_target"])
    report.add_metric("controls_drawn", picked.height)

    drawn = ad_cohort.add_rule_flags(
        picked.join(controls_with_visits, on="person_id"), gap_days, window_days, min_age, index_floor
    )
    drawn, control_counts = ad_cohort.apply_rules(drawn, data_dir, window_days)
    _add_counts(report, "controls", control_counts)

    achieved = ad_cohort.year_counts(drawn["anchor_date"])
    report.add_metric("achieved_control_anchor_year_counts", {str(y): n for y, n in achieved.items()})
    smd = ad_cohort.anchor_year_smd(capped, drawn["anchor_date"])
    report.add_metric("anchor_year_smd", None if smd is None else round(smd, 4))
    report.warn(
        f"control anchor years match capped case anchor years (|SMD| <= {SMD_WARN})",
        ok=smd is not None and abs(smd) <= SMD_WARN,
        detail=f"SMD={smd}",
    )
    for name, df in (("cases", cases), ("controls", drawn)):
        report.add_metric(f"{name}_mean_age_at_index", round(df["age"].mean(), 2) if df.height else None)

    out = pl.concat(
        [
            cases.select("person_id", "index_date", pl.lit(1, dtype=pl.Int64).alias("label")),
            drawn.select("person_id", "index_date", pl.lit(0, dtype=pl.Int64).alias("label")),
        ]
    )
    report.add_metric("excluded_by_filters", persons.height - out.height)
    return out.sort("person_id")


def restrict_to_tokenized(cohort: pl.DataFrame, prepared_dir: Path, report: StepReport) -> pl.DataFrame:
    """Keep only patients present in the full tokenized dataset (step 5b)."""
    from pipeline.dataset_checks import count_person_ids, datasets_available, resolve_prepared_path

    if not prepared_dir.exists() or resolve_prepared_path(prepared_dir) is None:
        report.warn("tokenized dataset available for verification", ok=False,
                    detail=f"{prepared_dir} not found; run step 5b first or use --intersect-with sequences")
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


def restrict_to_sequences(cohort: pl.DataFrame, sequence_dir: Path, report: StepReport) -> pl.DataFrame | None:
    """Keep only patients with a non-empty sequence in ``sequence_dir``; None (failed check) if it has none."""
    from pipeline.pretrain_data import sequence_person_ids, sequence_files

    if not report.add_check(
        "patient sequences available", bool(sequence_files(sequence_dir)), detail=f"{sequence_dir}; run step 2 first"
    ):
        return None
    sequence_ids = sequence_person_ids(sequence_dir, require_concepts=True)
    before = cohort.height
    cohort = cohort.filter(pl.col("person_id").is_in(list(sequence_ids)))
    report.add_metric("sequence_patients", len(sequence_ids))
    report.add_metric("dropped_no_sequence", before - cohort.height)
    report.add_check(
        "cohort patients remain after sequence intersect",
        cohort.height > 0,
        detail=f"{cohort.height} remain of {before}",
    )
    return cohort


def _bucket(seed: int, pid: int) -> int:
    # Stable per-patient hash in [0, 1_000_000); independent of row order / polars version.
    h = hashlib.md5(f"{seed}:{pid}".encode(), usedforsecurity=False).hexdigest()
    return int(h[:8], 16) % 1_000_000


def _hash_split(cohort: pl.DataFrame, fraction: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (rest, selected) where selected holds patients whose bucket is below ``fraction``."""
    threshold = int(fraction * 1_000_000)
    with_bucket = cohort.with_columns(
        pl.col("person_id").map_elements(lambda pid: _bucket(seed, pid), return_dtype=pl.Int64).alias("_bucket")
    )
    selected = with_bucket.filter(pl.col("_bucket") < threshold).drop("_bucket").sort("person_id")
    rest = with_bucket.filter(pl.col("_bucket") >= threshold).drop("_bucket").sort("person_id")
    return rest, selected


def split_train_val(finetune: pl.DataFrame, val_fraction: float, val_seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Patient-level split of the fine-tune cohort into (train, val), hashed with ``val_seed``."""
    return _hash_split(finetune, val_fraction, val_seed)


def build_splits_table(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """Stack the three splits into one table with a string ``split`` column."""
    parts = [df.with_columns(split=pl.lit(name)) for name, df in (("train", train), ("val", val), ("test", test))]
    return pl.concat(parts).select("person_id", "index_date", "label", "split").sort("person_id")


def split_finetune_test(cohort: pl.DataFrame, test_holdout: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Patient-level split into (finetune, test). No person_id appears in both.

    Deterministic: patients are ordered by a stable hash of (seed, person_id), and the
    first ``test_holdout`` fraction become the test set.
    """
    return _hash_split(cohort, test_holdout, seed)


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
    parser.add_argument("--splits-output", default=str(paths.COHORT_SPLITS_FILE),
                        help="Output parquet with the canonical train/val/test split per patient")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of the fine-tune cohort assigned to val (default: 0.1)")
    parser.add_argument("--val-seed", type=int, default=43,
                        help="Seed for the train/val split; differs from --split-seed so hash orders are independent")
    parser.add_argument("--gap-days", type=int, default=365, help="Lead time between index_date and first AD dx")
    parser.add_argument("--window-days", type=int, default=365,
                        help="Observation window before index_date (matches observation_window in the finetune config)")
    parser.add_argument("--min-age", type=int, default=50, help="Minimum age in years at index_date")
    parser.add_argument("--index-floor", type=dt.date.fromisoformat, default=dt.date(2000, 1, 1),
                        help="Earliest allowed index_date (YYYY-MM-DD)")
    parser.add_argument("--control-followup-days", type=int, default=365,
                        help="Observation required after a control anchor (last visit >= anchor + this)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the control anchor draw")
    parser.add_argument(
        "--intersect-with",
        choices=("sequences", "tokenized", "none"),
        default="sequences",
        help="Keep patients present in the raw sequences (default), the full tokenized "
        "dataset of step 5b, or skip the intersection",
    )
    parser.add_argument("--patient-sequence-dir", default=str(paths.PATIENT_SEQUENCE_DIR))
    parser.add_argument("--tokenized-dataset-path", default=str(paths.FULL_TOKENIZED_DIR))
    parser.add_argument("--skip-tokenized-check", action="store_true", help="Same as --intersect-with none")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("06_prepare_finetuning_data")))
    args = parser.parse_args(argv)

    report = StepReport("06_prepare_finetuning_data", title="Step 6 — Prepare finetuning data (AD cohort)")
    data_dir = Path(args.data_dir).resolve()
    report.add_metric("data_dir", str(data_dir))
    report.add_metric("gap_days", args.gap_days)
    report.add_metric("window_days", args.window_days)
    report.add_metric("min_age", args.min_age)
    report.add_metric("index_floor", str(args.index_floor))
    report.add_metric("control_followup_days", args.control_followup_days)
    report.add_metric("seed", args.seed)

    try:
        cohort = build_cohort(
            data_dir,
            args.gap_days,
            report,
            window_days=args.window_days,
            min_age=args.min_age,
            index_floor=args.index_floor,
            followup_days=args.control_followup_days,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        report.add_check("required OMOP tables present", False, detail=str(exc))
        return _finalize(report, args.report_dir)

    mode = "none" if args.skip_tokenized_check else args.intersect_with
    report.add_metric("intersect_with", mode)
    if mode == "none":
        report.note("sequence/tokenized-data intersection skipped")
    elif mode == "tokenized":
        cohort = restrict_to_tokenized(cohort, Path(args.tokenized_dataset_path), report)
    else:
        restricted = restrict_to_sequences(cohort, Path(args.patient_sequence_dir), report)
        if restricted is None:
            return _finalize(report, args.report_dir)
        cohort = restricted

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
    if not (0 < args.val_fraction < 1):
        report.add_check("val-fraction in (0, 1)", False, detail=f"val_fraction={args.val_fraction}")
        return _finalize(report, args.report_dir)

    # --- patient-level finetune/test split -----------------------------------
    finetune_path = Path(args.finetune_output)
    test_path = Path(args.test_output)
    report.add_metric("test_holdout", args.test_holdout)
    report.add_metric("split_seed", args.split_seed)
    report.add_metric("val_fraction", args.val_fraction)
    report.add_metric("val_seed", args.val_seed)

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

        train_df, val_df = split_train_val(finetune_df, args.val_fraction, args.val_seed)
        splits_df = build_splits_table(train_df, val_df, test_df)
        report.add_check("train/val/test cover the cohort exactly",
                         splits_df.height == cohort.height and splits_df["person_id"].n_unique() == cohort.height,
                         detail=f"{splits_df.height} rows for {cohort.height} patients")
        _split_stats(report, "split_train", train_df)
        _split_stats(report, "split_val", val_df)
        _split_stats(report, "split_test", test_df)
        splits_path = Path(args.splits_output)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_df.write_parquet(splits_path)
        report.add_artifact(splits_path)
        report.ok("canonical splits written", detail=str(splits_path))

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python 03b_build_pretraining_input.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
