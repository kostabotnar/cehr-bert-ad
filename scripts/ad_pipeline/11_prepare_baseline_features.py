#!/usr/bin/env python
"""Step 11: Build tabular feature matrices for the classical-ML baselines.

CEHR-BERT eats token sequences; L2 logistic regression and XGBoost need a fixed-width
design matrix. This step builds one from the OMOP source tables, using the *same* patient
partition CEHR-BERT used so the downstream comparison is apples-to-apples:

  * splits recovered by ``pipeline.baselines.recover_splits`` (test held out at step 6;
    val = CEHR-BERT's validation patients; train = finetune minus val)
  * features frozen strictly before each patient's ``index_date``, within a fixed
    look-back window (default 365 days = ad_finetune_config observation_window)
  * per-concept occurrence **counts** (condition / drug / procedure) + demographics
    (age at index, one-hot gender, one-hot race)
  * vocabulary fit on TRAIN only: concepts present in >= ``--min-prevalence`` of training
    patients are kept, then applied unchanged to val/test (no leakage)

Outputs (under build/baselines/features/)
  {train,val,test}_X.npz        scipy CSR design matrix
  {train,val,test}_meta.npz     y / person_ids / index_dates aligned to the matrix rows
  feature_spec.json             kept vocabulary + demographic categories + age median
  feature_names.json            column-aligned feature names
  build/reports/11_prepare_baseline_features/{status.json,report.md}

Usage
  python 11_prepare_baseline_features.py
  python 11_prepare_baseline_features.py --window-days 365 --min-prevalence 0.01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import baselines as bl  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

FEATURES_DIR = paths.BUILD_DIR / "baselines" / "features"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build baseline feature matrices")
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR))
    parser.add_argument("--finetune-dir", default=str(paths.COHORT_FINETUNE_DIR))
    parser.add_argument("--test-dir", default=str(paths.COHORT_TEST_DIR))
    parser.add_argument("--val-predictions", default=str(paths.VAL_PREDICTIONS_DIR),
                        help="CEHR-BERT validation_predictions folder (defines the val split)")
    parser.add_argument("--output-dir", default=str(FEATURES_DIR))
    parser.add_argument("--window-days", type=int, default=bl.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-prevalence", type=float, default=bl.DEFAULT_MIN_PREVALENCE)
    parser.add_argument("--report-dir", default=str(paths.build_report_dir_for("11_prepare_baseline_features")))
    args = parser.parse_args(argv)

    report = StepReport("11_prepare_baseline_features",
                        title="Step 11 - Prepare baseline feature matrices")
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir)
    report.add_metric("data_dir", str(data_dir))
    report.add_metric("window_days", args.window_days)
    report.add_metric("min_prevalence", args.min_prevalence)

    # --- recover the shared train/val/test partition -----------------------
    splits = bl.recover_splits(args.finetune_dir, args.test_dir, args.val_predictions)
    report.add_metric("val_split_source", splits.val_source)
    report.warn("val split taken from CEHR-BERT validation_predictions",
                ok=splits.val_source == "cehrbert_validation_predictions",
                detail=f"source={splits.val_source} (fallback = deterministic 10% of finetune)")
    for name, df in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        n_pos = int(df.filter(pl.col("label") == 1).height)
        report.add_metric(f"{name}_size", df.height)
        report.add_metric(f"{name}_positives", n_pos)
        report.add_check(f"{name} split non-empty with both classes",
                         df.height > 0 and 0 < n_pos < df.height,
                         detail=f"{df.height} patients, {n_pos} positive")

    # no patient leaks across the three splits
    train_ids = set(splits.train.get_column("person_id").to_list())
    val_ids = set(splits.val.get_column("person_id").to_list())
    test_ids = set(splits.test.get_column("person_id").to_list())
    leak = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    report.add_check("train/val/test patient-disjoint", not leak,
                     detail="ok" if not leak else f"{len(leak)} shared person_ids")

    # --- extract counts + demographics per split ---------------------------
    counts = {n: bl.build_counts(df, data_dir, args.window_days)
              for n, df in (("train", splits.train), ("val", splits.val), ("test", splits.test))}
    demo = {n: bl.build_demographics(df, data_dir)
            for n, df in (("train", splits.train), ("val", splits.val), ("test", splits.test))}

    # --- fit vocabulary on TRAIN, assemble every split ---------------------
    spec = bl.fit_feature_spec(
        counts["train"], demo["train"], n_train=splits.train.height,
        min_prevalence=args.min_prevalence, window_days=args.window_days,
    )
    report.add_metric("n_code_features", len(spec.code_features))
    report.add_metric("n_features_total", len(spec.feature_names))
    report.add_metric("n_gender_categories", len(spec.gender_categories))
    report.add_metric("n_race_categories", len(spec.race_categories))
    report.add_check("at least one code feature kept", len(spec.code_features) > 0,
                     detail=f"{len(spec.code_features)} concepts >= {args.min_prevalence:.1%} train prevalence")
    bl.save_feature_spec(out_dir, spec)

    cohorts = {"train": splits.train, "val": splits.val, "test": splits.test}
    for name in ("train", "val", "test"):
        X, y, pids, idates = bl.assemble_matrix(cohorts[name], counts[name], demo[name], spec)
        bl.save_split(out_dir, name, X, y, pids, idates)
        density = X.nnz / (X.shape[0] * X.shape[1]) if X.shape[1] else 0.0
        report.add_metric(f"{name}_matrix_shape", f"{X.shape[0]}x{X.shape[1]}")
        report.add_metric(f"{name}_density", round(density, 5))
        report.add_check(f"{name} matrix rows match cohort", X.shape[0] == cohorts[name].height,
                         detail=f"{X.shape[0]} rows")
        report.add_artifact(out_dir / f"{name}_X.npz")

    report.add_artifact(out_dir / "feature_spec.json")
    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    if report.passed:
        print("Next: python 12_train_baselines.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
