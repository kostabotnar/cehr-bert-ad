#!/usr/bin/env python
"""Step 02: Descriptive statistics of patient history length.

history_days = index_date - first_visit_date, where the cohort (index_date + label) comes from the
canonical ``splits.parquet`` and the first visit per person comes from ``visit_occurrence``.
Reported overall and split by label (cases vs controls). Reads only the OMOP tables and the cohort.

Outputs
-------
  build/data_qc/history_length/history_length_stats.json
  build/data_qc/history_length/figures/history_length_hist.png
  scripts/data_qc/reports/02_history_length_stats/{status.json,report.md}

Usage
-----
  python data_qc/02_history_length_stats.py
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import paths  # noqa: E402
from pipeline.omop_validation import table_glob  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

PART = "data_qc"
STEP = "02_history_length_stats"
PERCENTILES = [50, 75, 90, 95, 99]


def _describe(values: np.ndarray) -> dict:
    """Return count/mean/max plus p50/p75/p90/p95/p99 for a 1-D array."""
    values = np.asarray(values, dtype=np.float64)
    out: dict = {"count": int(values.size)}
    if values.size == 0:
        out["mean"] = None
        out["max"] = None
        for p in PERCENTILES:
            out[f"p{p}"] = None
        return out
    out["mean"] = float(values.mean())
    out["max"] = float(values.max())
    pct = np.percentile(values, PERCENTILES)
    for p, v in zip(PERCENTILES, pct):
        out[f"p{p}"] = float(v)
    return out


def _fmt(stats: dict) -> str:
    """Compact ASCII one-liner of a describe() dict for the report detail."""
    if stats.get("count", 0) == 0:
        return "n=0"
    return (
        f"n={stats['count']}, mean={stats['mean']:.0f}, "
        f"p50={stats['p50']:.0f}, p90={stats['p90']:.0f}, "
        f"p99={stats['p99']:.0f}, max={stats['max']:.0f}"
    )


def _load_cohort(report: StepReport, splits_file: Path) -> pl.DataFrame:
    """Read the canonical splits file as (person_id, index_date, label)."""
    if not splits_file.is_file():
        return pl.DataFrame(schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64})
    df = pl.read_parquet(splits_file, columns=["person_id", "index_date", "label"]).with_columns(
        pl.col("index_date").cast(pl.Date),
        pl.col("label").cast(pl.Int64),
    )
    report.add_metric("cohort_rows", df.height)
    return df


def _history_days(cohort: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    """Join cohort to first visit_start_date per person and compute history_days."""
    vo_glob = table_glob(data_dir, "visit_occurrence")
    if vo_glob is None:
        raise FileNotFoundError(f"visit_occurrence not found under {data_dir}")
    vo_files = sorted(globmod.glob(vo_glob))
    first_visit = (
        pl.read_parquet(vo_files, columns=["person_id", "visit_start_date"])
        .with_columns(pl.col("visit_start_date").cast(pl.Date))
        .group_by("person_id")
        .agg(pl.col("visit_start_date").min().alias("first_visit"))
    )
    joined = cohort.join(first_visit, on="person_id", how="inner").with_columns(
        (pl.col("index_date") - pl.col("first_visit")).dt.total_days().alias("history_days")
    )
    return joined.filter(pl.col("history_days").is_not_null())


def _plot_history_hist(history: pl.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    days = history.get_column("history_days").to_numpy().astype(np.float64)
    cases = history.filter(pl.col("label") == 1).get_column("history_days").to_numpy().astype(np.float64)
    controls = history.filter(pl.col("label") == 0).get_column("history_days").to_numpy().astype(np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    hi = float(np.percentile(days, 99)) if days.size else 1.0
    bins = np.linspace(0, max(hi, 1.0), 60)
    if controls.size:
        ax.hist(controls, bins=bins, alpha=0.55, label=f"controls (n={controls.size})", color="#3b6ea5")
    if cases.size:
        ax.hist(cases, bins=bins, alpha=0.55, label=f"cases (n={cases.size})", color="#c0504d")
    ax.set_xlabel("patient history length (days)")
    ax.set_ylabel("number of patients")
    ax.set_title("Patient history length: cases vs controls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patient history length statistics")
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR))
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE))
    parser.add_argument("--results-dir", default=str(paths.HISTORY_LENGTH_DIR))
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--report-dir", default=str(paths.report_dir_for(PART, STEP)))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"

    report = StepReport(STEP, title="Step 02 - Patient history length statistics")
    payload: dict = {}

    cohort = _load_cohort(report, Path(args.splits_file))
    history = pl.DataFrame(schema={"person_id": pl.Int64, "label": pl.Int64, "history_days": pl.Int64})
    cohort_ok = cohort.height > 0
    report.add_check("cohort readable", cohort_ok, detail=f"{cohort.height} cohort rows")
    if cohort_ok:
        try:
            history = _history_days(cohort, Path(args.data_dir))
        except Exception as exc:  # noqa: BLE001
            report.add_check("history_days computed", False, detail=str(exc))

    empty = np.asarray([], dtype=np.int64)

    def days_for(label: int | None) -> np.ndarray:
        if not history.height:
            return empty
        frame = history if label is None else history.filter(pl.col("label") == label)
        return frame.get_column("history_days").to_numpy().astype(np.int64)

    hist_all, cases, controls = days_for(None), days_for(1), days_for(0)
    payload["history_days"] = {
        "overall": _describe(hist_all),
        "cases": _describe(cases),
        "controls": _describe(controls),
    }
    if history.height:
        report.add_check("history_days computed", hist_all.size > 0, detail=f"overall {_fmt(_describe(hist_all))}")
        report.add_metric("n_patients_history", int(hist_all.size))
        report.add_metric("history_days_median", round(_describe(hist_all)["p50"]))
        report.add_metric("history_days_p90", round(_describe(hist_all)["p90"]))
        report.add_artifact(_plot_history_hist(history, figures_dir / "history_length_hist.png"))

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "history_length_stats.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    report.add_artifact(json_path)

    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
