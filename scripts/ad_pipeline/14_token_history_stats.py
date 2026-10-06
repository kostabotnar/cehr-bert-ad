#!/usr/bin/env python
"""Step 14: Descriptive statistics of token counts and patient history length.

Profiles two quantities that drive the modeling budget and cohort framing:

  (1) Token counts per patient sequence
      - Full-history  : number of concepts per row in the CEHR-BERT
                        ``patient_sequence`` parquet (``concept_ids`` list length).
      - Windowed      : number of ``input_ids`` per row in the tokenized
                        ``finetune_prepared`` DatasetDict (train + validation).
      For each, the distribution (mean/percentiles/max) plus the share of
      sequences exceeding the context length (``--ctx``) and exceeding 512.

  (2) Patient history length in days
      - history_days = index_date - first_visit_date, where the cohort
        (index_date + label) comes from the finetune + test cohort parquet and
        the first visit per person comes from ``visit_occurrence``.
      Reported overall and split by label (cases vs controls).

Outputs (under build/<run_label>/)
-------
  build/<label>/token_history_stats.json
  build/<label>/figures/token_length_hist.png
  build/<label>/figures/history_length_hist.png
  build/<label>/reports/14_token_history_stats/{status.json,report.md}

Usage
-----
  python 14_token_history_stats.py
  python 14_token_history_stats.py --window-days 365 --ctx 512
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline.dataset_checks import datasets_available, resolve_prepared_path  # noqa: E402
from pipeline.omop_validation import table_glob  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

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


def _pct_over(values: np.ndarray, threshold: float) -> float:
    """Share (0..1) of values strictly greater than ``threshold``."""
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float((values > threshold).mean())


def _fmt(stats: dict) -> str:
    """Compact ASCII one-liner of a describe() dict for the report detail."""
    if stats.get("count", 0) == 0:
        return "n=0"
    return (
        f"n={stats['count']}, mean={stats['mean']:.0f}, "
        f"p50={stats['p50']:.0f}, p90={stats['p90']:.0f}, "
        f"p99={stats['p99']:.0f}, max={stats['max']:.0f}"
    )


def _full_history_token_lengths(ps_dir: Path) -> np.ndarray:
    """Length of the ``concept_ids`` list per row across the patient_sequence parquet."""
    files = sorted(globmod.glob(str(ps_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {ps_dir}")
    lengths = (
        pl.read_parquet(files, columns=["concept_ids"])
        .select(pl.col("concept_ids").list.len().alias("n"))
        .get_column("n")
        .to_numpy()
    )
    return np.asarray(lengths, dtype=np.int64)


def _windowed_token_lengths(prepared_dir: Path) -> dict[str, np.ndarray]:
    """Per-split arrays of ``len(input_ids)`` from the tokenized finetune dataset."""
    from datasets import load_from_disk

    resolved = resolve_prepared_path(prepared_dir)
    if resolved is None:
        raise FileNotFoundError(f"no loadable DatasetDict under {prepared_dir}")
    dataset = load_from_disk(resolved)
    per_split: dict[str, np.ndarray] = {}
    splits = list(dataset.keys()) if hasattr(dataset, "keys") else ["__single__"]
    for name in splits:
        split = dataset[name] if name != "__single__" else dataset
        lengths = [len(x) for x in split["input_ids"]]
        per_split[name] = np.asarray(lengths, dtype=np.int64)
    return per_split


def _load_cohort(report: StepReport) -> pl.DataFrame:
    """Read finetune + test cohort parquet and concatenate (person_id, index_date, label)."""
    frames = []
    for tag, folder in (("finetune", paths.COHORT_FINETUNE_DIR), ("test", paths.COHORT_TEST_DIR)):
        files = sorted(globmod.glob(str(Path(folder) / "*.parquet")))
        if not files:
            continue
        df = pl.read_parquet(files, columns=["person_id", "index_date", "label"]).with_columns(
            pl.col("index_date").cast(pl.Date),
            pl.col("label").cast(pl.Int64),
        )
        report.add_metric(f"cohort_{tag}_rows", df.height)
        frames.append(df)
    if not frames:
        return pl.DataFrame(schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64})
    return pl.concat(frames, how="vertical")


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


def _plot_token_hist(full: np.ndarray, windowed: np.ndarray | None, ctx: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    full_pos = full[full > 0]
    lo = max(1, int(full_pos.min()) if full_pos.size else 1)
    hi = int(full.max()) if full.size else 1
    if windowed is not None and windowed.size:
        hi = max(hi, int(windowed.max()))
    bins = np.logspace(np.log10(lo), np.log10(max(hi, lo + 1)), 60)
    ax.hist(full_pos, bins=bins, alpha=0.55, label=f"full history (n={full.size})", color="#3b6ea5")
    if windowed is not None and windowed.size:
        win_pos = windowed[windowed > 0]
        ax.hist(win_pos, bins=bins, alpha=0.55, label=f"windowed (n={windowed.size})", color="#c0504d")
    ax.axvline(ctx, color="black", linestyle="--", linewidth=1.2, label=f"ctx = {ctx}")
    ax.set_xscale("log")
    ax.set_xlabel("tokens per patient sequence (log scale)")
    ax.set_ylabel("number of patients")
    ax.set_title("Token-count distribution: full history vs windowed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


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
    parser = argparse.ArgumentParser(description="Token-count and history-length statistics")
    parser.add_argument("--window-days", type=int, default=paths.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX)
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR))
    parser.add_argument("--patient-sequence-dir", default=str(paths.PATIENT_SEQUENCE_DIR))
    parser.add_argument("--prepared-dir", default=str(paths.FINETUNE_PREPARED_DIR))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    results_dir = Path(args.results_dir) if args.results_dir else paths.eval_results_dir(label)
    figures_dir = Path(args.figures_dir) if args.figures_dir else paths.figures_dir(label)
    report_dir = Path(args.report_dir) if args.report_dir else paths.build_report_dir_for("14_token_history_stats", label)

    report = StepReport(
        "14_token_history_stats",
        title="Step 14 - Token-count and history-length statistics",
    )
    report.add_metric("run_label", label)
    report.add_metric("ctx", args.ctx)

    payload: dict = {"run_label": label, "ctx": args.ctx}

    # --- (1a) full-history token counts -----------------------------------
    ps_dir = Path(args.patient_sequence_dir)
    full_lengths = np.asarray([], dtype=np.int64)
    try:
        full_lengths = _full_history_token_lengths(ps_dir)
        ok = full_lengths.size > 0
    except Exception as exc:  # noqa: BLE001
        ok = False
        report.add_check("patient_sequence readable", False, detail=str(exc))
    if full_lengths.size > 0:
        report.add_check("patient_sequence readable", True, detail=f"{full_lengths.size} sequences at {ps_dir}")
    elif ok is False and not any(c.name == "patient_sequence readable" for c in report.checks):
        report.add_check("patient_sequence readable", False, detail=str(ps_dir))

    full_stats = _describe(full_lengths)
    full_over_ctx = _pct_over(full_lengths, args.ctx)
    full_over_512 = _pct_over(full_lengths, 512)
    payload["full_history_tokens"] = {
        **full_stats,
        "pct_over_ctx": full_over_ctx,
        "pct_over_512": full_over_512,
    }
    report.add_check(
        "full-history token lengths computed",
        full_stats["count"] > 0,
        detail=_fmt(full_stats),
    )
    if full_stats["count"] > 0:
        report.add_metric("full_tokens_median", round(full_stats["p50"]))
        report.add_metric("full_tokens_p90", round(full_stats["p90"]))
        report.add_metric("full_tokens_p99", round(full_stats["p99"]))
        report.add_metric("full_pct_over_ctx", round(full_over_ctx, 4))
        report.add_metric("full_pct_over_512", round(full_over_512, 4))

    # --- (1b) windowed token counts ---------------------------------------
    windowed_combined: np.ndarray | None = None
    if not datasets_available():
        report.warn("windowed dataset available", ok=False, detail="`datasets` not installed; skipped")
        payload["windowed_tokens"] = None
    else:
        try:
            per_split = _windowed_token_lengths(Path(args.prepared_dir))
            windowed_combined = (
                np.concatenate(list(per_split.values())) if per_split else np.asarray([], dtype=np.int64)
            )
            win_block: dict = {"per_split": {}, "combined": {}}
            for name, arr in per_split.items():
                win_block["per_split"][name] = {
                    **_describe(arr),
                    "pct_over_ctx": _pct_over(arr, args.ctx),
                }
            win_block["combined"] = {
                **_describe(windowed_combined),
                "pct_over_ctx": _pct_over(windowed_combined, args.ctx),
            }
            payload["windowed_tokens"] = win_block
            combined_stats = win_block["combined"]
            report.add_check(
                "windowed token lengths computed",
                combined_stats["count"] > 0,
                detail=_fmt(combined_stats),
            )
            if combined_stats["count"] > 0:
                report.add_metric("windowed_tokens_median", round(combined_stats["p50"]))
                report.add_metric("windowed_tokens_p90", round(combined_stats["p90"]))
                report.add_metric("windowed_tokens_p99", round(combined_stats["p99"]))
                report.add_metric(
                    "windowed_pct_over_ctx", round(_pct_over(windowed_combined, args.ctx), 4)
                )
        except Exception as exc:  # noqa: BLE001
            report.warn("windowed dataset available", ok=False, detail=str(exc))
            payload["windowed_tokens"] = None

    # --- (2) history length in days ---------------------------------------
    cohort = _load_cohort(report)
    history = pl.DataFrame(schema={"person_id": pl.Int64, "label": pl.Int64, "history_days": pl.Int64})
    cohort_ok = cohort.height > 0
    report.add_check("cohort readable", cohort_ok, detail=f"{cohort.height} cohort rows")
    if cohort_ok:
        try:
            history = _history_days(cohort, Path(args.data_dir))
        except Exception as exc:  # noqa: BLE001
            report.add_check("history_days computed", False, detail=str(exc))

    hist_all = history.get_column("history_days").to_numpy().astype(np.int64) if history.height else np.asarray([], dtype=np.int64)
    cases = history.filter(pl.col("label") == 1).get_column("history_days").to_numpy().astype(np.int64) if history.height else np.asarray([], dtype=np.int64)
    controls = history.filter(pl.col("label") == 0).get_column("history_days").to_numpy().astype(np.int64) if history.height else np.asarray([], dtype=np.int64)

    payload["history_days"] = {
        "overall": _describe(hist_all),
        "cases": _describe(cases),
        "controls": _describe(controls),
    }
    if history.height:
        report.add_check(
            "history_days computed",
            hist_all.size > 0,
            detail=f"overall {_fmt(_describe(hist_all))}",
        )
        report.add_metric("n_patients_history", int(hist_all.size))
        report.add_metric("history_days_median", round(_describe(hist_all)["p50"]))
        report.add_metric("history_days_p90", round(_describe(hist_all)["p90"]))

    # --- figures -----------------------------------------------------------
    tok_fig = _plot_token_hist(full_lengths, windowed_combined, args.ctx, figures_dir / "token_length_hist.png")
    report.add_artifact(tok_fig)
    if history.height:
        hist_fig = _plot_history_hist(history, figures_dir / "history_length_hist.png")
        report.add_artifact(hist_fig)

    # --- write json --------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "token_history_stats.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    report.add_artifact(json_path)

    return _finalize(report, report_dir)


def _finalize(report: StepReport, report_dir: str | Path) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
