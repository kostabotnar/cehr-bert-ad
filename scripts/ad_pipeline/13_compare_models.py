#!/usr/bin/env python
"""Step 13: Compare classical-ML baselines against the CEHR-BERT classifier.

Evaluates the two classical baselines produced by step 12 (L2 logistic
regression and XGBoost) with the *exact same* evaluation code used for the
CEHR-BERT model (``10_evaluate.py``), then assembles a head-to-head comparison:

  * a structured ``comparison.json`` and a markdown ``comparison.md`` table
    (AUROC / AUPRC / Brier with 95% CIs, ECE, calibration slope, and the
    Youden operating point's sensitivity / specificity / PPV / F1)
  * overlaid ROC and PR curves (one line per model) at 300 DPI.

No metrics are recomputed here: each baseline is scored by invoking
``10_evaluate.py`` as a subprocess (which writes ``metrics.json`` next to its
predictions), and the CEHR-BERT ``metrics.json`` is read as-is. The overlaid
curves are drawn from each model's saved TEST predictions via
``pipeline.evaluation.load_predictions`` + sklearn's ``roc_curve`` /
``precision_recall_curve``.

The script is defensive: missing prediction folders or ``metrics.json`` files
are skipped with a warning rather than aborting, so a partial comparison still
renders.

Inputs
------
  CEHR-BERT metrics   : build/metrics.json           (--cehrbert-metrics)
  CEHR-BERT test preds: data/finetune_results/test_predictions (--cehrbert-predictions)
  baseline <m>        : build/baselines/<m>/test_predictions
                        build/baselines/<m>/validation_predictions

Outputs
-------
  build/baselines/comparison.json
  build/baselines/comparison.md
  build/figures/compare_roc.png, compare_pr.png
  build/reports/13_compare_models/{status.json,report.md}

Usage
-----
  python 13_compare_models.py
  python 13_compare_models.py --models lr xgboost
  python 13_compare_models.py --skip-eval          # reuse existing metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import evaluation as ev  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

# Human-readable labels for each known model key. CEHR-BERT is added separately.
MODEL_LABELS = {
    "lr": "L2 Logistic Regression",
    "xgboost": "XGBoost",
}
CEHRBERT_LABEL = "CEHR-BERT"

# Operating point reported in the comparison table (matches 10_evaluate's headline).
PRIMARY_METHOD = "youden"


# --------------------------------------------------------------------------- #
# Subprocess evaluation
# --------------------------------------------------------------------------- #
def _evaluate_baseline(model: str, baselines_dir: Path, report: StepReport) -> None:
    """Run 10_evaluate.py on one baseline's predictions; record success/failure."""
    model_dir = baselines_dir / model
    test_dir = model_dir / "test_predictions"
    val_dir = model_dir / "validation_predictions"
    evaluate_script = Path(__file__).resolve().parent / "10_evaluate.py"

    if not (test_dir.is_dir() and any(test_dir.glob("*.parquet"))):
        report.warn(
            f"{model} test predictions present",
            ok=False,
            detail=f"no parquet predictions at {test_dir}; skipping evaluation",
        )
        return

    cmd = [
        sys.executable,
        str(evaluate_script),
        "--predictions", str(test_dir),
        "--val-predictions", str(val_dir),
        "--results-dir", str(model_dir),
        "--figures-dir", str(model_dir / "figures"),
        "--report-dir", str(model_dir / "reports" / "10_evaluate"),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        report.add_check(f"{model} evaluation succeeded", ok=False,
                         detail=f"subprocess failed to launch: {exc}")
        return

    ok = proc.returncode == 0
    detail = f"return code {proc.returncode}"
    if not ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        detail += "; " + " | ".join(line.strip() for line in tail) if tail else ""
    report.add_check(f"{model} evaluation succeeded", ok=ok, detail=detail[:500])


# --------------------------------------------------------------------------- #
# Metrics loading + table formatting
# --------------------------------------------------------------------------- #
def _load_metrics(path: Path) -> dict | None:
    """Read a metrics.json file, returning None if absent or unreadable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _fmt_ci(point: float | None, ci: list | None, ndigits: int = 3) -> str:
    """Format a point estimate with its 95% CI, e.g. '0.812 [0.787, 0.838]'."""
    if point is None or (isinstance(point, float) and point != point):
        return "n/a"
    base = f"{point:.{ndigits}f}"
    if ci and len(ci) == 2 and all(c is not None and c == c for c in ci):
        return f"{base} [{ci[0]:.{ndigits}f}, {ci[1]:.{ndigits}f}]"
    return base


def _fmt_num(value: float | None, ndigits: int = 3) -> str:
    """Format a plain scalar; 'n/a' for None/NaN."""
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    return f"{value:.{ndigits}f}"


def _extract_row(label: str, metrics: dict) -> dict:
    """Pull the comparison fields out of a 10_evaluate metrics.json payload."""
    disc = metrics.get("discrimination", {}) or {}
    cal = metrics.get("calibration", {}) or {}
    op = (metrics.get("operating_points", {}) or {}).get(PRIMARY_METHOD, {}) or {}
    return {
        "model": label,
        "n_test": metrics.get("n_test"),
        "n_positive": metrics.get("n_positive"),
        "auroc": disc.get("auroc"),
        "auroc_ci95": disc.get("auroc_ci95"),
        "auprc": disc.get("auprc"),
        "auprc_ci95": disc.get("auprc_ci95"),
        "prevalence": disc.get("prevalence"),
        "brier": cal.get("brier"),
        "brier_ci95": cal.get("brier_ci95"),
        "ece": cal.get("ece"),
        "calibration_slope": cal.get("slope"),
        "operating_point": PRIMARY_METHOD,
        "sensitivity": op.get("sensitivity"),
        "specificity": op.get("specificity"),
        "ppv": op.get("ppv"),
        "f1": op.get("f1"),
    }


def _build_markdown(rows: list[dict]) -> str:
    """Render the per-model rows as a markdown pipe table plus a winner note."""
    header = (
        "| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | ECE | "
        "Cal. slope | Sens. | Spec. | PPV | F1 |"
    )
    divider = "|" + "|".join(["---"] * 10) + "|"
    lines = ["# Model comparison", "",
             f"Operating point for sensitivity / specificity / PPV / F1: `{PRIMARY_METHOD}`.",
             "", header, divider]
    for r in rows:
        lines.append(
            "| " + " | ".join([
                r["model"],
                _fmt_ci(r["auroc"], r["auroc_ci95"]),
                _fmt_ci(r["auprc"], r["auprc_ci95"]),
                _fmt_ci(r["brier"], r["brier_ci95"]),
                _fmt_num(r["ece"]),
                _fmt_num(r["calibration_slope"], 2),
                _fmt_num(r["sensitivity"]),
                _fmt_num(r["specificity"]),
                _fmt_num(r["ppv"]),
                _fmt_num(r["f1"]),
            ]) + " |"
        )

    winner = _auroc_winner(rows)
    lines += ["", winner, ""]
    return "\n".join(lines)


def _auroc_winner(rows: list[dict]) -> str:
    """One-line note naming the best model by AUROC (ASCII-only)."""
    scored = [r for r in rows if isinstance(r.get("auroc"), (int, float)) and r["auroc"] == r["auroc"]]
    if not scored:
        return "No AUROC available for any model."
    best = max(scored, key=lambda r: r["auroc"])
    return f"Best AUROC: {best['model']} ({best['auroc']:.3f})."


# --------------------------------------------------------------------------- #
# Overlaid curve figures
# --------------------------------------------------------------------------- #
def _load_curve_inputs(label: str, pred_dir: Path) -> tuple | None:
    """Load (y_true, y_prob) for a model's TEST predictions, or None if unavailable."""
    if not (pred_dir.is_dir() and any(pred_dir.glob("*.parquet"))) and not pred_dir.is_file():
        return None
    try:
        y_true, y_prob = ev.load_predictions(pred_dir)
    except Exception:  # noqa: BLE001
        return None
    if len(y_true) == 0 or len(set(y_true.tolist())) < 2:
        return None
    return label, y_true, y_prob


def plot_overlaid_roc(curve_inputs: list[tuple], out_path: Path) -> Path:
    """Overlay ROC curves for every model with available predictions."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, y_true, y_prob in curve_inputs:
        fpr, tpr, _ = ev.roc_curve(y_true, y_prob)
        auroc = ev.roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC curve comparison")
    ax.legend(loc="lower right", fontsize=8)
    return _save_fig(fig, out_path)


def plot_overlaid_pr(curve_inputs: list[tuple], out_path: Path) -> Path:
    """Overlay precision-recall curves; dashed line marks the prevalence baseline."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))
    prevalence = None
    for label, y_true, y_prob in curve_inputs:
        precision, recall, _ = ev.precision_recall_curve(y_true, y_prob)
        auprc = ev.average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{label} (AUPRC={auprc:.3f})")
        prevalence = float(np.mean(y_true))
    if prevalence is not None:
        ax.axhline(prevalence, linestyle="--", color="grey",
                   label=f"baseline (prev={prevalence:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall curve comparison")
    ax.legend(loc="upper right", fontsize=8)
    return _save_fig(fig, out_path)


def _save_fig(fig, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare baselines against CEHR-BERT")
    parser.add_argument("--models", nargs="+", default=["lr", "xgboost"],
                        help="Baseline model keys to evaluate and compare")
    parser.add_argument("--baselines-dir", default=str(paths.BUILD_DIR / "baselines"),
                        help="Directory holding per-baseline prediction/metric folders")
    parser.add_argument("--cehrbert-metrics", default=str(paths.BUILD_DIR / "metrics.json"),
                        help="Existing CEHR-BERT metrics.json")
    parser.add_argument("--cehrbert-predictions", default=str(paths.TEST_PREDICTIONS_DIR),
                        help="CEHR-BERT test predictions folder (for overlaid curves)")
    parser.add_argument("--figures-dir", default=str(paths.FIGURES_DIR),
                        help="Where to save compare_roc.png / compare_pr.png")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip re-running 10_evaluate; reuse existing metrics.json")
    parser.add_argument("--report-dir", default=str(paths.build_report_dir_for("13_compare_models")))
    args = parser.parse_args(argv)

    report = StepReport("13_compare_models",
                        title="Step 13 - Compare baselines vs CEHR-BERT")
    baselines_dir = Path(args.baselines_dir)
    figures_dir = Path(args.figures_dir)
    report.add_metric("models", list(args.models))
    report.add_metric("baselines_dir", str(baselines_dir))

    # --- 1. evaluate each baseline (unless skipped) ------------------------
    if args.skip_eval:
        report.note("--skip-eval set: reusing existing metrics.json for each baseline")
    else:
        for model in args.models:
            _evaluate_baseline(model, baselines_dir, report)

    # --- 2. load all metrics.json ------------------------------------------
    # (label, metrics_path, test_pred_dir)
    sources: list[tuple[str, Path, Path]] = [
        (CEHRBERT_LABEL, Path(args.cehrbert_metrics), Path(args.cehrbert_predictions)),
    ]
    for model in args.models:
        label = MODEL_LABELS.get(model, model)
        sources.append((label, baselines_dir / model / "metrics.json",
                        baselines_dir / model / "test_predictions"))

    rows: list[dict] = []
    curve_inputs: list[tuple] = []
    for label, metrics_path, pred_dir in sources:
        metrics = _load_metrics(metrics_path)
        if metrics is None:
            report.warn(f"{label} metrics available", ok=False,
                        detail=f"missing or unreadable metrics.json at {metrics_path}")
            continue
        row = _extract_row(label, metrics)
        rows.append(row)
        report.add_metric(f"auroc[{label}]", _fmt_num(row["auroc"]))
        report.add_metric(f"auprc[{label}]", _fmt_num(row["auprc"]))

        curve = _load_curve_inputs(label, pred_dir)
        if curve is not None:
            curve_inputs.append(curve)
        else:
            report.warn(f"{label} test predictions usable for curves", ok=False,
                        detail=f"no usable predictions at {pred_dir}")

    report.add_check("at least 2 models compared", ok=len(rows) >= 2,
                     detail=f"{len(rows)} model(s) with metrics: "
                            + ", ".join(r["model"] for r in rows))

    # --- 3. comparison table (json + markdown) -----------------------------
    baselines_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = baselines_dir / "comparison.json"
    comparison_md = baselines_dir / "comparison.md"
    comparison_json.write_text(
        json.dumps({"operating_point": PRIMARY_METHOD, "models": rows}, indent=2, default=float),
        encoding="utf-8",
    )
    comparison_md.write_text(_build_markdown(rows), encoding="utf-8")
    report.add_artifact(comparison_json)
    report.add_artifact(comparison_md)
    print(_auroc_winner(rows))

    # --- 4. overlaid curves -------------------------------------------------
    roc_ok = pr_ok = False
    if curve_inputs:
        try:
            roc_path = plot_overlaid_roc(curve_inputs, figures_dir / "compare_roc.png")
            report.add_artifact(roc_path)
            roc_ok = True
        except Exception as exc:  # noqa: BLE001
            report.warn("ROC comparison figure written", ok=False, detail=str(exc)[:300])
        try:
            pr_path = plot_overlaid_pr(curve_inputs, figures_dir / "compare_pr.png")
            report.add_artifact(pr_path)
            pr_ok = True
        except Exception as exc:  # noqa: BLE001
            report.warn("PR comparison figure written", ok=False, detail=str(exc)[:300])
    else:
        report.warn("predictions available for overlaid curves", ok=False,
                    detail="no model had usable test predictions; curves skipped")

    report.add_check("ROC comparison figure written", ok=roc_ok,
                     detail=str(figures_dir / "compare_roc.png"))
    report.add_check("PR comparison figure written", ok=pr_ok,
                     detail=str(figures_dir / "compare_pr.png"))

    # --- 5. finalize --------------------------------------------------------
    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
