#!/usr/bin/env python
"""Step 10: Evaluate the fine-tuned AD classifier on the held-out test predictions.

Consumes the prediction parquet folder written by step 9
(``data/finetune_results/test_predictions``) and computes:

  Discrimination : AUROC, AUPRC (vs prevalence baseline)
  Operating point: sensitivity, specificity, PPV, NPV, F1, accuracy at
                   {0.5, Youden-J, high-sensitivity} thresholds
  Calibration    : Brier score, ECE, calibration slope/intercept, reliability curve

All point estimates for AUROC / AUPRC / Brier and for the primary operating point's
sensitivity / specificity / PPV are reported with 95% bootstrap CIs.

Threshold selection
-------------------
Sensitivity/specificity/PPV depend on a probability cutoff. Choosing it on the same
data used to report them is optimistically biased, so this step prefers a *validation*
prediction folder (``--val-predictions``, default data/finetune_results/validation_predictions):
the Youden-J and high-sensitivity cutoffs are selected there and applied to the test set.
If that folder is absent, the step falls back to selecting on the test set and records a
non-blocking warning.

Outputs (all under the build/ folder)
-------
  build/metrics.json
  build/figures/roc_curve.png, pr_curve.png, calibration_curve.png, confusion_matrix.png
  build/reports/10_evaluate/{status.json,report.md}

Usage
-----
  python 10_evaluate.py
  python 10_evaluate.py --predictions data/finetune_results/test_predictions \
                        --val-predictions data/finetune_results/validation_predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import evaluation as ev  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

PRIMARY_METHOD = "youden"  # operating point used for the headline sens/spec/PPV + confusion fig


def _select_thresholds(report: StepReport, val_dir: Path, test_true, test_prob) -> dict:
    """Return {method: threshold}. Select on validation if available, else on test (warned)."""
    methods = ("youden", "high_sensitivity")
    if val_dir.is_dir() and any(val_dir.glob("*.parquet")):
        try:
            v_true, v_prob = ev.load_predictions(val_dir)
            report.add_check("validation predictions usable for threshold selection", len(v_true) > 0,
                             detail=f"{len(v_true)} validation rows at {val_dir}")
            report.add_metric("threshold_selected_on", "validation")
            return {m: ev.select_threshold(v_true, v_prob, method=m) for m in methods}
        except Exception as exc:  # noqa: BLE001
            report.warn("validation predictions readable", ok=False, detail=str(exc))
    report.warn(
        "threshold selected on validation set",
        ok=False,
        detail=f"no usable validation predictions at {val_dir}; selecting on TEST set "
               "(threshold-dependent metrics are mildly optimistic). Generate validation "
               "predictions to remove this bias.",
    )
    report.add_metric("threshold_selected_on", "test")
    return {m: ev.select_threshold(test_true, test_prob, method=m) for m in methods}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned AD classifier")
    parser.add_argument("--predictions", default=str(paths.TEST_PREDICTIONS_DIR),
                        help="Test predictions parquet folder")
    parser.add_argument("--val-predictions", default=str(paths.VAL_PREDICTIONS_DIR),
                        help="Validation predictions folder for unbiased threshold selection")
    parser.add_argument("--window-days", type=int, default=paths.DEFAULT_WINDOW_DAYS,
                        help="Look-back window in days for the run label (negative = all history).")
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX,
                        help="Token context for the run label.")
    parser.add_argument("--results-dir", default=None,
                        help="Override results dir; defaults to build/<run_label>/ (metrics.json).")
    parser.add_argument("--figures-dir", default=None,
                        help="Override figures dir; defaults to build/<run_label>/figures.")
    parser.add_argument("--report-dir", default=None,
                        help="Override report dir; defaults to build/<run_label>/reports/10_evaluate.")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--cal-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # Build outputs nest under build/<run_label>/; explicit overrides still win.
    label = paths.run_label(args.window_days, args.ctx)
    results_dir = Path(args.results_dir) if args.results_dir else paths.eval_results_dir(label)
    figures_dir = Path(args.figures_dir) if args.figures_dir else paths.figures_dir(label)
    report_dir = Path(args.report_dir) if args.report_dir else paths.build_report_dir_for("10_evaluate", label)

    report = StepReport("10_evaluate", title="Step 10 - Evaluate fine-tuned AD classifier")
    report.add_metric("run_label", label)
    pred_dir = Path(args.predictions)
    report.add_metric("predictions_dir", str(pred_dir))

    if not report.add_check("test predictions exist", pred_dir.is_dir() and any(pred_dir.glob("*.parquet")),
                            detail=str(pred_dir)):
        return _finalize(report, report_dir)

    y_true, y_prob = ev.load_predictions(pred_dir)
    report.add_metric("n_test", int(len(y_true)))
    report.add_metric("n_positive", int(y_true.sum()))
    if not report.add_check("test set has both classes", 0 < y_true.sum() < len(y_true),
                            detail=f"{int(y_true.sum())} positive of {len(y_true)}"):
        return _finalize(report, report_dir)

    # --- discrimination (with CIs) ----------------------------------------
    disc = ev.discrimination_metrics(y_true, y_prob)
    auroc_pt, auroc_lo, auroc_hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.roc_auc_score(t, p), args.n_bootstrap, args.seed)
    auprc_pt, auprc_lo, auprc_hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.average_precision_score(t, p), args.n_bootstrap, args.seed)
    report.add_check("AUROC in [0, 1]", 0.0 <= disc["auroc"] <= 1.0,
                     detail=f"AUROC={disc['auroc']:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")
    report.add_metric("auroc", round(disc["auroc"], 4))
    report.add_metric("auroc_ci95", [round(auroc_lo, 4), round(auroc_hi, 4)])
    report.add_metric("auprc", round(disc["auprc"], 4))
    report.add_metric("auprc_ci95", [round(auprc_lo, 4), round(auprc_hi, 4)])
    report.add_metric("prevalence", round(disc["prevalence"], 4))

    # --- thresholds + operating-point metrics -----------------------------
    thresholds = _select_thresholds(report, Path(args.val_predictions), y_true, y_prob)
    thresholds["fixed_0.5"] = 0.5
    operating = {name: ev.threshold_metrics(y_true, y_prob, thr) for name, thr in thresholds.items()}

    primary = operating[PRIMARY_METHOD]
    for key in ("sensitivity", "specificity", "ppv"):
        pt, lo, hi = ev.bootstrap_ci(
            y_true, y_prob,
            (lambda k: lambda t, p: ev.threshold_metrics(t, p, thresholds[PRIMARY_METHOD])[k])(key),
            args.n_bootstrap, args.seed)
        report.add_metric(f"{key}@{PRIMARY_METHOD}", round(pt, 4))
        report.add_metric(f"{key}@{PRIMARY_METHOD}_ci95", [round(lo, 4), round(hi, 4)])

    # --- calibration -------------------------------------------------------
    cal = ev.calibration_metrics(y_true, y_prob, n_bins=args.cal_bins)
    brier_pt, brier_lo, brier_hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.brier_score_loss(t, p), args.n_bootstrap, args.seed)
    report.add_metric("brier", round(cal.brier, 4))
    report.add_metric("brier_ci95", [round(brier_lo, 4), round(brier_hi, 4)])
    report.add_metric("ece", round(cal.ece, 4))
    report.add_metric("calibration_slope", round(cal.slope, 4))
    report.add_metric("calibration_intercept", round(cal.intercept, 4))
    report.warn("calibration slope near 1.0", ok=(0.7 <= cal.slope <= 1.3) if cal.slope == cal.slope else False,
                detail=f"slope={cal.slope:.3f} (1.0 = perfect; <1 overconfident)")

    # --- figures -----------------------------------------------------------
    fig_dir = figures_dir
    roc_p = ev.plot_roc(y_true, y_prob, fig_dir / "roc_curve.png", auroc=disc["auroc"])
    pr_p = ev.plot_pr(y_true, y_prob, fig_dir / "pr_curve.png", auprc=disc["auprc"])
    cal_p = ev.plot_calibration(cal, fig_dir / "calibration_curve.png")
    cm_p = ev.plot_confusion(primary, fig_dir / "confusion_matrix.png")
    for p in (roc_p, pr_p, cal_p, cm_p):
        report.add_artifact(p)

    # --- write full metrics.json ------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_test": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "discrimination": {
            "auroc": disc["auroc"], "auroc_ci95": [auroc_lo, auroc_hi],
            "auprc": disc["auprc"], "auprc_ci95": [auprc_lo, auprc_hi],
            "prevalence": disc["prevalence"],
        },
        "threshold_selected_on": report.metrics.get("threshold_selected_on"),
        "operating_points": operating,
        "calibration": {
            "brier": cal.brier, "brier_ci95": [brier_lo, brier_hi],
            "ece": cal.ece, "slope": cal.slope, "intercept": cal.intercept,
            "reliability_curve": {
                "mean_predicted": cal.bin_centers,
                "observed_rate": cal.bin_true,
                "bin_counts": cal.bin_counts,
            },
        },
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    report.add_artifact(metrics_path)

    return _finalize(report, report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
