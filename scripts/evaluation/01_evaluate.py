#!/usr/bin/env python
"""Step 01: Score every model listed in the prediction manifests.

The model parts (cehrbert_model, classical_ml) only write predictions and a ``predictions_manifest.json``
describing them. This step reads the manifests and, for each model, computes from its TEST predictions:

  Discrimination : AUROC, AUPRC (vs prevalence baseline)
  Operating point: sensitivity, specificity, PPV, NPV, F1, accuracy at
                   {0.5, Youden-J, high-sensitivity} thresholds
  Calibration    : Brier score, ECE, calibration slope/intercept, reliability curve

All point estimates for AUROC / AUPRC / Brier and for the primary operating point's
sensitivity / specificity / PPV are reported with 95% bootstrap CIs. The bootstrap size, seed and
calibration bins are the same for every model (constants in ``pipeline/evaluation.py``).

Threshold selection
-------------------
Sensitivity/specificity/PPV depend on a probability cutoff. Choosing it on the same
data used to report them is optimistically biased, so the Youden-J and high-sensitivity cutoffs
are selected on the model's VALIDATION predictions and applied to the test set. If they are
absent, the step falls back to selecting on the test set and records a non-blocking warning.

Models
------
Manifest mode (default): ``--manifest PATH`` (repeatable) evaluates every model of the given manifests.
Without ``--manifest`` the defaults are the cehrbert_model manifest for (``--window-days``, ``--ctx``) and the
classical_ml manifest for ``--window-days``; a missing default manifest is a warning and that family is
skipped. No model at all is a failure. One failing model fails the step, the others still run.

Single-model mode: ``--predictions`` (and ``--val-predictions``) score one prediction folder.

Outputs (comparison name = ``--name``, default the cehrbert run label ``w_<window>_ctx<ctx>``)
-------
  build/evaluation/<name>/models/<model>/metrics.json
  build/evaluation/<name>/models/<model>/figures/{roc_curve,pr_curve,calibration_curve,confusion_matrix}.png
  build/evaluation/<name>/models/<model>/reports/01_evaluate/{status.json,report.md}
  build/evaluation/<name>/reports/01_evaluate/{status.json,report.md}     overall result of the run

Usage
-----
  python evaluation/01_evaluate.py
  python evaluation/01_evaluate.py --manifest build/classical_ml/w_365/predictions_manifest.json --name classical
  python evaluation/01_evaluate.py --predictions data/finetune_results/test_predictions \
                        --val-predictions data/finetune_results/validation_predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import evaluation as ev  # noqa: E402
from pipeline import manifest  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

PRIMARY_METHOD = "youden"  # operating point used for the headline sens/spec/PPV + confusion fig


def _select_thresholds(report: StepReport, val_dir: Path, test_true, test_prob) -> dict:
    """Return {method: threshold}. Select on validation if available, else on test (warned)."""
    methods = ("youden", "high_sensitivity")
    if val_dir.is_dir() and any(val_dir.glob("*.parquet")):
        try:
            v_true, v_prob = ev.load_predictions(val_dir)
            report.add_check(
                "validation predictions usable for threshold selection",
                len(v_true) > 0,
                detail=f"{len(v_true)} validation rows at {val_dir}",
            )
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


def evaluate_model(
    label: str,
    pred_dir: Path,
    val_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    report_dir: Path,
    n_bootstrap: int = ev.EVAL_N_BOOTSTRAP,
    seed: int = ev.EVAL_SEED,
    cal_bins: int = ev.EVAL_CAL_BINS,
) -> int:
    """Score one model's prediction folders; writes metrics.json, figures and a report. Returns the exit code."""
    report = StepReport("01_evaluate", title=f"Step 01 - Evaluate {label}")
    report.add_metric("model", label)
    report.add_metric("predictions_dir", str(pred_dir))

    if not report.add_check(
        "test predictions exist", pred_dir.is_dir() and any(pred_dir.glob("*.parquet")), detail=str(pred_dir)
    ):
        return _finalize(report, report_dir)

    y_true, y_prob = ev.load_predictions(pred_dir)
    report.add_metric("n_test", int(len(y_true)))
    report.add_metric("n_positive", int(y_true.sum()))
    if not report.add_check(
        "test set has both classes",
        0 < y_true.sum() < len(y_true),
        detail=f"{int(y_true.sum())} positive of {len(y_true)}",
    ):
        return _finalize(report, report_dir)

    # --- discrimination (with CIs) ----------------------------------------
    disc = ev.discrimination_metrics(y_true, y_prob)
    auroc_pt, auroc_lo, auroc_hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.roc_auc_score(t, p), n_bootstrap, seed
    )
    auprc_pt, auprc_lo, auprc_hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.average_precision_score(t, p), n_bootstrap, seed
    )
    report.add_check(
        "AUROC in [0, 1]",
        0.0 <= disc["auroc"] <= 1.0,
        detail=f"AUROC={disc['auroc']:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]",
    )
    report.add_metric("auroc", round(disc["auroc"], 4))
    report.add_metric("auroc_ci95", [round(auroc_lo, 4), round(auroc_hi, 4)])
    report.add_metric("auprc", round(disc["auprc"], 4))
    report.add_metric("auprc_ci95", [round(auprc_lo, 4), round(auprc_hi, 4)])
    report.add_metric("prevalence", round(disc["prevalence"], 4))

    # --- thresholds + operating-point metrics -----------------------------
    thresholds = _select_thresholds(report, val_dir, y_true, y_prob)
    thresholds["fixed_0.5"] = 0.5
    operating = {name: ev.threshold_metrics(y_true, y_prob, thr) for name, thr in thresholds.items()}

    primary = operating[PRIMARY_METHOD]
    operating_cis: dict[str, list[float]] = {}
    for key in ("sensitivity", "specificity", "ppv"):
        pt, lo, hi = ev.bootstrap_ci(
            y_true,
            y_prob,
            (lambda k: lambda t, p: ev.threshold_metrics(t, p, thresholds[PRIMARY_METHOD])[k])(key),
            n_bootstrap,
            seed,
        )
        report.add_metric(f"{key}@{PRIMARY_METHOD}", round(pt, 4))
        report.add_metric(f"{key}@{PRIMARY_METHOD}_ci95", [round(lo, 4), round(hi, 4)])
        operating_cis[f"{key}_ci95"] = [lo, hi]

    # --- calibration -------------------------------------------------------
    cal = ev.calibration_metrics(y_true, y_prob, n_bins=cal_bins)
    brier_pt, brier_lo, brier_hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.brier_score_loss(t, p), n_bootstrap, seed
    )
    _, ece_lo, ece_hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.calibration_metrics(t, p, n_bins=cal_bins).ece, n_bootstrap, seed
    )
    _, slope_lo, slope_hi = ev.bootstrap_ci(
        y_true,
        y_prob,
        lambda t, p: ev.calibration_metrics(t, p, n_bins=cal_bins).slope,
        n_bootstrap,
        seed,
    )
    report.add_metric("brier", round(cal.brier, 4))
    report.add_metric("brier_ci95", [round(brier_lo, 4), round(brier_hi, 4)])
    report.add_metric("ece", round(cal.ece, 4))
    report.add_metric("calibration_slope", round(cal.slope, 4))
    report.add_metric("calibration_intercept", round(cal.intercept, 4))
    report.warn(
        "calibration slope near 1.0",
        ok=(0.7 <= cal.slope <= 1.3) if cal.slope == cal.slope else False,
        detail=f"slope={cal.slope:.3f} (1.0 = perfect; <1 overconfident)",
    )

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
            "auroc": disc["auroc"],
            "auroc_ci95": [auroc_lo, auroc_hi],
            "auprc": disc["auprc"],
            "auprc_ci95": [auprc_lo, auprc_hi],
            "prevalence": disc["prevalence"],
        },
        "threshold_selected_on": report.metrics.get("threshold_selected_on"),
        "operating_points": {
            **operating,
            PRIMARY_METHOD: {**operating[PRIMARY_METHOD], **operating_cis},
        },
        "calibration": {
            "brier": cal.brier,
            "brier_ci95": [brier_lo, brier_hi],
            "ece": cal.ece,
            "ece_ci95": [ece_lo, ece_hi],
            "slope": cal.slope,
            "slope_ci95": [slope_lo, slope_hi],
            "intercept": cal.intercept,
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


def _add_scoring_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-bootstrap", type=int, default=ev.EVAL_N_BOOTSTRAP)
    parser.add_argument("--cal-bins", type=int, default=ev.EVAL_CAL_BINS)
    parser.add_argument("--seed", type=int, default=ev.EVAL_SEED)


def _run_manifest_mode(args, models, findings, name: str) -> int:
    """Evaluate every manifest model into build/evaluation/<name>/models/<model>/; one failure does not stop the rest."""
    report_dir = Path(args.report_dir) if args.report_dir else paths.evaluation_report_dir("01_evaluate", name)
    report = StepReport("01_evaluate", title="Step 01 - Evaluate models")
    report.add_metric("comparison_name", name)
    report.add_metric("models", [m.name for m in models])
    for is_failure, check, detail in findings:
        (report.add_check if is_failure else report.warn)(check, ok=False, detail=detail)
    if not report.add_check("at least one model to evaluate", bool(models), detail=f"{len(models)} model(s)"):
        return _finalize(report, report_dir)

    for model in models:
        model_dir = paths.evaluation_model_dir(name, model.name)
        try:
            rc = evaluate_model(
                model.name,
                model.test_predictions,
                model.validation_predictions,
                model_dir,
                model_dir / "figures",
                model_dir / "reports" / "01_evaluate",
                args.n_bootstrap,
                args.seed,
                args.cal_bins,
            )
            detail = f"exit code {rc}; metrics in {model_dir}"
        except Exception as exc:  # noqa: BLE001
            rc, detail = 1, f"{type(exc).__name__}: {str(exc)[:300]}"
        report.add_check(f"{model.name}: evaluation succeeded", rc == 0, detail=detail)
    return _finalize(report, report_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AD classifiers from prediction folders or manifests")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Single-model mode: test predictions parquet folder (instead of manifests)",
    )
    parser.add_argument(
        "--val-predictions",
        default=str(paths.VAL_PREDICTIONS_DIR),
        help="Single-model mode: validation predictions folder for unbiased threshold selection",
    )
    parser.add_argument("--model-name", default="model", help="Single-model mode: model folder name under models/")
    manifest.add_selection_args(parser)
    parser.add_argument("--results-dir", default=None, help="Single-model mode: override the metrics.json folder.")
    parser.add_argument("--figures-dir", default=None, help="Single-model mode: override the figures folder.")
    parser.add_argument("--report-dir", default=None, help="Override the report folder.")
    _add_scoring_args(parser)
    args = parser.parse_args(argv)

    name = manifest.comparison_name(args)
    if args.predictions is None:
        if args.results_dir or args.figures_dir:
            parser.error("--results-dir and --figures-dir apply to single-model mode (--predictions)")
        models, findings = manifest.collect_models(args)
        return _run_manifest_mode(args, models, findings, name)

    model_dir = paths.evaluation_model_dir(name, args.model_name)
    return evaluate_model(
        args.model_name,
        Path(args.predictions),
        Path(args.val_predictions),
        Path(args.results_dir) if args.results_dir else model_dir,
        Path(args.figures_dir) if args.figures_dir else model_dir / "figures",
        Path(args.report_dir) if args.report_dir else model_dir / "reports" / "01_evaluate",
        args.n_bootstrap,
        args.seed,
        args.cal_bins,
    )


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
