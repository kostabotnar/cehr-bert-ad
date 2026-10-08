#!/usr/bin/env python
"""Step 02: Compare the frozen CEHR-BERT linear probe and the classical baselines.

Models compared (each evaluated with the *exact same* code, ``evaluation/01_evaluate.py``):

  * the fine-tuned CEHR-BERT classifier, when its metrics exist (skipped with a warning otherwise)
  * the linear probe on the frozen pretrained CEHR-BERT: the primary feature kind named in
    ``linear_probe/summary.json`` and, in a separate row marked secondary, the other kind
  * L2 logistic regression and XGBoost from classical_ml step 02

Before anything is scored, a parity check compares every included model's TEST predictions with
the canonical test split (person_ids, labels, row count including duplicates) and its VALIDATION
predictions with the canonical val split. Any difference, or missing validation predictions,
fails the step. Per-model counts are written to the step report and ``comparison.json``.

Outputs
-------
  build/<run_label>/comparison.json, comparison.md
      per-model metrics (each with 95% CI from the model's step 01 metrics.json) and paired
      AUROC / AUPRC differences between every pair of models (paired patient bootstrap)
  build/<run_label>/figures/compare_roc.png, compare_pr.png
  build/<run_label>/reports/02_compare_models/{status.json,report.md}

Inputs
------
  splits              : data/cohort/splits.parquet                   (--splits-file)
  linear probe        : build/<run_label>/linear_probe/{summary.json,<kind>/...}
  baseline <m>        : build/<run_label>/baselines/<m>/{test,validation}_predictions
  CEHR-BERT (optional): build/<run_label>/metrics.json, data/finetune_results/{test,validation}_predictions

Usage
-----
  python evaluation/02_compare_models.py
  python evaluation/02_compare_models.py --skip-eval          # reuse existing metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import baselines as bl  # noqa: E402
from pipeline import comparison as cmp  # noqa: E402
from pipeline import evaluation as ev  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

# Human-readable labels for each known baseline key.
MODEL_LABELS = {
    "lr": "L2 Logistic Regression",
    "xgboost": "XGBoost",
}
CEHRBERT_LABEL = "CEHR-BERT"

# Operating point reported in the comparison table (matches 01_evaluate's headline).
PRIMARY_METHOD = "youden"

# Every model must be scored with the same evaluation/01_evaluate.py arguments; pass them explicitly
# so a change to 01_evaluate's defaults cannot silently make the models incomparable.
EVAL_N_BOOTSTRAP = 2000
EVAL_SEED = 42
EVAL_CAL_BINS = 10


# --------------------------------------------------------------------------- #
# Subprocess evaluation
# --------------------------------------------------------------------------- #
def evaluate_command(
    model_dir: Path,
    n_bootstrap: int = EVAL_N_BOOTSTRAP,
    seed: int = EVAL_SEED,
    cal_bins: int = EVAL_CAL_BINS,
) -> list[str]:
    """Command line that runs evaluation/01_evaluate.py on one model's prediction folders."""
    evaluate_script = Path(__file__).resolve().parent / "01_evaluate.py"
    return [
        sys.executable,
        str(evaluate_script),
        "--predictions",
        str(model_dir / "test_predictions"),
        "--val-predictions",
        str(model_dir / "validation_predictions"),
        "--results-dir",
        str(model_dir),
        "--figures-dir",
        str(model_dir / "figures"),
        "--report-dir",
        str(model_dir / "reports" / "01_evaluate"),
        "--n-bootstrap",
        str(n_bootstrap),
        "--seed",
        str(seed),
        "--cal-bins",
        str(cal_bins),
    ]


def _evaluate_model(
    spec: cmp.ModelSpec,
    report: StepReport,
    n_bootstrap: int = EVAL_N_BOOTSTRAP,
    seed: int = EVAL_SEED,
    cal_bins: int = EVAL_CAL_BINS,
) -> None:
    """Run evaluation/01_evaluate.py on one model's predictions; record success/failure."""
    cmd = evaluate_command(spec.evaluate_dir, n_bootstrap, seed, cal_bins)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        report.add_check(f"{spec.label}: evaluation succeeded", ok=False, detail=f"subprocess failed to launch: {exc}")
        return

    ok = proc.returncode == 0
    detail = f"return code {proc.returncode}"
    if not ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        detail += "; " + " | ".join(line.strip() for line in tail) if tail else ""
    report.add_check(f"{spec.label}: evaluation succeeded", ok=ok, detail=detail[:500])


# --------------------------------------------------------------------------- #
# Metrics loading + table formatting
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict | None:
    """Read a json file, returning None if absent or unreadable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _is_nan(value) -> bool:
    return isinstance(value, float) and value != value


def _fmt_ci(point: float | None, ci: list | None, ndigits: int = 3) -> str:
    """Format a point estimate with its 95% CI, e.g. '0.812 [0.787, 0.838]'."""
    if point is None or _is_nan(point):
        return "n/a"
    base = f"{point:.{ndigits}f}"
    if ci and len(ci) == 2 and all(c is not None and c == c for c in ci):
        return f"{base} [{ci[0]:.{ndigits}f}, {ci[1]:.{ndigits}f}]"
    return base


def _fmt_num(value: float | None, ndigits: int = 3) -> str:
    """Format a plain scalar; 'n/a' for None/NaN."""
    if value is None or _is_nan(value):
        return "n/a"
    return f"{value:.{ndigits}f}"


def _extract_row(label: str, metrics: dict, secondary: bool = False) -> dict:
    """Pull the comparison fields out of a 01_evaluate metrics.json payload."""
    disc = metrics.get("discrimination", {}) or {}
    cal = metrics.get("calibration", {}) or {}
    op = (metrics.get("operating_points", {}) or {}).get(PRIMARY_METHOD, {}) or {}
    return {
        "model": label,
        "secondary": secondary,
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
        "ece_ci95": cal.get("ece_ci95"),
        "calibration_slope": cal.get("slope"),
        "calibration_slope_ci95": cal.get("slope_ci95"),
        "operating_point": PRIMARY_METHOD,
        "sensitivity": op.get("sensitivity"),
        "sensitivity_ci95": op.get("sensitivity_ci95"),
        "specificity": op.get("specificity"),
        "specificity_ci95": op.get("specificity_ci95"),
        "ppv": op.get("ppv"),
        "ppv_ci95": op.get("ppv_ci95"),
        "f1": op.get("f1"),
    }


def _build_markdown(rows: list[dict], paired: list[dict], n_resamples: int, seed: int) -> str:
    """Render the per-model table and the paired differences table."""
    header = (
        "| Model | N test (cases) | AUROC | AUPRC | Prevalence | Brier | ECE | Cal. slope | "
        "Sensitivity | Specificity | PPV |"
    )
    lines = [
        "# Model comparison",
        "",
        "All intervals are 95% percentile bootstrap intervals from each model's step 01 metrics.json. "
        f"Sensitivity, specificity and PPV use the `{PRIMARY_METHOD}` threshold selected on the validation split. "
        "Rows marked (secondary) were not chosen by validation AUROC.",
        "",
        header,
        "|" + "|".join(["---"] * 11) + "|",
    ]
    for r in rows:
        name = r["model"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{r['n_test']} ({r['n_positive']})",
                    _fmt_ci(r["auroc"], r["auroc_ci95"]),
                    _fmt_ci(r["auprc"], r["auprc_ci95"]),
                    _fmt_num(r["prevalence"]),
                    _fmt_ci(r["brier"], r["brier_ci95"]),
                    _fmt_ci(r["ece"], r["ece_ci95"]),
                    _fmt_ci(r["calibration_slope"], r["calibration_slope_ci95"], 2),
                    _fmt_ci(r["sensitivity"], r["sensitivity_ci95"]),
                    _fmt_ci(r["specificity"], r["specificity_ci95"]),
                    _fmt_ci(r["ppv"], r["ppv_ci95"]),
                ]
            )
            + " |"
        )
    lines += ["", _auroc_winner(rows), "", "## Paired differences on the test patients", ""]
    lines.append(
        f"Difference = first model minus second model. Patients are resampled jointly "
        f"({n_resamples} resamples, seed {seed}); an interval that excludes 0 indicates a real difference."
    )
    lines.append("")
    if not paired:
        lines.append("No pair of models could be compared.")
    else:
        lines += [
            "| First model | Second model | AUROC difference | AUPRC difference |",
            "|---|---|---|---|",
        ]
        for p in paired:
            lines.append(
                f"| {p['model_a']} | {p['model_b']} | "
                f"{_fmt_ci(p['auroc']['diff'], [p['auroc']['ci_low'], p['auroc']['ci_high']], 4)} | "
                f"{_fmt_ci(p['auprc']['diff'], [p['auprc']['ci_low'], p['auprc']['ci_high']], 4)} |"
            )
    lines.append("")
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
    """Load (label, y_true, y_prob) for a model's TEST predictions, or None if unavailable."""
    if not cmp.has_predictions(pred_dir):
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

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, y_true, y_prob in curve_inputs:
        fpr, tpr, _ = ev.roc_curve(y_true, y_prob)
        auroc = ev.roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC curve comparison")
    ax.legend(loc="lower right", fontsize=7)
    return _save_fig(fig, out_path)


def plot_overlaid_pr(curve_inputs: list[tuple], out_path: Path) -> Path:
    """Overlay precision-recall curves; dashed line marks the prevalence baseline."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    prevalence = None
    for label, y_true, y_prob in curve_inputs:
        precision, recall, _ = ev.precision_recall_curve(y_true, y_prob)
        auprc = ev.average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{label} (AUPRC={auprc:.3f})")
        prevalence = float(np.mean(y_true))
    if prevalence is not None:
        ax.axhline(prevalence, linestyle="--", color="grey", label=f"baseline (prev={prevalence:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall curve comparison")
    ax.legend(loc="upper right", fontsize=7)
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
# Model discovery
# --------------------------------------------------------------------------- #
def _linear_probe_specs(probe_dir: Path, report: StepReport) -> list[cmp.ModelSpec]:
    """Primary and secondary linear probe kinds named by summary.json; empty (with a warning) if absent."""
    summary = _load_json(probe_dir / "summary.json")
    kinds = [(summary or {}).get("primary_kind"), (summary or {}).get("secondary_kind")]
    if summary is None or not all(kinds):
        report.warn(
            "linear probe summary available",
            ok=False,
            detail=f"missing or unreadable summary.json at {probe_dir}; linear probe skipped",
        )
        return []
    specs = []
    for kind, role in zip(kinds, ("primary", "secondary")):
        kind_dir = probe_dir / kind
        if not cmp.has_predictions(kind_dir / "test_predictions"):
            report.warn(
                f"linear probe {kind} ({role}) test predictions present",
                ok=False,
                detail=f"no parquet predictions in {kind_dir}; skipped",
            )
            continue
        specs.append(
            cmp.ModelSpec(
                label=f"Linear probe ({kind}, {role})",
                test_dir=kind_dir / "test_predictions",
                val_dir=kind_dir / "validation_predictions",
                metrics_path=kind_dir / "metrics.json",
                evaluate_dir=kind_dir,
                secondary=role == "secondary",
            )
        )
    return specs


def _baseline_specs(models: list[str], baselines_dir: Path, report: StepReport) -> list[cmp.ModelSpec]:
    specs = []
    for model in models:
        model_dir = baselines_dir / model
        label = MODEL_LABELS.get(model, model)
        if not cmp.has_predictions(model_dir / "test_predictions"):
            report.warn(
                f"{label} test predictions present",
                ok=False,
                detail=f"no parquet predictions at {model_dir / 'test_predictions'}; skipping",
            )
            continue
        specs.append(
            cmp.ModelSpec(
                label=label,
                test_dir=model_dir / "test_predictions",
                val_dir=model_dir / "validation_predictions",
                metrics_path=model_dir / "metrics.json",
                evaluate_dir=model_dir,
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the linear probe and baselines (and CEHR-BERT if present)")
    parser.add_argument(
        "--models", nargs="+", default=["lr", "xgboost"], help="Baseline model keys to evaluate and compare"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=paths.DEFAULT_WINDOW_DAYS,
        help="Look-back window in days; negative => unbounded (all history)",
    )
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX, help="CEHR-BERT token cap; part of the run label")
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE), help="Canonical splits parquet")
    parser.add_argument(
        "--baselines-dir",
        default=None,
        help="Directory holding per-baseline prediction/metric folders (default: per-run baselines_dir)",
    )
    parser.add_argument(
        "--linear-probe-dir",
        default=None,
        help="Linear probe output folder with summary.json (default: per-run linear_probe_dir)",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Where comparison.json / comparison.md go (default: per-run folder)"
    )
    parser.add_argument(
        "--cehrbert-metrics",
        default=None,
        help="Fine-tuned CEHR-BERT metrics.json (default: per-run eval_results_dir/metrics.json)",
    )
    parser.add_argument(
        "--cehrbert-predictions",
        default=str(paths.TEST_PREDICTIONS_DIR),
        help="Fine-tuned CEHR-BERT test predictions folder",
    )
    parser.add_argument(
        "--cehrbert-val-predictions",
        default=str(paths.VAL_PREDICTIONS_DIR),
        help="Fine-tuned CEHR-BERT validation predictions folder",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Where to save compare_roc.png / compare_pr.png (default: per-run figures_dir)",
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip re-running 01_evaluate; reuse existing metrics.json"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=EVAL_N_BOOTSTRAP,
        help="Resamples for evaluation/01_evaluate.py (keep equal across models) and for the paired bootstrap",
    )
    parser.add_argument(
        "--seed", type=int, default=EVAL_SEED, help="Bootstrap seed, also passed to evaluation/01_evaluate.py"
    )
    parser.add_argument(
        "--cal-bins",
        type=int,
        default=EVAL_CAL_BINS,
        help="Passed to evaluation/01_evaluate.py; keep equal across models",
    )
    parser.add_argument("--report-dir", default=None, help="Report dir (default: per-run report dir for the run label)")
    args = parser.parse_args(argv)

    run = paths.run_label(args.window_days, args.ctx)
    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else paths.baselines_dir(run)
    probe_dir = Path(args.linear_probe_dir) if args.linear_probe_dir else paths.linear_probe_dir(run)
    output_dir = Path(args.output_dir) if args.output_dir else paths.run_dir(run)
    figures_dir = Path(args.figures_dir) if args.figures_dir else paths.figures_dir(run)
    cehrbert_metrics = (
        Path(args.cehrbert_metrics) if args.cehrbert_metrics else paths.eval_results_dir(run) / "metrics.json"
    )
    report_dir = Path(args.report_dir) if args.report_dir else paths.build_report_dir_for("02_compare_models", run)

    report = StepReport("02_compare_models", title="Step 02 - Compare models")
    report.add_metric("run_label", run)
    report.add_metric("baselines", list(args.models))
    report.add_metric("baselines_dir", str(baselines_dir))
    report.add_metric("linear_probe_dir", str(probe_dir))
    report.add_metric("eval_args", {"n_bootstrap": args.n_bootstrap, "seed": args.seed, "cal_bins": args.cal_bins})

    # --- 1. canonical splits -------------------------------------------------
    try:
        splits = bl.load_canonical_splits(args.splits_file)
    except Exception as exc:  # noqa: BLE001
        report.add_check("canonical splits readable", ok=False, detail=f"{args.splits_file}: {str(exc)[:300]}")
        report.write(report_dir)
        report.print_summary()
        return report.exit_code()
    report.add_check(
        "canonical splits readable",
        ok=splits.test.height > 0 and splits.val.height > 0,
        detail=f"{args.splits_file}: val={splits.val.height} test={splits.test.height}",
    )
    expected = {"val": splits.val.to_pandas(), "test": splits.test.to_pandas()}

    # --- 2. discover models --------------------------------------------------
    specs: list[cmp.ModelSpec] = []
    cehrbert_data = _load_json(cehrbert_metrics)
    if cehrbert_data is None:
        report.warn(
            f"{CEHRBERT_LABEL} metrics available",
            ok=False,
            detail=f"missing or unreadable metrics.json at {cehrbert_metrics}; fine-tuned model skipped",
        )
    else:
        specs.append(
            cmp.ModelSpec(
                label=CEHRBERT_LABEL,
                test_dir=Path(args.cehrbert_predictions),
                val_dir=Path(args.cehrbert_val_predictions),
                metrics_path=cehrbert_metrics,
            )
        )
    specs += _linear_probe_specs(probe_dir, report)
    specs += _baseline_specs(list(args.models), baselines_dir, report)

    # --- 3. parity with the canonical splits ---------------------------------
    parity: dict[str, dict] = {}
    parity_ok: dict[str, bool] = {}
    for spec in specs:
        results = {
            "test": cmp.check_parity(spec.test_dir, expected["test"], "test"),
            "val": cmp.check_parity(spec.val_dir, expected["val"], "val"),
        }
        parity[spec.label] = {name: res.to_dict() for name, res in results.items()}
        parity_ok[spec.label] = all(res.ok for res in results.values())
        for name, res in results.items():
            report.add_check(f"{spec.label}: {name} predictions match canonical split", res.ok, detail=res.detail())
            report.add_metric(f"parity[{spec.label}][{name}]", res.detail())

    # --- 4. evaluate with step 01 --------------------------------------------
    if args.skip_eval:
        report.note("--skip-eval set: reusing existing metrics.json for each model")
    else:
        for spec in specs:
            if spec.evaluate_dir is not None and parity_ok[spec.label]:
                _evaluate_model(spec, report, args.n_bootstrap, args.seed, args.cal_bins)

    # --- 5. load metrics and curve inputs ------------------------------------
    rows: list[dict] = []
    curve_inputs: list[tuple] = []
    frames: dict[str, object] = {}
    for spec in specs:
        metrics = _load_json(spec.metrics_path)
        if metrics is None:
            report.warn(
                f"{spec.label} metrics available",
                ok=False,
                detail=f"missing or unreadable metrics.json at {spec.metrics_path}",
            )
            continue
        row = _extract_row(spec.label, metrics, spec.secondary)
        rows.append(row)
        report.add_metric(f"auroc[{spec.label}]", _fmt_num(row["auroc"]))
        report.add_metric(f"auprc[{spec.label}]", _fmt_num(row["auprc"]))

        curve = _load_curve_inputs(spec.label, spec.test_dir)
        if curve is not None:
            curve_inputs.append(curve)
        else:
            report.warn(
                f"{spec.label} test predictions usable for curves",
                ok=False,
                detail=f"no usable predictions at {spec.test_dir}",
            )
        if parity_ok[spec.label]:
            frames[spec.label] = ev.load_prediction_frame(spec.test_dir)

    report.add_check(
        "at least 2 models compared",
        ok=len(rows) >= 2,
        detail=f"{len(rows)} model(s) with metrics: " + ", ".join(r["model"] for r in rows),
    )

    # --- 6. paired bootstrap -------------------------------------------------
    paired: list[dict] = []
    if len(frames) >= 2:
        try:
            paired = cmp.paired_difference_rows(frames, args.n_bootstrap, args.seed)
            report.add_check("paired bootstrap computed", ok=True, detail=f"{len(paired)} pair(s) of models")
        except Exception as exc:  # noqa: BLE001
            report.add_check("paired bootstrap computed", ok=False, detail=str(exc)[:300])
    else:
        report.warn("paired bootstrap computed", ok=False, detail="fewer than 2 models passed parity")

    # --- 7. comparison table (json + markdown) -------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = output_dir / "comparison.json"
    comparison_md = output_dir / "comparison.md"
    comparison_json.write_text(
        json.dumps(
            {
                "operating_point": PRIMARY_METHOD,
                "models": rows,
                "parity": parity,
                "paired_differences": {
                    "n_resamples": args.n_bootstrap,
                    "seed": args.seed,
                    "difference": "first model minus second model",
                    "pairs": paired,
                },
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    comparison_md.write_text(_build_markdown(rows, paired, args.n_bootstrap, args.seed), encoding="utf-8")
    report.add_artifact(comparison_json)
    report.add_artifact(comparison_md)
    print(_auroc_winner(rows))

    # --- 8. overlaid curves --------------------------------------------------
    roc_ok = pr_ok = False
    if curve_inputs:
        try:
            report.add_artifact(plot_overlaid_roc(curve_inputs, figures_dir / "compare_roc.png"))
            roc_ok = True
        except Exception as exc:  # noqa: BLE001
            report.warn("ROC comparison figure written", ok=False, detail=str(exc)[:300])
        try:
            report.add_artifact(plot_overlaid_pr(curve_inputs, figures_dir / "compare_pr.png"))
            pr_ok = True
        except Exception as exc:  # noqa: BLE001
            report.warn("PR comparison figure written", ok=False, detail=str(exc)[:300])
    else:
        report.warn(
            "predictions available for overlaid curves",
            ok=False,
            detail="no model had usable test predictions; curves skipped",
        )

    report.add_check("ROC comparison figure written", ok=roc_ok, detail=str(figures_dir / "compare_roc.png"))
    report.add_check("PR comparison figure written", ok=pr_ok, detail=str(figures_dir / "compare_pr.png"))

    # --- 9. finalize ---------------------------------------------------------
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
