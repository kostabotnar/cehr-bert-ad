#!/usr/bin/env python
"""Step 02: Compare the models listed in the prediction manifests.

Reads the same manifests as step 01 (same options and defaults: the cehrbert_model manifest for
``--window-days``/``--ctx`` and the classical_ml manifest for ``--window-days``). Nothing is re-scored here:
each model's ``metrics.json`` comes from step 01, which must have run for the same comparison name.

Before anything is compared, a parity check matches every model's TEST predictions with the canonical test
split (person_ids, labels, row count including duplicates) and its VALIDATION predictions with the canonical
val split. Any difference, missing validation predictions, or a model without step 01 metrics fails the
step. Per-model counts are written to the step report and ``comparison.json``.

Outputs (comparison name = ``--name``, default the cehrbert run label)
-------
  build/evaluation/<name>/comparison.json, comparison.md
      per-model metrics (each with 95% CI from the model's step 01 metrics.json) and paired
      AUROC / AUPRC differences between every pair of models (paired patient bootstrap)
  build/evaluation/<name>/figures/compare_roc.png, compare_pr.png
  build/evaluation/<name>/reports/02_compare_models/{status.json,report.md}

Inputs
------
  splits    : data/cohort/splits.parquet                                (--splits-file)
  manifests : predictions_manifest.json files                           (--manifest, repeatable)
  metrics   : build/evaluation/<name>/models/<model>/metrics.json       (evaluation step 01)

Usage
-----
  python evaluation/02_compare_models.py
  python evaluation/02_compare_models.py --manifest A/predictions_manifest.json --manifest B/predictions_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import baselines as bl  # noqa: E402
from pipeline import comparison as cmp  # noqa: E402
from pipeline import evaluation as ev  # noqa: E402
from pipeline import manifest  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

# Operating point reported in the comparison table (matches 01_evaluate's headline).
PRIMARY_METHOD = "youden"


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
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the models listed in the prediction manifests")
    manifest.add_selection_args(parser)
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE), help="Canonical splits parquet")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where comparison.json / comparison.md go (default: build/evaluation/<name>)",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Where to save compare_roc.png / compare_pr.png (default: build/evaluation/<name>/figures)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=ev.EVAL_N_BOOTSTRAP, help="Resamples for the paired bootstrap"
    )
    parser.add_argument("--seed", type=int, default=ev.EVAL_SEED, help="Paired bootstrap seed")
    parser.add_argument(
        "--report-dir", default=None, help="Report dir (default: build/evaluation/<name>/reports/02_compare_models)"
    )
    args = parser.parse_args(argv)

    name = manifest.comparison_name(args)
    output_dir = Path(args.output_dir) if args.output_dir else paths.evaluation_dir(name)
    figures_dir = Path(args.figures_dir) if args.figures_dir else paths.evaluation_figures_dir(name)
    report_dir = Path(args.report_dir) if args.report_dir else paths.evaluation_report_dir("02_compare_models", name)

    report = StepReport("02_compare_models", title="Step 02 - Compare models")
    report.add_metric("comparison_name", name)
    report.add_metric("paired_bootstrap", {"n_bootstrap": args.n_bootstrap, "seed": args.seed})

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

    # --- 2. models from the manifests ----------------------------------------
    manifest_models, findings = manifest.collect_models(args)
    manifest_models = manifest.primary_first(manifest_models)
    for is_failure, check, detail in findings:
        (report.add_check if is_failure else report.warn)(check, ok=False, detail=detail)
    specs = [
        cmp.ModelSpec(
            label=m.display_label,
            test_dir=m.test_predictions,
            val_dir=m.validation_predictions,
            metrics_path=paths.evaluation_model_dir(name, m.name) / "metrics.json",
            secondary=m.secondary,
        )
        for m in manifest_models
    ]
    report.add_metric("models", [m.name for m in manifest_models])

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

    # --- 4. load step 01 metrics and curve inputs ------------------------------------
    rows: list[dict] = []
    curve_inputs: list[tuple] = []
    frames: dict[str, object] = {}
    for spec in specs:
        metrics = _load_json(spec.metrics_path)
        if not report.add_check(
            f"{spec.label}: step 01 metrics available",
            metrics is not None,
            detail=f"missing or unreadable metrics.json at {spec.metrics_path}; run evaluation step 01 first",
        ):
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

    # --- 5. paired bootstrap -------------------------------------------------
    paired: list[dict] = []
    if len(frames) >= 2:
        try:
            paired = cmp.paired_difference_rows(frames, args.n_bootstrap, args.seed)
            report.add_check("paired bootstrap computed", ok=True, detail=f"{len(paired)} pair(s) of models")
        except Exception as exc:  # noqa: BLE001
            report.add_check("paired bootstrap computed", ok=False, detail=str(exc)[:300])
    else:
        report.warn("paired bootstrap computed", ok=False, detail="fewer than 2 models passed parity")

    # --- 6. comparison table (json + markdown) -------------------------------
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

    # --- 7. overlaid curves --------------------------------------------------
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

    # --- 8. finalize ---------------------------------------------------------
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
