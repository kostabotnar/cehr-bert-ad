#!/usr/bin/env python
"""Step 14: Train the linear probe on the frozen CEHR-BERT features from step 13.

For each feature kind (``cls``, ``mean``): StandardScaler + L2 logistic regression, tuned by
5-fold cross-validated AUROC over C and class_weight on the TRAIN split only (the same
protocol as the classical logistic regression in classical_ml step 02), refit on train, then scored on
val and test. The kind with the higher VALIDATION AUROC is the primary result; the other is
kept as a secondary result. Test is never used for any choice.

Predictions use CEHR-BERT's prediction-parquet schema. The step only describes them in the part's
``predictions_manifest.json`` (models ``linear_probe_cls`` and ``linear_probe_mean``, the lower validation AUROC
marked secondary); scoring is done by the evaluation part.

Outputs (under build/cehrbert_model/<run_label>/)
  predictions_manifest.json              entries upserted for both kinds
  linear_probe/ (all of the following)
  <kind>/test_predictions/predictions.parquet
  <kind>/validation_predictions/predictions.parquet
  <kind>/model.joblib, <kind>/best_params.json
  summary.json                           which kind is primary, with validation AUROCs
  reports/14_train_linear_probe/{status.json,report.md}

Usage
  python cehrbert_model/14_train_linear_probe.py
  python cehrbert_model/14_train_linear_probe.py --cv-folds 2 --n-jobs 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import linear_probe as lp  # noqa: E402
from pipeline import manifest as mf  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.baseline_predictions import write_predictions  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the frozen CEHR-BERT linear probe")
    parser.add_argument(
        "--features-dir", default=None, help="Step 13 features (default: per-run linear_probe/features)"
    )
    parser.add_argument("--output-dir", default=None, help="Output folder (default: per-run linear_probe)")
    parser.add_argument(
        "--window-days",
        type=int,
        default=paths.DEFAULT_WINDOW_DAYS,
        help="Look-back window in days; negative => unbounded (all history)",
    )
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX, help="CEHR-BERT token cap; part of the run label")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for the grid search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=None, help="Predictions manifest to update (default: per-run manifest)")
    parser.add_argument("--report-dir", default=None, help="Report dir (default: per-run report dir)")
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    features_dir = Path(args.features_dir) if args.features_dir else paths.cehrbert_linear_probe_features_dir(label)
    output_dir = Path(args.output_dir) if args.output_dir else paths.cehrbert_linear_probe_dir(label)
    report_dir = (
        Path(args.report_dir) if args.report_dir else paths.cehrbert_report_dir("14_train_linear_probe", label)
    )

    report = StepReport("14_train_linear_probe", title="Step 14 - Linear probe on frozen CEHR-BERT")
    for name, value in (
        ("run_label", label),
        ("features_dir", str(features_dir)),
        ("output_dir", str(output_dir)),
        ("cv_folds", args.cv_folds),
        ("seed", args.seed),
    ):
        report.add_metric(name, value)

    if not report.add_check(
        "feature files exist",
        all((features_dir / s / "meta.parquet").exists() for s in lp.SPLIT_NAMES),
        detail=str(features_dir),
    ):
        return _finalize(report, report_dir)

    val_auroc: dict[str, float] = {}
    summary: dict = {"kinds": {}}
    for kind in lp.FEATURE_KINDS:
        X_train, m_train = lp.load_split_features(features_dir, "train", kind)
        X_val, m_val = lp.load_split_features(features_dir, "val", kind)
        X_test, m_test = lp.load_split_features(features_dir, "test", kind)
        y_train, y_val, y_test = (m["label"].to_numpy() for m in (m_train, m_val, m_test))

        ok = all(0 < int(y.sum()) < len(y) for y in (y_train, y_val, y_test))
        if not report.add_check(
            f"{kind}: all splits non-empty with both classes",
            ok,
            detail=f"train={len(y_train)}/{int(y_train.sum())}+ val={len(y_val)}/{int(y_val.sum())}+ "
            f"test={len(y_test)}/{int(y_test.sum())}+",
        ):
            return _finalize(report, report_dir)

        search = lp.fit_on_train(X_train, y_train, args.cv_folds, args.seed, args.n_jobs)
        model = search.best_estimator_
        prob_val = model.predict_proba(X_val)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]

        val_auroc[kind] = float(roc_auc_score(y_val, prob_val))
        scores = {
            "cv_best_auroc": float(search.best_score_),
            "val_auroc": val_auroc[kind],
            "val_auprc": float(average_precision_score(y_val, prob_val)),
            "test_auroc": float(roc_auc_score(y_test, prob_test)),
            "test_auprc": float(average_precision_score(y_test, prob_test)),
        }
        for key, value in scores.items():
            report.add_metric(f"{kind}_{key}", round(value, 4))
        report.add_metric(f"{kind}_best_params", str(search.best_params_))
        report.warn(f"{kind}: test AUROC above 0.5 floor", ok=scores["test_auroc"] > 0.5)

        kind_dir = output_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        test_pred = write_predictions(
            kind_dir / "test_predictions", m_test["person_id"], m_test["index_date"], y_test, prob_test
        )
        val_pred = write_predictions(
            kind_dir / "validation_predictions", m_val["person_id"], m_val["index_date"], y_val, prob_val
        )
        model_path = kind_dir / "model.joblib"
        joblib.dump(model, model_path)
        params_path = lp.write_json(
            kind_dir / "best_params.json",
            {"best_params": search.best_params_, "cv_best_auroc": scores["cv_best_auroc"]},
        )
        for art in (test_pred, val_pred, model_path, params_path):
            report.add_artifact(art)
        summary["kinds"][kind] = {"best_params": search.best_params_, **scores}

    primary = lp.choose_primary_kind(val_auroc)
    summary.update(
        primary_kind=primary,
        secondary_kind=next(k for k in lp.FEATURE_KINDS if k != primary),
        selection_rule="higher validation AUROC",
    )
    report.add_metric("primary_kind", primary)
    report.add_artifact(lp.write_json(output_dir / "summary.json", summary))

    manifest_path = Path(args.manifest) if args.manifest else paths.cehrbert_manifest_path(label)
    secondary = summary["secondary_kind"]
    entries = [
        mf.make_entry(
            f"linear_probe_{kind}",
            f"Linear probe ({kind}, {'secondary' if kind == secondary else 'primary'})",
            "linear_probe",
            output_dir / kind / "test_predictions",
            output_dir / kind / "validation_predictions",
            secondary=kind == secondary,
        )
        for kind in lp.FEATURE_KINDS
    ]
    report.add_artifact(mf.upsert_models(manifest_path, entries))

    return _finalize(report, report_dir)


def _finalize(report: StepReport, report_dir: Path) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
