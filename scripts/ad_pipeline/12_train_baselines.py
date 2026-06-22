#!/usr/bin/env python
"""Step 12: Train the classical-ML baselines (L2 logistic regression, XGBoost).

These two models predict the *same* AD task as CEHR-BERT and are scored by the *same*
evaluation code (``pipeline.evaluation`` / ``10_evaluate.py``). This step consumes the
tabular feature matrices written by step 11 (``build/baselines/features/``), tunes each
model by cross-validated AUROC on the training split, refits on train, and writes
predictions in CEHR-BERT's prediction-parquet schema so the baseline folders are drop-in
inputs to ``10_evaluate.py``.

Models
------
  * **L2 logistic regression** : log1p(counts) -> MaxAbsScaler -> LogisticRegression(l2),
    tuned over C and class_weight with GridSearchCV.
  * **XGBoost** : XGBClassifier(hist) with a fixed scale_pos_weight for class imbalance,
    tuned with RandomizedSearchCV, then a final refit on train with early stopping on the
    validation split.

The sanity AUROC/AUPRC numbers printed here are *not* authoritative -- the headline metrics
(with bootstrap CIs, calibration, operating points) come from ``10_evaluate.py`` run against
the prediction folders this step writes.

Outputs (under build/baselines/)
  lr/test_predictions/predictions.parquet
  lr/validation_predictions/predictions.parquet
  lr/model.joblib, lr/best_params.json
  xgboost/test_predictions/predictions.parquet
  xgboost/validation_predictions/predictions.parquet
  xgboost/model.joblib, xgboost/best_params.json
  build/reports/12_train_baselines/{status.json,report.md}

Usage
  python 12_train_baselines.py
  python 12_train_baselines.py --cv-folds 5 --xgb-iter 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import baselines as bl  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.baseline_predictions import write_predictions  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

FEATURES_DIR = paths.BUILD_DIR / "baselines" / "features"
OUTPUT_DIR = paths.BUILD_DIR / "baselines"


def _build_lr_search(cv: StratifiedKFold) -> GridSearchCV:
    """L2 logistic regression pipeline + grid over C and class_weight."""
    pipe = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, accept_sparse=True)),
            ("scale", MaxAbsScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)),
        ]
    )
    grid = {
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    return GridSearchCV(
        pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )


def _build_xgb_search(cv: StratifiedKFold, scale_pos_weight: float, n_iter: int,
                      seed: int) -> RandomizedSearchCV:
    """XGBoost classifier + randomized search over the usual tree/boosting knobs."""
    base = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    param_dist = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "n_estimators": [200, 400, 600, 800],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.5, 1.0],
    }
    return RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=cv, scoring="roc_auc",
        n_jobs=-1, refit=True, random_state=seed,
    )


def _quick_scores(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sanity AUROC / AUPRC (authoritative metrics come from 10_evaluate.py)."""
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def _write_model_outputs(
    report: StepReport,
    model_dir: Path,
    estimator,
    best_params: dict,
    test_pids: np.ndarray,
    test_dates: np.ndarray,
    y_test: np.ndarray,
    prob_test: np.ndarray,
    val_pids: np.ndarray,
    val_dates: np.ndarray,
    y_val: np.ndarray,
    prob_val: np.ndarray,
) -> None:
    """Persist predictions, the fitted estimator, and best_params; register artifacts."""
    model_dir.mkdir(parents=True, exist_ok=True)

    test_pred = write_predictions(
        model_dir / "test_predictions", test_pids, test_dates, y_test, prob_test
    )
    val_pred = write_predictions(
        model_dir / "validation_predictions", val_pids, val_dates, y_val, prob_val
    )

    model_path = model_dir / "model.joblib"
    joblib.dump(estimator, model_path)

    params_path = model_dir / "best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")

    for art in (test_pred, val_pred, model_path, params_path):
        report.add_artifact(art)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train classical-ML baselines for the AD task")
    parser.add_argument("--features-dir", default=str(FEATURES_DIR),
                        help="Folder of {train,val,test}_X.npz / *_meta.npz from step 11")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Parent folder for lr/ and xgboost/ outputs")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--xgb-iter", type=int, default=30,
                        help="Number of RandomizedSearchCV samples for XGBoost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir",
                        default=str(paths.build_report_dir_for("12_train_baselines")))
    args = parser.parse_args(argv)

    report = StepReport("12_train_baselines", title="Step 12 - Train classical-ML baselines")
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    report.add_metric("features_dir", str(features_dir))
    report.add_metric("output_dir", str(output_dir))
    report.add_metric("cv_folds", args.cv_folds)
    report.add_metric("xgb_iter", args.xgb_iter)
    report.add_metric("seed", args.seed)

    # --- load the three splits --------------------------------------------
    if not report.add_check("feature matrices exist",
                            features_dir.is_dir() and (features_dir / "train_X.npz").exists(),
                            detail=str(features_dir)):
        return _finalize(report, args.report_dir)

    X_train, y_train, train_pids, train_dates = bl.load_split(features_dir, "train")
    X_val, y_val, val_pids, val_dates = bl.load_split(features_dir, "val")
    X_test, y_test, test_pids, test_dates = bl.load_split(features_dir, "test")

    def _both_classes(y: np.ndarray) -> bool:
        return len(y) > 0 and 0 < int(np.sum(y)) < len(y)

    for name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        report.add_metric(f"{name}_size", int(len(y)))
        report.add_metric(f"{name}_positives", int(np.sum(y)))

    splits_ok = _both_classes(y_train) and _both_classes(y_val) and _both_classes(y_test)
    if not report.add_check("all splits non-empty with both classes", splits_ok,
                            detail=f"train={len(y_train)}/{int(np.sum(y_train))}+ "
                                   f"val={len(y_val)}/{int(np.sum(y_val))}+ "
                                   f"test={len(y_test)}/{int(np.sum(y_test))}+"):
        return _finalize(report, args.report_dir)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # --- L2 logistic regression -------------------------------------------
    lr_search = _build_lr_search(cv)
    lr_search.fit(X_train, y_train)
    lr_model = lr_search.best_estimator_
    report.add_metric("lr_cv_best_auroc", round(float(lr_search.best_score_), 4))
    report.add_metric("lr_best_params", json.dumps(lr_search.best_params_, default=str))

    lr_prob_val = lr_model.predict_proba(X_val)[:, 1]
    lr_prob_test = lr_model.predict_proba(X_test)[:, 1]

    lr_val_auroc, lr_val_auprc = _quick_scores(y_val, lr_prob_val)
    lr_test_auroc, lr_test_auprc = _quick_scores(y_test, lr_prob_test)
    report.add_metric("lr_val_auroc", round(lr_val_auroc, 4))
    report.add_metric("lr_val_auprc", round(lr_val_auprc, 4))
    report.add_metric("lr_test_auroc", round(lr_test_auroc, 4))
    report.add_metric("lr_test_auprc", round(lr_test_auprc, 4))
    report.warn("lr test AUROC above 0.5 floor", ok=lr_test_auroc > 0.5,
                detail=f"AUROC={lr_test_auroc:.4f}")

    _write_model_outputs(
        report, output_dir / "lr", lr_model,
        {"best_params": lr_search.best_params_, "cv_best_auroc": float(lr_search.best_score_)},
        test_pids, test_dates, y_test, lr_prob_test,
        val_pids, val_dates, y_val, lr_prob_val,
    )

    # --- XGBoost ----------------------------------------------------------
    n_pos = int(np.sum(y_train))
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos else 1.0
    report.add_metric("xgb_scale_pos_weight", round(scale_pos_weight, 4))

    xgb_search = _build_xgb_search(cv, scale_pos_weight, args.xgb_iter, args.seed)
    xgb_search.fit(X_train, y_train)
    report.add_metric("xgb_cv_best_auroc", round(float(xgb_search.best_score_), 4))
    report.add_metric("xgb_best_params", json.dumps(xgb_search.best_params_, default=str))

    # Final refit on train with early stopping on the validation split.
    final_params = dict(xgb_search.best_params_)
    final_params["n_estimators"] = 1000
    xgb_model = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
        **final_params,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iteration = getattr(xgb_model, "best_iteration", None)
    if best_iteration is not None:
        report.add_metric("xgb_best_iteration", int(best_iteration))

    xgb_prob_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_prob_test = xgb_model.predict_proba(X_test)[:, 1]

    xgb_val_auroc, xgb_val_auprc = _quick_scores(y_val, xgb_prob_val)
    xgb_test_auroc, xgb_test_auprc = _quick_scores(y_test, xgb_prob_test)
    report.add_metric("xgb_val_auroc", round(xgb_val_auroc, 4))
    report.add_metric("xgb_val_auprc", round(xgb_val_auprc, 4))
    report.add_metric("xgb_test_auroc", round(xgb_test_auroc, 4))
    report.add_metric("xgb_test_auprc", round(xgb_test_auprc, 4))
    report.warn("xgb test AUROC above 0.5 floor", ok=xgb_test_auroc > 0.5,
                detail=f"AUROC={xgb_test_auroc:.4f}")

    _write_model_outputs(
        report, output_dir / "xgboost", xgb_model,
        {
            "best_params": xgb_search.best_params_,
            "cv_best_auroc": float(xgb_search.best_score_),
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
        },
        test_pids, test_dates, y_test, xgb_prob_test,
        val_pids, val_dates, y_val, xgb_prob_val,
    )

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python 13_compare_models.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
