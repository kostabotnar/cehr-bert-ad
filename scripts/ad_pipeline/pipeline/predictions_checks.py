"""Verify finetuning outputs: test predictions parquet + test_results.json metrics.

The finetune runner (hf_cehrbert_finetune_runner.do_predict) writes:
  <results>/test_predictions/<i>.parquet  columns:
      subject_id, prediction_time, predicted_boolean_probability,
      predicted_boolean_value, boolean_value
  <results>/test_results.json  {roc_auc, pr_auc, test_loss}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .reporting import StepReport

REQUIRED_PREDICTION_COLUMNS = [
    "subject_id",
    "prediction_time",
    "predicted_boolean_probability",
    "boolean_value",
]


def add_predictions_checks(report: StepReport, results_dir: Path) -> StepReport:
    """Check the test_predictions parquet folder: exists, columns, prob range, rows."""
    results_dir = Path(results_dir)
    pred_dir = results_dir / "test_predictions"
    if not report.add_check("test_predictions folder exists", pred_dir.is_dir(), detail=str(pred_dir)):
        return report

    parquet_files = sorted(pred_dir.glob("*.parquet"))
    if not report.add_check("prediction parquet files present", bool(parquet_files), detail=f"{len(parquet_files)} files"):
        return report

    try:
        df = pd.read_parquet(pred_dir)
    except Exception as exc:
        report.add_check("predictions readable", False, detail=str(exc))
        return report

    report.add_metric("n_predictions", int(len(df)))
    report.add_check("predictions non-empty", len(df) > 0)

    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    report.add_check("predictions have required columns", not missing,
                     detail="ok" if not missing else f"missing: {', '.join(missing)}")

    if "predicted_boolean_probability" in df.columns and len(df):
        probs = pd.to_numeric(df["predicted_boolean_probability"], errors="coerce")
        in_range = bool(((probs >= 0) & (probs <= 1)).all()) and not probs.isna().any()
        report.add_check(
            "probabilities in [0, 1]",
            in_range,
            detail=f"min={probs.min():.4f}, max={probs.max():.4f}" if len(probs) else "no values",
        )
        report.add_metric("prob_min", round(float(probs.min()), 6) if len(probs) else None)
        report.add_metric("prob_max", round(float(probs.max()), 6) if len(probs) else None)

    if "boolean_value" in df.columns and len(df):
        n_pos = int(pd.Series(df["boolean_value"]).astype("boolean").fillna(False).sum())
        report.add_metric("n_positive_labels", n_pos)
        report.warn("test set has both classes", ok=0 < n_pos < len(df),
                    detail=f"{n_pos} positive of {len(df)}")

    return report


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def add_test_results_checks(report: StepReport, results_dir: Path) -> StepReport:
    """Parse test_results.json and validate roc_auc / pr_auc / test_loss."""
    results_dir = Path(results_dir)
    path = results_dir / "test_results.json"
    metrics = _read_json(path)
    if not report.add_check("test_results.json present & valid", metrics is not None, detail=str(path)):
        return report

    for key in ("roc_auc", "pr_auc"):
        val = metrics.get(key)
        report.add_metric(key, val)
        ok = isinstance(val, (int, float)) and val == val and 0.0 <= val <= 1.0
        report.add_check(f"{key} present and in [0, 1]", ok, detail=f"{key}={val}")

    if "test_loss" in metrics:
        loss = metrics["test_loss"]
        report.add_metric("test_loss", loss)
        report.warn("test_loss is finite", ok=isinstance(loss, (int, float)) and loss == loss, detail=f"test_loss={loss}")

    return report
