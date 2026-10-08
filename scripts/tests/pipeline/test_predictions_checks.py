"""Tests for pipeline.predictions_checks."""

from __future__ import annotations

import json

import pandas as pd

from pipeline import predictions_checks as pc
from pipeline.reporting import StepReport


def test_valid_predictions_pass(finetune_results_dir):
    r = StepReport("t")
    pc.add_predictions_checks(r, finetune_results_dir)
    pc.add_test_results_checks(r, finetune_results_dir)
    assert r.status == "pass"
    assert r.metrics["n_predictions"] == 4
    assert r.metrics["roc_auc"] == 0.83


def test_missing_predictions_folder_fails(tmp_path):
    r = StepReport("t")
    pc.add_predictions_checks(r, tmp_path)
    assert r.status == "fail"


def test_probability_out_of_range_fails(tmp_path):
    pred = tmp_path / "test_predictions"
    pred.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "prediction_time": [None, None],
            "predicted_boolean_probability": [0.5, 1.7],  # out of range
            "boolean_value": [True, False],
        }
    ).to_parquet(pred / "0.parquet")
    r = StepReport("t")
    pc.add_predictions_checks(r, tmp_path)
    assert r.status == "fail"


def test_missing_required_column_fails(tmp_path):
    pred = tmp_path / "test_predictions"
    pred.mkdir()
    pd.DataFrame({"subject_id": [1], "predicted_boolean_probability": [0.5]}).to_parquet(pred / "0.parquet")
    r = StepReport("t")
    pc.add_predictions_checks(r, tmp_path)
    assert any(c.name == "predictions have required columns" and not c.ok for c in r.checks)


def test_null_auc_fails(tmp_path):
    (tmp_path / "test_results.json").write_text(
        json.dumps({"roc_auc": None, "pr_auc": None, "test_loss": 0.4}), encoding="utf-8"
    )
    r = StepReport("t")
    pc.add_test_results_checks(r, tmp_path)
    assert r.status == "fail"
