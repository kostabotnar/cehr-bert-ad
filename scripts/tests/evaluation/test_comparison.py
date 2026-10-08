"""Step 13 (model comparison) with synthetic predictions, and the parity check behind it."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pipeline import comparison as cmp
from pipeline.baseline_predictions import write_predictions

INDEX = dt.date(2020, 1, 1)
SPLIT_IDS = {"train": range(1, 61), "val": range(61, 81), "test": range(81, 101)}
MODELS = {
    "lr": "baselines/lr",
    "xgboost": "baselines/xgboost",
    "cls": "linear_probe/cls",
    "mean": "linear_probe/mean",
}


def _label(pid: int) -> int:
    return int(pid % 3 == 0)


def _write_split_predictions(folder: Path, ids, rng, signal: float) -> None:
    ids = np.array(list(ids))
    y = np.array([_label(p) for p in ids])
    prob = np.clip(rng.uniform(0.05, 0.6, size=len(ids)) + signal * y, 0.01, 0.99)
    write_predictions(folder, ids, np.array([INDEX] * len(ids), dtype="datetime64[D]"), y, prob)


@pytest.fixture
def run_dir(tmp_path):
    """Splits file plus predictions for lr, xgboost and both linear probe kinds."""
    ids = [p for part in SPLIT_IDS.values() for p in part]
    pl.DataFrame(
        {
            "person_id": ids,
            "index_date": [INDEX] * len(ids),
            "label": [_label(p) for p in ids],
            "split": [name for name, part in SPLIT_IDS.items() for _ in part],
        },
        schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64, "split": pl.Utf8},
    ).write_parquet(tmp_path / "splits.parquet")
    rng = np.random.default_rng(0)
    for signal, rel in zip((0.3, 0.4, 0.5, 0.2), MODELS.values()):
        _write_split_predictions(tmp_path / rel / "test_predictions", SPLIT_IDS["test"], rng, signal)
        _write_split_predictions(tmp_path / rel / "validation_predictions", SPLIT_IDS["val"], rng, signal)
    (tmp_path / "linear_probe" / "summary.json").write_text(
        json.dumps({"primary_kind": "cls", "secondary_kind": "mean"}), encoding="utf-8"
    )
    return tmp_path


def _run_step13(step, root: Path, *extra: str) -> int:
    return step("13_compare_models.py").main(
        [
            "--splits-file", str(root / "splits.parquet"),
            "--baselines-dir", str(root / "baselines"),
            "--linear-probe-dir", str(root / "linear_probe"),
            "--output-dir", str(root / "out"),
            "--figures-dir", str(root / "out" / "figures"),
            "--report-dir", str(root / "report"),
            "--cehrbert-metrics", str(root / "missing" / "metrics.json"),
            "--n-bootstrap", "20",
            *extra,
        ]
    )  # fmt: skip


def _status(root: Path) -> dict:
    return json.loads((root / "report" / "status.json").read_text())


def _failed_checks(root: Path) -> list[str]:
    return [c["name"] for c in _status(root)["checks"] if not c["ok"] and "match canonical split" in c["name"]]


def test_step13_passes_and_writes_outputs(step, run_dir):
    assert _run_step13(step, run_dir) == 0
    out = run_dir / "out"
    for name in ("comparison.json", "comparison.md", "figures/compare_roc.png", "figures/compare_pr.png"):
        assert (out / name).is_file(), name

    data = json.loads((out / "comparison.json").read_text())
    labels = [m["model"] for m in data["models"]]
    assert labels == [
        "Linear probe (cls, primary)",
        "Linear probe (mean, secondary)",
        "L2 Logistic Regression",
        "XGBoost",
    ]
    assert [m["secondary"] for m in data["models"]] == [False, True, False, False]
    for row in data["models"]:
        assert len(row["sensitivity_ci95"]) == 2 and len(row["calibration_slope_ci95"]) == 2
    assert len(data["paired_differences"]["pairs"]) == 6
    assert all(p["auroc"]["n_valid_resamples"] > 0 for p in data["paired_differences"]["pairs"])
    for res in data["parity"].values():
        assert res["test"]["ok"] and res["val"]["ok"]
        assert res["test"]["n_rows"] == res["test"]["n_expected"] == 20

    md = (out / "comparison.md").read_text()
    assert "Linear probe (mean, secondary)" in md and "Paired differences" in md
    assert all(ch not in md for ch in (chr(0x2014), chr(0x2013)))


def test_step13_without_linear_probe_still_runs(step, run_dir):
    (run_dir / "linear_probe" / "summary.json").unlink()
    assert _run_step13(step, run_dir) == 0
    data = json.loads((run_dir / "out" / "comparison.json").read_text())
    assert [m["model"] for m in data["models"]] == ["L2 Logistic Regression", "XGBoost"]


def _drop_test_patient(df):
    return df.iloc[1:]


def _duplicate_row(df):
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def _flip_label(df):
    df = df.copy()
    df.loc[df.index[0], "boolean_value"] = not df["boolean_value"].iloc[0]
    return df


@pytest.mark.parametrize("corrupt", [_drop_test_patient, _duplicate_row, _flip_label])
def test_step13_fails_when_test_predictions_differ_from_split(step, run_dir, corrupt):
    path = run_dir / "baselines" / "xgboost" / "test_predictions" / "predictions.parquet"
    corrupt(pd.read_parquet(path)).to_parquet(path, index=False)
    assert _run_step13(step, run_dir, "--skip-eval") == 1
    assert _failed_checks(run_dir) == ["XGBoost: test predictions match canonical split"]


def test_step13_fails_when_validation_predictions_missing(step, run_dir):
    (run_dir / "linear_probe" / "mean" / "validation_predictions" / "predictions.parquet").unlink()
    assert _run_step13(step, run_dir, "--skip-eval") == 1
    assert _failed_checks(run_dir) == ["Linear probe (mean, secondary): val predictions match canonical split"]


def test_step13_fails_when_validation_patients_differ(step, run_dir):
    path = run_dir / "baselines" / "lr" / "validation_predictions" / "predictions.parquet"
    pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    assert _run_step13(step, run_dir, "--skip-eval") == 1
    assert _failed_checks(run_dir) == ["L2 Logistic Regression: val predictions match canonical split"]


def test_check_parity_counts(tmp_path):
    expected = pd.DataFrame({"person_id": [1, 2, 3], "label": [0, 1, 0]})
    df = pd.DataFrame(
        {"subject_id": [1, 2, 2, 9], "boolean_value": [False, False, True, True], "predicted_boolean_probability": 0.5}
    )
    (tmp_path / "p").mkdir()
    df.to_parquet(tmp_path / "p" / "predictions.parquet", index=False)
    res = cmp.check_parity(tmp_path / "p", expected, "test")
    assert (res.n_rows, res.n_missing, res.n_extra, res.n_duplicate_rows, res.n_label_mismatch) == (4, 1, 1, 1, 1)
    assert not res.ok
    assert not cmp.check_parity(tmp_path / "none", expected, "val").ok
