"""Evaluation steps 01 and 02 on synthetic predictions described by manifests, and the parity check behind 02."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pipeline import comparison as cmp
from pipeline import manifest as mf
from pipeline import paths
from pipeline.baseline_predictions import write_predictions

INDEX = dt.date(2020, 1, 1)
SPLIT_IDS = {"train": range(1, 61), "val": range(61, 81), "test": range(81, 101)}
# name -> (display label, secondary, family)
MODELS = {
    "linear_probe_cls": ("Linear probe (cls, primary)", False, "linear_probe"),
    "linear_probe_mean": ("Linear probe (mean, secondary)", True, "linear_probe"),
    "lr": ("L2 Logistic Regression", False, "classical_ml"),
    "xgboost": ("XGBoost", False, "classical_ml"),
}


def _label(pid: int) -> int:
    return int(pid % 3 == 0)


def _write_split_predictions(folder: Path, ids, rng, signal: float) -> None:
    ids = np.array(list(ids))
    y = np.array([_label(p) for p in ids])
    prob = np.clip(rng.uniform(0.05, 0.6, size=len(ids)) + signal * y, 0.01, 0.99)
    write_predictions(folder, ids, np.array([INDEX] * len(ids), dtype="datetime64[D]"), y, prob)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Splits file, predictions for four models and one manifest per family; evaluation output is redirected."""
    monkeypatch.setattr(paths, "EVALUATION_BUILD_DIR", tmp_path / "evaluation")
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
    entries: dict[str, list[dict]] = {"cehrbert": [], "classical": []}
    for signal, (name, (label, secondary, family)) in zip((0.5, 0.2, 0.3, 0.4), MODELS.items()):
        folder = tmp_path / "preds" / name
        _write_split_predictions(folder / "test_predictions", SPLIT_IDS["test"], rng, signal)
        _write_split_predictions(folder / "validation_predictions", SPLIT_IDS["val"], rng, signal)
        group = "classical" if family == "classical_ml" else "cehrbert"
        entry = mf.make_entry(
            name, label, family, folder / "test_predictions", folder / "validation_predictions", secondary
        )
        entries[group].append(entry)
    for group, items in entries.items():
        mf.upsert_models(tmp_path / f"{group}_manifest.json", items)
    return tmp_path


def _manifest_args(root: Path) -> list[str]:
    return ["--manifest", str(root / "cehrbert_manifest.json"), "--manifest", str(root / "classical_manifest.json")]


def _run_step01(step, root: Path, *extra: str) -> int:
    return step("evaluation/01_evaluate.py").main(
        [*_manifest_args(root), "--name", "t", "--n-bootstrap", "20", "--report-dir", str(root / "report01"), *extra]
    )


def _run_step13(step, root: Path, *extra: str) -> int:
    return step("evaluation/02_compare_models.py").main(
        [
            *_manifest_args(root),
            "--name", "t",
            "--splits-file", str(root / "splits.parquet"),
            "--report-dir", str(root / "report"),
            "--n-bootstrap", "20",
            *extra,
        ]
    )  # fmt: skip


def _status(root: Path, report: str = "report") -> dict:
    return json.loads((root / report / "status.json").read_text())


def _failed_checks(root: Path, report: str = "report") -> list[str]:
    return [c["name"] for c in _status(root, report)["checks"] if not c["ok"] and c["level"] != "warning"]


def _failed_parity(root: Path) -> list[str]:
    return [n for n in _failed_checks(root) if "match canonical split" in n]


def _out(root: Path) -> Path:
    return root / "evaluation" / "t"


def test_step01_scores_every_manifest_model_into_its_own_folder(step, run_dir):
    assert _run_step01(step, run_dir) == 0
    for name in MODELS:
        model_dir = _out(run_dir) / "models" / name
        assert (model_dir / "metrics.json").is_file()
        assert (model_dir / "figures" / "roc_curve.png").is_file()
        assert (model_dir / "reports" / "01_evaluate" / "status.json").is_file()
    metrics = json.loads((_out(run_dir) / "models" / "lr" / "metrics.json").read_text())
    assert metrics["threshold_selected_on"] == "validation"


def test_step01_one_failing_model_fails_the_step_but_others_run(step, run_dir):
    (run_dir / "preds" / "lr" / "test_predictions" / "predictions.parquet").unlink()
    assert _run_step01(step, run_dir) == 1
    assert _failed_checks(run_dir, "report01") == ["lr: evaluation succeeded"]
    assert not (_out(run_dir) / "models" / "lr" / "metrics.json").exists()
    assert (_out(run_dir) / "models" / "xgboost" / "metrics.json").is_file()


def test_step01_missing_default_manifest_warns_and_no_model_fails(step, run_dir, monkeypatch):
    monkeypatch.setattr(paths, "CEHRBERT_MODEL_BUILD_DIR", run_dir / "no_cehrbert")
    monkeypatch.setattr(paths, "CLASSICAL_ML_BUILD_DIR", run_dir / "classical_ml")
    lr = MODELS["lr"]
    entry = mf.make_entry(
        "lr",
        lr[0],
        lr[2],
        run_dir / "preds" / "lr" / "test_predictions",
        run_dir / "preds" / "lr" / "validation_predictions",
    )
    mf.upsert_models(paths.classical_manifest_path(paths.window_label(365)), [entry])
    module = step("evaluation/01_evaluate.py")
    args = ["--name", "t", "--n-bootstrap", "20", "--report-dir", str(run_dir / "report01")]
    assert module.main(args) == 0
    warnings = [c["name"] for c in _status(run_dir, "report01")["checks"] if c["level"] == "warning" and not c["ok"]]
    assert any("cehrbert_model" in w for w in warnings)
    assert (_out(run_dir) / "models" / "lr" / "metrics.json").is_file()

    assert module.main([*args, "--window-days", "-1"]) == 1
    assert "at least one model to evaluate" in _failed_checks(run_dir, "report01")


def test_step01_single_model_mode_keeps_working(step, run_dir):
    out = run_dir / "single"
    rc = step("evaluation/01_evaluate.py").main(
        [
            "--predictions", str(run_dir / "preds" / "lr" / "test_predictions"),
            "--val-predictions", str(run_dir / "preds" / "lr" / "validation_predictions"),
            "--results-dir", str(out), "--figures-dir", str(out / "figures"), "--report-dir", str(out / "rep"),
            "--n-bootstrap", "20",
        ]
    )  # fmt: skip
    assert rc == 0 and (out / "metrics.json").is_file() and (out / "figures" / "pr_curve.png").is_file()


def test_step13_passes_and_writes_outputs(step, run_dir):
    assert _run_step01(step, run_dir) == 0
    assert _run_step13(step, run_dir) == 0
    out = _out(run_dir)
    for name in ("comparison.json", "comparison.md", "figures/compare_roc.png", "figures/compare_pr.png"):
        assert (out / name).is_file(), name

    data = json.loads((out / "comparison.json").read_text())
    labels = [m["model"] for m in data["models"]]
    assert labels == [label for label, _, _ in MODELS.values()]
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


def test_step13_with_one_manifest_compares_only_its_models(step, run_dir):
    assert _run_step01(step, run_dir) == 0
    rc = step("evaluation/02_compare_models.py").main(
        [
            "--manifest", str(run_dir / "classical_manifest.json"),
            "--name", "t",
            "--splits-file", str(run_dir / "splits.parquet"),
            "--report-dir", str(run_dir / "report"),
            "--n-bootstrap", "20",
        ]
    )  # fmt: skip
    assert rc == 0
    data = json.loads((_out(run_dir) / "comparison.json").read_text())
    assert [m["model"] for m in data["models"]] == ["L2 Logistic Regression", "XGBoost"]


def test_step13_fails_naming_the_model_when_step01_metrics_are_missing(step, run_dir):
    assert _run_step01(step, run_dir) == 0
    (_out(run_dir) / "models" / "xgboost" / "metrics.json").unlink()
    assert _run_step13(step, run_dir) == 1
    assert _failed_checks(run_dir) == ["XGBoost: step 01 metrics available"]
    detail = [c["detail"] for c in _status(run_dir)["checks"] if c["name"] == "XGBoost: step 01 metrics available"]
    assert "evaluation step 01" in detail[0]


def test_step13_does_not_launch_subprocesses(step, run_dir, monkeypatch):
    import subprocess

    assert _run_step01(step, run_dir) == 0

    def boom(*args, **kwargs):
        raise AssertionError("step 02 must not start a subprocess")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _run_step13(step, run_dir) == 0


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
    path = run_dir / "preds" / "xgboost" / "test_predictions" / "predictions.parquet"
    corrupt(pd.read_parquet(path)).to_parquet(path, index=False)
    assert _run_step01(step, run_dir) == 0
    assert _run_step13(step, run_dir) == 1
    assert _failed_parity(run_dir) == ["XGBoost: test predictions match canonical split"]


def test_step13_fails_when_validation_predictions_missing(step, run_dir):
    assert _run_step01(step, run_dir) == 0
    (run_dir / "preds" / "linear_probe_mean" / "validation_predictions" / "predictions.parquet").unlink()
    assert _run_step13(step, run_dir) == 1
    assert _failed_parity(run_dir) == ["Linear probe (mean, secondary): val predictions match canonical split"]


def test_step13_fails_when_validation_patients_differ(step, run_dir):
    path = run_dir / "preds" / "lr" / "validation_predictions" / "predictions.parquet"
    pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    assert _run_step01(step, run_dir) == 0
    assert _run_step13(step, run_dir) == 1
    assert _failed_parity(run_dir) == ["L2 Logistic Regression: val predictions match canonical split"]


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
