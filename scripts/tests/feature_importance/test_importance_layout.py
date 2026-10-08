"""Feature importance steps 01 and 03 read classical_ml artifacts and share the feature_importance build root."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from pipeline import paths

FEATURES = ["cond:1", "drug:2", "age"]


@pytest.fixture
def classical_dir(tmp_path):
    root = tmp_path / "classical_ml" / "w_365"
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + 0.1 * rng.normal(size=60) > 0).astype(int)
    (root / "features").mkdir(parents=True)
    (root / "features" / "feature_names.json").write_text(json.dumps(FEATURES))
    for name, model in (
        ("lr", Pipeline([("scale", MaxAbsScaler()), ("clf", LogisticRegression())])),
        ("xgboost", RandomForestClassifier(n_estimators=5, random_state=0)),
    ):
        (root / name).mkdir()
        joblib.dump(model.fit(X, y), root / name / "model.joblib")
    return root


def test_baseline_importance_reads_classical_ml_and_writes_to_feature_importance(step, classical_dir, tmp_path):
    out = tmp_path / "feature_importance" / "w_365_ctx512"
    rc = step("feature_importance/01_baseline_importance.py").main(
        ["--classical-dir", str(classical_dir), "--output-dir", str(out), "--concept-table", str(tmp_path / "none")]
    )
    assert rc == 0
    for model, column in (("lr", "coefficient"), ("xgboost", "importance")):
        df = pd.read_csv(out / model / "feature_importance.csv")
        assert set(df["feature"]) == set(FEATURES) and column in df.columns
    assert not list(classical_dir.rglob("feature_importance.csv"))


def test_baseline_importance_missing_features_lists_available_classical_folders(step, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "CLASSICAL_ML_BUILD_DIR", tmp_path / "classical_ml")
    (tmp_path / "classical_ml" / "w_30" / "features").mkdir(parents=True)
    assert step("feature_importance/01_baseline_importance.py").main([]) == 1
    out = capsys.readouterr().out
    assert "w_30" in out and "--window-days" in out


def test_top_feature_tables_read_rankings_from_the_feature_importance_root(step, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_IMPORTANCE_BUILD_DIR", tmp_path / "feature_importance")
    run = paths.feature_importance_dir(paths.run_label(365, 512))
    pd.DataFrame({"feature": FEATURES, "coefficient": [0.5, -0.2, 0.1]}).to_csv(
        _mk(run / "lr") / "feature_importance.csv", index=False
    )
    pd.DataFrame({"feature": FEATURES, "importance": [0.5, 0.3, 0.2]}).to_csv(
        _mk(run / "xgboost") / "feature_importance.csv", index=False
    )
    pd.DataFrame({"concept_id": ["1"], "gradxatt_mean": [0.9], "pct_patients_present": [50.0]}).to_csv(
        _mk(run / "attention") / "concept_attribution_pred_pos.csv", index=False
    )
    concepts = tmp_path / "concept.parquet"
    pd.DataFrame({"concept_id": [1, 2], "concept_name": ["ICD-10-CM:I10", "6719"]}).to_parquet(concepts)
    assert step("feature_importance/03_top_feature_tables.py").main(["--concept-table", str(concepts)]) == 0
    tables = run / "feature_tables"
    assert (tables / "lr_top10.csv").is_file()
    assert "| rank | feature |" in (tables / "top_features.md").read_text()


def _mk(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
