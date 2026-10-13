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


CONCEPT_ROWS = pd.DataFrame(
    {
        "concept_id": [0, 1, 2, 8527, 8507],
        "concept_name": ["No matching concept", "Amnesia", "donepezil", "White", "MALE"],
        "vocabulary_id": ["None", "SNOMED", "RxNorm", "Race", "Gender"],
        "concept_code": ["No matching concept", "48167000", "135447", "5", "M"],
    }
)


def test_baseline_importance_labels_race_gender_and_derived_features(step, tmp_path, classical_dir):
    names = ["cond:1", "race:8527", "race:0", "gender:0", "gender:8507", "age", "history_length_days", "drug:2"]
    (classical_dir / "features" / "feature_names.json").write_text(json.dumps(names))
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(60, len(names))), np.arange(60) % 2
    lr = Pipeline([("scale", MaxAbsScaler()), ("clf", LogisticRegression())]).fit(X, y)
    joblib.dump(lr, classical_dir / "lr" / "model.joblib")
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    joblib.dump(forest, classical_dir / "xgboost" / "model.joblib")
    concepts = tmp_path / "concept.parquet"
    CONCEPT_ROWS.to_parquet(concepts)
    out = tmp_path / "out"
    mod = step("feature_importance/01_baseline_importance.py")
    argv = ["--classical-dir", str(classical_dir), "--output-dir", str(out), "--concept-table", str(concepts)]
    assert mod.main(argv) == 0
    for model in ("lr", "xgboost"):
        got = pd.read_csv(out / model / "feature_importance.csv").set_index("feature")["label"].to_dict()
        assert got == {
            "cond:1": "cond: Amnesia",
            "race:8527": "race: White",
            "race:0": "race: unknown",
            "gender:0": "gender: unknown",
            "gender:8507": "gender: MALE",
            "age": "age at index date (years)",
            "history_length_days": "record length before index date (days)",
            "drug:2": "drug: donepezil",
        }


def test_top_feature_tables_use_vocabulary_code_and_concept_name(step, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_IMPORTANCE_BUILD_DIR", tmp_path / "feature_importance")
    run = paths.feature_importance_dir(paths.run_label(365, 512))
    features = ["cond:1", "drug:2", "age", "race:8527", "race:0", "gender:8507"]
    pd.DataFrame({"feature": features, "coefficient": [0.6, -0.5, 0.4, 0.3, 0.2, 0.1]}).to_csv(
        _mk(run / "lr") / "feature_importance.csv", index=False
    )
    pd.DataFrame({"feature": features, "importance": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}).to_csv(
        _mk(run / "xgboost") / "feature_importance.csv", index=False
    )
    tokens = ["1", "2", "D5", "i-LT", "Race/0", "age:70", "Visit/8527", "[CLS]"]
    pd.DataFrame(
        {"concept_id": tokens, "gradxatt_mean": np.linspace(1, 0.1, len(tokens)), "pct_patients_present": 50.0}
    ).to_csv(_mk(run / "attention") / "concept_attribution_pred_pos.csv", index=False)
    concepts = tmp_path / "concept.parquet"
    CONCEPT_ROWS.to_parquet(concepts)

    argv = ["--concept-table", str(concepts), "--top", "8"]
    assert step("feature_importance/03_top_feature_tables.py").main(argv) == 0

    tables = run / "feature_tables"
    lr = pd.read_csv(tables / "lr_top10.csv").set_index("feature")
    assert lr.loc["cond:1", "source_code"] == "SNOMED:48167000"
    assert lr.loc["cond:1", "description"] == "Amnesia"
    assert lr.loc["drug:2", "source_code"] == "RxNorm:135447"
    assert lr.loc["drug:2", "description"] == "Donepezil (cholinesterase inhibitor; anti-dementia)"
    assert lr.loc["race:8527", "source_code"] == "Race:5"
    assert lr.loc["race:8527", "description"] == "White"
    assert lr.loc["race:0", "description"] == "unknown race"
    assert lr.loc["race:0", "source_code"] == "race:0"
    assert lr.loc["gender:8507", "description"] == "Male"
    assert lr.loc["age", "source_code"] == "age" and lr.loc["age", "description"] == "Age at index date"
    assert not (lr["description"] == "(no description available)").any()
    assert set(pd.read_csv(tables / "xgboost_top10.csv")["source_code"]) >= {"SNOMED:48167000", "Race:5"}

    att = pd.read_csv(tables / "attention_top10.csv", dtype={"concept_id": str}).set_index("concept_id")
    assert att.loc["1", "source_code"] == "SNOMED:48167000" and att.loc["1", "description"] == "Amnesia"
    assert att.loc["D5", "description"] == "time gap: 5 days between visits"
    assert att.loc["i-LT", "source_code"] == "i-LT"
    assert att.loc["Race/0", "description"] == "race: unknown"
    assert att.loc["age:70", "description"] == "age at first visit: 70"
    assert att.loc["[CLS]", "description"] == "(no description available)"
    assert "SNOMED:48167000" in (tables / "top_features.md").read_text()
