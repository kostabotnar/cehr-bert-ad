"""Tests for the classical-ML baseline code merged from add-baseline-comparison."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from pipeline import baselines as bl
from pipeline import paths

D = dt.date


def _cohort(pids, label_fn=lambda p: int(p % 2 == 0), index_date=D(2020, 1, 1)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "person_id": list(pids),
            "index_date": [index_date] * len(pids),
            "label": [label_fn(p) for p in pids],
        },
        schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64},
    )


# --------------------------------------------------------------------------
# recover_splits
# --------------------------------------------------------------------------
def test_recover_splits_reads_canonical_file(tmp_path):
    splits_file = tmp_path / "splits.parquet"
    table = pl.concat(
        [
            _cohort([1, 2, 3, 4]).with_columns(split=pl.lit("train")),
            _cohort([5, 6]).with_columns(split=pl.lit("val")),
            _cohort([7, 8]).with_columns(split=pl.lit("test")),
        ]
    )
    table.write_parquet(splits_file)

    splits = bl.recover_splits(splits_file)

    assert splits.train["person_id"].to_list() == [1, 2, 3, 4]
    assert splits.val["person_id"].to_list() == [5, 6]
    assert splits.test["person_id"].to_list() == [7, 8]
    assert splits.train.columns == ["person_id", "index_date", "label"]
    assert splits.train.schema["index_date"] == pl.Date


def test_recover_splits_rejects_duplicate_person(tmp_path):
    splits_file = tmp_path / "splits.parquet"
    pl.concat(
        [
            _cohort([1, 2]).with_columns(split=pl.lit("train")),
            _cohort([2]).with_columns(split=pl.lit("test")),
        ]
    ).write_parquet(splits_file)
    with pytest.raises(ValueError, match="more than once"):
        bl.recover_splits(splits_file)


def test_recover_splits_requires_the_canonical_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="run step 6"):
        bl.recover_splits(tmp_path / "missing.parquet")


def test_run_label_formats():
    assert paths.run_label(365, 512) == "w_365_ctx512"
    assert paths.run_label(-1, 4096) == "w_all_ctx4096"
    assert paths.run_label(None, 512) == "w_all_ctx512"


def test_default_run_label_matches_trained_model():
    assert paths.run_label(paths.DEFAULT_WINDOW_DAYS, paths.DEFAULT_CTX) == "w_365_ctx512"


# --------------------------------------------------------------------------
# Feature extraction on a hand-built OMOP set
# --------------------------------------------------------------------------
INDEX = D(2020, 1, 1)  # 365-day window is [2019-01-01, 2020-01-01)


def _write_table(root: Path, table: str, rows: dict, schema: dict) -> None:
    d = root / table
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema).write_parquet(d / "data.parquet")


def _write_omop(root: Path, person, cond, drug, proc, visit) -> Path:
    i64, date = pl.Int64, pl.Date
    _write_table(
        root,
        "person",
        dict(zip(["person_id", "gender_concept_id", "race_concept_id", "year_of_birth"], zip(*person))),
        {"person_id": i64, "gender_concept_id": i64, "race_concept_id": i64, "year_of_birth": i64},
    )
    for table, concept_col, date_col, rows in (
        ("condition_occurrence", "condition_concept_id", "condition_start_date", cond),
        ("drug_exposure", "drug_concept_id", "drug_exposure_start_date", drug),
        ("procedure_occurrence", "procedure_concept_id", "procedure_date", proc),
    ):
        _write_table(
            root,
            table,
            dict(zip(["person_id", concept_col, date_col], zip(*rows))),
            {"person_id": i64, concept_col: i64, date_col: date},
        )
    _write_table(
        root,
        "visit_occurrence",
        dict(zip(["person_id", "visit_occurrence_id", "visit_start_date"], zip(*visit))),
        {"person_id": i64, "visit_occurrence_id": i64, "visit_start_date": date},
    )
    return root


@pytest.fixture
def small_omop(tmp_path) -> Path:
    """Persons 1 and 2 are train, person 9 is test. Comments give the expected treatment."""
    return _write_omop(
        tmp_path / "omop",
        person=[(1, 8507, 8527, 1950), (2, 8532, 8527, 1960), (9, 9999, 8527, 1955)],
        cond=[
            (1, 100, D(2019, 6, 1)),  # in window
            (1, 100, INDEX),  # on the index date: excluded
            (1, 101, D(2018, 12, 31)),  # one day before the window: excluded unless all history
            (1, 102, D(2019, 1, 1)),  # first day of the window: included
            (1, 103, D(2020, 3, 1)),  # after index: excluded
            (1, 0, D(2019, 6, 1)),  # concept 0: excluded
            (2, 100, D(2019, 5, 1)),
            (9, 999, D(2019, 5, 1)),  # only a test patient has this concept
        ],
        drug=[(1, 200, D(2019, 12, 31)), (1, 200, D(2019, 12, 31)), (1, 201, D(2020, 2, 1))],
        proc=[(1, 300, D(2019, 3, 1))],
        visit=[
            (1, 1, D(2015, 1, 1)),
            (1, 2, D(2019, 6, 1)),
            (1, 2, D(2019, 6, 1)),
            (1, 3, INDEX),
            (2, 4, D(2019, 5, 1)),
            (9, 5, D(2019, 5, 1)),
        ],
    )


def _counts_set(counts: pl.DataFrame) -> set:
    return set(counts.select("person_id", "feature", "count").iter_rows())


def test_build_counts_strictly_before_index_and_within_window(small_omop):
    cohort = _cohort([1, 2, 9], index_date=INDEX)
    counts = bl.build_counts(cohort, small_omop, 365)
    assert _counts_set(counts) == {
        (1, "cond:100", 1),
        (1, "cond:102", 1),
        (1, "drug:200", 2),
        (1, "proc:300", 1),
        (2, "cond:100", 1),
        (9, "cond:999", 1),
    }


def test_build_counts_all_history_still_stops_before_index(small_omop):
    counts = bl.build_counts(_cohort([1], index_date=INDEX), small_omop, -1)
    assert _counts_set(counts) == {
        (1, "cond:100", 1),
        (1, "cond:101", 1),
        (1, "cond:102", 1),
        (1, "drug:200", 2),
        (1, "proc:300", 1),
    }


def test_build_visit_features(small_omop):
    visit = bl.build_visit_features(_cohort([1, 2], index_date=INDEX), small_omop, 365).sort("person_id")
    assert visit["history_length_days"].to_list() == [(INDEX - D(2015, 1, 1)).days, (INDEX - D(2019, 5, 1)).days]
    assert visit["n_visits"].to_list() == [1, 1]


def _fit(small_omop, min_prevalence):
    train = _cohort([1, 2], index_date=INDEX)
    counts = bl.build_counts(train, small_omop, 365)
    demo = bl.build_demographics(train, small_omop)
    visit = bl.build_visit_features(train, small_omop, 365)
    return bl.fit_feature_spec(counts, demo, train.height, min_prevalence, 365, visit_features=visit)


def test_vocabulary_fit_on_train_only(small_omop):
    spec = _fit(small_omop, 0.5)
    assert spec.code_features == ["cond:100", "cond:102", "drug:200", "proc:300"]
    assert "cond:999" not in spec.feature_names
    assert spec.gender_categories == [8507, 8532]
    assert _fit(small_omop, 0.6).code_features == ["cond:100"]


def test_assemble_matrix_width_and_values(small_omop):
    spec = _fit(small_omop, 0.5)
    names = spec.feature_names

    test = _cohort([9], index_date=INDEX)
    X, y, pids, _ = bl.assemble_matrix(
        test,
        bl.build_counts(test, small_omop, 365),
        bl.build_demographics(test, small_omop),
        spec,
        visit=bl.build_visit_features(test, small_omop, 365),
    )
    assert X.shape == (1, len(names))
    row = dict(zip(names, X.toarray()[0]))
    assert all(row[f] == 0 for f in spec.code_features)
    assert row["gender:8507"] == 0 and row["gender:8532"] == 0  # unseen gender 9999
    assert row["age"] == 65.0
    assert pids.tolist() == [9] and y.tolist() == [0]

    train = _cohort([1, 2], index_date=INDEX)
    X, _, _, _ = bl.assemble_matrix(
        train, bl.build_counts(train, small_omop, 365), bl.build_demographics(train, small_omop), spec
    )
    assert X.shape == (2, len(names))
    p1 = dict(zip(names, X.toarray()[0]))
    assert p1["cond:100"] == 1 and p1["cond:102"] == 1
    assert p1["drug:200"] == 2  # drugs keep counts, conditions are binary
    assert p1["gender:8507"] == 1


# --------------------------------------------------------------------------
# Prediction writer
# --------------------------------------------------------------------------
def test_write_predictions_schema_and_loadable(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pipeline import evaluation as ev
    from pipeline.baseline_predictions import write_predictions

    out = tmp_path / "test_predictions"
    path = write_predictions(
        out,
        person_ids=np.array([3, 1, 2]),
        index_dates=np.array(["2020-01-01", "2019-05-02", "2021-03-04"], dtype="datetime64[D]"),
        y_true=np.array([1, 0, 1]),
        y_prob=np.array([0.9, 0.2, 0.6]),
    )
    table = pq.read_table(path)
    schema = table.schema
    assert schema.names == [
        "subject_id",
        "prediction_time",
        "predicted_boolean_probability",
        "predicted_boolean_value",
        "boolean_value",
    ]
    assert pa.types.is_integer(schema.field("subject_id").type)
    assert pa.types.is_timestamp(schema.field("prediction_time").type)
    assert pa.types.is_floating(schema.field("predicted_boolean_probability").type)
    assert pa.types.is_boolean(schema.field("predicted_boolean_value").type)
    assert table.column("predicted_boolean_value").null_count == 3
    assert pa.types.is_boolean(schema.field("boolean_value").type)
    assert table.column("subject_id").to_pylist() == [3, 1, 2]
    assert table.column("prediction_time").to_pylist()[1] == dt.datetime(2019, 5, 2)

    y_true, y_prob = ev.load_predictions(out)
    assert y_true.tolist() == [1.0, 0.0, 1.0]
    assert y_prob.tolist() == [0.9, 0.2, 0.6]


# --------------------------------------------------------------------------
# Steps 11, 12, 13 end to end on a small random OMOP set
# --------------------------------------------------------------------------
def _random_omop(root: Path, n: int, seed: int = 0) -> list[tuple[int, int]]:
    """Write a random OMOP set; returns (person_id, label). Concept 1 is enriched in cases."""
    import numpy as np

    rng = np.random.default_rng(seed)
    labels = [(pid, int(pid % 3 == 0)) for pid in range(1, n + 1)]
    person, cond, drug, proc, visit = [], [], [], [], []
    vid = 0
    for pid, label in labels:
        person.append((pid, int(rng.choice([8507, 8532])), 8527, int(rng.integers(1930, 1960))))
        for _ in range(int(rng.integers(2, 6))):
            day = INDEX - dt.timedelta(days=int(rng.integers(1, 360)))
            concept = 1 if (label and rng.random() < 0.7) else int(rng.integers(2, 8))
            cond.append((pid, concept, day))
            drug.append((pid, int(rng.integers(20, 24)), day))
            proc.append((pid, int(rng.integers(30, 33)), day))
            vid += 1
            visit.append((pid, vid, day))
    _write_omop(root, person, cond, drug, proc, visit)
    return labels


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    """Run steps 11 and 12 once on 60 synthetic patients; shared by the smoke tests."""
    from conftest import load_step

    root = tmp_path_factory.mktemp("baseline_run")
    labels = _random_omop(root / "omop", 60)
    split_of = {pid: ("test" if pid > 48 else "val" if pid > 36 else "train") for pid, _ in labels}
    pl.DataFrame(
        {
            "person_id": [p for p, _ in labels],
            "index_date": [INDEX] * len(labels),
            "label": [lab for _, lab in labels],
            "split": [split_of[p] for p, _ in labels],
        },
        schema={"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64, "split": pl.Utf8},
    ).write_parquet(root / "splits.parquet")

    features = root / "features"
    baselines = root / "baselines"
    rc11 = load_step("11_prepare_baseline_features.py").main(
        [
            "--data-dir", str(root / "omop"),
            "--splits-file", str(root / "splits.parquet"),
            "--output-dir", str(features),
            "--report-dir", str(root / "reports" / "11"),
            "--min-prevalence", "0.05",
        ]
    )  # fmt: skip
    rc12 = load_step("12_train_baselines.py").main(
        [
            "--features-dir", str(features),
            "--output-dir", str(baselines),
            "--report-dir", str(root / "reports" / "12"),
            "--cv-folds", "2",
            "--xgb-iter", "2",
        ]
    )  # fmt: skip
    return {"root": root, "features": features, "baselines": baselines, "rc11": rc11, "rc12": rc12}


def test_step11_writes_feature_matrices(baseline_run):
    import json

    assert baseline_run["rc11"] == 0
    features = baseline_run["features"]
    names = json.loads((features / "feature_names.json").read_text())
    for split, n_rows in (("train", 36), ("val", 12), ("test", 12)):
        X, _, pids, _ = bl.load_split(features, split)
        assert X.shape == (n_rows, len(names))
        assert len(pids) == n_rows
    status = json.loads((baseline_run["root"] / "reports" / "11" / "status.json").read_text())
    assert status["metrics"]["splits_file"].endswith("splits.parquet")


def test_step12_writes_model_outputs(baseline_run):
    from pipeline import evaluation as ev

    assert baseline_run["rc12"] == 0
    for model in ("lr", "xgboost"):
        d = baseline_run["baselines"] / model
        for name in ("model.joblib", "best_params.json"):
            assert (d / name).is_file()
        for split, n_rows in (("test_predictions", 12), ("validation_predictions", 12)):
            y_true, _ = ev.load_predictions(d / split)
            assert len(y_true) == n_rows


def test_step13_writes_comparison(baseline_run, step):
    root = baseline_run["root"]
    figures = root / "figures"
    rc = step("13_compare_models.py").main(
        [
            "--splits-file", str(root / "splits.parquet"),
            "--baselines-dir", str(baseline_run["baselines"]),
            "--linear-probe-dir", str(root / "missing_linear_probe"),
            "--output-dir", str(root / "comparison"),
            "--cehrbert-metrics", str(root / "missing_metrics.json"),
            "--cehrbert-predictions", str(root / "missing_predictions"),
            "--figures-dir", str(figures),
            "--report-dir", str(root / "reports" / "13"),
            "--n-bootstrap", "20",
        ]
    )  # fmt: skip
    assert rc == 0
    baselines = baseline_run["baselines"]
    for path in (
        root / "comparison" / "comparison.json",
        root / "comparison" / "comparison.md",
        baselines / "lr" / "metrics.json",
        baselines / "xgboost" / "metrics.json",
        figures / "compare_roc.png",
        figures / "compare_pr.png",
    ):
        assert path.is_file(), path


def test_step13_evaluate_command_uses_contract_arguments(step, tmp_path):
    mod = step("13_compare_models.py")
    cmd = mod.evaluate_command(tmp_path / "lr")
    args = dict(zip(cmd[2::2], cmd[3::2]))
    assert args["--n-bootstrap"] == "2000"
    assert args["--seed"] == "42"
    assert args["--cal-bins"] == "10"
    assert args["--predictions"] == str(tmp_path / "lr" / "test_predictions")
    assert args["--val-predictions"] == str(tmp_path / "lr" / "validation_predictions")
