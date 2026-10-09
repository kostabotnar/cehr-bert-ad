"""Tests for step 02: the CEHR-BERT cohort folders derived from splits.parquet."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
from conftest import write_spark_sequences

PERSONS = list(range(1, 13))
SPLIT = {p: "test" if p in (2, 5, 9) else "val" if p in (3, 7) else "train" for p in PERSONS}


def _splits(tmp_path: Path, persons=PERSONS) -> Path:
    path = tmp_path / "cohort" / "splits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "person_id": persons,
            "index_date": [dt.date(2020, 1, 1)] * len(persons),
            "label": [p % 2 for p in persons],
            "split": [SPLIT.get(p, "train") for p in persons],
        }
    ).write_parquet(path)
    return path


def _omop(tmp_path: Path, birth_years: dict[int, int | None] | None = None) -> Path:
    """OMOP dir whose person table holds persons 1..12 (born 1950 unless ``birth_years`` says otherwise)."""
    years = {p: 1950 for p in PERSONS} | (birth_years or {})
    path = tmp_path / "omop"
    (path / "person").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"person_id": list(years), "year_of_birth": list(years.values())},
        schema={"person_id": pl.Int64, "year_of_birth": pl.Int64},
    ).write_parquet(path / "person" / "data.parquet")
    return path


def _run(step, tmp_path, splits, seq, omop=None):
    mod = step("cehrbert_model/02_prepare_cohort_folders.py")
    rep = tmp_path / "rep"
    rc = mod.main(
        [
            "--splits-file", str(splits),
            "--data-dir", str(omop if omop is not None else _omop(tmp_path)),
            "--patient-sequence-dir", str(seq),
            "--finetune-dir", str(tmp_path / "cb" / "finetune"),
            "--test-dir", str(tmp_path / "cb" / "test"),
            "--report-dir", str(rep),
        ]
    )  # fmt: skip
    return rc, json.loads((rep / "status.json").read_text(encoding="utf-8"))


def test_writes_finetune_and_test_folders(step, tmp_path):
    seq = write_spark_sequences(tmp_path / "seq", [PERSONS[:6], PERSONS[6:]])
    rc, status = _run(step, tmp_path, _splits(tmp_path), seq)
    assert rc == 0 and status["status"] == "pass"
    ft = pl.read_parquet(tmp_path / "cb" / "finetune")
    te = pl.read_parquet(tmp_path / "cb" / "test")
    assert ft.columns == te.columns == ["person_id", "index_date", "label", "age_at_index"]
    assert (
        ft.schema
        == te.schema
        == {
            "person_id": pl.Int64,
            "index_date": pl.Date,
            "label": pl.Int64,
            "age_at_index": pl.Int64,
        }
    )
    assert set(ft["age_at_index"].to_list()) == {70} and set(te["age_at_index"].to_list()) == {70}
    assert ft["person_id"].to_list() == [p for p in PERSONS if SPLIT[p] != "test"]
    assert te["person_id"].to_list() == [2, 5, 9]
    assert [f.name for f in (tmp_path / "cb" / "finetune").iterdir()] == ["cohort.parquet"]
    assert status["metrics"]["finetune_rows"] == 9 and status["metrics"]["test_rows"] == 3


def test_replaces_stale_parquet_files(step, tmp_path):
    seq = write_spark_sequences(tmp_path / "seq", [PERSONS])
    stale = tmp_path / "cb" / "test" / "old.parquet"
    stale.parent.mkdir(parents=True)
    pl.DataFrame({"person_id": [99]}).write_parquet(stale)
    rc, _ = _run(step, tmp_path, _splits(tmp_path), seq)
    assert rc == 0 and not stale.exists()


def test_fails_for_persons_without_a_sequence_and_writes_nothing(step, tmp_path):
    # Person 4 has an empty sequence, person 6 none at all.
    seq = write_spark_sequences(tmp_path / "seq", [[p for p in PERSONS if p != 6]], n_visits={4: 0})
    rc, status = _run(step, tmp_path, _splits(tmp_path), seq)
    assert rc == 1
    assert status["metrics"]["cohort_without_sequence"] == 2
    gate = next(c for c in status["checks"] if c["name"].startswith("every cohort person"))
    assert not gate["ok"] and "4, 6" in gate["detail"]
    assert not (tmp_path / "cb").exists()


def test_failure_lists_at_most_twenty_person_ids(step, tmp_path):
    persons = list(range(1, 51))
    seq = write_spark_sequences(tmp_path / "seq", [[1, 2, 3]])
    rc, status = _run(step, tmp_path, _splits(tmp_path, persons), seq)
    assert rc == 1
    gate = next(c for c in status["checks"] if c["name"].startswith("every cohort person"))
    assert "47 without one; first 20: 4, 5" in gate["detail"] and gate["detail"].endswith("23")


def test_fails_without_splits_or_sequences(step, tmp_path):
    seq = write_spark_sequences(tmp_path / "seq", [PERSONS])
    rc, _ = _run(step, tmp_path, tmp_path / "missing.parquet", seq)
    assert rc == 1
    rc, _ = _run(step, tmp_path / "b", _splits(tmp_path / "b"), tmp_path / "no_seq")
    assert rc == 1


def test_age_at_index_is_index_year_minus_year_of_birth(step, tmp_path):
    seq = write_spark_sequences(tmp_path / "seq", [PERSONS])
    omop = _omop(tmp_path, {1: 2000, 2: 1931})
    rc, _ = _run(step, tmp_path, _splits(tmp_path), seq, omop)
    assert rc == 0
    ages = dict(pl.read_parquet(tmp_path / "cb" / "finetune").select("person_id", "age_at_index").iter_rows())
    ages |= dict(pl.read_parquet(tmp_path / "cb" / "test").select("person_id", "age_at_index").iter_rows())
    assert ages[1] == 20 and ages[2] == 89 and ages[3] == 70


def test_fails_for_persons_without_year_of_birth_and_writes_nothing(step, tmp_path):
    seq = write_spark_sequences(tmp_path / "seq", [PERSONS])
    omop = _omop(tmp_path, {4: None, 8: None})
    rc, status = _run(step, tmp_path, _splits(tmp_path), seq, omop)
    assert rc == 1 and status["metrics"]["cohort_without_age"] == 2
    gate = next(c for c in status["checks"] if c["name"].startswith("every cohort row has an age"))
    assert not gate["ok"] and "2 without one; first 2: 4, 8" in gate["detail"]
    assert not (tmp_path / "cb").exists()


def test_age_failure_lists_at_most_twenty_person_ids(step, tmp_path):
    persons = list(range(1, 51))
    seq = write_spark_sequences(tmp_path / "seq", [persons])
    rc, status = _run(step, tmp_path, _splits(tmp_path, persons), seq)  # person table only covers 1..12
    assert rc == 1
    gate = next(c for c in status["checks"] if c["name"].startswith("every cohort row has an age"))
    assert "38 without one; first 20: 13, 14" in gate["detail"] and gate["detail"].endswith("32")


def test_fails_without_a_person_table(step, tmp_path):
    seq = write_spark_sequences(tmp_path / "seq", [PERSONS])
    (tmp_path / "empty").mkdir()
    rc, _ = _run(step, tmp_path, _splits(tmp_path), seq, tmp_path / "empty")
    assert rc == 1 and not (tmp_path / "cb").exists()
