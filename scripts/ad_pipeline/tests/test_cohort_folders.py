"""Tests for step 02b: the CEHR-BERT cohort folders derived from splits.parquet."""

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


def _run(step, tmp_path, splits, seq):
    mod = step("02b_prepare_cohort_folders.py")
    rep = tmp_path / "rep"
    rc = mod.main(
        [
            "--splits-file", str(splits),
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
    assert ft.columns == te.columns == ["person_id", "index_date", "label"]
    assert ft.schema == te.schema == {"person_id": pl.Int64, "index_date": pl.Date, "label": pl.Int64}
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
