"""Tests for cohort/01_build_cohort.py main() on the synthetic OMOP fixture."""

from __future__ import annotations

import json


def test_build_cohort_writes_splits(step, omop_dir, tmp_path, monkeypatch):
    mod = step("cohort/01_build_cohort.py")
    monkeypatch.setattr(mod, "MIN_CONCEPT_PATIENTS", 1)
    splits = tmp_path / "cohort" / "splits.parquet"
    rep = tmp_path / "rep06"
    rc = mod.main(
        [
            "--data-dir",
            str(omop_dir),
            "--splits-output",
            str(splits),
            "--report-dir",
            str(rep),
        ]
    )
    assert rc == 0
    payload = json.loads((rep / "status.json").read_text(encoding="utf-8"))
    # Full cohort is 2 cases + 3 controls; the holdout split keeps both non-empty.
    assert payload["metrics"]["cases"] == 2
    assert payload["metrics"]["controls"] == 3
    assert payload["metrics"]["finetune_size"] + payload["metrics"]["test_size"] == 5
    assert payload["metrics"]["min_concept_patients"] == 1
    assert "excluded_by_event_rule" in payload["metrics"]

    import polars as pl

    table = pl.read_parquet(splits)
    assert table.columns == ["person_id", "index_date", "label", "split"]
    assert set(table["split"]) <= {"train", "val", "test"} and table.height == 5
    assert list(splits.parent.iterdir()) == [splits]
