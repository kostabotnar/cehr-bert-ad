"""Tests for the canonical train/val/test split written by step 6."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest


@pytest.fixture
def mod(step):
    return step("06_prepare_finetuning_data.py")


@pytest.fixture
def cohort() -> pl.DataFrame:
    n = 2000
    return pl.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "index_date": [dt.date(2020, 1, 1)] * n,
            "label": [int(i % 5 == 0) for i in range(n)],
        }
    )


def _splits(mod, cohort, val_seed=43):
    finetune, test = mod.split_finetune_test(cohort, 0.1, 42)
    train, val = mod.split_train_val(finetune, 0.1, val_seed)
    return train, val, test, mod.build_splits_table(train, val, test)


def test_splits_disjoint_and_cover_cohort(mod, cohort):
    train, val, test, table = _splits(mod, cohort)
    assert table.columns == ["person_id", "index_date", "label", "split"]
    assert table.schema["person_id"] == pl.Int64 and table.schema["label"] == pl.Int64
    assert table.schema["index_date"] == pl.Date and table.schema["split"] == pl.String
    assert table["person_id"].n_unique() == cohort.height
    assert table["person_id"].to_list() == cohort["person_id"].to_list()
    assert set(table["split"].unique()) == {"train", "val", "test"}
    assert train.height + val.height + test.height == cohort.height
    assert 100 < val.height < 300


def test_test_split_matches_existing_test_output(mod, cohort):
    _, test = mod.split_finetune_test(cohort, 0.1, 42)
    _, _, _, table = _splits(mod, cohort)
    assert table.filter(pl.col("split") == "test").drop("split").equals(test)


def test_deterministic(mod, cohort):
    assert _splits(mod, cohort)[3].equals(_splits(mod, cohort)[3])


def test_val_seed_changes_val_not_test(mod, cohort):
    _, val_a, test_a, _ = _splits(mod, cohort, 43)
    _, val_b, test_b, _ = _splits(mod, cohort, 44)
    assert not val_a.equals(val_b)
    assert test_a.equals(test_b)


def test_main_writes_splits_and_keeps_outputs(mod, omop_dir, tmp_path):
    def run(tag):
        out = {k: tmp_path / tag / f"{k}.parquet" for k in ("ft", "te", "sp")}
        rc = mod.main(
            [
                "--data-dir", str(omop_dir),
                "--finetune-output", str(out["ft"]),
                "--test-output", str(out["te"]),
                "--splits-output", str(out["sp"]),
                "--skip-tokenized-check",
                "--report-dir", str(tmp_path / tag / "rep"),
            ]
        )  # fmt: skip
        assert rc == 0
        return {k: pl.read_parquet(v) for k, v in out.items()}

    a = run("a")
    b = run("b")
    assert a["sp"].equals(b["sp"])
    assert a["sp"].filter(pl.col("split") == "test").drop("split").equals(a["te"])
    assert a["sp"].height == 5
    assert set(a["sp"].filter(pl.col("split") != "test")["person_id"]) == set(a["ft"]["person_id"])
