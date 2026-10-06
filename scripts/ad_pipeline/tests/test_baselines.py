"""Tests for the classical-ML baseline code merged from add-baseline-comparison."""

from __future__ import annotations

import datetime as dt
import logging
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


def _write_cohort(folder: Path, df: pl.DataFrame) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    df.write_parquet(folder / "cohort.parquet")
    return folder


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
    # Legacy cohort folders disagree with the split file; the file must win.
    finetune_dir = _write_cohort(tmp_path / "finetune", _cohort([1, 2, 3, 4, 5, 6, 9]))
    test_dir = _write_cohort(tmp_path / "test", _cohort([10]))

    splits = bl.recover_splits(
        finetune_dir, test_dir, tmp_path / "no_prepared", splits_file=splits_file
    )

    assert splits.val_source == "canonical_splits_file"
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
        bl.recover_splits(tmp_path, tmp_path, tmp_path, splits_file=splits_file)


def test_recover_splits_falls_back_with_warning(tmp_path, caplog):
    finetune_dir = _write_cohort(tmp_path / "finetune", _cohort(range(1, 201)))
    test_dir = _write_cohort(tmp_path / "test", _cohort([300, 301]))
    with caplog.at_level(logging.WARNING, logger="pipeline.baselines"):
        splits = bl.recover_splits(
            finetune_dir, test_dir, tmp_path / "no_prepared", splits_file=tmp_path / "missing.parquet"
        )
    assert splits.val_source == "deterministic_fallback"
    assert "missing.parquet" in caplog.text
    assert splits.train.height + splits.val.height == 200
    assert 0 < splits.val.height < 60
    assert splits.test["person_id"].to_list() == [300, 301]


def test_run_label_formats():
    assert paths.run_label(365, 512) == "w_365_ctx512"
    assert paths.run_label(-1, 4096) == "w_all_ctx4096"
    assert paths.run_label(None, 512) == "w_all_ctx512"


def test_default_run_label_matches_trained_model():
    assert paths.run_label(paths.DEFAULT_WINDOW_DAYS, paths.DEFAULT_CTX) == "w_365_ctx512"
