"""Tests for data_qc/02_history_length_stats.py: index_date minus first visit, overall and by label."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl


def _write_splits(path, rows):
    pl.DataFrame(
        {
            "person_id": [r[0] for r in rows],
            "index_date": [r[1] for r in rows],
            "label": [r[2] for r in rows],
            "split": ["train"] * len(rows),
        }
    ).write_parquet(path)


def test_describe_handles_empty_and_filled_arrays(step):
    mod = step("data_qc/02_history_length_stats.py")
    empty = mod._describe(np.asarray([]))
    assert empty["count"] == 0 and empty["mean"] is None and empty["p50"] is None
    full = mod._describe(np.arange(1, 101))
    assert full["count"] == 100 and full["max"] == 100.0 and full["p50"] == 50.5


def test_history_days_uses_the_first_visit(step, omop_dir):
    mod = step("data_qc/02_history_length_stats.py")
    cohort = pl.DataFrame(
        {"person_id": [1, 2, 99], "index_date": [dt.date(2019, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 1)]}
    ).with_columns(label=pl.Series([1, 0, 0]))
    out = mod._history_days(cohort, omop_dir).sort("person_id")
    # p1 first visit 2010-01-01, p2 first visit 2011-01-01; p99 has no visits and is dropped.
    assert out["person_id"].to_list() == [1, 2]
    assert out["history_days"].to_list() == [
        (dt.date(2019, 1, 1) - dt.date(2010, 1, 1)).days,
        (dt.date(2020, 1, 1) - dt.date(2011, 1, 1)).days,
    ]


def test_main_writes_stats_split_by_label(step, omop_dir, tmp_path):
    mod = step("data_qc/02_history_length_stats.py")
    splits = tmp_path / "splits.parquet"
    _write_splits(splits, [(1, dt.date(2019, 1, 1), 1), (2, dt.date(2020, 1, 1), 1), (3, dt.date(2020, 1, 1), 0)])
    results, report = tmp_path / "qc", tmp_path / "rep"
    rc = mod.main(
        [
            "--data-dir",
            str(omop_dir),
            "--splits-file",
            str(splits),
            "--results-dir",
            str(results),
            "--report-dir",
            str(report),
        ]
    )
    assert rc == 0
    payload = json.loads((results / "history_length_stats.json").read_text(encoding="utf-8"))
    assert payload["history_days"]["overall"]["count"] == 3
    assert payload["history_days"]["cases"]["count"] == 2
    assert payload["history_days"]["controls"]["count"] == 1
    assert (results / "figures" / "history_length_hist.png").is_file()
    assert json.loads((report / "status.json").read_text(encoding="utf-8"))["status"] == "pass"


def test_main_fails_without_a_cohort(step, omop_dir, tmp_path):
    mod = step("data_qc/02_history_length_stats.py")
    rc = mod.main(
        [
            "--data-dir",
            str(omop_dir),
            "--splits-file",
            str(tmp_path / "missing.parquet"),
            "--results-dir",
            str(tmp_path / "qc"),
            "--report-dir",
            str(tmp_path / "rep"),
        ]
    )
    assert rc == 1
