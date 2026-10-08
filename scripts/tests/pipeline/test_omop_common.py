"""Tests for pipeline/omop_common.py (helpers shared by step 6 and step 18)."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import omop_common as oc  # noqa: E402


def test_is_ad_source_value_matches_icd_prefixes():
    df = pl.DataFrame(
        {"condition_source_value": ["ICD-10-CM:G30.9", "ICD-9-CM:331.0", "ICD-9-CM:3310", "ICD-10-CM:I10"]}
    )
    assert df.filter(oc.is_ad_source_value()).height == 2


def test_birth_date_and_age_expressions():
    df = pl.DataFrame(
        {
            "year_of_birth": [1950, None],
            "month_of_birth": [None, 1],
            "day_of_birth": [31, 1],
            "index_date": [dt.date(2000, 7, 1)] * 2,
        }
    ).with_columns(oc.birth_date_expr().alias("birth_date"))
    assert df["birth_date"][0] == dt.date(1950, 7, 28)
    assert df["birth_date"][1] is None
    age = df.with_columns(oc.age_years_expr().alias("age"))["age"][0]
    assert age == pytest.approx(49.93, abs=0.01)


def test_scan_table_missing_table_and_columns(tmp_path):
    assert oc.scan_table(tmp_path, "person", ["person_id"]) is None
    (tmp_path / "visit_occurrence").mkdir()
    pl.DataFrame({"person_id": [1], "visit_start_date": [dt.datetime(2020, 1, 2, 3)]}).write_parquet(
        tmp_path / "visit_occurrence" / "a.parquet"
    )
    out = oc.scan_table(tmp_path, "visit_occurrence", ["person_id", "visit_start_date", "visit_end_date"]).collect()
    assert out["visit_start_date"].dtype == pl.Date
    assert out["visit_end_date"].null_count() == 1


def test_fit_stratum_weights_moves_expected_counts_to_target():
    # Two controls, each with one candidate in year 2000 and one in 2001; cases want 3:1.
    cand = pl.DataFrame({"person_id": [1, 1, 2, 2], "band": [0] * 4, "year": [2000, 2001, 2000, 2001]})
    strata = pl.DataFrame({"band": [0, 0], "year": [2000, 2001], "n_case": [3, 1]})
    w = oc.fit_stratum_weights(cand, strata).sort("year")
    assert w["w"][0] / w["w"][1] == pytest.approx(3.0, rel=0.05)
