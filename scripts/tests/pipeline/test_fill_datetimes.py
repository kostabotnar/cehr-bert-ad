"""Tests for step 01b (fill null datetimes) and the step 02 visit datetime preflight."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from pipeline import fill_datetimes as fd
from pipeline import omop_validation as ov
from pipeline.reporting import StepReport

D = dt.date
DT = dt.datetime
TS = pl.Datetime("us")
UTC = dt.timezone.utc


UTC_TS = pl.Datetime("us", "UTC")


def _utc(*args):
    return DT(*args, tzinfo=UTC)


def _visits(ids, dates, datetimes):
    return pl.DataFrame(
        {
            "visit_occurrence_id": ids,
            "visit_start_date": pl.Series(dates, dtype=pl.Date),
            "visit_start_datetime": pl.Series(datetimes, dtype=TS),
            "visit_end_date": pl.Series(dates, dtype=pl.Date),
            "visit_end_datetime": pl.Series([None] * len(ids), dtype=TS),
            "other": [f"x{i}" for i in ids],
        }
    )


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "omop"
    (root / "visit_occurrence").mkdir(parents=True)
    # two files: one all null, one mixed null and non-null
    _visits([1, 2], [D(2020, 1, 1), D(2020, 1, 2)], [None, None]).write_parquet(root / "visit_occurrence" / "a.parquet")
    _visits([3, 4], [D(2020, 2, 1), D(2020, 2, 2)], [DT(2020, 2, 1, 13, 30), None]).write_parquet(
        root / "visit_occurrence" / "b.parquet"
    )
    # single-file table in the flat layout
    pl.DataFrame(
        {
            "person_id": [1, 2, 3],
            "year_of_birth": [1950, 1960, 1970],
            "month_of_birth": pl.Series([None, 5, None], dtype=pl.Int64),
            "day_of_birth": pl.Series([None, 17, None], dtype=pl.Int64),
            "birth_datetime": pl.Series([None, None, DT(1970, 3, 3, 8)], dtype=TS),
        }
    ).write_parquet(root / "person.parquet")
    # datetime column stored with the Null type (all null)
    pl.DataFrame({"measurement_id": [1, 2], "measurement_date": [D(2021, 5, 5), D(2021, 5, 6)]}).with_columns(
        measurement_datetime=pl.lit(None)
    ).write_parquet(root / "measurement.parquet")
    return root


def _read(root: Path, table: str) -> pl.DataFrame:
    if (root / table).is_dir():
        return pl.read_parquet(root / table / "*.parquet")
    return pl.read_parquet(root / f"{table}.parquet")


def test_fills_with_date_midnight_and_keeps_existing(data_dir):
    before = _read(data_dir, "visit_occurrence").sort("visit_occurrence_id")
    fd.process_table(data_dir, "visit_occurrence", dry_run=False)
    after = _read(data_dir, "visit_occurrence").sort("visit_occurrence_id")
    assert after.height == before.height
    assert after["other"].to_list() == before["other"].to_list()
    assert after.schema == before.schema | {c: UTC_TS for c in ("visit_start_datetime", "visit_end_datetime")}
    assert after["visit_start_datetime"].to_list() == [
        _utc(2020, 1, 1),
        _utc(2020, 1, 2),
        _utc(2020, 2, 1, 13, 30),
        _utc(2020, 2, 2),
    ]
    assert after["visit_end_datetime"].to_list() == [
        _utc(2020, 1, 1),
        _utc(2020, 1, 2),
        _utc(2020, 2, 1),
        _utc(2020, 2, 2),
    ]
    assert sorted(p.name for p in (data_dir / "visit_occurrence").iterdir()) == ["a.parquet", "b.parquet"]


def test_birth_datetime_rule(data_dir):
    fd.process_table(data_dir, "person", dry_run=False)
    p = _read(data_dir, "person").sort("person_id")
    assert p["birth_datetime"].to_list() == [_utc(1950, 1, 1), _utc(1960, 5, 17), _utc(1970, 3, 3, 8)]
    assert p["year_of_birth"].to_list() == [1950, 1960, 1970]


def test_null_typed_column_becomes_datetime(data_dir):
    res = fd.process_table(data_dir, "measurement", dry_run=False)
    m = _read(data_dir, "measurement")
    assert m.schema["measurement_datetime"] == UTC_TS
    assert m["measurement_datetime"].to_list() == [_utc(2021, 5, 5), _utc(2021, 5, 6)]
    assert res.columns["measurement_datetime"].filled == 2


def test_stats_and_idempotent(data_dir):
    first = fd.process_table(data_dir, "visit_occurrence", dry_run=False)
    s = first.columns["visit_start_datetime"]
    assert (s.rows, s.nulls_before, s.filled, s.nulls_after) == (4, 3, 3, 0)
    mtimes = {p: p.stat().st_mtime_ns for p in (data_dir / "visit_occurrence").iterdir()}
    snapshot = _read(data_dir, "visit_occurrence").sort("visit_occurrence_id")
    second = fd.process_table(data_dir, "visit_occurrence", dry_run=False)
    assert all(c.filled == 0 and c.nulls_before == 0 for c in second.columns.values())
    assert second.files_rewritten == 0
    assert {p: p.stat().st_mtime_ns for p in (data_dir / "visit_occurrence").iterdir()} == mtimes
    assert _read(data_dir, "visit_occurrence").sort("visit_occurrence_id").equals(snapshot)


def _is_utc_adjusted(path: Path, column: str) -> bool:
    schema = pq.ParquetFile(path).schema
    logical = json.loads(schema.column(schema.names.index(column)).logical_type.to_json())
    return logical.get("Type") == "Timestamp" and logical["isAdjustedToUTC"]


def test_written_columns_are_utc_adjusted(data_dir):
    for table in ("visit_occurrence", "person", "measurement"):
        fd.process_table(data_dir, table, dry_run=False)
    for rel, cols in {
        "visit_occurrence/a.parquet": ["visit_start_datetime", "visit_end_datetime"],
        "visit_occurrence/b.parquet": ["visit_start_datetime", "visit_end_datetime"],
        "person.parquet": ["birth_datetime"],
        "measurement.parquet": ["measurement_datetime"],
    }.items():
        for c in cols:
            assert _is_utc_adjusted(data_dir / rel, c), (rel, c)


def test_multiple_batches_keep_arrow_schema(data_dir, monkeypatch):
    path = data_dir / "visit_occurrence" / "a.parquet"
    before = pq.ParquetFile(path).schema_arrow
    monkeypatch.setattr(fd, "BATCH_ROWS", 1)
    fd.process_table(data_dir, "visit_occurrence", dry_run=False)
    after = pq.ParquetFile(path).schema_arrow
    assert pq.ParquetFile(path).metadata.num_row_groups == 2
    assert [f.name for f in after] == [f.name for f in before]
    assert after.field("other").type == before.field("other").type
    assert after.field("visit_occurrence_id").type == before.field("visit_occurrence_id").type


def test_values_are_midnight_utc_of_the_date(data_dir):
    fd.process_table(data_dir, "visit_occurrence", dry_run=False)
    v = _read(data_dir, "visit_occurrence").filter(pl.col("visit_occurrence_id") != 3)
    assert (
        v["visit_start_datetime"].dt.epoch("us").to_list()
        == v["visit_start_date"].cast(pl.Datetime("us")).dt.epoch("us").to_list()
    )


def test_converts_filled_naive_columns_then_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "visit_occurrence").mkdir()
    path = tmp_path / "visit_occurrence" / "v.parquet"
    full = pl.DataFrame(
        {
            "visit_start_date": [D(2020, 1, 1), D(2020, 1, 2)],
            "visit_start_datetime": pl.Series([DT(2020, 1, 1), DT(2020, 1, 2, 7, 5)], dtype=TS),
        }
    )
    full.write_parquet(path)
    assert not _is_utc_adjusted(path, "visit_start_datetime")
    first = fd.process_table(tmp_path, "visit_occurrence", dry_run=False)
    stats = first.columns["visit_start_datetime"]
    assert first.files_rewritten == 1 and stats.tz_converted and stats.filled == 0
    assert _is_utc_adjusted(path, "visit_start_datetime")
    assert pl.read_parquet(path)["visit_start_datetime"].to_list() == [_utc(2020, 1, 1), _utc(2020, 1, 2, 7, 5)]
    mtime = path.stat().st_mtime_ns
    second = fd.process_table(tmp_path, "visit_occurrence", dry_run=False)
    assert second.files_rewritten == 0 and not second.columns["visit_start_datetime"].tz_converted
    assert path.stat().st_mtime_ns == mtime


def test_dry_run_reports_conversion_without_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "visit_occurrence").mkdir()
    path = tmp_path / "visit_occurrence" / "v.parquet"
    pl.DataFrame(
        {"visit_start_date": [D(2020, 1, 1)], "visit_start_datetime": pl.Series([DT(2020, 1, 1)], dtype=TS)}
    ).write_parquet(path)
    before = path.read_bytes()
    res = fd.process_table(tmp_path, "visit_occurrence", dry_run=True)
    assert res.columns["visit_start_datetime"].tz_converted and res.files_rewritten == 0
    assert path.read_bytes() == before


def _spark_or_skip():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None:
        pytest.skip("Java is not available")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[1]")
        .appName("test_fill_datetimes")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


def test_spark_reads_filled_columns_as_timestamps(data_dir):
    spark = _spark_or_skip()
    try:
        from pyspark.sql.types import TimestampType

        fd.process_table(data_dir, "visit_occurrence", dry_run=False)
        df = spark.read.parquet(str(data_dir / "visit_occurrence" / "b.parquet"))
        assert isinstance(df.schema["visit_start_datetime"].dataType, TimestampType)
        assert isinstance(df.schema["visit_end_datetime"].dataType, TimestampType)
        rows = {r["visit_occurrence_id"]: r for r in df.collect()}
        assert rows[3]["visit_start_datetime"].timestamp() == _utc(2020, 2, 1, 13, 30).timestamp()
        assert rows[4]["visit_start_datetime"].timestamp() == _utc(2020, 2, 2).timestamp()
        assert rows[4]["visit_end_datetime"].timestamp() == _utc(2020, 2, 2).timestamp()
        epoch = df.selectExpr("visit_occurrence_id", "unix_timestamp(visit_start_datetime) as s").collect()
        assert {r["visit_occurrence_id"]: r["s"] for r in epoch}[4] == int(_utc(2020, 2, 2).timestamp())
    finally:
        spark.stop()


def test_dry_run_writes_nothing(data_dir):
    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    snap = {p: p.read_bytes() for p in files}
    res = fd.process_table(data_dir, "visit_occurrence", dry_run=True)
    assert res.columns["visit_start_datetime"].nulls_before == 3
    assert res.columns["visit_start_datetime"].filled == 3
    assert sorted(p for p in data_dir.rglob("*") if p.is_file()) == files
    assert {p: p.read_bytes() for p in files} == snap


def test_no_tmp_left_and_snappy(data_dir):
    fd.process_table(data_dir, "visit_occurrence", dry_run=False)
    assert not list(data_dir.rglob(f"*{fd.TMP_SUFFIX}"))
    md = pq.ParquetFile(data_dir / "visit_occurrence" / "a.parquet").metadata
    assert md.row_group(0).column(0).compression == "SNAPPY"


def test_unsupported_dtype_reported_not_modified(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "t").mkdir()
    pl.DataFrame({"x_date": [D(2020, 1, 1)], "x_datetime": ["2020-01-01"]}).write_parquet(tmp_path / "t" / "x.parquet")
    res = fd.process_table(tmp_path, "t", dry_run=False)
    assert res.skipped and res.skipped[0][0] == "x_datetime"
    assert pl.read_parquet(tmp_path / "t" / "x.parquet")["x_datetime"].to_list() == ["2020-01-01"]


def test_guard_refuses_derived(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "data" / "derived" / "omop").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    assert fd.guard_data_dir(home / "data" / "derived" / "omop") is not None
    link = tmp_path / "omop_data"
    link.symlink_to(home / "data" / "derived" / "omop")
    assert fd.guard_data_dir(link) is not None
    assert fd.guard_data_dir(tmp_path) is None


def test_step_dry_run_then_real(step, data_dir, tmp_path):
    mod = step("01b_fill_datetimes.py")
    rd = str(tmp_path / "rep")
    assert mod.main(["--data-dir", str(data_dir), "--dry-run", "--report-dir", rd]) == 0
    assert _read(data_dir, "visit_occurrence")["visit_start_datetime"].null_count() == 3
    assert mod.main(["--data-dir", str(data_dir), "--report-dir", rd]) == 0
    assert _read(data_dir, "visit_occurrence")["visit_start_datetime"].null_count() == 0


def test_step_refuses_derived(step, tmp_path, monkeypatch):
    home = tmp_path / "home"
    src = home / "data" / "derived" / "omop"
    (src / "person").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    pl.DataFrame({"person_id": [1]}).write_parquet(src / "person" / "p.parquet")
    mod = step("01b_fill_datetimes.py")
    assert mod.main(["--data-dir", str(src), "--report-dir", str(tmp_path / "rep")]) != 0


def test_preflight_fails_on_all_null_then_passes_after_fill(data_dir):
    r = StepReport("t")
    ov.add_visit_datetime_check(r, data_dir)
    assert r.status == "pass"  # file b has a non-null value
    fd.process_table(data_dir, "visit_occurrence", dry_run=False)
    r = StepReport("t")
    ov.add_visit_datetime_check(r, data_dir)
    assert r.status == "pass"


def test_preflight_all_null_and_missing_column(tmp_path):
    (tmp_path / "visit_occurrence").mkdir()
    _visits([1], [D(2020, 1, 1)], [None]).write_parquet(tmp_path / "visit_occurrence" / "v.parquet")
    r = StepReport("t")
    ov.add_visit_datetime_check(r, tmp_path)
    assert r.status == "fail"
    assert "01b" in r.checks[-1].detail
    pl.DataFrame({"visit_start_date": [D(2020, 1, 1)]}).write_parquet(tmp_path / "visit_occurrence" / "v.parquet")
    r = StepReport("t")
    ov.add_visit_datetime_check(r, tmp_path)
    assert r.status == "pass" and not r.checks


def test_step02_fails_on_all_null_visit_datetimes(step, omop_dir, tmp_path):
    vo = omop_dir / "visit_occurrence" / "data.parquet"
    df = pl.read_parquet(vo).with_columns(visit_start_datetime=pl.lit(None, dtype=TS))
    df.write_parquet(vo)
    mod = step("02_prepare_pretraining_data.py")
    rc = mod.main(
        [
            "--input-folder",
            str(omop_dir),
            "--staging-dir",
            str(tmp_path / "stage"),
            "--skip-launch",
            "--report-dir",
            str(tmp_path / "rep"),
        ]
    )
    assert rc != 0
