"""Tests for pipeline.omop_validation."""

from __future__ import annotations

import polars as pl

from pipeline import omop_validation as ov
from pipeline.reporting import StepReport


def _check(report: StepReport, name: str):
    for c in report.checks:
        if c.name == name:
            return c
    return None


def test_valid_dataset_passes(omop_dir):
    r = StepReport("t")
    ov.add_dataset_checks(r, omop_dir, require_concept=True)
    assert r.status == "pass"
    assert r.metrics["person_rows"] == 6
    assert r.metrics["condition_occurrence_rows"] == 5


def test_missing_required_table_fails(omop_dir):
    # Remove the concept table entirely.
    for f in (omop_dir / "concept").glob("*.parquet"):
        f.unlink()
    (omop_dir / "concept").rmdir()
    r = StepReport("t")
    ov.add_dataset_checks(r, omop_dir, require_concept=True)
    assert r.status == "fail"
    assert _check(r, "table 'concept' present").ok is False


def test_concept_optional_when_disabled(omop_dir):
    for f in (omop_dir / "concept").glob("*.parquet"):
        f.unlink()
    (omop_dir / "concept").rmdir()
    r = StepReport("t")
    ov.add_dataset_checks(r, omop_dir, require_concept=False)
    # concept is no longer checked at error level -> dataset still passes
    assert r.status == "pass"


def test_missing_required_column_fails(omop_dir, tmp_path):
    # Rewrite person without person_id.
    pdir = omop_dir / "person"
    for f in pdir.glob("*.parquet"):
        f.unlink()
    pl.DataFrame({"wrong_id": [1, 2, 3]}).write_parquet(pdir / "data.parquet")
    r = StepReport("t")
    ov.add_dataset_checks(r, omop_dir)
    assert r.status == "fail"
    assert _check(r, "table 'person' has required columns").ok is False


def test_missing_recommended_table_is_warning(omop_dir):
    for f in (omop_dir / "measurement").glob("*.parquet"):
        f.unlink()
    (omop_dir / "measurement").rmdir()
    r = StepReport("t")
    ov.add_dataset_checks(r, omop_dir)
    assert r.status == "warn"  # only a recommended table missing


def test_table_glob_finds_flat_file(tmp_path):
    pl.DataFrame({"person_id": [1]}).write_parquet(tmp_path / "person.parquet")
    glob = ov.table_glob(tmp_path, "person")
    assert glob is not None
    assert ov.count_rows(glob) == 1
    assert "person_id" in ov.read_columns(glob)
