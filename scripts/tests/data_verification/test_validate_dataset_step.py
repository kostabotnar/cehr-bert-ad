"""Tests for data_verification/01_validate_dataset.py main()."""

from __future__ import annotations

import json


def _status(report_dir):
    return json.loads((report_dir / "status.json").read_text(encoding="utf-8"))["status"]


def test_validate_dataset_validate_dataset_passes(step, omop_dir, tmp_path):
    mod = step("data_verification/01_validate_dataset.py")
    rep = tmp_path / "rep01"
    rc = mod.main(["--data-dir", str(omop_dir), "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) == "pass"


def test_validate_dataset_missing_table_fails(step, omop_dir, tmp_path):
    for f in (omop_dir / "person").glob("*.parquet"):
        f.unlink()
    (omop_dir / "person").rmdir()
    mod = step("data_verification/01_validate_dataset.py")
    rep = tmp_path / "rep01b"
    rc = mod.main(["--data-dir", str(omop_dir), "--report-dir", str(rep)])
    assert rc == 1
    assert _status(rep) == "fail"


def test_validate_dataset_link_identity_validates_in_place(step, omop_dir, tmp_path, monkeypatch):
    # --link pointing at a folder that is already data/omop_data must not error;
    # it should validate the folder in place.
    mod = step("data_verification/01_validate_dataset.py")
    monkeypatch.setattr(mod.paths, "OMOP_DIR", omop_dir)
    rep = tmp_path / "rep01_link"
    rc = mod.main(["--link", str(omop_dir), "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) == "pass"
    payload = json.loads((rep / "status.json").read_text(encoding="utf-8"))
    names = {c["name"] for c in payload["checks"] if c["ok"]}
    assert "data already at data/omop_data (no link needed)" in names


def test_validate_dataset_link_missing_source_fails(step, tmp_path, monkeypatch):
    mod = step("data_verification/01_validate_dataset.py")
    monkeypatch.setattr(mod.paths, "OMOP_DIR", tmp_path / "omop_target")
    rep = tmp_path / "rep01_missing"
    rc = mod.main(["--link", str(tmp_path / "does_not_exist"), "--report-dir", str(rep)])
    assert rc == 1
