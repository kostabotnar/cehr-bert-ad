"""Tests for pipeline.reporting.StepReport."""

from __future__ import annotations

import json

from pipeline.reporting import StepReport


def test_all_pass_is_pass_and_exit_zero():
    r = StepReport("s", "Title")
    r.ok("a")
    r.ok("b")
    assert r.status == "pass"
    assert r.exit_code() == 0


def test_failed_error_check_fails_and_exits_nonzero():
    r = StepReport("s")
    r.ok("a")
    r.fail("b", "boom")
    assert r.status == "fail"
    assert r.exit_code() == 1


def test_failed_warning_only_is_warn_and_exits_zero():
    r = StepReport("s")
    r.ok("a")
    r.warn("b", ok=False, detail="meh")
    assert r.status == "warn"
    assert r.exit_code() == 0
    assert r.passed is True


def test_error_outranks_warning():
    r = StepReport("s")
    r.warn("w", ok=False)
    r.fail("e")
    assert r.status == "fail"


def test_add_check_returns_ok_value():
    r = StepReport("s")
    assert r.add_check("x", True) is True
    assert r.add_check("y", False) is False


def test_write_emits_json_and_markdown(tmp_path):
    r = StepReport("s", "My Step")
    r.ok("check one")
    r.add_metric("rows", 5)
    r.add_artifact(tmp_path / "out.parquet")
    r.note("a note")
    out = r.write(tmp_path / "rep", timestamp="2026-01-01T00:00:00")

    payload = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["metrics"]["rows"] == 5
    assert payload["timestamp"] == "2026-01-01T00:00:00"
    assert payload["n_checks"] == 1

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "# My Step" in md
    assert "PASS" in md
    assert "rows" in md
