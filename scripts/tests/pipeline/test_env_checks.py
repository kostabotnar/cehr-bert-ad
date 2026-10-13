"""Tests for pipeline/env_checks.py and the 00_check_environment step of every part."""

from __future__ import annotations

import json

import pytest

from pipeline import env_checks, paths

PARTS = list(paths.PARTS)


def _names(report):
    return {c.name: c for c in report.checks}


def test_missing_required_module_fails():
    report = env_checks.build_report(["json", "no_such_module_xyz"])
    checks = _names(report)
    assert checks["required package 'json' importable"].ok
    assert not checks["required package 'no_such_module_xyz' importable"].ok
    assert report.status == "fail"
    assert report.exit_code() == 1


def test_missing_optional_module_only_warns():
    report = env_checks.build_report(["json"], ["no_such_module_xyz", "os"])
    checks = _names(report)
    assert not checks["optional package 'no_such_module_xyz'"].ok
    assert checks["optional package 'no_such_module_xyz'"].level == "warning"
    assert checks["optional package 'os'"].ok
    assert report.status == "warn"
    assert report.exit_code() == 0


def test_all_present_passes_and_records_python_version():
    report = env_checks.build_report(["json", "os"])
    assert report.status == "pass"
    assert "python" in report.metrics


def test_gpu_and_java_checks_are_opt_in_and_never_fail(monkeypatch, tmp_path):
    plain = env_checks.build_report(["json"])
    assert not any("CUDA" in c.name or "Java" in c.name for c in plain.checks)

    monkeypatch.setattr(env_checks.shutil, "which", lambda name: None)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(env_checks.paths, "PROJECT_ROOT", tmp_path)
    report = env_checks.build_report(["json"], check_gpu=True, check_java=True)
    java = [c for c in report.checks if c.name.startswith("Java")]
    assert len(java) == 1 and not java[0].ok and java[0].level == "warning"
    assert report.exit_code() == 0


def _fake_java(root):
    java = root / "bin" / "java"
    java.parent.mkdir(parents=True)
    java.write_text("#!/bin/sh\necho 'openjdk version \"1.8.0\"' >&2\n", encoding="utf-8")
    java.chmod(0o755)
    return str(java)


def test_find_java_prefers_path(monkeypatch, tmp_path):
    monkeypatch.setattr(env_checks.shutil, "which", lambda name: "/usr/bin/java")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "home"))
    assert env_checks._find_java() == "/usr/bin/java"


def test_find_java_uses_java_home_when_not_on_path(monkeypatch, tmp_path):
    expected = _fake_java(tmp_path / "home")
    monkeypatch.setattr(env_checks.shutil, "which", lambda name: None)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(env_checks.paths, "PROJECT_ROOT", tmp_path / "empty")
    assert env_checks._find_java() == expected


def test_find_java_uses_project_local_jdk_last(monkeypatch, tmp_path):
    expected = _fake_java(tmp_path / ".tools" / "jdk8")
    monkeypatch.setattr(env_checks.shutil, "which", lambda name: None)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(env_checks.paths, "PROJECT_ROOT", tmp_path)
    assert env_checks._find_java() == expected

    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "missing"))
    assert env_checks._find_java() == expected

    report = env_checks.build_report(["json"], check_java=True)
    java = [c for c in report.checks if c.name.startswith("Java")][0]
    assert java.ok and expected in java.detail


def test_find_java_none_found(monkeypatch, tmp_path):
    monkeypatch.setattr(env_checks.shutil, "which", lambda name: None)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(env_checks.paths, "PROJECT_ROOT", tmp_path)
    assert env_checks._find_java() is None


def test_run_writes_report_and_returns_exit_code(tmp_path):
    rc = env_checks.run("cohort", ["json"], [], argv=["--report-dir", str(tmp_path / "rep")])
    assert rc == 0
    payload = json.loads((tmp_path / "rep" / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"

    rc = env_checks.run("cohort", ["no_such_module_xyz"], [], argv=["--report-dir", str(tmp_path / "rep2")])
    assert rc == 1


@pytest.mark.parametrize("part", PARTS)
def test_part_environment_step_lists_modules(step, part, tmp_path, monkeypatch):
    mod = step(f"{part}/00_check_environment.py")
    assert mod.REQUIRED, "every part declares required modules"
    assert not set(mod.REQUIRED) & set(mod.OPTIONAL)

    monkeypatch.setattr(env_checks, "_installed", lambda name: True)
    assert mod.main(["--report-dir", str(tmp_path / "ok")]) == 0

    monkeypatch.setattr(env_checks, "_installed", lambda name: name != mod.REQUIRED[0])
    assert mod.main(["--report-dir", str(tmp_path / "missing")]) == 1


def test_data_verification_requires_only_the_common_modules(step):
    mod = step("data_verification/00_check_environment.py")
    assert set(mod.REQUIRED) == {"polars", "pandas", "pyarrow", "yaml", "numpy"}


def test_cehrbert_model_requires_the_runtime(step):
    mod = step("cehrbert_model/00_check_environment.py")
    assert set(mod.REQUIRED) == {"datasets", "transformers", "torch", "cehrbert", "cehrbert_data"}
