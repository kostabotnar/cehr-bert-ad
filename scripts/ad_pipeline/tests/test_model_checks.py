"""Tests for pipeline.model_checks."""

from __future__ import annotations

from pipeline import model_checks as mc
from pipeline.reporting import StepReport


def test_valid_model_dir_passes(model_dir):
    r = StepReport("t")
    mc.add_model_checks(r, model_dir, require_tokenizer=True)
    assert r.status == "pass"


def test_missing_model_dir_fails(tmp_path):
    r = StepReport("t")
    mc.add_model_checks(r, tmp_path / "nope")
    assert r.status == "fail"


def test_missing_weights_fails(model_dir):
    (model_dir / "model.safetensors").unlink()
    r = StepReport("t")
    mc.add_model_checks(r, model_dir, require_tokenizer=False)
    assert r.status == "fail"


def test_train_results_metrics_surfaced(model_dir):
    r = StepReport("t")
    mc.add_train_results_checks(r, model_dir)
    assert r.metrics["train_loss"] == 0.42
    assert r.metrics["best_metric"] == 0.31
    assert r.metrics["global_step"] == 500
    assert r.status == "pass"


def test_missing_train_results_fails(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    r = StepReport("t")
    mc.add_train_results_checks(r, tmp_path)
    assert r.status == "fail"
