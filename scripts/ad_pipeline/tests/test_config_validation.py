"""Tests for pipeline.config_validation (semantic + path layer; structural layer is
skipped here because the cehrbert/transformers deps are not installed in test env)."""

from __future__ import annotations

import yaml

from pipeline import config_validation as cv
from pipeline.paths import resolve_under_root
from pipeline.reporting import StepReport


def _names_failed(report):
    return {c.name for c in report.checks if not c.ok and c.level == "error"}


def test_valid_pretrain_config_passes(pretrain_config):
    r = StepReport("t")
    cv.add_pretrain_config_checks(r, pretrain_config, resolve_under_root)
    # Structural parse is skipped (warning), so overall status is at worst 'warn'.
    assert r.passed is True
    assert not _names_failed(r)


def test_bad_pretrain_values_fail(tmp_path, patient_sequence_dir):
    cfg = {
        "data_folder": str(patient_sequence_dir),
        "num_train_epochs": 0,       # invalid
        "learning_rate": 5.0,        # invalid (>=1)
        "vocab_size": 50000,
        "do_train": True,
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = StepReport("t")
    cv.add_pretrain_config_checks(r, path, resolve_under_root)
    assert r.status == "fail"
    failed = _names_failed(r)
    assert "num_train_epochs valid" in failed
    assert "learning_rate valid" in failed


def test_pretrain_missing_data_folder_fails(tmp_path):
    cfg = {"data_folder": str(tmp_path / "does_not_exist"), "do_train": True}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = StepReport("t")
    cv.add_pretrain_config_checks(r, path, resolve_under_root)
    assert r.status == "fail"


def test_valid_finetune_config_passes(finetune_config):
    r = StepReport("t")
    cv.add_finetune_config_checks(r, finetune_config, resolve_under_root)
    assert r.passed is True
    assert not _names_failed(r)


def test_finetune_bad_validation_pct_fails(tmp_path, model_dir, fake_tokenized_dataset, cohort_dir):
    cfg = {
        "model_name_or_path": str(model_dir),
        "tokenizer_name_or_path": str(model_dir),
        "tokenized_full_dataset_path": str(fake_tokenized_dataset),
        "cohort_folder": str(cohort_dir),
        "observation_window": 365,
        "do_train": True,
        "do_predict": False,
        "validation_split_percentage": 1.5,  # invalid
    }
    path = tmp_path / "ft.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = StepReport("t")
    cv.add_finetune_config_checks(r, path, resolve_under_root)
    assert r.status == "fail"
    assert "validation_split_percentage in (0, 1)" in _names_failed(r)


def test_finetune_do_predict_true_warns(tmp_path, model_dir, fake_tokenized_dataset, cohort_dir):
    # Fine-tuning should not predict (that's step 9) -> warning, not a hard failure.
    cfg = {
        "model_name_or_path": str(model_dir),
        "tokenizer_name_or_path": str(model_dir),
        "tokenized_full_dataset_path": str(fake_tokenized_dataset),
        "cohort_folder": str(cohort_dir),
        "observation_window": 365,
        "do_train": True,
        "do_predict": True,
        "validation_split_percentage": 0.1,
    }
    path = tmp_path / "ft.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = StepReport("t")
    cv.add_finetune_config_checks(r, path, resolve_under_root)
    assert r.passed is True  # warning only
    assert any(c.name == "do_predict disabled for fine-tuning" and not c.ok for c in r.checks)


def test_write_config_roundtrip(tmp_path):
    cfg = {"do_predict": True, "cohort_folder": "x", "validation_split_percentage": 0.1}
    dst = tmp_path / "sub" / "predict.yaml"
    cv.write_config(cfg, dst)
    assert dst.exists()
    assert yaml.safe_load(dst.read_text(encoding="utf-8")) == cfg


def test_finetune_bad_cohort_schema_fails(tmp_path, model_dir, fake_tokenized_dataset):
    import polars as pl

    bad_cohort = tmp_path / "ad_cohort"
    bad_cohort.mkdir()
    pl.DataFrame({"person_id": [1], "wrong": [2]}).write_parquet(bad_cohort / "cohort.parquet")
    cfg = {
        "model_name_or_path": str(model_dir),
        "tokenizer_name_or_path": str(model_dir),
        "tokenized_full_dataset_path": str(fake_tokenized_dataset),
        "cohort_folder": str(bad_cohort),
        "observation_window": 365,
        "do_predict": True,
    }
    path = tmp_path / "ft.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = StepReport("t")
    cv.add_finetune_config_checks(r, path, resolve_under_root)
    assert "cohort has required columns" in _names_failed(r)
