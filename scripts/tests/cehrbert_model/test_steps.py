"""Tests for the config validation, pretraining verification and prediction steps of cehrbert_model.

No training is run. Launcher steps are covered in test_launchers.py.
"""

from __future__ import annotations

import json


def _status(report_dir):
    return json.loads((report_dir / "status.json").read_text(encoding="utf-8"))["status"]


def test_validate_pretrain_config(step, pretrain_config, tmp_path):
    mod = step("cehrbert_model/04_validate_pretrain_config.py")
    rep = tmp_path / "rep03"
    rc = mod.main(["--config", str(pretrain_config), "--report-dir", str(rep)])
    assert rc == 0  # warn (structural skipped) still exits 0
    assert _status(rep) in {"pass", "warn"}


def test_verify_pretraining(step, model_dir, prepared_dataset, tmp_path):
    import polars as pl

    mod = step("cehrbert_model/06_verify_pretraining.py")
    rep = tmp_path / "rep05"
    splits = tmp_path / "splits.parquet"
    pl.DataFrame({"person_id": [1, 5, 9], "split": ["train", "val", "test"]}).write_parquet(splits)
    rc = mod.main(
        [
            "--results-dir",
            str(model_dir),
            "--prepared-dir",
            str(prepared_dataset),
            "--splits-file",
            str(splits),
            "--report-dir",
            str(rep),
        ]
    )
    # Model + train_results pass; prepared-dataset content check may warn without datasets.
    assert rc == 0
    assert _status(rep) in {"pass", "warn"}


def test_verify_pretraining_no_model_fails(step, tmp_path):
    mod = step("cehrbert_model/06_verify_pretraining.py")
    rep = tmp_path / "rep05b"
    rc = mod.main(
        [
            "--results-dir",
            str(tmp_path / "nope"),
            "--prepared-dir",
            str(tmp_path / "nope2"),
            "--report-dir",
            str(rep),
        ]
    )
    assert rc == 1


def test_validate_finetune_config(step, finetune_config, tmp_path):
    mod = step("cehrbert_model/08_validate_finetune_config.py")
    rep = tmp_path / "rep07"
    rc = mod.main(["--config", str(finetune_config), "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) in {"pass", "warn"}


def test_predict_verify_only(step, finetune_results_dir, tmp_path):
    # --skip-launch verifies existing predictions/metrics without running inference.
    mod = step("cehrbert_model/10_predict.py")
    rep = tmp_path / "rep09"
    rc = mod.main(["--results-dir", str(finetune_results_dir), "--skip-launch", "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) == "pass"


def test_predict_no_results_fails(step, tmp_path):
    mod = step("cehrbert_model/10_predict.py")
    rep = tmp_path / "rep09b"
    rc = mod.main(["--results-dir", str(tmp_path / "nope"), "--skip-launch", "--report-dir", str(rep)])
    assert rc == 1
