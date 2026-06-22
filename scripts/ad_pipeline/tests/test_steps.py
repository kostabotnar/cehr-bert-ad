"""End-to-end-ish tests for each step script's main(), using synthetic fixtures.

No training is run. Heavy launcher steps (02/04/08) are covered in test_launchers.py.
"""

from __future__ import annotations

import json



def _status(report_dir):
    return json.loads((report_dir / "status.json").read_text(encoding="utf-8"))["status"]


def test_step01_validate_dataset_passes(step, omop_dir, tmp_path):
    mod = step("01_validate_dataset.py")
    rep = tmp_path / "rep01"
    rc = mod.main(["--data-dir", str(omop_dir), "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) == "pass"


def test_step01_missing_table_fails(step, omop_dir, tmp_path):
    for f in (omop_dir / "person").glob("*.parquet"):
        f.unlink()
    (omop_dir / "person").rmdir()
    mod = step("01_validate_dataset.py")
    rep = tmp_path / "rep01b"
    rc = mod.main(["--data-dir", str(omop_dir), "--report-dir", str(rep)])
    assert rc == 1
    assert _status(rep) == "fail"


def test_step01_link_identity_validates_in_place(step, omop_dir, tmp_path, monkeypatch):
    # --link pointing at a folder that is already data/omop_data must not error;
    # it should validate the folder in place.
    mod = step("01_validate_dataset.py")
    monkeypatch.setattr(mod.paths, "OMOP_DIR", omop_dir)
    rep = tmp_path / "rep01_link"
    rc = mod.main(["--link", str(omop_dir), "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) == "pass"
    payload = json.loads((rep / "status.json").read_text(encoding="utf-8"))
    names = {c["name"] for c in payload["checks"] if c["ok"]}
    assert "data already at data/omop_data (no link needed)" in names


def test_step01_link_missing_source_fails(step, tmp_path, monkeypatch):
    mod = step("01_validate_dataset.py")
    monkeypatch.setattr(mod.paths, "OMOP_DIR", tmp_path / "omop_target")
    rep = tmp_path / "rep01_missing"
    rc = mod.main(["--link", str(tmp_path / "does_not_exist"), "--report-dir", str(rep)])
    assert rc == 1


def test_step03_evaluate_pretrain_config(step, pretrain_config, tmp_path):
    mod = step("03_evaluate_pretrain_config.py")
    rep = tmp_path / "rep03"
    rc = mod.main(["--config", str(pretrain_config), "--report-dir", str(rep)])
    assert rc == 0  # warn (structural skipped) still exits 0
    assert _status(rep) in {"pass", "warn"}


def test_step05_verify_pretraining(step, model_dir, prepared_dataset, tmp_path):
    mod = step("05_verify_pretraining.py")
    rep = tmp_path / "rep05"
    rc = mod.main([
        "--results-dir", str(model_dir),
        "--prepared-dir", str(prepared_dataset),
        "--report-dir", str(rep),
    ])
    # Model + train_results pass; prepared-dataset content check may warn without datasets.
    assert rc == 0
    assert _status(rep) in {"pass", "warn"}


def test_step05_no_model_fails(step, tmp_path):
    mod = step("05_verify_pretraining.py")
    rep = tmp_path / "rep05b"
    rc = mod.main([
        "--results-dir", str(tmp_path / "nope"),
        "--prepared-dir", str(tmp_path / "nope2"),
        "--report-dir", str(rep),
    ])
    assert rc == 1


def test_step06_prepare_finetuning_data(step, omop_dir, tmp_path):
    mod = step("06_prepare_finetuning_data.py")
    ft = tmp_path / "ft" / "cohort.parquet"
    te = tmp_path / "te" / "cohort.parquet"
    rep = tmp_path / "rep06"
    rc = mod.main([
        "--data-dir", str(omop_dir),
        "--finetune-output", str(ft),
        "--test-output", str(te),
        "--skip-tokenized-check",
        "--report-dir", str(rep),
    ])
    assert rc == 0
    assert ft.exists() and te.exists()
    payload = json.loads((rep / "status.json").read_text(encoding="utf-8"))
    # Full cohort is 2 cases + 3 controls; the holdout split keeps both non-empty.
    assert payload["metrics"]["cases"] == 2
    assert payload["metrics"]["controls"] == 3
    assert payload["metrics"]["finetune_size"] + payload["metrics"]["test_size"] == 5

    import polars as pl

    ft_ids = set(pl.read_parquet(ft)["person_id"].to_list())
    te_ids = set(pl.read_parquet(te)["person_id"].to_list())
    assert ft_ids.isdisjoint(te_ids)  # patient-level, no leakage


def test_step07_validate_finetune_config(step, finetune_config, tmp_path):
    mod = step("07_validate_finetune_config.py")
    rep = tmp_path / "rep07"
    rc = mod.main(["--config", str(finetune_config), "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) in {"pass", "warn"}


def test_step09_verify_only(step, finetune_results_dir, tmp_path):
    # --skip-launch verifies existing predictions/metrics without running inference.
    mod = step("09_predict.py")
    rep = tmp_path / "rep09"
    rc = mod.main(["--results-dir", str(finetune_results_dir), "--skip-launch", "--report-dir", str(rep)])
    assert rc == 0
    assert _status(rep) == "pass"


def test_step09_no_results_fails(step, tmp_path):
    mod = step("09_predict.py")
    rep = tmp_path / "rep09b"
    rc = mod.main(["--results-dir", str(tmp_path / "nope"), "--skip-launch", "--report-dir", str(rep)])
    assert rc == 1
