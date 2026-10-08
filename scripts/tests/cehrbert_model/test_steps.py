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


def test_validation_step_registers_the_finetuned_model_in_the_manifest(step, tmp_path):
    from pipeline import manifest as mf
    from pipeline.reporting import StepReport

    mod = step("cehrbert_model/11_validation_predictions.py")
    results = tmp_path / "finetune_results"
    for sub in ("test_predictions", "validation_predictions"):
        (results / sub).mkdir(parents=True)
        (results / sub / "0.parquet").write_bytes(b"")
    manifest_path = tmp_path / "manifest.json"
    mf.upsert_models(manifest_path, [mf.make_entry("linear_probe_cls", "Probe", "linear_probe", "a", "b")])

    report = StepReport("11")
    mod._update_manifest(report, manifest_path, results)
    models = {m.name: m for m in mf.read_manifest(manifest_path)}
    assert list(models) == ["linear_probe_cls", "cehrbert_finetuned"]
    assert models["cehrbert_finetuned"].test_predictions == results / "test_predictions"
    assert models["cehrbert_finetuned"].validation_predictions == results / "validation_predictions"
    assert report.passed


def test_validation_step_without_test_predictions_adds_nothing_and_warns(step, tmp_path):
    from pipeline.reporting import StepReport

    mod = step("cehrbert_model/11_validation_predictions.py")
    (tmp_path / "validation_predictions").mkdir()
    report = StepReport("11")
    mod._update_manifest(report, tmp_path / "manifest.json", tmp_path)
    assert not (tmp_path / "manifest.json").exists()
    assert [c.level for c in report.checks if not c.ok] == ["warning"]
