"""Tests for the heavy launcher steps (02/04/08): preflight gating + launch wiring.

The cehrbert runners and Spark are never executed — pipeline.launch.run_command is
monkeypatched. We assert that preflight failures prevent any launch, and that on a
mocked-successful launch the step verifies outputs and reports success.
"""

from __future__ import annotations

import json

import pandas as pd
import yaml


class CallRecorder:
    """Stand-in for launch.run_command; records calls and returns a fixed code."""

    def __init__(self, returncode=0, side_effect=None):
        self.returncode = returncode
        self.side_effect = side_effect
        self.calls = []

    def __call__(self, cmd, cwd=None, env=None):
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env or {}})
        if self.side_effect:
            self.side_effect(cmd, cwd, env)
        return self.returncode


def _status(report_dir):
    return json.loads((report_dir / "status.json").read_text(encoding="utf-8"))["status"]


# --- step 02 ---------------------------------------------------------------
def test_step02_launch_success(step, omop_dir, tmp_path, monkeypatch):
    mod = step("02_prepare_pretraining_data.py")
    out_folder = tmp_path / "cehrbert_data"

    def make_sequences(cmd, cwd, env):
        seq = out_folder / "patient_sequence"
        seq.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"person_id": [1, 2]}).to_parquet(seq / "0.parquet")

    rec = CallRecorder(returncode=0, side_effect=make_sequences)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep02"
    rc = mod.main([
        "--input-folder", str(omop_dir),
        "--output-folder", str(out_folder),
        "--report-dir", str(rep),
    ])
    assert rc == 0
    assert _status(rep) == "pass"
    assert len(rec.calls) == 1
    # Spark env is passed through to the wrapper.
    assert "SPARK_DRIVER_MEMORY" in rec.calls[0]["env"]
    assert rec.calls[0]["cmd"][0] == "bash"


def test_step02_preflight_gate_blocks_launch(step, tmp_path, monkeypatch):
    mod = step("02_prepare_pretraining_data.py")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    empty = tmp_path / "empty_omop"
    empty.mkdir()
    rep = tmp_path / "rep02b"
    rc = mod.main(["--input-folder", str(empty), "--output-folder", str(tmp_path / "o"), "--report-dir", str(rep)])
    assert rc == 1
    assert _status(rep) == "fail"
    assert rec.calls == []  # never launched


# --- step 04 ---------------------------------------------------------------
def test_step04_launch_success(step, pretrain_config, patient_sequence_dir, tmp_path, monkeypatch):
    mod = step("04_pretrain.py")
    monkeypatch.setattr(mod.paths, "PATIENT_SEQUENCE_DIR", patient_sequence_dir)
    monkeypatch.setattr(mod.paths, "PRETRAIN_PREPARED_DIR", tmp_path / "pp")
    monkeypatch.setattr(mod.paths, "PRETRAIN_RESULTS_DIR", tmp_path / "pr")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep04"
    rc = mod.main(["--config", str(pretrain_config), "--report-dir", str(rep)])
    assert rc == 0
    assert len(rec.calls) == 1


def test_step04_missing_sequences_gates_launch(step, pretrain_config, tmp_path, monkeypatch):
    mod = step("04_pretrain.py")
    monkeypatch.setattr(mod.paths, "PATIENT_SEQUENCE_DIR", tmp_path / "no_sequences")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep04b"
    rc = mod.main(["--config", str(pretrain_config), "--report-dir", str(rep)])
    assert rc == 1
    assert rec.calls == []


# --- step 08 ---------------------------------------------------------------
def test_step08_launch_success(step, finetune_config, tmp_path, monkeypatch):
    mod = step("08_finetune.py")
    monkeypatch.setattr(mod.paths, "FINETUNE_PREPARED_DIR", tmp_path / "fp")
    monkeypatch.setattr(mod.paths, "FINETUNE_RESULTS_DIR", tmp_path / "fr")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep08"
    rc = mod.main(["--config", str(finetune_config), "--report-dir", str(rep)])
    assert rc == 0
    assert len(rec.calls) == 1


def test_step08_preflight_gate_blocks_launch(step, tmp_path, monkeypatch):
    mod = step("08_finetune.py")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    # Config referencing non-existent model/cohort -> preflight fails.
    cfg = {
        "model_name_or_path": str(tmp_path / "nope"),
        "tokenizer_name_or_path": str(tmp_path / "nope"),
        "tokenized_full_dataset_path": str(tmp_path / "nope"),
        "cohort_folder": str(tmp_path / "nope"),
        "observation_window": 365,
        "do_train": True,
        "do_predict": False,
    }
    path = tmp_path / "bad_ft.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rep = tmp_path / "rep08b"
    rc = mod.main(["--config", str(path), "--report-dir", str(rep)])
    assert rc == 1
    assert rec.calls == []


# --- step 09 (predict + verify) -------------------------------------------
def test_step09_predict_then_verify_success(step, finetune_config, finetune_results_dir, tmp_path, monkeypatch):
    mod = step("09_predict.py")
    # Point pipeline outputs at fixtures; cohort_test + finetuned model must exist for preflight.
    cohort_test = tmp_path / "ad_cohort" / "test"
    cohort_test.mkdir(parents=True)
    import polars as pl

    pl.DataFrame({"person_id": [1], "index_date": ["2020-01-01"], "label": [1]}).write_parquet(
        cohort_test / "cohort.parquet"
    )
    model_dir = finetune_results_dir
    (model_dir / "config.json").write_text("{}", encoding="utf-8")  # make it look like a model dir

    monkeypatch.setattr(mod.paths, "COHORT_TEST_DIR", cohort_test)
    monkeypatch.setattr(mod.paths, "FINETUNE_RESULTS_DIR", model_dir)
    monkeypatch.setattr(mod.paths, "PREDICT_PREPARED_DIR", tmp_path / "pp")
    rec = CallRecorder(returncode=0)  # predictions already exist in the fixture
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep09"
    rc = mod.main(["--finetune-config", str(finetune_config), "--results-dir", str(model_dir),
                   "--report-dir", str(rep)])
    assert rc == 0
    assert len(rec.calls) == 1  # prediction launched
    assert _status(rep) == "pass"


def test_step09_preflight_gate_blocks_launch(step, finetune_config, tmp_path, monkeypatch):
    mod = step("09_predict.py")
    monkeypatch.setattr(mod.paths, "COHORT_TEST_DIR", tmp_path / "no_cohort")
    monkeypatch.setattr(mod.paths, "FINETUNE_RESULTS_DIR", tmp_path / "no_model")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep09b"
    rc = mod.main(["--finetune-config", str(finetune_config), "--report-dir", str(rep)])
    assert rc == 1
    assert rec.calls == []  # never launched
