"""Tests for the heavy launcher steps (02/04/08): preflight gating + launch wiring.

The cehrbert runners and Spark are never executed — pipeline.launch.run_command is
monkeypatched. We assert that preflight failures prevent any launch, and that on a
mocked-successful launch the step verifies outputs and reports success.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from pipeline.omop_common import MIN_CONCEPT_PATIENTS


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


def _metrics(report_dir):
    return json.loads((report_dir / "status.json").read_text(encoding="utf-8"))["metrics"]


def _status(report_dir):
    return json.loads((report_dir / "status.json").read_text(encoding="utf-8"))["status"]


# --- step 01 ---------------------------------------------------------------
def test_step02_launch_success(step, omop_dir, tmp_path, monkeypatch):
    mod = step("cehrbert_model/01_generate_sequences.py")
    out_folder = tmp_path / "cehrbert_data"
    staging = out_folder / "omop_input"
    omop_before = sorted(p.name for p in omop_dir.iterdir())

    def make_sequences(cmd, cwd, env):
        seq = out_folder / "patient_sequence"
        seq.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"person_id": [1, 2]}).to_parquet(seq / "0.parquet")

    rec = CallRecorder(returncode=0, side_effect=make_sequences)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep02"
    rc = mod.main(
        [
            "--input-folder",
            str(omop_dir),
            "--output-folder",
            str(out_folder),
            "--staging-dir",
            str(staging),
            "--report-dir",
            str(rep),
        ]
    )
    assert rc == 0
    assert _status(rep) == "pass"
    assert len(rec.calls) == 1
    # The wrapper gets the staging folder, never the OMOP folder, and the OMOP folder stays untouched.
    assert rec.calls[0]["env"]["INPUT_FOLDER"] == str(staging)
    assert omop_before == sorted(p.name for p in omop_dir.iterdir())
    assert (staging / "person").is_symlink() and (staging / "person").resolve() == (omop_dir / "person").resolve()
    # Spark env is passed through to the wrapper.
    assert "SPARK_DRIVER_MEMORY" in rec.calls[0]["env"]
    assert rec.calls[0]["cmd"][0] == "bash"
    assert rec.calls[0]["env"]["MIN_CONCEPT_PATIENTS"] == str(MIN_CONCEPT_PATIENTS)
    assert _metrics(rep)["min_concept_patients"] == MIN_CONCEPT_PATIENTS


def test_step01_wrapper_fails_without_min_concept_patients(tmp_path):
    marker = tmp_path / "called"
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(f"#!/bin/bash\ntouch {marker}\n")
    fake_python.chmod(0o755)
    wrapper = Path(__file__).resolve().parents[2] / "cehrbert_model" / "01_generate_sequences.sh"
    env = {k: v for k, v in os.environ.items() if k != "MIN_CONCEPT_PATIENTS"}
    env.update(
        {"PYTHON": str(fake_python), "OUTPUT_FOLDER": str(tmp_path / "out"), "INPUT_FOLDER": str(tmp_path / "in")}
    )
    for value in (None, ""):
        if value is not None:
            env["MIN_CONCEPT_PATIENTS"] = value
        proc = subprocess.run(["bash", str(wrapper)], env=env, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "MIN_CONCEPT_PATIENTS" in proc.stderr
    assert not marker.exists()
    assert not (tmp_path / "out").exists()


def test_step02_preflight_gate_blocks_launch(step, tmp_path, monkeypatch):
    mod = step("cehrbert_model/01_generate_sequences.py")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    empty = tmp_path / "empty_omop"
    empty.mkdir()
    rep = tmp_path / "rep02b"
    rc = mod.main(
        [
            "--input-folder",
            str(empty),
            "--output-folder",
            str(tmp_path / "o"),
            "--staging-dir",
            str(tmp_path / "stage"),
            "--report-dir",
            str(rep),
        ]
    )
    assert rc == 1
    assert _status(rep) == "fail"
    assert rec.calls == []  # never launched
    assert not (tmp_path / "stage").exists()


# --- step 05 ---------------------------------------------------------------
def _splits(tmp_path, test_ids=(7,)):
    import polars as pl

    path = tmp_path / "splits.parquet"
    ids = [1, 2, *test_ids]
    pl.DataFrame({"person_id": ids, "split": ["train", "val"] + ["test"] * len(test_ids)}).write_parquet(path)
    return path


def test_step04_launch_success(step, pretrain_config, tmp_path, monkeypatch):
    mod = step("cehrbert_model/05_pretrain.py")
    monkeypatch.setattr(mod.paths, "PRETRAIN_PREPARED_DIR", tmp_path / "pp")
    monkeypatch.setattr(mod.paths, "PRETRAIN_RESULTS_DIR", tmp_path / "pr")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep04"
    rc = mod.main(["--config", str(pretrain_config), "--splits-file", str(_splits(tmp_path)), "--report-dir", str(rep)])
    assert rc == 0
    assert len(rec.calls) == 1


def test_step04_missing_sequences_gates_launch(step, pretrain_config, tmp_path, monkeypatch):
    mod = step("cehrbert_model/05_pretrain.py")
    cfg = yaml.safe_load(pretrain_config.read_text(encoding="utf-8"))
    cfg["data_folder"] = str(tmp_path / "no_sequences")
    pretrain_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep04b"
    rc = mod.main(["--config", str(pretrain_config), "--splits-file", str(_splits(tmp_path)), "--report-dir", str(rep)])
    assert rc == 1
    assert rec.calls == []


def test_step04_missing_splits_gates_launch(step, pretrain_config, tmp_path, monkeypatch):
    mod = step("cehrbert_model/05_pretrain.py")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rc = mod.main(
        [
            "--config", str(pretrain_config),
            "--splits-file", str(tmp_path / "none.parquet"),
            "--report-dir", str(tmp_path / "rep04c"),
        ]
    )  # fmt: skip
    assert rc == 1
    assert rec.calls == []


# --- step 09 ---------------------------------------------------------------
def test_step08_launch_success(step, finetune_config, tmp_path, monkeypatch):
    mod = step("cehrbert_model/09_finetune.py")
    monkeypatch.setattr(mod.paths, "FINETUNE_PREPARED_DIR", tmp_path / "fp")
    monkeypatch.setattr(mod.paths, "FINETUNE_RESULTS_DIR", tmp_path / "fr")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep08"
    rc = mod.main(["--config", str(finetune_config), "--report-dir", str(rep)])
    assert rc == 0
    assert len(rec.calls) == 1


def test_step08_preflight_gate_blocks_launch(step, tmp_path, monkeypatch):
    mod = step("cehrbert_model/09_finetune.py")
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


# --- step 10 (predict + verify) -------------------------------------------
def test_step09_predict_then_verify_success(step, finetune_config, finetune_results_dir, tmp_path, monkeypatch):
    mod = step("cehrbert_model/10_predict.py")
    # Point pipeline outputs at fixtures; cohort_test + finetuned model must exist for preflight.
    cohort_test = tmp_path / "cohort" / "test"
    cohort_test.mkdir(parents=True)
    import polars as pl

    pl.DataFrame({"person_id": [1], "index_date": ["2020-01-01"], "label": [1]}).write_parquet(
        cohort_test / "cohort.parquet"
    )
    model_dir = finetune_results_dir
    (model_dir / "config.json").write_text("{}", encoding="utf-8")  # make it look like a model dir

    monkeypatch.setattr(mod.paths, "CEHRBERT_COHORT_TEST_DIR", cohort_test)
    monkeypatch.setattr(mod.paths, "FINETUNE_RESULTS_DIR", model_dir)
    monkeypatch.setattr(mod.paths, "PREDICT_PREPARED_DIR", tmp_path / "pp")
    rec = CallRecorder(returncode=0)  # predictions already exist in the fixture
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep09"
    rc = mod.main(
        ["--finetune-config", str(finetune_config), "--results-dir", str(model_dir), "--report-dir", str(rep)]
    )
    assert rc == 0
    assert len(rec.calls) == 1  # prediction launched
    assert _status(rep) == "pass"


def test_step09_preflight_gate_blocks_launch(step, finetune_config, tmp_path, monkeypatch):
    mod = step("cehrbert_model/10_predict.py")
    monkeypatch.setattr(mod.paths, "CEHRBERT_COHORT_TEST_DIR", tmp_path / "no_cohort")
    monkeypatch.setattr(mod.paths, "FINETUNE_RESULTS_DIR", tmp_path / "no_model")
    rec = CallRecorder(returncode=0)
    monkeypatch.setattr(mod.launch, "run_command", rec)

    rep = tmp_path / "rep09b"
    rc = mod.main(["--finetune-config", str(finetune_config), "--report-dir", str(rep)])
    assert rc == 1
    assert rec.calls == []  # never launched
