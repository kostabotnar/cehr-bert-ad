"""Shared pytest fixtures: synthetic OMOP tables, fake model/results dirs, configs.

No real training is ever run. Tests that need the ``datasets`` library skip themselves
via ``pytest.importorskip`` so the suite runs in a minimal environment too.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

AD_PIPELINE_DIR = Path(__file__).resolve().parents[1]
# Make `import pipeline` work and allow loading digit-prefixed step scripts.
sys.path.insert(0, str(AD_PIPELINE_DIR))


# --------------------------------------------------------------------------
# Step-script loader (filenames start with digits, so use importlib).
# --------------------------------------------------------------------------
def load_step(filename: str):
    """Import a step script (e.g. '01_validate_dataset.py') as a module."""
    path = AD_PIPELINE_DIR / filename
    mod_name = "step_" + path.stem
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def step():
    return load_step


# --------------------------------------------------------------------------
# Synthetic OMOP dataset
# --------------------------------------------------------------------------
@pytest.fixture
def omop_dir(tmp_path: Path) -> Path:
    """A tiny but schema-complete OMOP dataset that yields a 2-case / 3-control cohort."""
    root = tmp_path / "omop"
    root.mkdir()

    def write(table: str, df: pl.DataFrame) -> None:
        d = root / table
        d.mkdir()
        df.write_parquet(d / "data.parquet")

    D = dt.date
    write("person", pl.DataFrame({"person_id": [1, 2, 3, 4, 5, 6]}))

    write(
        "visit_occurrence",
        pl.DataFrame(
            {
                "person_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                "visit_start_date": [
                    D(2010, 1, 1), D(2021, 1, 1),   # p1
                    D(2011, 1, 1), D(2021, 6, 1),   # p2
                    D(2012, 1, 1), D(2020, 1, 1),   # p3
                    D(2010, 1, 1), D(2022, 1, 1),   # p4
                    D(2015, 1, 1), D(2022, 6, 1),   # p5
                    D(2016, 1, 1), D(2016, 2, 1),   # p6 (too little history -> excluded)
                ],
            }
        ),
    )

    # concept_id 1001/1002 = AD; 2002 = non-AD
    write(
        "condition_occurrence",
        pl.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "condition_concept_id": [1001, 1002, 2002, 2002, 2002],
                "condition_start_date": [
                    D(2020, 1, 1), D(2020, 6, 1), D(2019, 1, 1), D(2018, 1, 1), D(2019, 6, 1)
                ],
            }
        ),
    )

    write(
        "concept",
        pl.DataFrame(
            {
                "concept_id": [1001, 1002, 2002],
                "concept_code": ["ICD-10-CM:G30.9", "ICD-9-CM:331.0", "ICD-10-CM:E11.9"],
            }
        ),
    )

    write("drug_exposure", pl.DataFrame({"person_id": [1, 2], "drug_concept_id": [50, 51]}))
    write("procedure_occurrence", pl.DataFrame({"person_id": [1, 3], "procedure_concept_id": [60, 61]}))
    write("measurement", pl.DataFrame({"person_id": [1, 4], "measurement_concept_id": [70, 71]}))

    return root


# --------------------------------------------------------------------------
# Fake model directory (mimics HF Trainer.save_model output)
# --------------------------------------------------------------------------
@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pretrain_results"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "cehrbert"}), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00")  # presence is all we check
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "train_results.json").write_text(json.dumps({"train_loss": 0.42, "epoch": 10.0}), encoding="utf-8")
    (d / "trainer_state.json").write_text(
        json.dumps({"best_metric": 0.31, "global_step": 500, "epoch": 10.0}), encoding="utf-8"
    )
    return d


# --------------------------------------------------------------------------
# Fake finetuning results (predictions + metrics)
# --------------------------------------------------------------------------
@pytest.fixture
def finetune_results_dir(tmp_path: Path) -> Path:
    import pandas as pd

    d = tmp_path / "finetune_results"
    pred = d / "test_predictions"
    pred.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [dt.datetime(2020, 1, 1)] * 4,
            "predicted_boolean_probability": [0.1, 0.9, 0.4, 0.7],
            "predicted_boolean_value": [None] * 4,
            "boolean_value": [False, True, False, True],
        }
    ).to_parquet(pred / "0.parquet")
    (d / "test_results.json").write_text(
        json.dumps({"roc_auc": 0.83, "pr_auc": 0.71, "test_loss": 0.35}), encoding="utf-8"
    )
    return d


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------
@pytest.fixture
def patient_sequence_dir(tmp_path: Path) -> Path:
    """A folder that looks like generated patient sequences (has a parquet)."""
    import pandas as pd

    d = tmp_path / "cehrbert_data" / "patient_sequence"
    d.mkdir(parents=True)
    pd.DataFrame({"person_id": [1, 2], "concept_ids": [[1, 2], [3, 4]]}).to_parquet(d / "0.parquet")
    return d


@pytest.fixture
def pretrain_config(tmp_path: Path, patient_sequence_dir: Path) -> Path:
    cfg = {
        "model_name_or_path": str(tmp_path / "pretrain_results"),
        "tokenizer_name_or_path": str(tmp_path / "pretrain_results"),
        "data_folder": str(patient_sequence_dir),
        "dataset_prepared_path": str(tmp_path / "pretrain_prepared"),
        "output_dir": str(tmp_path / "pretrain_results"),
        "do_train": True,
        "num_train_epochs": 10,
        "learning_rate": 5e-05,
        "per_device_train_batch_size": 32,
        "vocab_size": 50000,
        "max_position_embeddings": 512,
        "num_hidden_layers": 6,
        "weight_decay": 0.01,
    }
    path = tmp_path / "pretrain.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def fake_tokenized_dataset(tmp_path: Path) -> Path:
    """A base dir containing one resolvable DatasetDict subdir (marker file only)."""
    base = tmp_path / "pretrain_prepared"
    sub = base / "omop_abc123"
    sub.mkdir(parents=True)
    (sub / "dataset_dict.json").write_text("{}", encoding="utf-8")
    return base


@pytest.fixture
def prepared_dataset(tmp_path: Path) -> Path:
    """A base dir with a real saved DatasetDict when `datasets` is available.

    Falls back to a structural marker (like fake_tokenized_dataset) otherwise, so the
    fixture works in both environments.
    """
    base = tmp_path / "pretrain_prepared_real"
    base.mkdir()
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        sub = base / "omop_hash"
        sub.mkdir()
        (sub / "dataset_dict.json").write_text("{}", encoding="utf-8")
        return base
    DatasetDict(
        {
            "train": Dataset.from_dict({"person_id": [1, 2, 3, 4], "concept_ids": [[1]] * 4}),
            "validation": Dataset.from_dict({"person_id": [5, 6], "concept_ids": [[1]] * 2}),
        }
    ).save_to_disk(str(base / "omop_hash"))
    return base


@pytest.fixture
def cohort_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ad_cohort"
    d.mkdir()
    pl.DataFrame(
        {"person_id": [1, 2], "index_date": [dt.date(2019, 1, 1), dt.date(2020, 1, 1)], "label": [1, 0]}
    ).write_parquet(d / "cohort.parquet")
    return d


@pytest.fixture
def finetune_config(tmp_path: Path, model_dir: Path, fake_tokenized_dataset: Path, cohort_dir: Path) -> Path:
    cfg = {
        "model_name_or_path": str(model_dir),
        "tokenizer_name_or_path": str(model_dir),
        "tokenized_full_dataset_path": str(fake_tokenized_dataset),
        "data_folder": str(tmp_path / "cehrbert_data" / "patient_sequence"),
        "cohort_folder": str(cohort_dir),
        "dataset_prepared_path": str(tmp_path / "finetune_prepared"),
        "output_dir": str(tmp_path / "finetune_results"),
        "observation_window": 365,
        "validation_split_percentage": 0.1,
        "do_train": True,
        "do_eval": True,
        "do_predict": False,  # prediction is a separate step (step 9) on the held-out test cohort
        "num_train_epochs": 20,
        "learning_rate": 2e-05,
        "per_device_train_batch_size": 16,
        "vocab_size": 50000,
        "max_position_embeddings": 512,
        "num_hidden_layers": 6,
        "weight_decay": 0.01,
    }
    path = tmp_path / "finetune.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path
