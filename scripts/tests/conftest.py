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
    write("person", pl.DataFrame({"person_id": [1, 2, 3, 4, 5, 6], "year_of_birth": [1950] * 6}))

    # Cases p1, p2 (anchors 2020, indexes 2019). Controls p3..p5 each have exactly one eligible anchor in 2020;
    # p6 has no eligible anchor. The data cutoff is the latest visit (2023-01-01, p1), so the cap is 2022-01-01.
    write(
        "visit_occurrence",
        pl.DataFrame(
            {
                "person_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6],
                "visit_start_date": [
                    D(2010, 1, 1),
                    D(2018, 6, 1),
                    D(2023, 1, 1),  # p1
                    D(2011, 1, 1),
                    D(2018, 12, 1),
                    D(2021, 6, 1),  # p2
                    D(2012, 1, 1),
                    D(2018, 6, 1),
                    D(2020, 3, 1),
                    D(2022, 6, 1),  # p3
                    D(2010, 1, 1),
                    D(2018, 9, 1),
                    D(2020, 6, 1),
                    D(2022, 1, 1),  # p4
                    D(2015, 1, 1),
                    D(2018, 12, 1),
                    D(2020, 5, 1),
                    D(2022, 6, 1),  # p5
                    D(2016, 1, 1),
                    D(2016, 2, 1),  # p6 (too little history -> excluded)
                ],
            }
        ),
    )

    # Real layout: condition_concept_id is a standard (SNOMED) id; ICD strings are in condition_source_value.
    write(
        "condition_occurrence",
        pl.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "visit_occurrence_id": [1, 1, 1, 1, 1],
                "condition_concept_id": [3001, 3001, 3002, 3002, 3002],
                "condition_source_value": [
                    "ICD-10-CM:G30.9",
                    "ICD-9-CM:331.0",
                    "ICD-10-CM:E11.9",
                    "ICD-10-CM:G31.0",
                    "X-ICD-10-CM:G30",
                ],
                "condition_start_date": [D(2020, 1, 1), D(2020, 6, 1), D(2019, 1, 1), D(2018, 1, 1), D(2019, 6, 1)],
            }
        ),
    )

    write(
        "concept",
        pl.DataFrame(
            {
                "concept_id": [1001, 1002, 2002],
                "vocabulary_id": ["ICD10CM", "ICD9CM", "ICD10CM"],
                "concept_code": ["G30.9", "331.0", "E11.9"],
            }
        ),
    )

    write(
        "drug_exposure",
        pl.DataFrame(
            {
                "person_id": [1, 2, 5],
                "visit_occurrence_id": [1, 1, 1],
                "drug_concept_id": [50, 51, 52],
                "drug_exposure_start_date": [D(2018, 6, 1), D(2019, 1, 1), D(2019, 1, 1)],
            }
        ),
    )
    write(
        "procedure_occurrence",
        pl.DataFrame(
            {
                "person_id": [1, 3],
                "visit_occurrence_id": [1, 1],
                "procedure_concept_id": [60, 61],
                "procedure_date": [D(2015, 1, 1), D(2018, 12, 1)],
            }
        ),
    )
    write(
        "measurement",
        pl.DataFrame(
            {
                "person_id": [1, 4],
                "visit_occurrence_id": [1, 1],
                "measurement_concept_id": [70, 71],
                "measurement_date": [D(2016, 1, 1), D(2018, 9, 1)],
            }
        ),
    )

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


# --------------------------------------------------------------------------
# Spark-style patient_sequence folder (same schema as cehrbert_data writes)
# --------------------------------------------------------------------------
def _sequence_record(person_id: int, n_visits: int) -> dict:
    concepts = ["year:2010", "age:60", "8507" if person_id % 2 else "8532", "8527"] if n_visits else []
    for v in range(n_visits):
        concepts += ["[VS]", "9201", f"c{(person_id * 7 + v) % 11}", "[VE]"]
        if v < n_visits - 1:
            concepts.append(f"W{v % 3}")
    n = len(concepts)
    head = min(n, 4)
    return {
        "cohort_member_id": person_id,
        "person_id": person_id,
        "concept_ids": concepts,
        "visit_segments": [0] * head + [1 + (i // 5) % 2 for i in range(n - head)],
        "orders": list(range(n)),
        "dates": [0] * head + [2100 + i for i in range(n - head)],
        "ages": [-1] * head + [60 + i // 5 for i in range(n - head)],
        "visit_concept_orders": [0] * head + [1 + i // 5 for i in range(n - head)],
        "num_of_visits": n_visits,
        "num_of_concepts": n,
        "concept_value_masks": [0] * n,
        "number_as_values": [float("nan")] * n,
        "concept_as_values": [None] * n,
        "is_numeric_types": [0] * n,
        "mlm_skip_values": [0] * n,
        "priorities": [0.0] * n,
        "visit_concept_ids": [9201] * n,
        "visit_rank_orders": [0] * n,
        "concept_orders": list(range(n)),
        "record_ranks": list(range(n)),
        "units": [None] * n,
        "epoch_times": [0] * head + [(2100 + i) * 7 * 86400 for i in range(n - head)],
    }


SEQUENCE_SCHEMA_FIELDS = [
    ("cohort_member_id", "int64", True), ("person_id", "int64", True),
    ("concept_ids", "string", False), ("visit_segments", "int32", False), ("orders", "int32", False),
    ("dates", "int32", False), ("ages", "int64", False), ("visit_concept_orders", "int32", False),
    ("num_of_visits", "int32", True), ("num_of_concepts", "int64", None),
    ("concept_value_masks", "int32", False), ("number_as_values", "float64", False),
    ("concept_as_values", "string", False), ("is_numeric_types", "int32", False),
    ("mlm_skip_values", "int32", False), ("priorities", "float64", False),
    ("visit_concept_ids", "int64", False), ("visit_rank_orders", "int32", False),
    ("concept_orders", "int32", False), ("record_ranks", "int32", False),
    ("units", "string", False), ("epoch_times", "int64", False),
]  # fmt: skip


def write_spark_sequences(folder: Path, persons_per_file: list[list[int]], n_visits=None) -> Path:
    """Write a Spark-like patient_sequence folder: part files, snappy, Spark schema metadata, _SUCCESS.

    The third field entry is True for a nullable scalar, None for a non-null scalar and False
    for a non-null list column, as in the files cehrbert_data writes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = []
    for name, typ, kind in SEQUENCE_SCHEMA_FIELDS:
        arrow_type = getattr(pa, typ)()
        if kind is False:
            fields.append(pa.field(name, pa.list_(arrow_type), nullable=False))
        else:
            fields.append(pa.field(name, arrow_type, nullable=bool(kind)))
    schema = pa.schema(fields, metadata={b"org.apache.spark.version": b"3.1.2"})
    folder.mkdir(parents=True, exist_ok=True)
    n_visits = n_visits or {}
    for i, persons in enumerate(persons_per_file):
        rows = [_sequence_record(p, n_visits.get(p, 2 + p % 4)) for p in persons]
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, folder / f"part-{i:05d}-abc-c000.snappy.parquet", compression="snappy")
    (folder / "_SUCCESS").write_text("", encoding="utf-8")
    return folder


@pytest.fixture
def spark_sequences(tmp_path: Path) -> Path:
    """Persons 1..12 over three part files; person 12 has an empty sequence (num_of_concepts 0)."""
    return write_spark_sequences(
        tmp_path / "cehrbert_data" / "patient_sequence",
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        n_visits={12: 0},
    )
