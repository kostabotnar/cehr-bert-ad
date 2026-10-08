"""Tests for pretraining without the test patients: cehrbert_model steps 03, 05 (preflight), 06 and 07."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml

from pipeline import pretrain_data as pd_

TEST_IDS = {2, 5, 9, 10, 11}
VAL_IDS = {3, 7}


def _status(report_dir: Path) -> dict:
    return json.loads((report_dir / "status.json").read_text(encoding="utf-8"))


def _splits_file(tmp_path: Path, persons=range(1, 13), test=TEST_IDS, val=VAL_IDS) -> Path:
    persons = list(persons)
    split = ["test" if p in test else "val" if p in val else "train" for p in persons]
    path = tmp_path / "splits.parquet"
    pl.DataFrame(
        {
            "person_id": persons,
            "index_date": [dt.date(2020, 1, 1)] * len(persons),
            "label": [p % 2 for p in persons],
            "split": split,
        }
    ).write_parquet(path)
    return path


# --------------------------------------------------------------------------- #
# cehrbert_model step 01: staging folder for cehrbert_data
# --------------------------------------------------------------------------- #
def _omop_tables(root: Path, names) -> Path:
    for name in names:
        (root / name).mkdir(parents=True)
        pl.DataFrame({"person_id": [1]}).write_parquet(root / name / "data.parquet")
    return root


def test_stage_omop_input_links_tables_and_replaces_stale_links(tmp_path):
    omop = _omop_tables(tmp_path / "omop", ["person", "concept", "measurement"])
    # CEHR-BERT outputs and stray folders that an earlier run left in the OMOP folder are not tables.
    _omop_tables(omop, ["processed_measurement", "qualified_concept_list"])
    (omop / "notes").mkdir()
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "stale").symlink_to(omop / "person", target_is_directory=True)
    (staging / "qualified_concept_list").symlink_to(omop / "concept", target_is_directory=True)

    tables = pd_.stage_omop_input(omop, staging)

    assert tables == ["concept", "measurement", "person"]
    assert sorted(p.name for p in staging.iterdir()) == tables
    assert all((staging / t).is_symlink() and (staging / t).resolve() == (omop / t).resolve() for t in tables)
    pd_.stage_omop_input(omop, staging)  # idempotent
    assert sorted(p.name for p in staging.iterdir()) == tables


def test_stage_omop_input_keeps_cehrbert_output_and_never_writes_to_omop(tmp_path):
    omop = _omop_tables(tmp_path / "omop", ["person"])
    staging = tmp_path / "stage"
    pd_.stage_omop_input(omop, staging)
    (staging / "processed_measurement").mkdir()  # what cehrbert_data writes into its input folder
    pd_.stage_omop_input(omop, staging)
    assert (staging / "processed_measurement").is_dir() and not (staging / "processed_measurement").is_symlink()
    assert sorted(p.name for p in omop.iterdir()) == ["person"]


def test_stage_omop_input_refuses_to_replace_a_real_folder(tmp_path):
    omop = _omop_tables(tmp_path / "omop", ["person"])
    staging = tmp_path / "stage"
    (staging / "person").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        pd_.stage_omop_input(omop, staging)


# --------------------------------------------------------------------------- #
# cehrbert_model step 03: pretraining input without the test persons
# --------------------------------------------------------------------------- #
def test_step03_removes_exactly_the_test_persons(step, spark_sequences, tmp_path):
    mod = step("cehrbert_model/03_build_pretraining_input.py")
    splits = _splits_file(tmp_path)
    out = tmp_path / "cehrbert_data" / "patient_sequence_pretrain"
    rep = tmp_path / "rep03b"
    rc = mod.main(
        [
            "--patient-sequence-dir", str(spark_sequences),
            "--splits-file", str(splits),
            "--output-dir", str(out),
            "--report-dir", str(rep),
        ]
    )  # fmt: skip
    assert rc == 0
    metrics = _status(rep)["metrics"]
    assert (metrics["sequences_total"], metrics["sequences_removed"], metrics["sequences_remaining"]) == (12, 5, 7)

    src_files = pd_.sequence_files(spark_sequences)
    assert [f.name for f in pd_.sequence_files(out)] == [f.name for f in src_files]
    for src in src_files:
        original = pq.read_table(src)
        written = pq.read_table(out / src.name)
        assert written.schema.equals(original.schema, check_metadata=False)
        assert written.schema.metadata[b"org.apache.spark.version"] == b"3.1.2"
        assert pq.ParquetFile(out / src.name).metadata.row_group(0).column(0).compression == "SNAPPY"
        keep = [p not in TEST_IDS for p in original.column("person_id").to_pylist()]
        # repr compares the NaN values in number_as_values, which == does not.
        assert repr(written.to_pylist()) == repr(original.filter(keep).to_pylist())
    assert pd_.sequence_person_ids(out, require_concepts=False) == set(range(1, 13)) - TEST_IDS
    assert not (out.with_name(out.name + ".tmp")).exists()


def test_step03_output_loads_like_the_source(spark_sequences, tmp_path):
    pytest.importorskip("datasets")
    from cehrbert.runners.runner_util import load_parquet_as_dataset

    out = tmp_path / "pretrain_input"
    # Persons 9..12 fill the third file, which is then skipped: datasets cannot read an empty one.
    counts = pd_.write_sequences_without(spark_sequences, out, {9, 10, 11, 12})
    assert (counts["files"], counts["files_written"]) == (3, 2)
    assert not (out / pd_.sequence_files(spark_sequences)[2].name).exists()
    dataset = load_parquet_as_dataset(str(out))
    assert sorted(dataset["person_id"]) == list(range(1, 9))
    source = load_parquet_as_dataset(str(spark_sequences))
    assert dataset.features == source.features


def test_step03_rerun_replaces_the_output(spark_sequences, tmp_path):
    out = tmp_path / "pretrain_input"
    pd_.write_sequences_without(spark_sequences, out, {1})
    (out / "stale.parquet").write_bytes(b"x")
    counts = pd_.write_sequences_without(spark_sequences, out, {2, 3})
    assert counts["rows_removed"] == 2
    assert not (out / "stale.parquet").exists()
    assert pd_.sequence_person_ids(out, require_concepts=False) == set(range(1, 13)) - {2, 3}


def test_step03_fails_without_splits(step, spark_sequences, tmp_path):
    mod = step("cehrbert_model/03_build_pretraining_input.py")
    rc = mod.main(
        [
            "--patient-sequence-dir", str(spark_sequences),
            "--splits-file", str(tmp_path / "missing.parquet"),
            "--output-dir", str(tmp_path / "out"),
            "--report-dir", str(tmp_path / "rep"),
        ]
    )  # fmt: skip
    assert rc == 1
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- #
# Test-exclusion check on the pretraining prepared dataset (cehrbert_model step 06)
# --------------------------------------------------------------------------- #
def _dataset_dict(path: Path, train, validation) -> Path:
    datasets = pytest.importorskip("datasets")
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict({"person_id": list(train), "concept_ids": [[1]] * len(train)}),
            "validation": datasets.Dataset.from_dict(
                {"person_id": list(validation), "concept_ids": [[1]] * len(validation)}
            ),
        }
    ).save_to_disk(str(path))
    return path


def test_leak_check_passes_without_test_persons(tmp_path):
    from pipeline.reporting import StepReport

    ds = _dataset_dict(tmp_path / "ds", [1, 3, 4], [6])
    report = StepReport("x")
    assert pd_.add_test_exclusion_check(report, ds, _splits_file(tmp_path))
    assert report.passed


@pytest.mark.parametrize("train,validation", [([1, 2, 3], [4]), ([1, 3], [4, 9])])
def test_leak_check_fails_with_a_test_person(tmp_path, train, validation):
    from pipeline.reporting import StepReport

    ds = _dataset_dict(tmp_path / "ds", train, validation)
    report = StepReport("x")
    assert not pd_.add_test_exclusion_check(report, ds, _splits_file(tmp_path))
    assert not report.passed
    assert pd_.leaked_person_ids(ds, TEST_IDS) == {"train": int(2 in train), "validation": int(9 in validation)}


def test_step05_fails_when_a_test_person_was_pretrained_on(step, model_dir, tmp_path):
    mod = step("cehrbert_model/06_verify_pretraining.py")
    base = tmp_path / "pretrain_prepared"
    _dataset_dict(base / "patient_sequence_pretrain_abc", [1, 2, 3], [4])
    rep = tmp_path / "rep05"
    args = ["--results-dir", str(model_dir), "--prepared-dir", str(base), "--report-dir", str(rep)]
    assert mod.main(args + ["--splits-file", str(_splits_file(tmp_path))]) == 1
    assert _status(rep)["metrics"]["test_persons_in_train"] == 1
    assert mod.main(args + ["--splits-file", str(_splits_file(tmp_path, test={9}))]) == 0


def test_step05_default_prepared_dir_comes_from_the_config(step, model_dir, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    mod = step("cehrbert_model/06_verify_pretraining.py")
    cfg = _pretrain_config(tmp_path, tmp_path / "seq_pretrain", model_dir)
    prepared = pd_.pretrain_prepared_path(cfg, mod.paths.resolve_under_root)
    assert prepared.parent == tmp_path / "pretrain_prepared"
    assert prepared.name.startswith("seq_pretrain_")
    _dataset_dict(prepared, [1, 3], [4])
    # A second, older prepared dataset in the same folder must not make the check ambiguous.
    _dataset_dict(tmp_path / "pretrain_prepared" / "patient_sequence_old", [2], [5])
    rep = tmp_path / "rep05"
    rc = mod.main(
        ["--results-dir", str(model_dir), "--config", str(cfg), "--splits-file", str(_splits_file(tmp_path)),
         "--report-dir", str(rep)]
    )  # fmt: skip
    assert rc == 0
    assert _status(rep)["metrics"]["prepared_dir"] == str(prepared)


def test_prepared_path_hash_changes_with_keep_demographic_prompt(tmp_path, model_dir):
    pytest.importorskip("transformers")
    off = _pretrain_config(tmp_path / "a", tmp_path / "seq", model_dir, keep_demographic_prompt=False)
    on = _pretrain_config(tmp_path / "b", tmp_path / "seq", model_dir, keep_demographic_prompt=True)
    for cfg in (off, on):
        text = yaml.safe_load(cfg.read_text())
        text["dataset_prepared_path"] = str(tmp_path / "pp")
        cfg.write_text(yaml.safe_dump(text))
    a = pd_.pretrain_prepared_path(off, Path)
    b = pd_.pretrain_prepared_path(on, Path)
    assert a.parent == b.parent and a != b


# --------------------------------------------------------------------------- #
# cehrbert_model step 05 preflight: input excludes test persons, no stale artifacts
# --------------------------------------------------------------------------- #
def _pretrain_config(tmp_path: Path, data_folder: Path, model_dir: Path, **extra) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_name_or_path": str(model_dir),
        "tokenizer_name_or_path": str(model_dir),
        "data_folder": str(data_folder),
        "dataset_prepared_path": str(tmp_path / "pretrain_prepared"),
        "output_dir": str(model_dir),
        "do_train": True,
        "validation_split_percentage": 0.05,
        "preprocessing_num_workers": 1,
        "preprocessing_batch_size": 4,
        "keep_demographic_prompt": True,
        "max_position_embeddings": 32,
        **extra,
    }
    path = tmp_path / "pretrain.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, cwd=None, env=None):
        self.calls.append(cmd)
        return 0


def _run04(step, cfg, splits, tmp_path, monkeypatch):
    mod = step("cehrbert_model/05_pretrain.py")
    monkeypatch.setattr(mod.paths, "PRETRAIN_PREPARED_DIR", tmp_path / "pp")
    monkeypatch.setattr(mod.paths, "PRETRAIN_RESULTS_DIR", tmp_path / "pr")
    rec = _Recorder()
    monkeypatch.setattr(mod.launch, "run_command", rec)
    rep = tmp_path / "rep04"
    rc = mod.main(["--config", str(cfg), "--splits-file", str(splits), "--report-dir", str(rep)])
    return rc, rec, rep


def test_step04_refuses_input_with_test_persons(step, spark_sequences, tmp_path, monkeypatch):
    cfg = _pretrain_config(tmp_path, spark_sequences, tmp_path / "results")
    rc, rec, rep = _run04(step, cfg, _splits_file(tmp_path), tmp_path, monkeypatch)
    assert rc == 1 and rec.calls == []
    checks = {c["name"]: c["ok"] for c in _status(rep)["checks"]}
    assert checks["pretraining input holds no test person"] is False


def test_step04_refuses_stale_artifacts(step, spark_sequences, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    splits = _splits_file(tmp_path)
    results = tmp_path / "results"
    (results / "checkpoint-10").mkdir(parents=True)
    (results / "tokenizer.json").write_text("{}")
    old = 1_000_000_000
    for path in (results / "checkpoint-10", results / "tokenizer.json"):
        os.utime(path, (old, old))
    data = tmp_path / "seq_pretrain"
    pd_.write_sequences_without(spark_sequences, data, TEST_IDS)
    cfg = _pretrain_config(tmp_path, data, results)

    rc, rec, rep = _run04(step, cfg, splits, tmp_path, monkeypatch)
    assert rc == 1 and rec.calls == []
    detail = {c["name"]: c["detail"] for c in _status(rep)["checks"]}["no stale pretraining artifacts"]
    assert "checkpoint-10" in detail and "tokenizer.json" in detail

    # Artifacts written after the input (an interrupted run being resumed) are fine.
    for path in (results / "checkpoint-10", results / "tokenizer.json"):
        os.utime(path, None)
    rc, rec, _ = _run04(step, cfg, splits, tmp_path, monkeypatch)
    assert rc == 0 and len(rec.calls) == 1


def test_stale_prepared_dataset_is_reported(spark_sequences, tmp_path):
    pytest.importorskip("transformers")
    cfg = _pretrain_config(tmp_path, spark_sequences, tmp_path / "results")
    prepared = pd_.pretrain_prepared_path(cfg, Path)
    prepared.mkdir(parents=True)
    os.utime(prepared, (1_000_000_000, 1_000_000_000))
    assert pd_.stale_pretrain_artifacts(cfg, Path) == [prepared]


# --------------------------------------------------------------------------- #
# cehrbert_model step 07: full tokenized dataset
# --------------------------------------------------------------------------- #
@pytest.fixture
def trained_tokenizer_dir(spark_sequences, tmp_path):
    """A tokenizer trained, as the pretrain runner does, on the input without the test persons."""
    pytest.importorskip("datasets")
    from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
    from cehrbert.runners.runner_util import load_parquet_as_dataset

    data = tmp_path / "seq_pretrain"
    pd_.write_sequences_without(spark_sequences, data, TEST_IDS)
    results = tmp_path / "results"
    cfg = _pretrain_config(tmp_path, data, results)
    data_args, _ = pd_.parse_pretrain_args(cfg)
    tokenizer = CehrBertTokenizer.train_tokenizer(
        load_parquet_as_dataset(str(data)), feature_names=["concept_ids"], concept_name_mapping={}, data_args=data_args
    )
    tokenizer.save_pretrained(str(results))
    return cfg, results


def _runner_processing(cfg: Path, data_folder: Path):
    """What hf_cehrbert_pretrain_runner.main does from the parquet folder to the prepared dataset."""
    from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_pretraining_dataset
    from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
    from cehrbert.runners.runner_util import load_parquet_as_dataset
    from datasets import DatasetDict

    data_args, model_args = pd_.parse_pretrain_args(cfg)
    dataset = load_parquet_as_dataset(str(data_folder), split="train", streaming=False)
    dataset = dataset.train_test_split(test_size=data_args.validation_split_percentage, seed=42)
    dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    tokenizer = CehrBertTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    return create_cehrbert_pretraining_dataset(dataset=dataset, concept_tokenizer=tokenizer, data_args=data_args)


def _rows_by_person(dataset_dict) -> dict:
    rows = {}
    for split in dataset_dict.values():
        for row in split:
            rows[row["person_id"]] = row
    return rows


def test_step07_tokenizes_every_patient_like_the_runner(step, spark_sequences, trained_tokenizer_dir, tmp_path):
    cfg, _ = trained_tokenizer_dir
    mod = step("cehrbert_model/07_tokenize_full_dataset.py")
    splits = _splits_file(tmp_path, persons=range(1, 12))
    out = tmp_path / "full_tokenized"
    rep = tmp_path / "rep05b"
    rc = mod.main(
        [
            "--config", str(cfg),
            "--patient-sequence-dir", str(spark_sequences),
            "--splits-file", str(splits),
            "--output-dir", str(out),
            "--report-dir", str(rep),
        ]
    )  # fmt: skip
    assert rc == 0, _status(rep)["checks"]

    from datasets import load_from_disk

    full = load_from_disk(str(out))
    assert set(full.keys()) == {"train", "validation"}
    assert set(full["validation"]["person_id"]) == VAL_IDS
    # Every person with events is present, the test persons included; person 12 has no events.
    assert set(full["train"]["person_id"]) | VAL_IDS == set(range(1, 12))
    assert TEST_IDS <= set(full["train"]["person_id"])

    runner = _runner_processing(cfg, spark_sequences)
    expected = _rows_by_person(runner)
    got = _rows_by_person(full)
    assert set(got) == set(expected)
    assert full["train"].column_names == runner["train"].column_names
    for person_id, row in expected.items():
        assert got[person_id]["input_ids"] == row["input_ids"], person_id
        for column in ("concept_ids", "ages", "dates", "visit_concept_orders", "labels", "epoch_times"):
            assert got[person_id][column] == row[column], (person_id, column)


def test_step07_fails_without_a_tokenizer(step, spark_sequences, tmp_path):
    mod = step("cehrbert_model/07_tokenize_full_dataset.py")
    cfg = _pretrain_config(tmp_path, spark_sequences, tmp_path / "no_results")
    rc = mod.main(
        [
            "--config", str(cfg),
            "--patient-sequence-dir", str(spark_sequences),
            "--splits-file", str(_splits_file(tmp_path)),
            "--output-dir", str(tmp_path / "full_tokenized"),
            "--report-dir", str(tmp_path / "rep"),
        ]
    )  # fmt: skip
    assert rc == 1
    assert not (tmp_path / "full_tokenized").exists()


def test_step07_fails_when_a_cohort_person_has_no_sequence(step, spark_sequences, trained_tokenizer_dir, tmp_path):
    cfg, _ = trained_tokenizer_dir
    mod = step("cehrbert_model/07_tokenize_full_dataset.py")
    rep = tmp_path / "rep"
    rc = mod.main(
        [
            "--config", str(cfg),
            "--patient-sequence-dir", str(spark_sequences),
            "--splits-file", str(_splits_file(tmp_path, persons=range(1, 14))),
            "--output-dir", str(tmp_path / "full_tokenized"),
            "--report-dir", str(rep),
        ]
    )  # fmt: skip
    assert rc == 1
    metrics = _status(rep)["metrics"]
    # Person 12 has an empty sequence and person 13 none at all; both are train persons here.
    assert metrics["cohort_train_missing"] == 2


def test_full_tokenized_dataset_feeds_extract_cohort_sequences(spark_sequences, trained_tokenizer_dir, tmp_path):
    from cehrbert.runners.data_utils import extract_cohort_sequences
    from cehrbert.runners.hf_runner_argument_dataclass import CehrBertArguments, DataTrainingArguments

    cfg, _ = trained_tokenizer_dir
    out = tmp_path / "full_tokenized"
    pd_.tokenize_full_dataset(cfg, spark_sequences, out, VAL_IDS, Path)
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    cohort_ids = [2, 3, 5, 7, 8]
    pl.DataFrame(
        {"person_id": cohort_ids, "index_date": [dt.datetime(2060, 1, 1)] * 5, "label": [1, 0, 1, 0, 1]}
    ).write_parquet(cohort_dir / "cohort.parquet")
    data_args = DataTrainingArguments(
        data_folder="x",
        dataset_prepared_path="y",
        cohort_folder=str(cohort_dir),
        observation_window=0,
        preprocessing_num_workers=1,
        preprocessing_batch_size=4,
        keep_demographic_prompt=True,
    )
    processed = extract_cohort_sequences(data_args, CehrBertArguments(tokenized_full_dataset_path=str(out)))
    assert set(processed["validation"]["person_id"]) == {3, 7}
    assert set(processed["train"]["person_id"]) == {2, 5, 8}
