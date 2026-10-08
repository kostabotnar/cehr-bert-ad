"""Tests for cehrbert_model/12_token_stats.py: token counts of patient_sequence and the windowed dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_pct_over_counts_strictly_greater(step):
    mod = step("cehrbert_model/12_token_stats.py")
    assert mod._pct_over(np.asarray([1, 2, 3, 4]), 2) == 0.5
    assert mod._pct_over(np.asarray([]), 2) == 0.0


def test_full_history_lengths_match_concept_counts(step, spark_sequences):
    mod = step("cehrbert_model/12_token_stats.py")
    lengths = mod._full_history_token_lengths(spark_sequences)
    assert len(lengths) == 12
    assert int(lengths.min()) == 0  # person 12 has an empty sequence
    assert int(lengths.max()) > 4


def test_full_history_lengths_fail_without_parquet(step, tmp_path):
    mod = step("cehrbert_model/12_token_stats.py")
    with pytest.raises(FileNotFoundError):
        mod._full_history_token_lengths(tmp_path)


def test_windowed_lengths_read_input_ids_per_split(step, tmp_path):
    datasets = pytest.importorskip("datasets")
    mod = step("cehrbert_model/12_token_stats.py")
    base = tmp_path / "finetune_prepared" / "x_hash"
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict({"input_ids": [[1, 2, 3], [1]]}),
            "validation": datasets.Dataset.from_dict({"input_ids": [[1, 2]]}),
        }
    ).save_to_disk(str(base))
    per_split = mod._windowed_token_lengths(tmp_path / "finetune_prepared")
    assert per_split["train"].tolist() == [3, 1]
    assert per_split["validation"].tolist() == [2]


def test_main_writes_token_stats_only(step, spark_sequences, tmp_path):
    mod = step("cehrbert_model/12_token_stats.py")
    results, report = tmp_path / "build", tmp_path / "rep"
    rc = mod.main(
        [
            "--patient-sequence-dir",
            str(spark_sequences),
            "--prepared-dir",
            str(tmp_path / "no_such_prepared"),
            "--results-dir",
            str(results),
            "--figures-dir",
            str(results / "figures"),
            "--report-dir",
            str(report),
            "--ctx",
            "8",
        ]
    )
    assert rc == 0  # a missing windowed dataset is a warning
    payload = json.loads((results / "token_stats.json").read_text(encoding="utf-8"))
    assert payload["full_history_tokens"]["count"] == 12
    assert "history_days" not in payload
    assert (results / "figures" / "token_length_hist.png").is_file()
    assert not (results / "figures" / "history_length_hist.png").exists()
