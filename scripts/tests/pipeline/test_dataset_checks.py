"""Tests for pipeline.dataset_checks.

Resolution logic is testable without `datasets`; the load/split/carve tests require it
and skip when it is unavailable.
"""

from __future__ import annotations

import pytest

from pipeline import dataset_checks as dc
from pipeline.reporting import StepReport


def test_resolve_prepared_path_direct(tmp_path):
    (tmp_path / "dataset_dict.json").write_text("{}", encoding="utf-8")
    assert dc.resolve_prepared_path(tmp_path) == str(tmp_path)


def test_resolve_prepared_path_single_subdir(fake_tokenized_dataset):
    resolved = dc.resolve_prepared_path(fake_tokenized_dataset)
    assert resolved is not None
    assert resolved.endswith("omop_abc123")


def test_resolve_prepared_path_ambiguous_returns_none(tmp_path):
    for name in ("a", "b"):
        sub = tmp_path / name
        sub.mkdir()
        (sub / "dataset_dict.json").write_text("{}", encoding="utf-8")
    assert dc.resolve_prepared_path(tmp_path) is None


def test_prepared_checks_without_datasets_is_structural(fake_tokenized_dataset, monkeypatch):
    # Force the "datasets not available" branch regardless of the host env.
    monkeypatch.setattr(dc, "datasets_available", lambda: False)
    r = StepReport("t")
    dc.add_prepared_dataset_checks(r, fake_tokenized_dataset)
    # Structural checks pass; content verification is a warning.
    assert r.passed is True
    assert any(c.name == "dataset contents verified" and not c.ok for c in r.checks)


# --- datasets-dependent tests ---------------------------------------------
def _make_dataset_dict(base, n_train=4, n_val=2, n_test=0):
    from datasets import Dataset, DatasetDict

    pid = iter(range(1, 10_000))
    splits = {
        "train": Dataset.from_dict({"person_id": [next(pid) for _ in range(n_train)]}),
        "validation": Dataset.from_dict({"person_id": [next(pid) for _ in range(n_val)]}),
    }
    if n_test:
        splits["test"] = Dataset.from_dict({"person_id": [next(pid) for _ in range(n_test)]})
    sub = base / "omop_hash"
    DatasetDict(splits).save_to_disk(str(sub))
    return sub



def test_prepared_checks_with_datasets(tmp_path):
    pytest.importorskip("datasets")
    base = tmp_path / "prepared"
    base.mkdir()
    _make_dataset_dict(base)
    r = StepReport("t")
    dc.add_prepared_dataset_checks(r, base, expected_splits=["train", "validation"], require_person_id=True)
    assert r.status == "pass"
    assert r.metrics["train_rows"] == 4


def test_count_person_ids(tmp_path):
    pytest.importorskip("datasets")
    base = tmp_path / "prepared"
    base.mkdir()
    _make_dataset_dict(base)
    assert len(dc.count_person_ids(base)) == 6


def test_collapse_to_test_split(tmp_path):
    pytest.importorskip("datasets")
    from datasets import load_from_disk

    base = tmp_path / "prepared"
    base.mkdir()
    _make_dataset_dict(base, n_train=4, n_val=2)  # train+validation, no test

    patched = dc.collapse_to_test_split(base)
    assert patched == ["omop_hash"]

    ds = load_from_disk(str(base / "omop_hash"))
    assert set(ds.keys()) == {"test"}
    assert ds["test"].num_rows == 6  # all rows pooled into test

    # Idempotent: already exactly {"test"} -> no-op.
    assert dc.collapse_to_test_split(base) == []


