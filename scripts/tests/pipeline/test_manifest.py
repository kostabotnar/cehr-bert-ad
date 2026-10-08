"""Tests for pipeline/manifest.py: entry paths, validation and upsert semantics."""

from __future__ import annotations

import json

import pytest

from pipeline import manifest as mf
from pipeline import paths


def _entry(name, label=None, secondary=False):
    return mf.make_entry(name, label or name.upper(), "fam", f"/x/{name}/t", f"/x/{name}/v", secondary)


def test_to_manifest_path_is_relative_inside_the_project_and_absolute_outside(tmp_path):
    inside = paths.PROJECT_ROOT / "data" / "finetune_results" / "test_predictions"
    assert mf.to_manifest_path(inside) == "data/finetune_results/test_predictions"
    assert mf.to_manifest_path("data/x") == "data/x"
    assert mf.to_manifest_path(tmp_path / "p") == str(tmp_path / "p")


def test_read_manifest_resolves_paths(tmp_path):
    path = tmp_path / "m.json"
    mf.upsert_models(
        path,
        [mf.make_entry("a", "A", "f", paths.PROJECT_ROOT / "data" / "t", tmp_path / "v"), _entry("b", secondary=True)],
    )
    models = mf.read_manifest(path)
    assert [m.name for m in models] == ["a", "b"]
    assert models[0].test_predictions == paths.PROJECT_ROOT / "data" / "t"
    assert models[0].validation_predictions == tmp_path / "v"
    assert models[1].secondary is True


def test_upsert_replaces_by_name_keeps_order_and_others(tmp_path):
    path = tmp_path / "sub" / "m.json"
    mf.upsert_models(path, [_entry("a"), _entry("b")])
    mf.upsert_models(path, [_entry("b", label="NEW", secondary=True), _entry("c")])
    data = json.loads(path.read_text())
    assert [m["name"] for m in data["models"]] == ["a", "b", "c"]
    assert data["models"][1]["display_label"] == "NEW" and data["models"][1]["secondary"] is True


def test_read_rejects_missing_keys_and_duplicates(tmp_path):
    path = tmp_path / "m.json"
    good = _entry("a")
    path.write_text(json.dumps({"models": [{k: v for k, v in good.items() if k != "family"}]}))
    with pytest.raises(mf.ManifestError, match="family"):
        mf.read_manifest(path)
    path.write_text(json.dumps({"models": [good, good]}))
    with pytest.raises(mf.ManifestError, match="duplicate"):
        mf.read_manifest(path)


@pytest.mark.parametrize("text", ["not json", "[]", '{"models": 3}'])
def test_read_rejects_malformed_files(tmp_path, text):
    path = tmp_path / "m.json"
    path.write_text(text)
    with pytest.raises(mf.ManifestError):
        mf.read_manifest(path)
    with pytest.raises(mf.ManifestError):
        mf.read_manifest(tmp_path / "absent.json")
