"""Step 02 writes readable concept names (not ids) in its tables and figure."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import pytest

pytest.importorskip("cehrbert")

from pipeline.concept_labels import ConceptLabels  # noqa: E402


@pytest.fixture
def mod(step):
    return step("feature_importance/02_attention_analysis.py")


@pytest.fixture
def labels(tmp_path):
    path = tmp_path / "concept.parquet"
    pd.DataFrame(
        {
            "concept_id": [439147, 8527],
            "concept_name": ["Amnesia", "White"],
            "vocabulary_id": ["SNOMED", "Race"],
            "concept_code": ["48167000", "5"],
        }
    ).to_parquet(path)
    return ConceptLabels.load(path)


def _accumulators(tokens):
    acc = {m: defaultdict(float) for m in ("raw", "rollout", "gradxatt")}
    for i, tok in enumerate(tokens):
        for m in acc:
            acc[m][tok] = 1.0 / (i + 1)
    return acc, {tok: 1 for tok in tokens}


def test_group_csv_resolves_concept_ids_and_structural_tokens(mod, labels, tmp_path):
    tokens = ["439147", "D12", "i-LT", "Race/8527", "year:2015", "999"]
    acc, present = _accumulators(tokens)
    _, df, _ = mod._write_group_csv(tmp_path, "pred_pos", acc, present, 3, labels, top_k=3)
    names = dict(zip(df["concept_id"].astype(str), df["concept_name"]))
    assert names["439147"] == "Amnesia"
    assert names["D12"] == "time gap: 12 days between visits"
    assert names["i-LT"] == "time gap: long-term (1080 days or more) within an inpatient visit"
    assert names["Race/8527"] == "race: White"
    assert names["year:2015"] == "year of first visit: 2015"
    assert names["999"] == "999"
    assert (tmp_path / "concept_attribution_pred_pos.csv").is_file()


def test_figure_labels_use_names_without_concept_ids(mod, labels, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    tokens = ["439147", "D12"]
    acc, present = _accumulators(tokens)
    _, _, tops = mod._write_group_csv(tmp_path, "pred_pos", acc, present, 3, labels, top_k=2)
    seen = []
    from matplotlib.axes import Axes

    original = Axes.set_yticklabels

    def spy(self, labs, *args, **kwargs):
        seen.append(list(labs))
        return original(self, labs, *args, **kwargs)

    Axes.set_yticklabels = spy
    try:
        fig_path = tmp_path / "fig" / "attention_top_concepts.png"
        mod._plot_top_concepts(tops, 3, fig_path, top_k=2)
    finally:
        Axes.set_yticklabels = original
    assert fig_path.is_file()
    flat = [label for labs in seen for label in labs]
    assert "Amnesia" in flat and "time gap: 12 days between visits" in flat
    assert not any("(439147)" in label for label in flat)
