"""Shared concept label lookup: OMOP ids, race/gender placeholders and CEHR-BERT structural tokens."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.concept_labels import ConceptLabels, numeric_ids

ROWS = [
    (0, "No matching concept", "None", "No matching concept"),
    (439147, "Amnesia", "SNOMED", "48167000"),
    (715997, "donepezil", "RxNorm", "135447"),
    (8527, "White", "Race", "5"),
    (8515, "Asian", "Race", "2"),
    (8507, "MALE", "Gender", "M"),
    (9203, "Emergency Room Visit", "Visit", "ER"),
    (8536, "Home", "Visit", "Home"),
]


@pytest.fixture
def concept_path(tmp_path):
    path = tmp_path / "concept.parquet"
    pd.DataFrame(ROWS, columns=["concept_id", "concept_name", "vocabulary_id", "concept_code"]).assign(
        domain_id="x"
    ).to_parquet(path)
    return path


@pytest.fixture
def labels(concept_path):
    return ConceptLabels.load(concept_path)


def test_numeric_ids_and_names(labels):
    assert labels.concept_name("439147") == "Amnesia"
    assert labels.concept_name(715997) == "donepezil"
    assert labels.concept_name("123") is None
    assert labels.token_label("439147") == "Amnesia"
    assert labels.token_label("123") == "123"


def test_source_code_is_vocabulary_and_code(labels):
    assert labels.source_code("439147") == "SNOMED:48167000"
    assert labels.source_code("715997") == "RxNorm:135447"
    assert labels.source_code("0") is None
    assert labels.feature_source_code("cond:439147") == "SNOMED:48167000"
    assert labels.feature_source_code("715997") == "RxNorm:135447"
    assert labels.feature_source_code("cond:999") == "cond:999"
    assert labels.feature_source_code("age") == "age"


def test_load_filters_to_requested_ids(concept_path):
    sub = ConceptLabels.load(concept_path, ids=[439147])
    assert len(sub) == 1 and sub.concept_name(715997) is None


def test_load_missing_file_or_empty_ids_gives_empty_labels(tmp_path, concept_path):
    assert not ConceptLabels.load(tmp_path / "missing.parquet")
    assert not ConceptLabels.load(concept_path, ids=[])
    assert ConceptLabels().token_label("D5").startswith("time gap")


def test_load_accepts_a_table_with_only_names(tmp_path):
    path = tmp_path / "names.parquet"
    pd.DataFrame({"concept_id": [1], "concept_name": ["x"]}).to_parquet(path)
    sub = ConceptLabels.load(path)
    assert sub.concept_name(1) == "x" and sub.source_code(1) is None


@pytest.mark.parametrize(
    "feature, expected",
    [
        ("cond:439147", "cond: Amnesia"),
        ("drug:715997", "drug: donepezil"),
        ("race:8527", "race: White"),
        ("race:8515", "race: Asian"),
        ("race:0", "race: unknown"),
        ("gender:8507", "gender: MALE"),
        ("gender:0", "gender: unknown"),
        ("cond:999", "cond:999"),
        ("age", "age at index date (years)"),
        ("history_length_days", "record length before index date (days)"),
        ("other", "other"),
    ],
)
def test_feature_label(labels, feature, expected):
    assert labels.feature_label(feature) == expected


def test_unlisted_prefix_with_a_resolvable_id_is_labelled(labels):
    assert labels.feature_label("visit:9203") == "visit: Emergency Room Visit"


@pytest.mark.parametrize(
    "token, expected",
    [
        ("D0", "time gap: 0 days between visits"),
        ("D1", "time gap: 1 day between visits"),
        ("D273", "time gap: 273 days between visits"),
        ("i-D7", "time gap: 7 days within an inpatient visit"),
        ("W2", "time gap: 2 weeks between visits"),
        ("M3", "time gap: 3 months between visits"),
        ("LT", "time gap: long-term (1080 days or more) between visits"),
        ("i-LT", "time gap: long-term (1080 days or more) within an inpatient visit"),
        ("year:2019", "year of first visit: 2019"),
        ("age:67", "age at first visit: 67"),
        ("Gender/0", "gender: unknown"),
        ("Gender/8507", "gender: MALE"),
        ("Race/0", "race: unknown"),
        ("Race/8527", "race: White"),
        ("Visit/0", "visit type: unknown"),
        ("Visit/9203", "visit type: Emergency Room Visit"),
        ("Visit/ERIP", "visit type: Emergency Room and Inpatient Visit"),
        ("Discharge/0", "discharge destination: unknown"),
        ("Discharge/8536", "discharge destination: Home"),
        ("Discharge/9999", "discharge destination: 9999"),
        ("Visit/XYZ", "visit type: XYZ"),
        ("[CLS]", "[CLS]"),
        ("something-else", "something-else"),
    ],
)
def test_token_label(labels, token, expected):
    assert labels.token_label(token) == expected


def test_describe(labels):
    assert labels.describe("cond:439147") == "Amnesia"
    assert labels.describe("race:0") == "unknown race"
    assert labels.describe("age") == "age at index date (years)"
    assert labels.describe("Gender/0") == "gender: unknown"
    assert labels.describe("cond:999") is None
    assert labels.describe("mystery") is None


def test_numeric_ids_helper():
    got = numeric_ids(["cond:1", "race:0", "Race/8527", "715997", "D5", "age", "age:67", "year:2019", "i-LT"])
    assert got == {1, 0, 8527, 715997}
