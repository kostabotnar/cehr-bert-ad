"""Tests for pipeline/omop_profile.py and step 18 on a small synthetic OMOP dataset."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import polars as pl
import pytest

from pipeline import omop_common as oc
from pipeline import omop_profile as op

D = dt.date


# --------------------------------------------------------------------------
# Synthetic dataset
#   AD cases: 1 (inpatient first AD, proxy and donepezil before), 2 (single ICD-9 row), 3 (AD 90 days
#   after the first visit). Controls: 4 (dies within a year of the last visit), 5 (memantine), 6 (no
#   visits), 7 (age 30), 8 (151 days of history), 9 (clean), 10 (last visit before 1990).
# --------------------------------------------------------------------------
def _write(root: Path, table: str, df: pl.DataFrame) -> None:
    d = root / table
    d.mkdir()
    df.write_parquet(d / "data.parquet")


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    root = tmp_path / "omop"
    root.mkdir()
    birth = {1: 1940, 2: 1935, 3: 1950, 4: 1940, 5: 1938, 6: 1945, 7: 1990, 8: 1941, 9: 1942, 10: 1943}
    ids = sorted(birth)
    _write(
        root,
        "person",
        pl.DataFrame(
            {
                "person_id": ids,
                "gender_concept_id": [8532 if i % 2 else 8507 for i in ids],
                "year_of_birth": [birth[i] for i in ids],
                "month_of_birth": [None] * len(ids),
                "day_of_birth": [None] * len(ids),
                "race_concept_id": [8527 if i < 8 else 0 for i in ids],
                "ethnicity_concept_id": [38003564] * len(ids),
                "location_id": [None] * len(ids),
                "care_site_id": [None] * len(ids),
            },
            schema_overrides={
                "location_id": pl.Int64,
                "care_site_id": pl.Int64,
                "month_of_birth": pl.Int64,
                "day_of_birth": pl.Int64,
            },
        ),
    )
    _write(root, "death", pl.DataFrame({"person_id": [4], "death_date": [D(2020, 6, 1)]}))
    _write(
        root,
        "observation_period",
        pl.DataFrame(
            {
                "person_id": ids,
                "observation_period_start_date": [D(1980, 1, 1)] * len(ids),
                "observation_period_end_date": [D(2020, 1, 1) if i == 9 else D(2026, 1, 1) for i in ids],
            }
        ),
    )
    visits = [
        # (visit_occurrence_id, person_id, date, visit_concept_id)
        (11, 1, D(2010, 1, 1), 9202),
        (12, 1, D(2015, 6, 1), 9202),
        (13, 1, D(2019, 3, 1), 9201),
        (14, 1, D(2019, 4, 1), 9202),
        (21, 2, D(2012, 1, 1), 9202),
        (22, 2, D(2020, 6, 1), 9202),
        (31, 3, D(2018, 1, 1), 9202),
        (32, 3, D(2018, 4, 1), 9202),
        (41, 4, D(2005, 1, 1), 9202),
        (42, 4, D(2019, 6, 1), 0),
        (43, 4, D(2020, 1, 1), 9202),
        (51, 5, D(2010, 1, 1), 9202),
        (52, 5, D(2019, 3, 1), 9202),
        (71, 7, D(2000, 1, 1), 9202),
        (72, 7, D(2020, 1, 1), 9202),
        (81, 8, D(2015, 1, 1), 9202),
        (82, 8, D(2015, 6, 1), 9202),
        (91, 9, D(2008, 1, 1), 9202),
        (92, 9, D(2012, 1, 1), 9203),
        (93, 9, D(2018, 12, 1), 9202),
        (94, 9, D(2019, 12, 31), 9202),
        (101, 10, D(1985, 1, 1), 9202),
        (102, 10, D(1988, 1, 1), 9202),
    ]
    _write(
        root,
        "visit_occurrence",
        pl.DataFrame(
            {
                "visit_occurrence_id": [v[0] for v in visits],
                "person_id": [v[1] for v in visits],
                "visit_start_date": [v[2] for v in visits],
                "visit_concept_id": [v[3] for v in visits],
                "visit_type_concept_id": [32035] * len(visits),
                "provider_id": [v[1] * 10 + (v[0] % 2) for v in visits],
                "care_site_id": [None] * len(visits),
            },
            schema_overrides={"care_site_id": pl.Int64},
        ),
    )
    conds = [
        # (person, concept, source, date, visit id)
        (1, 1, "ICD-10-CM:G30.9", D(2019, 3, 1), 13),
        (1, 1, "ICD-10-CM:G30.0", D(2019, 5, 1), None),
        (1, 2, "ICD-10-CM:R41.3", D(2018, 1, 1), None),
        (1, 3, "ICD-10-CM:I10", D(2015, 6, 1), 12),
        (2, 1, "ICD-9-CM:331.0", D(2020, 6, 1), 22),
        (3, 1, "ICD-10-CM:G30.1", D(2018, 4, 1), 32),
        (4, 3, "ICD-10-CM:I10", D(2019, 6, 1), 42),
        (9, 4, "ICD-10-CM:E11.9", D(2015, 1, 1), None),
        (9, 4, "ICD-10-CM:E11.9", D(2015, 1, 1), None),
        (9, 4, "ICD-10-CM:E11.9", D(2019, 12, 31), 94),
        (8, 5, "X-ICD-10-CM:G30", D(2015, 3, 1), None),
    ]
    _write(
        root,
        "condition_occurrence",
        pl.DataFrame(
            {
                "person_id": [c[0] for c in conds],
                "condition_concept_id": [c[1] for c in conds],
                "condition_source_value": [c[2] for c in conds],
                "condition_start_date": [c[3] for c in conds],
                "condition_type_concept_id": [32020] * len(conds),
                "visit_occurrence_id": [c[4] for c in conds],
            },
            schema_overrides={"visit_occurrence_id": pl.Int64},
        ),
    )
    _write(
        root,
        "concept",
        pl.DataFrame(
            {
                "concept_id": [100, 101, 102],
                "concept_name": ["Donepezil", "Memantine", "Donepezil 5 MG Oral Tablet"],
                "vocabulary_id": ["RxNorm", "RxNorm", "RxNorm"],
                "concept_class_id": ["Ingredient", "Ingredient", "Clinical Drug"],
                "standard_concept": ["S", "S", "S"],
            }
        ),
    )
    _write(
        root,
        "concept_ancestor",
        pl.DataFrame({"ancestor_concept_id": [100, 100, 101, 101], "descendant_concept_id": [100, 1001, 101, 1011]}),
    )
    _write(
        root,
        "drug_exposure",
        pl.DataFrame(
            {
                "person_id": [1, 1, 5, 9],
                "drug_concept_id": [1001, 1001, 1011, 999],
                "drug_exposure_start_date": [D(2018, 2, 1), D(2019, 6, 1), D(2016, 1, 1), D(2015, 1, 1)],
            }
        ),
    )
    _write(
        root,
        "procedure_occurrence",
        pl.DataFrame(
            {
                "person_id": [1, 9],
                "procedure_date": [D(2018, 5, 1), D(2015, 1, 1)],
                "procedure_source_value": ["CPT:70551", "CPT:99213"],
                "procedure_concept_id": [0, 5],
            }
        ),
    )
    _write(
        root,
        "measurement",
        pl.DataFrame(
            {
                "person_id": [1, 9],
                "measurement_date": [D(2018, 5, 1), D(2015, 1, 1)],
                "measurement_source_value": ["72106-8", "2160-0"],
                "measurement_concept_id": [0, 7],
            }
        ),
    )
    _write(
        root,
        "cdm_source",
        pl.DataFrame({"source_release_date": [D(2026, 7, 8)]}),
    )
    return root


def _persons(root: Path):
    persons0 = op.load_persons(
        oc.scan_table(
            root,
            "person",
            [
                "person_id",
                "gender_concept_id",
                "year_of_birth",
                "month_of_birth",
                "day_of_birth",
                "race_concept_id",
                "ethnicity_concept_id",
                "location_id",
                "care_site_id",
            ],
        ),
        oc.scan_table(root, "death", ["person_id", "death_date"]),
        oc.scan_table(
            root, "observation_period", ["person_id", "observation_period_start_date", "observation_period_end_date"]
        ),
    )
    visits = oc.scan_table(
        root,
        "visit_occurrence",
        [
            "visit_occurrence_id",
            "person_id",
            "visit_start_date",
            "visit_concept_id",
            "visit_type_concept_id",
            "provider_id",
            "care_site_id",
        ],
    ).collect()
    cond = (
        oc.scan_table(
            root,
            "condition_occurrence",
            ["person_id", "condition_start_date", "condition_source_value", "visit_occurrence_id"],
        )
        .rename({"condition_start_date": "date", "condition_source_value": "source"})
        .collect()
    )
    ad_rows = cond.filter(pl.col("source").str.contains(oc.AD_SOURCE_REGEX))
    ad_first = op.ad_first_dates(ad_rows)
    persons = op.assign_cohort(persons0, ad_first)
    return persons, visits, op.visit_summary(visits), ad_rows, ad_first


# --------------------------------------------------------------------------
# Code matching
# --------------------------------------------------------------------------
def test_expand_code_range():
    assert op.expand_code_range("E08-E13") == ["E08", "E09", "E10", "E11", "E12", "E13"]
    assert op.expand_code_range("401-405") == ["401", "402", "403", "404", "405"]
    assert op.expand_code_range("G30") == ["G30"]
    with pytest.raises(ValueError):
        op.expand_code_range("E13-E08")


def test_prefix_matching_with_vocabulary():
    rx = op.code_prefix_regex({"ICD-10-CM": ["G30"], "ICD-9-CM": ["331.0"]})
    match = lambda s: pl.Series([s]).str.contains(rx)[0]  # noqa: E731
    assert match("ICD-10-CM:G30.9")
    assert match("ICD-10-CM:G30")
    assert not match("ICD-10-CM:G31.0")
    assert not match("ICD-9-CM:G30.9")  # wrong vocabulary
    assert not match("X-ICD-10-CM:G30")  # anchored at the start
    assert not match("G30.9")  # vocabulary prefix is required
    assert match("ICD-9-CM:331.0")
    assert match("ICD-9-CM:331.00")  # plain prefix semantics: documented, such a code does not exist
    assert not match("ICD-9-CM:331.1")
    assert not match("ICD-9-CM:3310")
    assert match("ICD-10-CM:G300")  # plain prefix semantics: documented, no such ICD-10-CM code exists


def test_range_prefix_regex():
    rx = op.code_prefix_regex({"ICD-10-CM": ["I10-I16"]})
    s = pl.Series(["ICD-10-CM:I10", "ICD-10-CM:I16.9", "ICD-10-CM:I17"])
    assert s.str.contains(rx).to_list() == [True, True, False]


def test_default_config_loads():
    cfg = op.load_concept_config(Path(__file__).resolve().parents[1] / "configs" / "omop_profile_concepts.yaml")
    items = op.regex_items(cfg["conditions"])
    assert "dementia_proxies__mild_cognitive_impairment" in items
    assert "comorbidities__falls" in items
    assert cfg["drugs"]["ad_drugs"][0] == "donepezil"
    assert "72106-8" in op.measurement_items(cfg["measurements"])["labs_vitals__mmse"][1]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_smd_continuous_hand_computed():
    # a: mean 10, sd 2; b: mean 12, sd 4 -> 2 / sqrt((4 + 16) / 2) = 2 / sqrt(10)
    assert oc.smd_continuous(10, 2, 12, 4) == pytest.approx(2 / math.sqrt(10))
    assert oc.smd_continuous(10, 0, 10, 0) == 0.0
    assert oc.smd_continuous(None, 1, 1, 1) is None
    d = op.describe([1, 2, 3, 4, 5])
    assert (
        d["mean"] == 3
        and d["sd"] == pytest.approx(math.sqrt(2.5))
        and d["median"] == 3
        and (d["q25"], d["q75"]) == (2, 4)
    )


def test_smd_binary_hand_computed():
    # p_a = 0.2, p_b = 0.5 -> 0.3 / sqrt((0.16 + 0.25) / 2)
    assert op.smd_binary(0.2, 0.5) == pytest.approx(0.3 / math.sqrt(0.205))
    assert op.smd_binary(0.5, 0.5) == 0.0
    assert op.smd_binary(0.0, 0.0) == 0.0
    assert op.smd_binary(None, 0.5) is None


# --------------------------------------------------------------------------
# Cohorts and index dates
# --------------------------------------------------------------------------
def test_cohort_split_and_ad_index_is_first_ad_date(profile_dir):
    persons, _, _, _, ad_first = _persons(profile_dir)
    cohorts = dict(zip(persons["person_id"], persons["cohort"]))
    assert {p for p, c in cohorts.items() if c == "ad"} == {1, 2, 3}
    assert cohorts[8] == "control"  # "X-ICD-10-CM:G30" does not match the anchored regex
    row1 = ad_first.filter(pl.col("person_id") == 1).row(0, named=True)
    assert row1["first_ad_date"] == D(2019, 3, 1) and row1["n_ad_rows"] == 2 and row1["first_ad_visit_id"] == 13


def test_control_last_visit_index_and_visitless_controls(profile_dir):
    persons, visits, vsum, _, _ = _persons(profile_dir)
    controls = visits.join(
        persons.filter(pl.col("cohort") == "control").select("person_id"), on="person_id", how="semi"
    )
    table, info = op.build_index_table(persons, vsum, controls.select("person_id", "visit_start_date"), seed=1)
    last = table.filter(pl.col("view") == op.VIEW_LAST)
    assert dict(zip(last["person_id"], last["index_date"]))[9] == D(2019, 12, 31)
    assert 6 not in last["person_id"].to_list()
    assert info["n_controls_without_visits"] == 1 and info["n_controls"] == 7
    ad = table.filter(pl.col("view") == op.VIEW_AD)
    assert dict(zip(ad["person_id"], ad["index_date"])) == {1: D(2019, 3, 1), 2: D(2020, 6, 1), 3: D(2018, 4, 1)}


def test_matched_index_is_seeded_and_uses_own_visit_dates(profile_dir):
    persons, visits, vsum, _, _ = _persons(profile_dir)
    controls = visits.join(
        persons.filter(pl.col("cohort") == "control").select("person_id"), on="person_id", how="semi"
    )
    cv = controls.select("person_id", "visit_start_date")
    t1, _ = op.build_index_table(persons, vsum, cv, seed=7)
    t2, _ = op.build_index_table(persons, vsum, cv, seed=7)
    m1 = t1.filter(pl.col("view") == op.VIEW_MATCHED).sort("person_id")
    m2 = t2.filter(pl.col("view") == op.VIEW_MATCHED).sort("person_id")
    assert m1.equals(m2)
    own = {(r["person_id"], r["visit_start_date"]) for r in cv.iter_rows(named=True)}
    assert all((r["person_id"], r["index_date"]) in own for r in m1.iter_rows(named=True))
    assert m1["person_id"].n_unique() == m1.height == 6  # all controls with visits, one pseudo-index each


def test_matched_index_prefers_case_strata():
    # Cases: age 79 in 2020. Control visits: one in the case stratum, one far from it.
    cases = pl.DataFrame({"person_id": [1, 2], "index_date": [D(2020, 4, 1)] * 2, "birth_date": [D(1940, 7, 1)] * 2})
    cv = pl.DataFrame(
        {
            "person_id": [10, 10, 11, 11],
            "visit_start_date": [D(2020, 3, 1), D(2005, 3, 1), D(2020, 5, 1), D(2001, 1, 1)],
            "birth_date": [D(1940, 7, 1)] * 4,
        }
    )
    for seed in range(5):
        picked, fallback = op.pick_matched_index(cases, cv, seed)
        assert fallback == 0
        assert dict(zip(picked["person_id"], picked["index_date"])) == {10: D(2020, 3, 1), 11: D(2020, 5, 1)}


def test_matched_index_ignores_strata_without_cases():
    # 60% of cases index in 2020 and 40% in 2010 (all aged 79 at index). Every control was born in
    # 1941 and has a visit in 2020 (age 79), 2010 (age 69, no cases in that stratum) and 2000.
    n_cases = 100
    years = [2020] * 60 + [2010] * 40
    cases = pl.DataFrame(
        {
            "person_id": list(range(n_cases)),
            "index_date": [D(y, 6, 1) for y in years],
            "birth_date": [D(y - 79, 1, 1) for y in years],
        }
    )
    n_ctl = 400
    rows = [(1000 + i, D(y, 6, 1), D(1941, 1, 1)) for i in range(n_ctl) for y in (2020, 2010, 2000)]
    cv = pl.DataFrame(rows, schema=["person_id", "visit_start_date", "birth_date"], orient="row")
    picked, _ = op.pick_matched_index(cases, cv, seed=11)
    # controls born in 1941 only reach the age band of the 2020 cases, the 2010 cases are not reachable
    assert set(picked["index_date"].dt.year().to_list()) == {2020}


def test_matched_index_fallback_when_no_stratum_matches():
    cases = pl.DataFrame({"person_id": [1], "index_date": [D(2020, 7, 1)], "birth_date": [D(1940, 7, 1)]})
    cv = pl.DataFrame(
        {"person_id": [10, 10], "visit_start_date": [D(2001, 1, 1), D(2002, 1, 1)], "birth_date": [D(1990, 1, 1)] * 2}
    )
    picked, fallback = op.pick_matched_index(cases, cv, 3)
    assert fallback == 1 and picked.height == 1 and picked["index_date"][0] in (D(2001, 1, 1), D(2002, 1, 1))


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def test_visit_windows_are_half_open():
    idx_date = D(2020, 1, 1)
    offsets = [0, 1, 365, 366, 730, 731, 1825, 1826, -5]  # days before the index; 0 and -5 are not history
    visits = pl.DataFrame(
        {
            "person_id": [1] * len(offsets),
            "visit_start_date": [idx_date - dt.timedelta(days=o) for o in offsets],
            "visit_concept_id": [9202] * len(offsets),
            "provider_id": [None] * len(offsets),
            "care_site_id": [None] * len(offsets),
        },
        schema_overrides={"provider_id": pl.Int64, "care_site_id": pl.Int64},
    )
    per, monthly = op.visit_features(visits, pl.DataFrame({"person_id": [1], "index_date": [idx_date]}))
    r = per.row(0, named=True)
    assert r["n_visits_before"] == 7
    assert r["n_w1"] == 2  # 1 and 365 days before: [index - 365, index)
    assert r["n_w2"] == 2  # 366 and 730
    assert r["n_w3"] == 2  # 731 and 1825
    assert r["n_outpatient"] == 7
    m = dict(zip(monthly["month_offset"], monthly["n"]))
    # block m holds days_before in (30 m, 30 (m + 1)]: 1 -> 0, 365 -> 12, 366 -> 12, 730 -> 24, 731 -> 24
    assert m[0] == 1 and m[12] == 2 and m[24] == 2 and 36 not in m and len(m) == 3


def test_event_windows_are_half_open():
    ev = pl.DataFrame(
        {
            "person_id": [1, 1, 1, 1],
            "date": [D(2019, 12, 31), D(2019, 1, 1), D(2018, 12, 31), D(2020, 1, 1)],
            "group": ["g"] * 4,
            "item": ["g__x"] * 4,
        }
    )
    idx = pl.DataFrame({"person_id": [1], "index_date": [D(2020, 1, 1)]})
    assert op.events_before_index(ev, idx).height == 3  # the event on the index date is excluded
    assert op.events_before_index(ev, idx, 365).height == 2  # 2019-01-01 is exactly 365 days before
    assert op.events_before_index(ev, idx, 364).height == 1
    p = op.prevalence(ev, idx, 365, ["g__x", "g__y"])
    assert p["items"]["g__x"]["n"] == 1 and p["items"]["g__y"]["n"] == 0 and p["n"] == 1


# --------------------------------------------------------------------------
# Clinical items, exclusion flow, outcome validity
# --------------------------------------------------------------------------
def test_drug_resolution_uses_ingredient_descendants(profile_dir):
    concept = oc.scan_table(
        profile_dir, "concept", ["concept_id", "concept_name", "vocabulary_id", "concept_class_id", "standard_concept"]
    )
    anc = oc.scan_table(profile_dir, "concept_ancestor", ["ancestor_concept_id", "descendant_concept_id"])
    mapping, unresolved = op.resolve_drug_concepts(concept, anc, {"ad_drugs": ["donepezil", "memantine", "lecanemab"]})
    got = {(r["drug_concept_id"], r["item"]) for r in mapping.iter_rows(named=True)}
    assert got == {
        (100, "ad_drugs__donepezil"),
        (1001, "ad_drugs__donepezil"),
        (101, "ad_drugs__memantine"),
        (1011, "ad_drugs__memantine"),
    }
    assert unresolved == ["lecanemab"]


def test_exclusion_flow_counts(profile_dir):
    persons, _, vsum, _, _ = _persons(profile_dir)
    flow = op.exclusion_flow(persons, vsum, contaminated={5}, date_floor="1990-01-01")

    def rem(cohort):
        sub = flow.filter(pl.col("cohort") == cohort)
        return [(r["step"], r["n_removed"], r["n_remaining"]) for r in sub.iter_rows(named=True)]

    assert rem("control") == [
        ("candidates", 0, 7),
        ("no_visits", 1, 6),
        ("birth_year_missing_or_implausible_age", 1, 5),
        ("index_before_date_floor", 1, 4),
        ("history_lt_365_days", 1, 3),
        ("control_dementia_proxy_or_ad_drug", 1, 2),
        ("control_death_within_365_days_of_last_visit", 1, 1),
    ]
    assert rem("ad") == [
        ("candidates", 0, 3),
        ("no_visits", 0, 3),
        ("birth_year_missing_or_implausible_age", 0, 3),
        ("index_before_date_floor", 0, 3),
        ("history_lt_365_days", 1, 2),
        ("ad_first_diagnosis_within_182_days_of_first_visit", 0, 2),
    ]
    alone = flow.filter(
        (pl.col("cohort") == "ad") & (pl.col("step") == "ad_first_diagnosis_within_182_days_of_first_visit")
    )
    assert alone["n_meeting_alone"][0] == 0


def test_outcome_validity_on_fixture(profile_dir):
    persons, visits, vsum, ad_rows, ad_first = _persons(profile_dir)
    proxy = pl.DataFrame(
        {"person_id": [1], "date": [D(2018, 1, 1)], "group": ["dementia_proxies"], "item": ["dementia_proxies__x"]}
    )
    res = op.outcome_validity(ad_rows.rename({"source": "source"}), ad_first, vsum, visits, proxy)
    assert res["n_cases"] == 3
    assert res["code_counts"]["G30.9"]["n_rows"] == 1 and res["code_counts"]["G30.0"]["n_rows"] == 1
    assert res["code_counts"]["331.0"]["n_persons"] == 1 and res["code_counts"]["G30.1"]["n_rows"] == 1
    assert res["share_exactly_one_ad_row"] == pytest.approx(2 / 3)
    assert res["days_first_proxy_to_first_ad"]["median"] == (D(2019, 3, 1) - D(2018, 1, 1)).days
    assert res["share_with_proxy_before_first_ad"] == pytest.approx(1 / 3)
    assert res["share_first_ad_on_inpatient_visit"] == pytest.approx(1 / 3)
    assert res["share_first_ad_within_182_days_of_first_visit"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def test_step18_end_to_end(step, profile_dir, tmp_path):
    mod = step("18_profile_omop.py")
    out = tmp_path / "out"
    rep = tmp_path / "rep"
    rc = mod.main(["--data-dir", str(profile_dir), "--out-dir", str(out), "--report-dir", str(rep), "--seed", "3"])
    assert rc == 0
    status = json.loads((rep / "status.json").read_text())
    assert status["status"] in ("pass", "warn")
    assert any(c["name"] == "outputs written" and c["ok"] for c in status["checks"])
    prof = json.loads((out / "profile.json").read_text())
    assert prof["meta"]["n_ad"] == 3 and prof["meta"]["n_control"] == 7
    assert prof["demographics"]["by_view"]["ad"]["n"] == 3
    assert prof["demographics"]["by_view"]["control_last_visit"]["n"] == 6
    cc = prof["clinical_content"]["all_history"]["ad"]["items"]
    assert cc["comorbidities__hypertension"]["n"] == 1  # person 1, before index
    assert cc["ad_drugs__donepezil"]["n"] == 1  # only the exposure before the index counts
    assert cc["imaging__brain_mri"]["n"] == 1 and cc["labs_vitals__mmse"]["n"] == 1
    assert prof["clinical_content"]["ad_drug_first_exposure"]["ad"]["n_exposed"] == 1
    dm = prof["data_quality"]["tables"]["condition_occurrence"]["all"]
    assert dm["n_rows"] == 11 and dm["n_duplicate_rows"] == 1
    assert prof["data_quality"]["tables"]["visit_occurrence"]["all"]["n_before_floor"] == 2
    assert prof["control_contamination"]["control_last_visit"]["n_any_proxy_or_ad_drug"] == 1
    assert prof["site_source"]["control"]["person_care_site_id"]["share_null"] == 1.0
    flow = pl.read_csv(out / "exclusion_flow.csv")
    assert flow.filter(pl.col("cohort") == "control")["n_remaining"].min() == 1
    t1 = pl.read_csv(out / "table1.csv")
    assert {"variable", "level", "ad", "control_last_visit", "control_matched_index", "smd_control_last_visit"} <= set(
        t1.columns
    )
    assert (out / "table1.md").read_text().startswith("| variable")
    assert len(list((out / "figures").glob("*.png"))) >= 8


def test_step18_skip_sections_and_missing_tables(step, profile_dir, tmp_path):
    mod = step("18_profile_omop.py")
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--data-dir",
            str(profile_dir),
            "--out-dir",
            str(out),
            "--report-dir",
            str(tmp_path / "rep"),
            "--skip-sections",
            "data_quality,measurement,site_source",
        ]
    )
    assert rc == 0
    prof = json.loads((out / "profile.json").read_text())
    assert "data_quality" not in prof and "site_source" not in prof
    assert "labs_vitals__mmse" not in prof["clinical_content"]["all_history"]["ad"]["items"]

    empty = tmp_path / "empty"
    empty.mkdir()
    rc = mod.main(["--data-dir", str(empty), "--out-dir", str(tmp_path / "o2"), "--report-dir", str(tmp_path / "rep2")])
    assert rc == 1
