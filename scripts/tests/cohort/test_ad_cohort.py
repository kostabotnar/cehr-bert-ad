"""Tests for the step 6 cohort definition: anchors, control candidates, the year draw and inclusion rules."""

from __future__ import annotations

import datetime as dt
from datetime import date as D
from datetime import timedelta as TD

import numpy as np
import polars as pl
import pytest

from pipeline import ad_cohort
from pipeline.reporting import StepReport

GAP = 365
WINDOW = 365
FOLLOWUP = 365
FLOOR = D(2000, 1, 1)
AD = "ICD-10-CM:G30.9"


@pytest.fixture
def mod(step):
    return step("06_prepare_finetuning_data.py")


def ev(person, date, concept=100, visit=1):
    """An event row; by default it is visit linked with concept 100."""
    return (person, date, concept, visit)


def write_omop(root, persons, visits, ad=(), events=None):
    """Minimal OMOP folder. persons: {id: birth_year}; visits: {id: [dates]}; ad: {id: date};
    events: {table: [(id, date) or (id, date, concept, visit_occurrence_id)]} for drug_exposure,
    procedure_occurrence, measurement, condition_occurrence. A pair gets concept 100 and visit 1."""
    events = {t: [e if len(e) == 4 else ev(*e) for e in rows] for t, rows in (events or {}).items()}

    def write(table, df):
        d = root / table
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / "data.parquet")

    def frame(rows, date_col, concept_col, source=None):
        cols = {
            "person_id": [r[0] for r in rows],
            date_col: [r[1] for r in rows],
            concept_col: [r[2] for r in rows],
            "visit_occurrence_id": [r[3] for r in rows],
        }
        schema = {
            "person_id": pl.Int64,
            date_col: pl.Date,
            concept_col: pl.Int64,
            "visit_occurrence_id": pl.Int64,
        }
        if source is not None:
            cols["condition_source_value"] = source
            schema["condition_source_value"] = pl.String
        return pl.DataFrame(cols, schema=schema)

    write("person", pl.DataFrame({"person_id": list(persons), "year_of_birth": list(persons.values())}))
    rows = [(p, d) for p, d in ((p, d) for p, ds in visits.items() for d in ds)]
    write(
        "visit_occurrence",
        pl.DataFrame(
            {"person_id": [r[0] for r in rows], "visit_start_date": [r[1] for r in rows]},
            schema={"person_id": pl.Int64, "visit_start_date": pl.Date},
        ),
    )
    ad_rows = [ev(p, d, 9) for p, d in dict(ad).items()]
    other = events.get("condition_occurrence", [])
    source = [AD] * len(ad_rows) + ["ICD-10-CM:E11.9"] * len(other)
    write("condition_occurrence", frame(ad_rows + other, "condition_start_date", "condition_concept_id", source))
    for table, date_col, concept_col in (
        ("drug_exposure", "drug_exposure_start_date", "drug_concept_id"),
        ("procedure_occurrence", "procedure_date", "procedure_concept_id"),
        ("measurement", "measurement_date", "measurement_concept_id"),
    ):
        write(table, frame(events.get(table, []), date_col, concept_col))


def candidates(visits, birth=D(1950, 7, 1)):
    """Eligible anchors of a single control with the given visit dates."""
    vd = pl.DataFrame({"person_id": [1] * len(visits), "visit_date": visits}).unique()
    ctrl = pl.DataFrame(
        {"person_id": [1], "birth_date": [birth], "first_visit": [min(visits)], "last_visit": [max(visits)]}
    )
    out, counts = ad_cohort.control_candidates(vd, ctrl, GAP, WINDOW, 50, FLOOR, FOLLOWUP)
    return out["anchor_date"].to_list(), counts


# --------------------------------------------------------------------------
# Control candidate filtering
# --------------------------------------------------------------------------
def test_candidate_needs_one_year_of_followup():
    base = [D(2010, 1, 1), D(2013, 6, 1), D(2015, 1, 1)]
    assert D(2015, 1, 1) not in candidates(base + [D(2015, 12, 31)])[0]  # d + 364
    assert D(2015, 1, 1) in candidates(base + [D(2016, 1, 1)])[0]  # d + 365


def test_candidate_age_boundary():
    # Index = 2015-01-01 - 365 days = 2014-01-01. Born 1964-01-01 is 50.0 years old then, 1964-02-01 is 49.9.
    visits = [D(2010, 1, 1), D(2013, 6, 1), D(2015, 1, 1), D(2016, 1, 1)]
    assert D(2015, 1, 1) in candidates(visits, birth=D(1963, 7, 1))[0]
    assert D(2015, 1, 1) not in candidates(visits, birth=D(1964, 7, 1))[0]


def test_candidate_index_floor_is_inclusive():
    # 2000 is a leap year, so 2000-12-31 minus 365 days is 2000-01-01.
    ok = [D(1998, 1, 1), D(1999, 6, 1), D(2000, 12, 31), D(2002, 1, 1)]
    assert D(2000, 12, 31) in candidates(ok, birth=D(1940, 7, 1))[0]
    early = [D(1998, 1, 1), D(1999, 6, 1), D(2000, 12, 30), D(2002, 1, 1)]
    assert D(2000, 12, 30) not in candidates(early, birth=D(1940, 7, 1))[0]


def test_candidate_history_must_cover_window():
    # Anchor 2015-01-01 -> index 2014-01-01 -> window starts 2013-01-01.
    tail = [D(2013, 6, 1), D(2015, 1, 1), D(2016, 1, 1)]
    assert D(2015, 1, 1) in candidates([D(2013, 1, 1)] + tail)[0]
    assert D(2015, 1, 1) not in candidates([D(2013, 1, 2)] + tail)[0]


def test_candidate_needs_visit_in_half_open_window():
    # Only visit before the index sits on the index date (2014-01-01): excluded from [index - 365, index).
    visits = [D(2012, 1, 1), D(2014, 1, 1), D(2015, 1, 1), D(2016, 1, 1)]
    anchors, _ = candidates(visits)
    assert D(2015, 1, 1) not in anchors
    # 2014-01-01 as anchor: index 2013-01-01, the visit at 2012-01-01 is exactly index - 366 days (window starts
    # 2012-01-02), so it is out; move it to 2012-01-02 and it is in.
    anchors, _ = candidates([D(2012, 1, 2), D(2014, 1, 1), D(2015, 1, 1), D(2016, 1, 1)])
    assert D(2014, 1, 1) in anchors


def test_candidate_counts_are_sequential():
    visits = [D(2010, 1, 1), D(2013, 6, 1), D(2015, 1, 1), D(2016, 1, 1)]
    _, counts = candidates(visits)
    assert counts["anchor_candidates"] == 4
    seq = [
        counts[k] for k in ("anchor_candidates", "after_followup", "after_age", "after_index_floor", "after_history")
    ]
    assert seq == sorted(seq, reverse=True)
    assert counts["after_visit_in_window"] <= counts["after_history"]


# --------------------------------------------------------------------------
# The two stage draw
# --------------------------------------------------------------------------
def synthetic_candidates(n_people, years, seed=0, per_year=(1, 4)):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_people + 1):
        for y in rng.choice(years, size=rng.integers(2, 6), replace=False):
            for _ in range(rng.integers(*per_year)):
                rows.append((pid, D(int(y), 1, 1) + TD(days=int(rng.integers(0, 365)))))
    return (
        pl.DataFrame(rows, schema=["person_id", "anchor_date"], orient="row").unique().sort("person_id", "anchor_date")
    )


def target_frame(counts):
    return pl.DataFrame(
        {"year": list(counts), "n_case": list(counts.values())}, schema={"year": pl.Int64, "n_case": pl.Int64}
    )


def test_draw_is_deterministic_and_uses_own_dates():
    cand = synthetic_candidates(300, list(range(2005, 2015)))
    target = target_frame({y: 10 + y - 2005 for y in range(2005, 2015)})
    a, _ = ad_cohort.draw_control_anchors(cand, target, seed=7)
    b, _ = ad_cohort.draw_control_anchors(cand, target, seed=7)
    c, _ = ad_cohort.draw_control_anchors(cand, target, seed=8)
    assert a.equals(b)
    assert not a.equals(c)
    assert a["person_id"].n_unique() == a.height == 300
    assert a.join(cand, on=["person_id", "anchor_date"], how="anti").is_empty()


def test_draw_year_distribution_approaches_target():
    years = list(range(2005, 2020))
    cand = synthetic_candidates(4000, years, seed=1)
    # Rising target with a skewed shape that plain uniform sampling over each control's dates would miss.
    counts = {y: 20 * (y - 2004) ** 2 for y in years}
    picked, info = ad_cohort.draw_control_anchors(cand, target_frame(counts), seed=3)
    assert info["n_zero_target"] == 0
    achieved = ad_cohort.year_counts(picked["anchor_date"])
    total_t, total_a = sum(counts.values()), picked.height
    tv = 0.5 * sum(abs(counts[y] / total_t - achieved.get(y, 0) / total_a) for y in years)
    assert tv < 0.03


def test_draw_excludes_and_counts_controls_with_only_zero_target_years():
    cand = pl.DataFrame(
        {
            "person_id": [1, 2, 3],
            "anchor_date": [D(2010, 5, 1), D(2011, 5, 1), D(2010, 6, 1)],
        }
    )
    picked, info = ad_cohort.draw_control_anchors(cand, target_frame({2010: 5, 2012: 3}), seed=1)
    assert picked["person_id"].to_list() == [1, 3]
    assert info == {"n_with_candidates": 3, "n_zero_target": 1}


def test_draw_with_empty_target_draws_nobody():
    cand = pl.DataFrame({"person_id": [1], "anchor_date": [D(2010, 5, 1)]})
    picked, info = ad_cohort.draw_control_anchors(cand, target_frame({}), seed=1)
    assert picked.is_empty() and info["n_zero_target"] == 1


def test_anchor_year_smd_is_zero_for_equal_distributions():
    s = pl.Series("anchor_date", [D(2010, 1, 1), D(2012, 1, 1), D(2014, 1, 1)])
    assert ad_cohort.anchor_year_smd(s, s) == 0.0


# --------------------------------------------------------------------------
# Event in window (half-open) for both groups, end to end
# --------------------------------------------------------------------------
ANCHOR = D(2020, 6, 1)
INDEX = ANCHOR - TD(days=GAP)


def case_visits():
    return [D(2012, 1, 1), INDEX - TD(days=100), ANCHOR, D(2023, 1, 1)]


def control_visits():
    return [D(2012, 1, 1), INDEX - TD(days=100), ANCHOR, ANCHOR + TD(days=400)]


def test_event_window_boundaries_for_cases_and_controls(mod, tmp_path):
    persons = {p: 1950 for p in range(1, 7)}
    visits = {
        1: case_visits(),
        2: case_visits(),
        3: case_visits(),
        4: control_visits(),
        5: control_visits(),
        6: control_visits(),
    }
    ad = {p: ANCHOR for p in (1, 2, 3)}
    events = {
        "procedure_occurrence": [(1, INDEX), (2, INDEX - TD(days=WINDOW)), (3, INDEX - TD(days=WINDOW + 1))],
        "drug_exposure": [(4, INDEX), (5, INDEX - TD(days=WINDOW)), (6, INDEX - TD(days=WINDOW + 1))],
    }
    write_omop(tmp_path, persons, visits, ad, events)
    report = StepReport("t")
    cohort = mod.build_cohort(tmp_path, GAP, report, min_concept_patients=0)
    assert dict(zip(cohort["person_id"], cohort["label"])) == {2: 1, 5: 0}
    m = report.metrics
    assert m["cases_excluded_event"] == 2 and m["cases_remaining"] == 1
    assert m["controls_excluded_event"] == 2 and m["controls_remaining"] == 1
    assert cohort.filter(pl.col("person_id") == 2)["index_date"].item() == INDEX


def test_events_in_window_checks_every_event_table(tmp_path):
    persons = {p: 1950 for p in range(1, 5)}
    events = {
        "condition_occurrence": [(1, INDEX - TD(days=1))],
        "drug_exposure": [(2, INDEX - TD(days=1))],
        "procedure_occurrence": [(3, INDEX - TD(days=1))],
        "measurement": [(4, INDEX - TD(days=1))],
    }
    write_omop(tmp_path, persons, {p: [D(2012, 1, 1)] for p in persons}, events=events)
    idx = pl.DataFrame({"person_id": [1, 2, 3, 4, 5], "index_date": [INDEX] * 5})
    assert sorted(ad_cohort.events_in_window(tmp_path, idx, WINDOW).to_list()) == [1, 2, 3, 4]


def test_events_without_visit_or_with_concept_zero_do_not_count(tmp_path):
    persons = {p: 1950 for p in range(1, 6)}
    day = INDEX - TD(days=1)
    events = {
        "drug_exposure": [ev(1, day, visit=None), ev(2, day, concept=0), ev(3, day, concept=None), ev(4, day)],
        "measurement": [ev(5, day, visit=None)],
    }
    write_omop(tmp_path, persons, {p: [D(2012, 1, 1)] for p in persons}, events=events)
    idx = pl.DataFrame({"person_id": [1, 2, 3, 4, 5], "index_date": [INDEX] * 5})
    assert ad_cohort.events_in_window(tmp_path, idx, WINDOW).to_list() == [4]


def test_qualified_concepts_pool_visit_linked_persons_across_tables(tmp_path):
    day = INDEX - TD(days=1)
    events = {
        # concept 100: persons 1, 2 (drug) and 3 (procedure) -> 3 persons; person 1's repeat counts once
        "drug_exposure": [ev(1, day), ev(1, day - TD(days=1)), ev(2, day)],
        "procedure_occurrence": [ev(3, day)],
        # concept 200: persons 1, 2 visit linked, 3 without a visit -> 2 persons
        "condition_occurrence": [ev(1, day, 200), ev(2, day, 200), ev(3, day, 200, visit=None)],
        # concept 0 never counts; measurement rows count even without a date
        "measurement": [ev(p, day, 0) for p in (1, 2, 3)] + [ev(p, None, 300) for p in (1, 2, 3)],
    }
    write_omop(tmp_path, {p: 1950 for p in (1, 2, 3)}, {p: [D(2012, 1, 1)] for p in (1, 2, 3)}, events=events)
    assert sorted(ad_cohort.qualified_concepts(tmp_path, 3).to_list()) == [100, 300]
    assert sorted(ad_cohort.qualified_concepts(tmp_path, 2).to_list()) == [100, 200, 300]
    assert ad_cohort.qualified_concepts(tmp_path, 0) is None


def test_event_with_unqualified_concept_does_not_count(tmp_path):
    day = INDEX - TD(days=1)
    # Person 1 has only a rare concept in the window, person 2 a common one (shared with persons 3, 4).
    events = {"drug_exposure": [ev(1, day, 7), ev(2, day, 100), ev(3, day, 100), ev(4, day, 100)]}
    write_omop(tmp_path, {p: 1950 for p in range(1, 5)}, {p: [D(2012, 1, 1)] for p in range(1, 5)}, events=events)
    idx = pl.DataFrame({"person_id": [1, 2], "index_date": [INDEX] * 2})
    qualified = ad_cohort.qualified_concepts(tmp_path, 3)
    assert ad_cohort.events_in_window(tmp_path, idx, WINDOW, qualified).to_list() == [2]
    assert sorted(ad_cohort.events_in_window(tmp_path, idx, WINDOW).to_list()) == [1, 2]


def test_build_cohort_records_event_rule_exclusions(mod, tmp_path):
    persons = {p: 1950 for p in range(1, 5)}
    visits = {1: case_visits(), 2: case_visits(), 3: control_visits(), 4: control_visits()}
    day = INDEX - TD(days=5)
    events = {
        "drug_exposure": [ev(1, day, 100), ev(2, day, 7), ev(3, day, 100, visit=None), ev(4, day, 100)],
        "procedure_occurrence": [ev(p, D(2015, 1, 1), 100) for p in (1, 2, 3, 4)],
    }
    write_omop(tmp_path, persons, visits, ad={1: ANCHOR, 2: ANCHOR}, events=events)
    report = StepReport("t")
    cohort = mod.build_cohort(tmp_path, GAP, report, min_concept_patients=3)
    m = report.metrics
    assert sorted(cohort["person_id"]) == [1, 4]
    assert m["min_concept_patients"] == 3 and m["qualified_concepts"] == 1
    assert m["cases_excluded_event"] == 1 and m["controls_excluded_event"] == 1
    assert m["excluded_by_event_rule"] == 2


# --------------------------------------------------------------------------
# Anchors, cap and exclusion counts end to end
# --------------------------------------------------------------------------
def test_case_anchor_is_first_ad_date_and_index_is_anchor_minus_gap(mod, tmp_path):
    persons = {1: 1950}
    write_omop(
        tmp_path,
        persons,
        {1: case_visits()},
        ad={1: ANCHOR},
        events={"drug_exposure": [(1, INDEX - TD(days=10))], "condition_occurrence": [(1, INDEX - TD(days=20))]},
    )
    # A later AD row must not move the anchor.
    extra = pl.DataFrame(
        {
            "person_id": [1],
            "visit_occurrence_id": [1],
            "condition_concept_id": [9],
            "condition_source_value": [AD],
            "condition_start_date": [D(2022, 1, 1)],
        },
    ).select(pl.read_parquet(tmp_path / "condition_occurrence" / "data.parquet").columns)
    path = tmp_path / "condition_occurrence" / "data.parquet"
    pl.concat([pl.read_parquet(path), extra]).write_parquet(path)
    cohort = mod.build_cohort(tmp_path, GAP, StepReport("t"), min_concept_patients=0)
    assert cohort.row(0) == (1, INDEX, 1)


def test_capped_target_excludes_late_case_anchors_but_keeps_the_cases(mod, tmp_path):
    late = D(2022, 6, 1)  # cutoff is the last visit, 2023-01-01, so the cap is 2022-01-01
    late_index = late - TD(days=GAP)
    persons = {1: 1950, 2: 1950, 3: 1950}
    visits = {
        1: case_visits(),
        2: [D(2012, 1, 1), late_index - TD(days=100), late, D(2023, 1, 1)],
        3: control_visits(),
    }
    events = {
        "drug_exposure": [(1, INDEX - TD(days=5)), (2, late_index - TD(days=5)), (3, INDEX - TD(days=5))],
    }
    write_omop(tmp_path, persons, visits, ad={1: ANCHOR, 2: late}, events=events)
    report = StepReport("t")
    cohort = mod.build_cohort(tmp_path, GAP, report, min_concept_patients=0)
    m = report.metrics
    assert m["data_cutoff"] == "2023-01-01" and m["target_cap_date"] == "2022-01-01"
    assert m["target_anchor_year_counts"] == {"2020": 1}
    assert m["cases_beyond_cap"] == 1
    assert sorted(cohort.filter(pl.col("label") == 1)["person_id"]) == [1, 2]
    assert cohort.filter(pl.col("person_id") == 3)["index_date"].item() == INDEX


def test_exclusion_counts_per_group(mod, tmp_path):
    def case(birth=1940, anchor=ANCHOR, first=D(2012, 1, 1), event=True, visits=True):
        idx = anchor - TD(days=GAP)
        v = [first, idx - TD(days=100), anchor, D(2023, 1, 1)] if visits else []
        return birth, v, anchor, [idx - TD(days=5)] if event else []

    cases = {
        1: case(),
        2: case(birth=1990),
        3: case(anchor=D(2000, 6, 1)),
        4: case(first=D(2019, 12, 1)),
        5: case(visits=False),
        6: case(event=False),
    }
    persons, visits, ad, drugs = {}, {}, {}, []
    for pid, (birth, v, anchor, ev) in cases.items():
        persons[pid], visits[pid], ad[pid] = birth, v, anchor
        drugs += [(pid, e) for e in ev]
    # 4: the history rule must fail on first visit, so its window visit stays inside the record.
    visits[4] = [D(2019, 12, 1), INDEX - TD(days=100), ANCHOR, D(2023, 1, 1)]

    persons.update({11: 1950, 12: 1950, 13: 1950, 14: 1950, 15: 1950})
    visits[11] = control_visits()
    visits[12] = []
    visits[13] = [D(2012, 1, 1), D(2013, 1, 1)]
    visits[14] = [D(2010, 1, 1), D(2014, 1, 1), D(2015, 6, 1), D(2017, 1, 1)]  # only eligible in 2015
    visits[15] = control_visits()
    drugs += [(11, INDEX - TD(days=5)), (14, D(2014, 3, 1))]

    write_omop(tmp_path, persons, visits, ad, {"drug_exposure": drugs})
    report = StepReport("t")
    cohort = mod.build_cohort(tmp_path, GAP, report, min_concept_patients=0)
    m = report.metrics

    assert (m["cases_candidates"], m["cases_excluded_age"], m["cases_excluded_index_floor"]) == (6, 1, 1)
    assert (m["cases_excluded_history"], m["cases_excluded_event"], m["cases_remaining"]) == (2, 1, 1)
    assert m["controls_no_ad_persons"] == 5 and m["controls_no_visits"] == 1
    assert m["controls_no_eligible_candidate"] == 1
    assert m["controls_zero_target_weight_only"] == 1
    assert m["controls_drawn"] == 2
    assert (m["controls_excluded_age"], m["controls_excluded_index_floor"], m["controls_excluded_history"]) == (0, 0, 0)
    assert m["controls_excluded_event"] == 1 and m["controls_remaining"] == 1
    assert m["cases_mean_age_at_index"] > 60 and m["controls_mean_age_at_index"] > 60
    assert sorted(cohort["person_id"]) == [1, 11]
    assert "anchor_year_smd" in m and "achieved_control_anchor_year_counts" in m
