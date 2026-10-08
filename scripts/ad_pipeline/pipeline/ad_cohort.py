"""Cohort construction for the labeled AD dataset (step 6).

Anchor and index dates
----------------------
* Case: anchor = first AD ``condition_start_date``. Index = anchor - gap.
* Control: no AD code and at least one visit. The anchor is one of the control's own distinct visit
  dates ``d`` with ``last_visit >= d + followup`` (at least ``followup`` days of observation after the
  anchor). Index = anchor - gap.

Controls are drawn in two stages so that their anchor years follow the case anchor years:
(1) pick an anchor year among the years in which the control has eligible candidates, with per-year
weights fitted by iterative proportional fitting (``omop_common.fit_stratum_weights``, one stratum
per calendar year) so the expected number of control anchors per year approaches the target;
(2) pick one eligible visit date of that year uniformly. Both draws use one seeded NumPy generator over
rows sorted by (person, year, date), so the result is deterministic. Controls whose eligible years all
have zero target weight are not drawn.

Inclusion rules, applied identically on the final index
-------------------------------------------------------
1. age >= ``min_age`` at the index date; 2. index >= ``index_floor``; 3. first visit <=
index - ``window_days``; 4. at least one event (condition, drug, procedure or measurement) in the
half-open window ``[index - window_days, index)``. Rules 1 to 3 only need dates already at hand, so
controls are pre-filtered on them at the candidate level (``control_candidates``).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from .omop_common import age_years_expr, birth_date_expr, fit_stratum_weights, scan_table, smd_continuous

EVENT_TABLES = (
    ("condition_occurrence", "condition_start_date"),
    ("drug_exposure", "drug_exposure_start_date"),
    ("procedure_occurrence", "procedure_date"),
    ("measurement", "measurement_date"),
)

RULES = ("age", "index_floor", "history", "event")  # ok_<rule> flags exist for the first three

# Year-only IPF strata reuse the (band, year) layout of the age matched draw with a constant band.
_BAND = 0


def read_persons(data_dir: Path) -> pl.DataFrame:
    """person_id and birth_date for every person."""
    lf = scan_table(data_dir, "person", ["person_id", "year_of_birth", "month_of_birth", "day_of_birth"])
    if lf is None:
        raise FileNotFoundError(f"No parquet found for table 'person' under {data_dir}")
    return lf.select("person_id", birth_date_expr().alias("birth_date")).unique("person_id").collect(engine="streaming")


def read_release_date(data_dir: Path) -> dt.date | None:
    """``cdm_source.source_release_date`` when the table and a value exist."""
    try:
        lf = scan_table(data_dir, "cdm_source", ["source_release_date"])
        if lf is None:
            return None
        return lf.select(pl.col("source_release_date").max()).collect().item()
    except Exception:  # noqa: BLE001
        return None


def add_rule_flags(
    df: pl.DataFrame, gap_days: int, window_days: int, min_age: int, index_floor: dt.date
) -> pl.DataFrame:
    """Add ``index_date`` (anchor - gap) and the booleans for rules 1 to 3 (null inputs fail)."""
    return (
        df.with_columns((pl.col("anchor_date") - pl.duration(days=gap_days)).alias("index_date"))
        .with_columns(
            age_years_expr("index_date", "birth_date").alias("age"),
            (pl.col("index_date") >= index_floor).fill_null(False).alias("ok_index_floor"),
            (pl.col("first_visit") <= pl.col("index_date") - pl.duration(days=window_days))
            .fill_null(False)
            .alias("ok_history"),
        )
        .with_columns((pl.col("age") >= min_age).fill_null(False).alias("ok_age"))
    )


def events_in_window(data_dir: Path, idx: pl.DataFrame, window_days: int) -> pl.Series:
    """person_ids in ``idx`` (person_id, index_date) with any event in ``[index - window, index)``.

    Tables are semi-joined one at a time against the persons still without an event, so memory stays
    bounded by the index table.
    """
    found: list[pl.Series] = []
    remaining = idx.select("person_id", "index_date")
    for table, date_col in EVENT_TABLES:
        if remaining.is_empty():
            break
        lf = scan_table(data_dir, table, ["person_id", date_col])
        if lf is None:
            continue
        hit = (
            lf.join(remaining.lazy(), on="person_id", how="inner")
            .filter(
                (pl.col(date_col) < pl.col("index_date"))
                & (pl.col(date_col) >= pl.col("index_date") - pl.duration(days=window_days))
            )
            .select("person_id")
            .unique()
            .collect(engine="streaming")["person_id"]
        )
        found.append(hit)
        remaining = remaining.filter(~pl.col("person_id").is_in(hit.implode()))
    if not found:
        return pl.Series("person_id", [], dtype=pl.Int64)
    return pl.concat(found).unique()


def apply_rules(df: pl.DataFrame, data_dir: Path, window_days: int) -> tuple[pl.DataFrame, dict[str, int]]:
    """Apply rules 1 to 4 in order to a frame with the flags from ``add_rule_flags``.

    Returns the kept rows and ``{"candidates", "excluded_<rule>" ..., "remaining"}``.
    """
    counts = {"candidates": df.height}
    for rule in RULES[:3]:
        before = df.height
        df = df.filter(pl.col(f"ok_{rule}"))
        counts[f"excluded_{rule}"] = before - df.height
    with_event = events_in_window(data_dir, df, window_days)
    before = df.height
    df = df.filter(pl.col("person_id").is_in(with_event.implode()))
    counts["excluded_event"] = before - df.height
    counts["remaining"] = df.height
    return df, counts


def control_candidates(
    visit_dates: pl.DataFrame,
    controls: pl.DataFrame,
    gap_days: int,
    window_days: int,
    min_age: int,
    index_floor: dt.date,
    followup_days: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Eligible (person_id, anchor_date) pairs of controls, plus candidate counts after each filter.

    ``visit_dates``: distinct (person_id, visit_date). ``controls``: person_id, birth_date,
    first_visit, last_visit. A candidate date ``d`` must leave ``followup_days`` of observation after
    it and satisfy rules 1 to 3 plus a visit in ``[index - window, index)`` at index = d - gap.
    """
    cand = visit_dates.rename({"visit_date": "anchor_date"}).join(controls, on="person_id")
    counts = {"anchor_candidates": cand.height}

    cand = cand.filter(pl.col("last_visit") >= pl.col("anchor_date") + pl.duration(days=followup_days))
    counts["after_followup"] = cand.height
    cand = add_rule_flags(cand, gap_days, window_days, min_age, index_floor)
    for name in ("ok_age", "ok_index_floor", "ok_history"):
        cand = cand.filter(pl.col(name))
        counts["after_" + name[3:]] = cand.height

    # Latest visit strictly before the index (asof key index - 1 day); it is in the window when >= index - window.
    right = visit_dates.rename({"visit_date": "prev_visit"}).sort("prev_visit")
    cand = (
        cand.with_columns((pl.col("index_date") - pl.duration(days=1)).alias("_key"))
        .sort("_key")
        .join_asof(
            right, left_on="_key", right_on="prev_visit", by="person_id", strategy="backward", check_sortedness=False
        )
        .filter(pl.col("prev_visit") >= pl.col("index_date") - pl.duration(days=window_days))
    )
    counts["after_visit_in_window"] = cand.height
    return cand.select("person_id", "anchor_date").sort("person_id", "anchor_date"), counts


def draw_control_anchors(
    candidates: pl.DataFrame, target: pl.DataFrame, seed: int
) -> tuple[pl.DataFrame, dict[str, int]]:
    """One anchor per control by the two stage draw in the module docstring.

    ``candidates``: distinct (person_id, anchor_date). ``target``: ``year`` and ``n_case`` (target counts
    per calendar year). Returns (person_id, anchor_date) and ``{"n_with_candidates", "n_zero_target"}``.
    """
    empty = pl.DataFrame(schema={"person_id": pl.Int64, "anchor_date": pl.Date})
    cand = candidates.with_columns(pl.col("anchor_date").dt.year().cast(pl.Int64).alias("year"))
    n_with = cand["person_id"].n_unique()
    target = target.filter(pl.col("n_case") > 0).select(pl.col("year").cast(pl.Int64), "n_case")
    reachable = cand.join(target.select("year"), on="year")
    n_zero = n_with - reachable["person_id"].n_unique()
    info = {"n_with_candidates": n_with, "n_zero_target": n_zero}
    if reachable.is_empty():
        return empty, info

    person_year = (
        reachable.select("person_id", "year")
        .unique()
        .sort("person_id", "year")
        .with_columns(pl.lit(_BAND).alias("band"))
    )
    weights = fit_stratum_weights(person_year, target.with_columns(pl.lit(_BAND).alias("band")))
    person_year = person_year.join(weights.select("year", "w"), on="year")

    rng = np.random.default_rng(seed)
    u = 1.0 - rng.random(person_year.height)
    keys = -np.log(u) / person_year["w"].to_numpy()
    chosen = (
        person_year.with_columns(pl.Series("key", keys))
        .sort("person_id", "key")
        .unique("person_id", keep="first", maintain_order=True)
        .sort("person_id")
        .select("person_id", "year")
    )
    chosen = chosen.with_columns(pl.Series("u_date", rng.random(chosen.height)))

    dates = (
        reachable.join(chosen, on=["person_id", "year"])
        .sort("person_id", "anchor_date")
        .with_columns(
            pl.int_range(pl.len()).over("person_id").alias("pos"),
            pl.len().over("person_id").alias("n"),
        )
    )
    picked = dates.filter(pl.col("pos") == (pl.col("u_date") * pl.col("n")).floor().cast(pl.Int64))
    return picked.select("person_id", "anchor_date").sort("person_id"), info


def year_counts(anchor_dates: pl.Series) -> dict[int, int]:
    """Number of anchors per calendar year, ordered by year."""
    counts = pl.DataFrame({"year": anchor_dates.dt.year()}).group_by("year").len().sort("year")
    return dict(zip(counts["year"].to_list(), counts["len"].to_list()))


def anchor_year_smd(case_dates: pl.Series, control_dates: pl.Series) -> float | None:
    """Standardized mean difference of the anchor calendar year, controls minus cases."""

    def stats(s: pl.Series):
        y = s.dt.year().cast(pl.Float64)
        return (y.mean(), y.std()) if y.len() > 1 else (None, None)

    ma, sa = stats(case_dates)
    mb, sb = stats(control_dates)
    return smd_continuous(ma, sa, mb, sb)
