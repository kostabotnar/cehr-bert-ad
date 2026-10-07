"""Profile raw OMOP data for Alzheimer's disease (AD) cases versus controls.

Pure functions that take Polars frames and return DataFrames or plain dicts; the CLI, figures and
report live in ``18_profile_omop.py``. Nothing here depends on step 6 or later outputs.

Cohorts and index dates
-----------------------
* AD: any ``condition_source_value`` matching ``AD_SOURCE_REGEX`` (same semantics as step 6).
  Index = first AD ``condition_start_date``.
* Control: every other person. Two index views are profiled:
  ``control_last_visit`` (index = last visit date, controls without visits are dropped from
  index-based metrics) and ``control_matched_index``.

Matched index method (``pick_matched_index``)
---------------------------------------------
Cases are binned into strata of (5-year age band, calendar year) at their index date. For every
control, each distinct visit date is a candidate pseudo-index with a stratum defined the same way.
A candidate is weighted by ``case_count(stratum) / candidate_count(stratum)``, so strata with many
cases are favoured and strata with many candidate visits are not over-represented. One candidate per
control is drawn with probability proportional to its weight (exponential keys, ``-log(u) / w``,
smallest key wins) using a seeded NumPy generator over candidates sorted by (person, date), which
makes the draw deterministic. A control whose candidates all fall in strata without cases (or who has
no birth year) is drawn uniformly among its own visit dates and counted in ``n_fallback``.

Windows are half-open: an event at ``d`` is inside ``[index - k, index)`` when
``index - k <= d < index``. Events on the index date itself are never counted as history.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
import yaml

from .omop_validation import table_glob

COHORT_AD = "ad"
COHORT_CONTROL = "control"
VIEW_AD = "ad"
VIEW_LAST = "control_last_visit"
VIEW_MATCHED = "control_matched_index"
VIEWS = (VIEW_AD, VIEW_LAST, VIEW_MATCHED)
VIEW_COHORT = {VIEW_AD: COHORT_AD, VIEW_LAST: COHORT_CONTROL, VIEW_MATCHED: COHORT_CONTROL}
CONTROL_VIEWS = (VIEW_LAST, VIEW_MATCHED)

# Same semantics as step 6 (AD_CODE_PATTERNS / AD_SOURCE_REGEX); duplicated because step scripts are
# not importable modules.
AD_CODE_PATTERNS = [r"ICD-9-CM:331\.0", r"ICD-10-CM:G30"]
AD_SOURCE_REGEX = "^(" + "|".join(AD_CODE_PATTERNS) + ")"

AGE_BAND_YEARS = 5
MIN_PLAUSIBLE_AGE = 40
MAX_PLAUSIBLE_AGE = 110
DEFAULT_DATE_FLOOR = "1990-01-01"
MIN_HISTORY_DAYS = 365
AD_EARLY_DAYS = 182
VISIT_TYPE_LABELS = {9201: "inpatient", 9202: "outpatient", 9203: "emergency", 0: "unknown"}
VISIT_TYPE_ORDER = ["inpatient", "outpatient", "emergency", "unknown", "other"]
GENDER_LABELS = {8507: "male", 8532: "female"}
RACE_LABELS = {
    8527: "white",
    8516: "black",
    8515: "asian",
    8557: "pacific_islander",
    8657: "native_american",
    0: "unknown",
}
ETHNICITY_LABELS = {38003563: "hispanic", 38003564: "not_hispanic", 0: "unknown"}

# table -> (date column, concept column used for concept-0 share and duplicate detection)
QUALITY_TABLES = {
    "visit_occurrence": ("visit_start_date", "visit_concept_id"),
    "condition_occurrence": ("condition_start_date", "condition_concept_id"),
    "drug_exposure": ("drug_exposure_start_date", "drug_concept_id"),
    "measurement": ("measurement_date", "measurement_concept_id"),
    "procedure_occurrence": ("procedure_date", "procedure_concept_id"),
    "observation": ("observation_date", "observation_concept_id"),
}


# --------------------------------------------------------------------------
# Scanning helpers
# --------------------------------------------------------------------------
def sample_expr(fraction: float, seed: int) -> pl.Expr | None:
    """Deterministic hash-based person_id sample predicate, ``None`` when the fraction is 1."""
    if fraction >= 1.0:
        return None
    buckets = 10_000
    threshold = max(1, int(round(fraction * buckets)))
    return (pl.col("person_id").hash(seed=seed) % buckets) < threshold


def scan_table(
    data_dir: Path, table: str, columns: Iterable[str], sample: pl.Expr | None = None
) -> pl.LazyFrame | None:
    """Lazy scan of ``columns`` of a table, ``None`` when absent.

    Columns missing from the parquet schema come back as nulls and ``*_date`` columns are cast to
    ``Date``, so callers can rely on a fixed schema. The person sample filter is applied first.
    """
    glob = table_glob(Path(data_dir), table)
    if glob is None:
        return None
    lf = pl.scan_parquet(glob)
    have = set(lf.collect_schema().names())
    cols = list(columns)
    if sample is not None and "person_id" in have:
        lf = lf.filter(sample)
    exprs = []
    for c in cols:
        e = pl.col(c) if c in have else pl.lit(None, dtype=pl.Int64).alias(c)
        if c.endswith("_date"):
            e = e.cast(pl.Date)
        exprs.append(e)
    return lf.select(exprs)


# --------------------------------------------------------------------------
# Concept configuration and code matching
# --------------------------------------------------------------------------
_RANGE_RE = re.compile(r"^([A-Za-z]*)(\d+)-([A-Za-z]*)(\d+)$")


def expand_code_range(token: str) -> list[str]:
    """Expand ``"E08-E13"`` or ``"401-405"`` to prefixes; any other token is returned unchanged."""
    m = _RANGE_RE.match(token)
    if not m:
        return [token]
    pre1, lo, pre2, hi = m.groups()
    if pre1 != pre2 or int(hi) < int(lo):
        raise ValueError(f"invalid code range: {token}")
    width = len(lo)
    return [f"{pre1}{n:0{width}d}" for n in range(int(lo), int(hi) + 1)]


def code_prefix_regex(vocab_codes: dict[str, list[str]]) -> str:
    """Anchored regex for ``{vocabulary: [code prefixes or ranges]}`` against source values."""
    parts = []
    for vocab, codes in vocab_codes.items():
        prefixes = [re.escape(p) for c in codes for p in expand_code_range(str(c))]
        parts.append(re.escape(f"{vocab}:") + "(?:" + "|".join(prefixes) + ")")
    return "^(?:" + "|".join(parts) + ")"


def load_concept_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("conditions", "drugs", "procedures", "measurements"):
        cfg.setdefault(key, {})
    return cfg


def item_id(group: str, item: str) -> str:
    return f"{group}__{item}"


def regex_items(section: dict[str, dict[str, dict[str, list[str]]]]) -> dict[str, tuple[str, str]]:
    """Map ``group__item`` -> (group, regex) for a conditions or procedures config section."""
    out = {}
    for group, items in section.items():
        for item, vocab_codes in items.items():
            out[item_id(group, item)] = (group, code_prefix_regex(vocab_codes))
    return out


def is_ad_source_value() -> pl.Expr:
    return pl.col("condition_source_value").str.contains(AD_SOURCE_REGEX)


def item_group(item: str) -> str:
    return item.split("__", 1)[0]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def describe(values: Iterable[float] | pl.Series | None) -> dict[str, Any]:
    """n, mean, sd (sample), median, q25, q75, min, max of the non-null values."""
    arr = (
        np.array([], dtype=float)
        if values is None
        else pl.Series(values).cast(pl.Float64).drop_nulls().drop_nans().to_numpy()
    )
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def smd_continuous(mean_a: float | None, sd_a: float | None, mean_b: float | None, sd_b: float | None):
    """``(mean_b - mean_a) / sqrt((sd_a^2 + sd_b^2) / 2)``; ``None`` when undefined."""
    if None in (mean_a, sd_a, mean_b, sd_b):
        return None
    pooled = ((sd_a**2 + sd_b**2) / 2) ** 0.5
    return 0.0 if pooled == 0 and mean_a == mean_b else (None if pooled == 0 else (mean_b - mean_a) / pooled)


def smd_binary(p_a: float | None, p_b: float | None):
    """``(p_b - p_a) / sqrt((p_a(1-p_a) + p_b(1-p_b)) / 2)``; ``None`` when undefined."""
    if p_a is None or p_b is None:
        return None
    pooled = ((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2) ** 0.5
    return 0.0 if pooled == 0 and p_a == p_b else (None if pooled == 0 else (p_b - p_a) / pooled)


def share(count: int, total: int) -> float | None:
    return None if total == 0 else count / total


# --------------------------------------------------------------------------
# Persons, cohorts and index dates
# --------------------------------------------------------------------------
def load_persons(person: pl.LazyFrame, death: pl.LazyFrame | None, obs_period: pl.LazyFrame | None) -> pl.DataFrame:
    """One row per person: demographics, birth_date, death_date and observation period bounds."""
    month = pl.col("month_of_birth").fill_null(7).clip(1, 12)
    day = pl.col("day_of_birth").fill_null(1).clip(1, 28)
    out = person.unique("person_id").select(
        "person_id",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
        "location_id",
        "care_site_id",
        pl.col("year_of_birth").alias("birth_year"),
        pl.when(pl.col("year_of_birth").is_not_null())
        .then(pl.date(pl.col("year_of_birth"), month, day))
        .alias("birth_date"),
    )
    if death is not None:
        out = out.join(death.group_by("person_id").agg(pl.col("death_date").min()), on="person_id", how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Date).alias("death_date"))
    if obs_period is not None:
        out = out.join(
            obs_period.group_by("person_id").agg(
                pl.col("observation_period_start_date").min().alias("obs_start"),
                pl.col("observation_period_end_date").max().alias("obs_end"),
            ),
            on="person_id",
            how="left",
        )
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.Date).alias("obs_start"), pl.lit(None, dtype=pl.Date).alias("obs_end")
        )
    return out.collect(engine="streaming")


def visit_summary(visits: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """first_visit, last_visit and n_visits per person."""
    lf = visits.lazy() if isinstance(visits, pl.DataFrame) else visits
    return (
        lf.group_by("person_id")
        .agg(
            pl.col("visit_start_date").min().alias("first_visit"),
            pl.col("visit_start_date").max().alias("last_visit"),
            pl.len().alias("n_visits"),
        )
        .collect(engine="streaming")
    )


def ad_first_dates(ad_rows: pl.DataFrame) -> pl.DataFrame:
    """person_id, first_ad_date, n_ad_rows and the visit of the first AD row (earliest, then lowest id)."""
    if ad_rows.is_empty():
        return pl.DataFrame(
            schema={
                "person_id": pl.Int64,
                "first_ad_date": pl.Date,
                "n_ad_rows": pl.UInt32,
                "first_ad_visit_id": pl.Int64,
            }
        )
    return (
        ad_rows.sort("person_id", "date", "visit_occurrence_id", nulls_last=True)
        .group_by("person_id", maintain_order=True)
        .agg(
            pl.col("date").first().alias("first_ad_date"),
            pl.len().alias("n_ad_rows"),
            pl.col("visit_occurrence_id").first().alias("first_ad_visit_id"),
        )
    )


def assign_cohort(persons: pl.DataFrame, ad_first: pl.DataFrame) -> pl.DataFrame:
    """Add ``cohort`` ('ad' or 'control') and ``first_ad_date`` to persons."""
    out = persons.join(ad_first.select("person_id", "first_ad_date"), on="person_id", how="left")
    return out.with_columns(
        pl.when(pl.col("first_ad_date").is_not_null())
        .then(pl.lit(COHORT_AD))
        .otherwise(pl.lit(COHORT_CONTROL))
        .alias("cohort")
    )


def age_years_expr(index_col: str = "index_date", birth_col: str = "birth_date") -> pl.Expr:
    return (pl.col(index_col) - pl.col(birth_col)).dt.total_days() / 365.25


def age_band_expr(age: pl.Expr, width: int = AGE_BAND_YEARS) -> pl.Expr:
    return (age / width).floor().cast(pl.Int64) * width


def pick_matched_index(case_index: pl.DataFrame, control_visits: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, int]:
    """Pseudo-index per control, see the module docstring.

    ``case_index``: person_id, index_date, birth_date. ``control_visits``: person_id,
    visit_start_date, birth_date. Returns (person_id, index_date) and the fallback count.
    """
    strata = (
        case_index.drop_nulls(["index_date", "birth_date"])
        .with_columns(age_band_expr(age_years_expr()).alias("band"), pl.col("index_date").dt.year().alias("year"))
        .group_by("band", "year")
        .agg(pl.len().alias("n_case"))
    )
    cand = (
        control_visits.select("person_id", pl.col("visit_start_date").alias("index_date"), "birth_date")
        .drop_nulls(["index_date"])
        .unique(["person_id", "index_date"])
        .sort("person_id", "index_date")
        .with_columns(age_band_expr(age_years_expr()).alias("band"), pl.col("index_date").dt.year().alias("year"))
    )
    if cand.is_empty():
        return pl.DataFrame(schema={"person_id": pl.Int64, "index_date": pl.Date}), 0
    cand = cand.join(strata, on=["band", "year"], how="left").with_columns(pl.col("n_case").fill_null(0))
    n_cand = cand.group_by("band", "year").agg(pl.len().alias("n_cand"))
    cand = cand.join(n_cand, on=["band", "year"], how="left").with_columns(
        (pl.col("n_case") / pl.col("n_cand")).fill_null(0.0).alias("w")
    )
    has_weight = cand.group_by("person_id").agg((pl.col("w").sum() > 0).alias("has_w"))
    cand = cand.join(has_weight, on="person_id").with_columns(
        pl.when(pl.col("has_w")).then(pl.col("w")).otherwise(pl.lit(1.0)).alias("w")
    )
    rng = np.random.default_rng(seed)
    u = 1.0 - rng.random(cand.height)
    cand = cand.with_columns(
        pl.Series("key", -np.log(u) / np.where(cand["w"].to_numpy() > 0, cand["w"].to_numpy(), np.nan))
    )
    picked = cand.filter(pl.col("key").is_not_nan()).sort("person_id", "key").unique("person_id", keep="first")
    n_fallback = int(has_weight.filter(~pl.col("has_w")).height)
    return picked.select("person_id", "index_date").sort("person_id"), n_fallback


def build_index_table(
    persons: pl.DataFrame,
    vsum: pl.DataFrame,
    control_visit_dates: pl.DataFrame,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Long table (person_id, view, cohort, index_date) for the three index views plus info."""
    ad = persons.filter(pl.col("cohort") == COHORT_AD).select(
        "person_id", pl.col("first_ad_date").alias("index_date"), "birth_date"
    )
    controls = persons.filter(pl.col("cohort") == COHORT_CONTROL)
    last = controls.join(vsum.select("person_id", pl.col("last_visit").alias("index_date")), on="person_id").select(
        "person_id", "index_date"
    )
    cv = control_visit_dates.join(controls.select("person_id", "birth_date"), on="person_id")
    matched, n_fallback = pick_matched_index(ad, cv, seed)

    def tag(df: pl.DataFrame, view: str) -> pl.DataFrame:
        return df.select(
            "person_id",
            pl.lit(view).alias("view"),
            pl.lit(VIEW_COHORT[view]).alias("cohort"),
            "index_date",
        )

    table = pl.concat([tag(ad, VIEW_AD), tag(last, VIEW_LAST), tag(matched, VIEW_MATCHED)])
    info = {
        "n_controls": controls.height,
        "n_controls_without_visits": controls.height - last.height,
        "n_matched_fallback": n_fallback,
    }
    return table, info


# --------------------------------------------------------------------------
# Visit based features
# --------------------------------------------------------------------------
def visit_type_bucket() -> pl.Expr:
    c = pl.col("visit_concept_id")
    return (
        pl.when(c == 9201)
        .then(pl.lit("inpatient"))
        .when(c == 9202)
        .then(pl.lit("outpatient"))
        .when(c == 9203)
        .then(pl.lit("emergency"))
        .when(c == 0)
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("other"))
    )


def visit_features(visits: pl.DataFrame, idx: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-person visit features before ``idx`` plus the monthly profile.

    ``visits``: person_id, visit_start_date, visit_concept_id, provider_id, care_site_id.
    ``idx``: person_id, index_date. Only visits strictly before the index count. Returns
    (per-person frame, monthly frame with month_offset 0..35 and total n). ``month_offset`` m
    covers ``[index - 30 (m + 1), index - 30 m)``.
    """
    joined = (
        visits.join(idx.select("person_id", "index_date"), on="person_id")
        .filter(pl.col("visit_start_date") < pl.col("index_date"))
        .with_columns(
            (pl.col("index_date") - pl.col("visit_start_date")).dt.total_days().alias("days_before"),
            visit_type_bucket().alias("bucket"),
        )
    )
    aggs = [
        pl.len().alias("n_visits_before"),
        (pl.col("days_before") <= 365).sum().alias("n_w1"),
        ((pl.col("days_before") > 365) & (pl.col("days_before") <= 730)).sum().alias("n_w2"),
        ((pl.col("days_before") > 730) & (pl.col("days_before") <= 1825)).sum().alias("n_w3"),
        pl.col("provider_id").drop_nulls().n_unique().alias("n_providers"),
        pl.col("care_site_id").drop_nulls().n_unique().alias("n_care_sites"),
        pl.col("visit_start_date").unique().sort().diff().dt.total_days().max().alias("max_gap_days"),
    ]
    aggs += [(pl.col("bucket") == b).sum().alias(f"n_{b}") for b in VISIT_TYPE_ORDER]
    per_person = (
        joined.group_by("person_id")
        .agg(aggs)
        .with_columns(
            (pl.col("n_w1")).cast(pl.Float64).alias("visits_per_year_w1"),
            (pl.col("n_w2")).cast(pl.Float64).alias("visits_per_year_w2"),
            (pl.col("n_w3") / 3.0).alias("visits_per_year_w3"),
        )
    )
    monthly = (
        joined.filter(pl.col("days_before") <= 36 * 30)
        .with_columns(((pl.col("days_before") - 1) // 30).alias("month_offset"))
        .group_by("month_offset")
        .agg(pl.len().alias("n"))
        .sort("month_offset")
    )
    return per_person, monthly


def view_features(
    view: str, persons: pl.DataFrame, idx: pl.DataFrame, vsum: pl.DataFrame, vfeat: pl.DataFrame
) -> pl.DataFrame:
    """One row per person of the view with demographics, index, history and utilization features."""
    count_cols = [c for c in vfeat.columns if c.startswith("n_") or c.startswith("visits_per_year")]
    df = (
        idx.select("person_id", "index_date")
        .join(persons, on="person_id", how="left")
        .join(vsum.select("person_id", "first_visit", "last_visit"), on="person_id", how="left")
        .join(vfeat, on="person_id", how="left")
    )
    has_visits = pl.col("first_visit").is_not_null()
    df = df.with_columns(
        [pl.when(has_visits).then(pl.col(c).fill_null(0)).otherwise(None).alias(c) for c in count_cols]
    )
    age = age_years_expr()
    df = df.with_columns(
        pl.lit(view).alias("view"),
        age.alias("age"),
        pl.col("index_date").dt.year().alias("index_year"),
        (pl.col("index_date") - pl.col("first_visit")).dt.total_days().alias("history_days"),
        (pl.max_horizontal(pl.col("last_visit"), pl.col("death_date")) - pl.col("index_date"))
        .dt.total_days()
        .alias("followup_days"),
        (pl.col("obs_end") - pl.col("index_date")).dt.total_days().alias("days_to_obs_end"),
        (pl.col("death_date") - pl.col("index_date")).dt.total_days().alias("days_to_death"),
        (pl.col("death_date") - pl.col("last_visit")).dt.total_days().alias("last_visit_to_death"),
        pl.col("gender_concept_id").replace_strict(GENDER_LABELS, default="other", return_dtype=pl.String).alias("sex"),
        pl.col("race_concept_id").replace_strict(RACE_LABELS, default="other", return_dtype=pl.String).alias("race"),
        pl.col("ethnicity_concept_id")
        .replace_strict(ETHNICITY_LABELS, default="other", return_dtype=pl.String)
        .alias("ethnicity"),
    ).with_columns(
        pl.when(pl.col("age").is_not_null()).then(age_band_expr(pl.col("age"))).alias("age_band"),
        (pl.col("history_days") / 365.25).alias("history_years"),
        pl.when(pl.col("age").is_not_null())
        .then(
            age_band_expr(pl.col("age")).cast(pl.String)
            + pl.lit("-")
            + (age_band_expr(pl.col("age")) + AGE_BAND_YEARS - 1).cast(pl.String)
        )
        .alias("age_band_label"),
        pl.col("days_to_death").is_between(0, 365).fill_null(False).alias("died_within_365"),
        pl.col("last_visit_to_death").is_between(0, 182).fill_null(False).alias("last_visit_within_182_of_death"),
    )
    return df


# --------------------------------------------------------------------------
# Clinical events
# --------------------------------------------------------------------------
def condition_item_events(rows: pl.DataFrame, items: dict[str, tuple[str, str]], source_col: str) -> pl.DataFrame:
    """Long (person_id, date, group, item) events from rows with a source value column."""
    frames = [
        rows.filter(pl.col(source_col).str.contains(regex)).select(
            "person_id", "date", pl.lit(group).alias("group"), pl.lit(name).alias("item")
        )
        for name, (group, regex) in items.items()
    ]
    return _concat_events(frames)


def exact_item_events(rows: pl.DataFrame, items: dict[str, tuple[str, list[str]]], source_col: str) -> pl.DataFrame:
    frames = [
        rows.filter(pl.col(source_col).is_in(codes)).select(
            "person_id", "date", pl.lit(group).alias("group"), pl.lit(name).alias("item")
        )
        for name, (group, codes) in items.items()
    ]
    return _concat_events(frames)


def _concat_events(frames: list[pl.DataFrame]) -> pl.DataFrame:
    schema = {"person_id": pl.Int64, "date": pl.Date, "group": pl.String, "item": pl.String}
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


def measurement_items(section: dict[str, dict[str, list[str]]]) -> dict[str, tuple[str, list[str]]]:
    return {item_id(g, i): (g, [str(c) for c in codes]) for g, items in section.items() for i, codes in items.items()}


def union_regex(items: dict[str, tuple[str, str]]) -> str:
    return "|".join(f"(?:{regex})" for _, regex in items.values())


def resolve_drug_concepts(
    concept: pl.LazyFrame, ancestor: pl.LazyFrame, groups: dict[str, list[str]]
) -> tuple[pl.DataFrame, list[str]]:
    """Map drug_concept_id -> item (``group__ingredient``) via ingredient descendants.

    Ingredients are standard RxNorm 'Ingredient' concepts matched by lowercase name. Returns the
    mapping and the list of names that did not resolve.
    """
    names = {item_id(g, n.lower()): (g, n.lower()) for g, ns in groups.items() for n in ns}
    wanted = sorted({n for _, n in names.values()})
    ingredients = (
        concept.filter(
            (pl.col("vocabulary_id") == "RxNorm")
            & (pl.col("standard_concept") == "S")
            & (pl.col("concept_class_id") == "Ingredient")
            & (pl.col("concept_name").str.to_lowercase().is_in(wanted))
        )
        .select(
            pl.col("concept_id").alias("ancestor_concept_id"), pl.col("concept_name").str.to_lowercase().alias("name")
        )
        .unique()
        .collect()
    )
    desc = (
        ancestor.filter(pl.col("ancestor_concept_id").is_in(ingredients["ancestor_concept_id"].implode()))
        .select("ancestor_concept_id", pl.col("descendant_concept_id").alias("drug_concept_id"))
        .collect(engine="streaming")
        .join(ingredients, on="ancestor_concept_id")
    )
    rows = [{"name": n, "item": i, "group": g} for i, (g, n) in names.items()]
    mapping = desc.join(pl.DataFrame(rows), on="name").select("drug_concept_id", "group", "item").unique()
    unresolved = sorted(set(wanted) - set(ingredients["name"].to_list()))
    return mapping, unresolved


def events_before_index(events: pl.DataFrame, idx: pl.DataFrame, window_days: int | None = None) -> pl.DataFrame:
    """Events inside ``[index - window_days, index)`` (all earlier history when ``window_days`` is None)."""
    j = events.join(idx.select("person_id", "index_date"), on="person_id").filter(pl.col("date") < pl.col("index_date"))
    if window_days is not None:
        j = j.filter(pl.col("date") >= pl.col("index_date") - pl.duration(days=window_days))
    return j


def prevalence(
    events: pl.DataFrame, idx: pl.DataFrame, window_days: int | None, items: Iterable[str]
) -> dict[str, Any]:
    """Persons with at least one event in the window per item and per group, with the denominator."""
    n = idx.height
    j = events_before_index(events, idx, window_days)
    per_item = j.group_by("item").agg(pl.col("person_id").n_unique().alias("n"))
    per_group = j.group_by("group").agg(pl.col("person_id").n_unique().alias("n"))
    item_counts = {r["item"]: r["n"] for r in per_item.iter_rows(named=True)}
    group_counts = {r["group"]: r["n"] for r in per_group.iter_rows(named=True)}
    return {
        "n": n,
        "items": {i: {"n": item_counts.get(i, 0), "share": share(item_counts.get(i, 0), n)} for i in items},
        "groups": {g: {"n": c, "share": share(c, n)} for g, c in group_counts.items()},
    }


# --------------------------------------------------------------------------
# Section summaries over per-view feature frames
# --------------------------------------------------------------------------
def counts_by(df: pl.DataFrame, col: str) -> dict[str, int]:
    out = df.group_by(col).agg(pl.len().alias("n")).sort(col)
    return {str(r[col]): r["n"] for r in out.iter_rows(named=True)}


def demographics_section(feats: dict[str, pl.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for view, df in feats.items():
        n = df.height
        bins = df.drop_nulls("age_band").group_by("age_band").agg(pl.len().alias("n")).sort("age_band")
        out[view] = {
            "n": n,
            "age": describe(df["age"]),
            "age_bands": {
                f"{r['age_band']}-{r['age_band'] + AGE_BAND_YEARS - 1}": r["n"] for r in bins.iter_rows(named=True)
            },
            "birth_year": describe(df["birth_year"]),
            "sex": counts_by(df, "sex"),
            "race": counts_by(df, "race"),
            "ethnicity": counts_by(df, "ethnicity"),
        }
    return out


def smd_section(feats: dict[str, pl.DataFrame], continuous: list[str], categorical: list[str]) -> dict[str, Any]:
    """SMD (control view minus AD) per continuous variable and per categorical level."""
    out: dict[str, Any] = {}
    ad = feats.get(VIEW_AD)
    for view in CONTROL_VIEWS:
        ctl = feats.get(view)
        if ad is None or ctl is None:
            continue
        res: dict[str, Any] = {}
        for var in continuous:
            a, c = describe(ad[var]), describe(ctl[var])
            res[var] = smd_continuous(a["mean"], a["sd"], c["mean"], c["sd"])
        for var in categorical:
            levels = sorted(set(ad[var].drop_nulls().to_list()) | set(ctl[var].drop_nulls().to_list()))
            res[var] = {
                lv: smd_binary(
                    share(int((ad[var] == lv).sum()), ad.height), share(int((ctl[var] == lv).sum()), ctl.height)
                )
                for lv in levels
            }
        out[view] = res
    return out


def age_band_prevalence(ad: pl.DataFrame, control_last: pl.DataFrame) -> dict[str, Any]:
    """AD cases per persons in each age band, using the AD index and the last visit control index."""
    a = ad.drop_nulls("age_band").group_by("age_band").agg(pl.len().alias("n_ad"))
    c = control_last.drop_nulls("age_band").group_by("age_band").agg(pl.len().alias("n_control"))
    j = a.join(c, on="age_band", how="full", coalesce=True).fill_null(0).sort("age_band")
    return {
        f"{r['age_band']}-{r['age_band'] + AGE_BAND_YEARS - 1}": {
            "n_ad": r["n_ad"],
            "n_control": r["n_control"],
            "ad_prevalence": share(r["n_ad"], r["n_ad"] + r["n_control"]),
        }
        for r in j.iter_rows(named=True)
    }


def index_date_section(feats: dict[str, pl.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for view, df in feats.items():
        n = df.height
        with_death = df.filter(pl.col("death_date").is_not_null())
        out[view] = {
            "n": n,
            "index_year_counts": counts_by(df.drop_nulls("index_year"), "index_year"),
            "year_by_age_band": {
                f"{r['index_year']}|{r['age_band']}": r["n"]
                for r in df.drop_nulls(["index_year", "age_band"])
                .group_by("index_year", "age_band")
                .agg(pl.len().alias("n"))
                .sort("index_year", "age_band")
                .iter_rows(named=True)
            },
            "days_to_obs_end": describe(df["days_to_obs_end"]),
            "days_to_death": describe(with_death["days_to_death"]),
            "share_died": share(with_death.height, n),
            "share_died_within_365": share(int(df["died_within_365"].sum()), n),
            "share_last_visit_within_182_before_death": share(int(df["last_visit_within_182_of_death"].sum()), n),
            "share_last_visit_within_182_before_death_among_died": share(
                int(with_death["last_visit_within_182_of_death"].sum()), with_death.height
            ),
        }
    return out


def history_section(feats: dict[str, pl.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for view, df in feats.items():
        h = df.drop_nulls("history_days")
        out[view] = {
            "n": df.height,
            "n_with_visits": h.height,
            "history_years": describe(h["history_years"]),
            "followup_days": describe(df["followup_days"]),
            "share_history_ge_years": {
                str(y): share(int((h["history_days"] >= y * 365).sum()), h.height) for y in (1, 2, 3, 5)
            },
            "n_index_before_first_visit": int((h["history_days"] < 0).sum()),
            "max_gap_days": describe(df["max_gap_days"]),
        }
    return out


def utilization_section(feats: dict[str, pl.DataFrame], monthly: dict[str, pl.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for view, df in feats.items():
        n_vis = df.drop_nulls("n_visits_before")
        total = int(n_vis["n_visits_before"].sum()) if n_vis.height else 0
        mdf = monthly.get(view)
        months = (
            {int(r["month_offset"]): r["n"] / df.height for r in mdf.iter_rows(named=True)}
            if mdf is not None and df.height
            else {}
        )
        out[view] = {
            "n": df.height,
            "visits_per_year": {w: describe(df[f"visits_per_year_{w}"]) for w in ("w1", "w2", "w3")},
            "visit_type_share": {b: share(int(n_vis[f"n_{b}"].sum()), total) for b in VISIT_TYPE_ORDER},
            "visit_type_mean_per_patient": {
                b: (float(n_vis[f"n_{b}"].mean()) if n_vis.height else None) for b in VISIT_TYPE_ORDER
            },
            "monthly_visits_per_patient": {str(m): months.get(m, 0.0) for m in range(36)},
            "n_providers": describe(df["n_providers"]),
            "n_care_sites": describe(df["n_care_sites"]),
        }
    return out


# --------------------------------------------------------------------------
# Exclusion flow
# --------------------------------------------------------------------------
def exclusion_flow(
    persons: pl.DataFrame,
    vsum: pl.DataFrame,
    contaminated: set[int],
    date_floor: str,
    min_history_days: int = MIN_HISTORY_DAYS,
    ad_early_days: int = AD_EARLY_DAYS,
) -> pl.DataFrame:
    """Sequential candidate exclusions per cohort (control index = last visit). Reported only."""
    floor = pl.lit(date_floor).str.to_date()
    base = (
        persons.join(vsum.select("person_id", "first_visit", "last_visit"), on="person_id", how="left")
        .with_columns(
            pl.when(pl.col("cohort") == COHORT_AD)
            .then(pl.col("first_ad_date"))
            .otherwise(pl.col("last_visit"))
            .alias("index_date")
        )
        .with_columns(age_years_expr().alias("age"))
        .with_columns(
            (pl.col("index_date") - pl.col("first_visit")).dt.total_days().alias("history_days"),
            pl.col("person_id").is_in(pl.Series(sorted(contaminated), dtype=pl.Int64).implode()).alias("contaminated"),
            (pl.col("death_date") - pl.col("last_visit")).dt.total_days().alias("death_lag"),
        )
    )
    steps: list[tuple[str, str | None, pl.Expr]] = [
        ("no_visits", None, pl.col("first_visit").is_null()),
        (
            "birth_year_missing_or_implausible_age",
            None,
            pl.col("age").is_null() | (pl.col("age") < MIN_PLAUSIBLE_AGE) | (pl.col("age") > MAX_PLAUSIBLE_AGE),
        ),
        ("index_before_date_floor", None, pl.col("index_date") < floor),
        (f"history_lt_{min_history_days}_days", None, pl.col("history_days") < min_history_days),
        ("control_dementia_proxy_or_ad_drug", COHORT_CONTROL, pl.col("contaminated")),
        ("control_death_within_365_days_of_last_visit", COHORT_CONTROL, pl.col("death_lag") <= 365),
        (
            f"ad_first_diagnosis_within_{ad_early_days}_days_of_first_visit",
            COHORT_AD,
            pl.col("history_days") < ad_early_days,
        ),
    ]
    rows = []
    for cohort in (COHORT_AD, COHORT_CONTROL):
        remaining = base.filter(pl.col("cohort") == cohort)
        rows.append(
            {
                "cohort": cohort,
                "step": "candidates",
                "n_removed": 0,
                "n_remaining": remaining.height,
                "n_meeting_alone": 0,
            }
        )
        for name, only, cond in steps:
            if only is not None and only != cohort:
                continue
            alone = int(remaining.select(cond.fill_null(False).sum()).item())
            remaining = remaining.filter(~cond.fill_null(False))
            rows.append(
                {
                    "cohort": cohort,
                    "step": name,
                    "n_removed": 0,
                    "n_remaining": remaining.height,
                    "n_meeting_alone": alone,
                }
            )
        # n_removed from consecutive remaining counts
        prev = None
        for r in rows:
            if r["cohort"] != cohort:
                continue
            if prev is not None:
                r["n_removed"] = prev - r["n_remaining"]
            prev = r["n_remaining"]
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------
def table_quality(
    lf: pl.LazyFrame,
    persons: pl.DataFrame,
    date_col: str,
    concept_col: str,
    release_date: Any,
    date_floor: str,
    duplicates: bool = True,
) -> dict[str, Any]:
    """Row counts, concept-0 share, date problems, observation period misses and duplicates by cohort.

    ``lf`` must hold person_id, ``date_col`` and ``concept_col``. Duplicates are rows sharing
    (person, concept, date) beyond the first one. The result has the keys 'all', 'ad', 'control'
    and a per-year row count ('rows_per_year').
    """
    ctx = persons.lazy().select("person_id", "cohort", "birth_year", "death_date", "obs_start", "obs_end")
    d = pl.col(date_col)
    floor = pl.lit(date_floor).str.to_date()
    aggs = [
        pl.len().alias("n_rows"),
        (pl.col(concept_col) == 0).sum().alias("n_concept_zero"),
        d.is_null().sum().alias("n_null_date"),
        (d.dt.year() < pl.col("birth_year")).sum().alias("n_before_birth_year"),
        (d > pl.col("death_date")).sum().alias("n_after_death"),
        (d < floor).sum().alias("n_before_floor"),
        ((d < pl.col("obs_start")) | (d > pl.col("obs_end"))).sum().alias("n_outside_observation_period"),
    ]
    if release_date is not None:
        aggs.append((d > pl.lit(release_date)).sum().alias("n_after_release"))
    else:
        aggs.append(pl.lit(0).alias("n_after_release"))
    if duplicates:
        aggs.append(pl.struct("person_id", concept_col, date_col).hash().n_unique().alias("n_unique_keys"))
    else:
        aggs.append(pl.lit(None, dtype=pl.UInt32).alias("n_unique_keys"))
    by_year = (
        lf.join(ctx, on="person_id").group_by("cohort", d.dt.year().alias("year")).agg(aggs).collect(engine="streaming")
    )
    persons_with_rows = (
        lf.select("person_id")
        .unique()
        .join(ctx.select("person_id", "cohort"), on="person_id")
        .group_by("cohort")
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    pw = {r["cohort"]: r["n"] for r in persons_with_rows.iter_rows(named=True)}
    n_persons = {c: int(persons.filter(pl.col("cohort") == c).height) for c in (COHORT_AD, COHORT_CONTROL)}
    sum_cols = [c for c in by_year.columns if c.startswith("n_") and c != "n_unique_keys"]

    def summarize(frame: pl.DataFrame, persons_with: int, persons_total: int) -> dict[str, Any]:
        tot = {c: int(frame[c].sum() or 0) for c in sum_cols}
        n_rows = tot["n_rows"]
        out: dict[str, Any] = {
            **tot,
            "n_persons": persons_total,
            "n_persons_with_rows": persons_with,
            "rows_per_person": share(n_rows, persons_total),
            "share_concept_zero": share(tot["n_concept_zero"], n_rows),
        }
        if duplicates:
            uniq = int(frame["n_unique_keys"].sum() or 0)
            out["n_duplicate_rows"] = n_rows - uniq
            out["share_duplicate_rows"] = share(n_rows - uniq, n_rows)
        for k in (
            "n_before_birth_year",
            "n_after_death",
            "n_before_floor",
            "n_after_release",
            "n_outside_observation_period",
        ):
            out["share" + k[1:]] = share(tot[k], n_rows)
        return out

    result: dict[str, Any] = {}
    for cohort in (COHORT_AD, COHORT_CONTROL):
        result[cohort] = summarize(by_year.filter(pl.col("cohort") == cohort), pw.get(cohort, 0), n_persons[cohort])
    result["all"] = summarize(by_year, sum(pw.values()), sum(n_persons.values()))
    per_year = by_year.group_by("cohort", "year").agg(pl.col("n_rows").sum()).sort("year")
    result["rows_per_year"] = {
        c: {str(r["year"]): r["n_rows"] for r in per_year.filter(pl.col("cohort") == c).iter_rows(named=True)}
        for c in (COHORT_AD, COHORT_CONTROL)
    }
    return result


def person_field_quality(persons: pl.DataFrame) -> dict[str, Any]:
    """Share of concept id 0 for gender, race and ethnicity, plus missing birth year, by cohort."""
    out: dict[str, Any] = {}
    for label, df in (
        ("all", persons),
        (COHORT_AD, persons.filter(pl.col("cohort") == COHORT_AD)),
        (COHORT_CONTROL, persons.filter(pl.col("cohort") == COHORT_CONTROL)),
    ):
        n = df.height
        out[label] = {
            "n_persons": n,
            **{
                f"share_{c}_zero": share(int((df[c] == 0).sum()), n)
                for c in ("gender_concept_id", "race_concept_id", "ethnicity_concept_id")
            },
            "share_birth_year_missing": share(int(df["birth_year"].is_null().sum()), n),
            "share_death_recorded": share(int(df["death_date"].is_not_null().sum()), n),
        }
    return out


# --------------------------------------------------------------------------
# Outcome validity, contamination, site and source
# --------------------------------------------------------------------------
def ad_code_label(source_value: str) -> str:
    """Bucket an AD source value: G30.0, G30.1, G30.8, G30.9, G30_other or 331.0."""
    if source_value.startswith("ICD-9-CM:"):
        return "331.0"
    code = source_value.split(":", 1)[-1]
    return code[:5] if code[:5] in ("G30.0", "G30.1", "G30.8", "G30.9") else "G30_other"


def outcome_validity(
    ad_rows: pl.DataFrame,
    ad_first: pl.DataFrame,
    vsum: pl.DataFrame,
    visits: pl.DataFrame,
    proxy_events: pl.DataFrame,
    early_days: int = AD_EARLY_DAYS,
) -> dict[str, Any]:
    """Validity of the AD outcome among cases.

    ``ad_rows``: person_id, date, source, visit_occurrence_id. ``proxy_events`` are the dementia
    proxy events (person_id, date). Proxy lag is first AD date minus first proxy date.
    """
    n = ad_first.height
    labelled = ad_rows.with_columns(pl.col("source").map_elements(ad_code_label, return_dtype=pl.String).alias("code"))
    codes = (
        labelled.group_by("code")
        .agg(pl.len().alias("n_rows"), pl.col("person_id").n_unique().alias("n_persons"))
        .sort("code")
    )
    first_proxy = proxy_events.group_by("person_id").agg(pl.col("date").min().alias("first_proxy"))
    lag = ad_first.join(first_proxy, on="person_id", how="left").with_columns(
        (pl.col("first_ad_date") - pl.col("first_proxy")).dt.total_days().alias("proxy_to_ad_days")
    )
    with_proxy = lag.drop_nulls("proxy_to_ad_days")
    linked = ad_first.join(
        visits.select("visit_occurrence_id", "visit_concept_id").unique("visit_occurrence_id"),
        left_on="first_ad_visit_id",
        right_on="visit_occurrence_id",
        how="left",
    )
    linked_ok = linked.drop_nulls("visit_concept_id")
    early = ad_first.join(vsum.select("person_id", "first_visit"), on="person_id", how="inner").with_columns(
        (pl.col("first_ad_date") - pl.col("first_visit")).dt.total_days().alias("days_from_first_visit")
    )
    return {
        "n_cases": n,
        "code_counts": {
            r["code"]: {"n_rows": r["n_rows"], "n_persons": r["n_persons"]} for r in codes.iter_rows(named=True)
        },
        "ad_rows_per_patient": describe(ad_first["n_ad_rows"]),
        "share_exactly_one_ad_row": share(int((ad_first["n_ad_rows"] == 1).sum()), n),
        "days_first_proxy_to_first_ad": describe(with_proxy["proxy_to_ad_days"]),
        "share_with_proxy_before_first_ad": share(int((lag["proxy_to_ad_days"] > 0).sum()), n),
        "share_first_ad_on_inpatient_visit": share(
            int((linked_ok["visit_concept_id"] == 9201).sum()), linked_ok.height
        ),
        "n_first_ad_without_linked_visit": n - linked_ok.height,
        f"share_first_ad_within_{early_days}_days_of_first_visit": share(
            int((early["days_from_first_visit"] < early_days).sum()), early.height
        ),
    }


def first_exposure_days(events: pl.DataFrame, idx: pl.DataFrame, group: str) -> dict[str, Any]:
    """Days from the first event of ``group`` before the index to the index, among exposed persons."""
    j = events_before_index(events.filter(pl.col("group") == group), idx)
    first = j.group_by("person_id").agg(pl.col("date").min().alias("first"), pl.col("index_date").first())
    days = (first["index_date"] - first["first"]).dt.total_days()
    return {"n_exposed": first.height, "days_first_exposure_to_index": describe(days)}


def ever_prevalence(events: pl.DataFrame, persons_in_view: pl.DataFrame, items: Iterable[str]) -> dict[str, Any]:
    """Persons with at least one event at any date, per item and per group."""
    ids = persons_in_view.select("person_id")
    j = events.join(ids, on="person_id", how="semi")
    n = ids.height
    per_item = {
        r["item"]: r["n"]
        for r in j.group_by("item").agg(pl.col("person_id").n_unique().alias("n")).iter_rows(named=True)
    }
    per_group = {
        r["group"]: r["n"]
        for r in j.group_by("group").agg(pl.col("person_id").n_unique().alias("n")).iter_rows(named=True)
    }
    return {
        "n": n,
        "items": {i: {"n": per_item.get(i, 0), "share": share(per_item.get(i, 0), n)} for i in items},
        "groups": {g: {"n": c, "share": share(c, n)} for g, c in per_group.items()},
    }


def contaminated_controls(events: pl.DataFrame, control_ids: pl.Series) -> set[int]:
    """Control person_ids with any dementia proxy or AD drug event (any date)."""
    hit = events.filter(pl.col("group").is_in(["dementia_proxies", "ad_drugs"]))
    return set(hit.filter(pl.col("person_id").is_in(control_ids.implode()))["person_id"].unique().to_list())


def contamination_section(
    feats: dict[str, pl.DataFrame], events: pl.DataFrame, items: Iterable[str], min_history_days: int = MIN_HISTORY_DAYS
) -> dict[str, Any]:
    """Control contamination: dementia proxies and AD drugs, early death and short history.

    ``control_last_visit`` counts events at any date, ``control_matched_index`` only before the
    pseudo-index.
    """
    items = [i for i in items if item_group(i) in ("dementia_proxies", "ad_drugs")]
    out: dict[str, Any] = {}
    for view in CONTROL_VIEWS:
        df = feats.get(view)
        if df is None:
            continue
        if view == VIEW_LAST:
            prev = ever_prevalence(events, df, items)
        else:
            prev = prevalence(events, df, None, items)
        any_hit = events.filter(pl.col("group").is_in(["dementia_proxies", "ad_drugs"]))
        if view == VIEW_LAST:
            hit_ids = any_hit.join(df.select("person_id"), on="person_id", how="semi")["person_id"].n_unique()
        else:
            hit_ids = events_before_index(any_hit, df)["person_id"].n_unique()
        n = df.height
        out[view] = {
            "n": n,
            "n_any_proxy_or_ad_drug": int(hit_ids),
            "share_any_proxy_or_ad_drug": share(int(hit_ids), n),
            "by_group": prev["groups"],
            "by_item": prev["items"],
            "share_died_within_365_of_index": share(int(df["died_within_365"].sum()), n),
            f"share_history_lt_{min_history_days}_days": share(int((df["history_days"] < min_history_days).sum()), n),
        }
    return out


def _null_and_top(df: pl.DataFrame, col: str, top: int = 5) -> dict[str, Any]:
    n = df.height
    counts = df.drop_nulls(col).group_by(col).agg(pl.len().alias("n")).sort("n", descending=True).head(top)
    return {
        "share_null": share(int(df[col].is_null().sum()), n),
        "top": {str(r[col]): r["n"] for r in counts.iter_rows(named=True)},
    }


def mix_by_cohort(df: pl.DataFrame, col: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cohort in (COHORT_AD, COHORT_CONTROL):
        sub = df.filter(pl.col("cohort") == cohort)
        counts = sub.group_by(col).agg(pl.len().alias("n")).sort(col)
        out[cohort] = {str(r[col]): r["n"] for r in counts.iter_rows(named=True)}
    return out


def site_source_section(
    persons: pl.DataFrame, visits: pl.DataFrame, condition_types: pl.DataFrame | None
) -> dict[str, Any]:
    """Site fields and record-type mix by cohort. ``condition_types``: cohort, condition_type_concept_id, n."""
    v = visits.join(persons.select("person_id", "cohort"), on="person_id", how="inner")
    out: dict[str, Any] = {}
    for cohort in (COHORT_AD, COHORT_CONTROL):
        p = persons.filter(pl.col("cohort") == cohort)
        vc = v.filter(pl.col("cohort") == cohort)
        out[cohort] = {
            "person_care_site_id": _null_and_top(p, "care_site_id"),
            "person_location_id": _null_and_top(p, "location_id"),
            "visit_care_site_id": _null_and_top(vc, "care_site_id"),
        }
    out["visit_type_concept_id_mix"] = mix_by_cohort(v, "visit_type_concept_id")
    if condition_types is not None:
        out["condition_type_concept_id_mix"] = {
            c: {
                str(r["condition_type_concept_id"]): r["n"]
                for r in condition_types.filter(pl.col("cohort") == c)
                .sort("condition_type_concept_id")
                .iter_rows(named=True)
            }
            for c in (COHORT_AD, COHORT_CONTROL)
        }
    return out


# --------------------------------------------------------------------------
# Table 1
# --------------------------------------------------------------------------
def _level_key(level: str) -> tuple:
    head = level.split("-")[0]
    return (0, int(head), level) if head.isdigit() else (1, 0, level)


def _fmt_count(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else f"{n} (NA)"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    return "NA" if x is None else f"{x:.{digits}f}"


def build_table1(
    feats: dict[str, pl.DataFrame],
    continuous: list[str],
    categorical: list[str],
    binary: dict[str, dict[str, tuple[int, int]]] | None = None,
) -> list[dict[str, Any]]:
    """Rows of Table 1: one per variable/level with a column per view and SMD vs AD.

    ``binary`` maps a variable name to ``{view: (n_positive, n_total)}``. SMD is the control view
    minus AD (positive: the control mean is higher).
    """
    views = [v for v in VIEWS if v in feats]
    rows: list[dict[str, Any]] = []

    def row(variable: str, level: str, cells: dict[str, str], smds: dict[str, float | None] | None = None) -> None:
        r: dict[str, Any] = {"variable": variable, "level": level}
        for v in VIEWS:
            r[v] = cells.get(v, "")
        for v in CONTROL_VIEWS:
            r[f"smd_{v}"] = (smds or {}).get(v)
        rows.append(r)

    row("n", "", {v: str(feats[v].height) for v in views})
    for var in continuous:
        stats = {v: describe(feats[v][var]) for v in views}
        smds = {
            v: smd_continuous(stats[VIEW_AD]["mean"], stats[VIEW_AD]["sd"], stats[v]["mean"], stats[v]["sd"])
            for v in CONTROL_VIEWS
            if VIEW_AD in stats and v in stats
        }
        row(var, "mean (sd)", {v: f"{_fmt_num(s['mean'])} ({_fmt_num(s['sd'])})" for v, s in stats.items()}, smds)
        row(
            var,
            "median [q25, q75]",
            {v: f"{_fmt_num(s['median'])} [{_fmt_num(s['q25'])}, {_fmt_num(s['q75'])}]" for v, s in stats.items()},
        )
    for var in categorical:
        filled = {v: feats[v][var].cast(pl.String).fill_null("missing") for v in views}
        levels = sorted({lv for s in filled.values() for lv in s.unique().to_list()}, key=_level_key)
        for lv in levels:
            counts = {v: int((filled[v] == lv).sum()) for v in views}
            props = {v: share(counts[v], feats[v].height) for v in views}
            smds = {v: smd_binary(props[VIEW_AD], props[v]) for v in CONTROL_VIEWS if VIEW_AD in props and v in props}
            row(var, lv, {v: _fmt_count(counts[v], feats[v].height) for v in views}, smds)
    for name, per_view in (binary or {}).items():
        props = {v: share(c, n) for v, (c, n) in per_view.items()}
        smds = {
            v: smd_binary(props.get(VIEW_AD), props.get(v)) for v in CONTROL_VIEWS if v in props and VIEW_AD in props
        }
        row(name, "yes", {v: _fmt_count(c, n) for v, (c, n) in per_view.items()}, smds)
    return rows


def table1_markdown(rows: list[dict[str, Any]]) -> str:
    header = ["variable", "level", *VIEWS, *[f"smd_{v}" for v in CONTROL_VIEWS]]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [str(r["variable"]), str(r["level"]), *[str(r[v]) for v in VIEWS]]
        cells += ["" if r[f"smd_{v}"] is None else f"{r[f'smd_{v}']:.3f}" for v in CONTROL_VIEWS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
