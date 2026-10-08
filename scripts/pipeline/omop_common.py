"""OMOP helpers shared by the cohort (step 6), the raw profile (step 18) and their tests.

Table scanning, AD code matching, birth date and age expressions, standardized mean differences
and the iterative proportional fitting of sampling weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from .omop_validation import table_glob

AD_CODE_PATTERNS = [r"ICD-9-CM:331\.0", r"ICD-10-CM:G30"]
AD_SOURCE_REGEX = "^(" + "|".join(AD_CODE_PATTERNS) + ")"


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


def is_ad_source_value() -> pl.Expr:
    return pl.col("condition_source_value").str.contains(AD_SOURCE_REGEX)


def smd_continuous(mean_a: float | None, sd_a: float | None, mean_b: float | None, sd_b: float | None):
    """``(mean_b - mean_a) / sqrt((sd_a^2 + sd_b^2) / 2)``; ``None`` when undefined."""
    if None in (mean_a, sd_a, mean_b, sd_b):
        return None
    pooled = ((sd_a**2 + sd_b**2) / 2) ** 0.5
    return 0.0 if pooled == 0 and mean_a == mean_b else (None if pooled == 0 else (mean_b - mean_a) / pooled)


def birth_date_expr() -> pl.Expr:
    """Birth date from year/month/day of birth; missing month defaults to 7, missing day to 1."""
    month = pl.col("month_of_birth").fill_null(7).clip(1, 12)
    day = pl.col("day_of_birth").fill_null(1).clip(1, 28)
    return pl.when(pl.col("year_of_birth").is_not_null()).then(pl.date(pl.col("year_of_birth"), month, day))


def age_years_expr(index_col: str = "index_date", birth_col: str = "birth_date") -> pl.Expr:
    return (pl.col(index_col) - pl.col(birth_col)).dt.total_days() / 365.25


def fit_stratum_weights(cand: pl.DataFrame, strata: pl.DataFrame, n_iter: int = 20) -> pl.DataFrame:
    """Per-stratum sampling weights ``w`` by iterative proportional fitting.

    Each control picks one of its candidate visits with probability proportional to ``w`` of the
    candidate's stratum. Starting from the case counts, ``w`` is rescaled so the expected number of
    pseudo-indexes per stratum approaches the case distribution (scaled to the number of controls).
    Strata no control can reach keep their last weight.
    """
    per_person = cand.group_by("person_id", "band", "year").agg(pl.len().alias("k"))
    n_controls = per_person["person_id"].n_unique()
    w = strata.with_columns(
        pl.col("n_case").cast(pl.Float64).alias("w"),
        (pl.col("n_case") / pl.col("n_case").sum() * n_controls).alias("target"),
    )
    for _ in range(n_iter):
        j = per_person.join(w.select("band", "year", "w"), on=["band", "year"], how="left").with_columns(
            (pl.col("k") * pl.col("w").fill_null(0.0)).alias("kw")
        )
        j = j.with_columns((pl.col("kw") / pl.col("kw").sum().over("person_id")).alias("p"))
        expected = j.group_by("band", "year").agg(pl.col("p").sum().alias("expected"))
        w = (
            w.join(expected, on=["band", "year"], how="left")
            .with_columns(
                pl.when(pl.col("expected") > 1e-9)
                .then(pl.col("w") * pl.col("target") / pl.col("expected"))
                .otherwise(pl.col("w"))
                .alias("w")
            )
            .drop("expected")
        )
    return w.select("band", "year", "w")
