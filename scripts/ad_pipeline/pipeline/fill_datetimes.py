"""Fill null ``*_datetime`` columns of OMOP parquet tables from the matching ``*_date`` column.

Files are rewritten one at a time with polars streaming (bounded memory), into a temporary
file next to the original, verified, then atomically swapped in. Non-null datetimes are
never changed, so a second run is a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import pyarrow.parquet as pq

# Must not end in .parquet, so an interrupted run never leaves a file the table glob picks up.
TMP_SUFFIX = ".fill_tmp"
DERIVED_SUBPATH = Path("data") / "derived"
BIRTH_COLUMN = ("person", "birth_datetime")


@dataclass
class ColumnStats:
    table: str
    column: str
    source: str
    rows: int = 0
    nulls_before: int = 0
    filled: int = 0
    nulls_after: int = 0


@dataclass
class TableResult:
    table: str
    files: List[Path] = field(default_factory=list)
    columns: Dict[str, ColumnStats] = field(default_factory=dict)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (column, reason)
    files_rewritten: int = 0


def derived_dir() -> Path:
    return (Path.home() / DERIVED_SUBPATH).resolve()


def guard_data_dir(data_dir: Path) -> Optional[str]:
    """Return an error message when ``data_dir`` (symlinks resolved) is under ~/data/derived."""
    resolved = Path(data_dir).resolve()
    root = derived_dir()
    if resolved == root or root in resolved.parents:
        return f"{resolved} is under {root}; the original source data must never be rewritten"
    return None


def table_files(data_dir: Path, table: str) -> List[Path]:
    sub = data_dir / table
    if sub.is_dir():
        return sorted(sub.glob("*.parquet"))
    flat = data_dir / f"{table}.parquet"
    return [flat] if flat.exists() else []


def list_tables(data_dir: Path) -> List[str]:
    names = {p.name for p in data_dir.iterdir() if p.is_dir() and any(p.glob("*.parquet"))}
    names |= {p.stem for p in data_dir.glob("*.parquet")}
    return sorted(names)


def fill_expr(table: str, column: str, schema: Dict[str, pl.DataType]) -> Tuple[Optional[pl.Expr], str]:
    """Return (fill value cast to the column's dtype, source description).

    Returns (None, reason) when the column cannot be handled. An all-null column stored
    with the Null type becomes Datetime("us"); any other non-datetime dtype is refused.
    """
    dtype = schema[column]
    if dtype == pl.Null:
        target: pl.DataType = pl.Datetime("us")
    elif isinstance(dtype, pl.Datetime):
        target = dtype
    else:
        return None, f"unsupported dtype {dtype} (expected a datetime type)"

    if (table, column) == BIRTH_COLUMN:
        if "year_of_birth" not in schema:
            return None, "no year_of_birth column"
        year = pl.col("year_of_birth")
        jan1 = pl.datetime(year, 1, 1)
        if "month_of_birth" in schema:
            month = pl.col("month_of_birth")
            day = pl.col("day_of_birth").fill_null(1) if "day_of_birth" in schema else pl.lit(1)
            full = pl.format("{}-{}-{}", year, month, day).str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime("us"))
            value = pl.when(month.is_null()).then(jan1).otherwise(full.fill_null(jan1))
            source = "year_of_birth, month_of_birth, day_of_birth (January 1 when month is null)"
        else:
            value, source = jan1, "year_of_birth (January 1)"
    else:
        date_col = column[: -len("datetime")] + "date"
        if date_col not in schema:
            return None, f"no matching {date_col} column"
        if schema[date_col] != pl.Date and not isinstance(schema[date_col], pl.Datetime):
            return None, f"{date_col} has unsupported dtype {schema[date_col]}"
        value = pl.col(date_col).cast(pl.Date).cast(pl.Datetime("us"))
        source = date_col

    value = value.cast(pl.Datetime(target.time_unit))
    if target.time_zone:
        value = value.dt.replace_time_zone(target.time_zone)
    return value, source


def _null_counts(path: Path, exprs: Dict[str, pl.Expr]) -> Dict[str, int]:
    row = pl.scan_parquet(path).select([e.alias(c) for c, e in exprs.items()]).collect()
    return {c: int(row[c][0]) for c in exprs}


def process_table(data_dir: Path, table: str, dry_run: bool) -> TableResult:
    result = TableResult(table=table, files=table_files(data_dir, table))
    for path in result.files:
        schema = dict(pl.scan_parquet(path).collect_schema())
        fills: Dict[str, pl.Expr] = {}
        for column in schema:
            if not column.endswith("_datetime"):
                continue
            expr, source = fill_expr(table, column, schema)
            if expr is None:
                if all(c != column for c, _ in result.skipped):
                    result.skipped.append((column, source))
                continue
            fills[column] = expr
            result.columns.setdefault(column, ColumnStats(table, column, source))
        if not fills:
            continue

        n_rows = pq.ParquetFile(path).metadata.num_rows
        before = _null_counts(path, {c: pl.col(c).null_count() for c in fills})
        todo = [c for c in fills if before[c] > 0]
        for c in fills:
            result.columns[c].rows += n_rows
            result.columns[c].nulls_before += before[c]

        if not todo:
            after = before
        elif dry_run:
            fillable = _null_counts(
                path, {c: pl.when(pl.col(c).is_null()).then(fills[c]).is_not_null().sum() for c in todo}
            )
            after = {c: before[c] - fillable.get(c, 0) for c in fills}
        else:
            after = _rewrite(path, fills, todo, schema, n_rows)
            result.files_rewritten += 1
        for c in fills:
            result.columns[c].nulls_after += after[c]

    for stats in result.columns.values():
        stats.filled = stats.nulls_before - stats.nulls_after
    return result


def _rewrite(
    path: Path, fills: Dict[str, pl.Expr], todo: List[str], schema: Dict[str, pl.DataType], n_rows: int
) -> Dict[str, int]:
    tmp = path.with_name(path.name + TMP_SUFFIX)
    try:
        lf = pl.scan_parquet(path).with_columns(
            [(fills[c] if schema[c] == pl.Null else pl.coalesce(pl.col(c), fills[c])).alias(c) for c in todo]
        )
        lf.sink_parquet(tmp, compression="snappy")
        if pq.ParquetFile(tmp).metadata.num_rows != n_rows:
            raise RuntimeError(f"row count mismatch rewriting {path}")
        after = _null_counts(tmp, {c: pl.col(c).null_count() for c in fills})
        os.replace(tmp, path)
        return after
    finally:
        if tmp.exists():
            tmp.unlink()
