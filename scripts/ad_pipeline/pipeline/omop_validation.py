"""Validate the raw OMOP dataset: tables present, readable, non-empty, right columns.

Only parquet *schema* and row counts are read (not the full data), so this is cheap
even on large exports. Uses pyarrow, which is always available in the pipeline env.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .reporting import StepReport

# concept is required: the cohort build (step 6) and the concept-list generation
# (step 2) both need it. The original reorganize script treated it as optional.
REQUIRED_TABLES = ["person", "visit_occurrence", "condition_occurrence", "concept"]
RECOMMENDED_TABLES = ["drug_exposure", "procedure_occurrence", "measurement"]

# Minimal columns each table must expose for the downstream cehrbert_data tooling
# and the cohort builder. Kept intentionally small to avoid coupling to OMOP CDM
# versions that add/rename optional columns.
EXPECTED_COLUMNS = {
    "person": ["person_id"],
    "visit_occurrence": ["person_id", "visit_start_date"],
    "condition_occurrence": ["person_id", "condition_concept_id", "condition_start_date"],
    "concept": ["concept_id", "concept_code"],
    "drug_exposure": ["person_id", "drug_concept_id"],
    "procedure_occurrence": ["person_id", "procedure_concept_id"],
    "measurement": ["person_id", "measurement_concept_id"],
}


def table_glob(data_dir: Path, table: str) -> Optional[str]:
    """Return a parquet glob for ``<table>/*.parquet`` or ``<table>.parquet``, else None."""
    sub = data_dir / table
    if sub.is_dir() and any(sub.glob("*.parquet")):
        return str(sub / "*.parquet")
    flat = data_dir / f"{table}.parquet"
    if flat.exists():
        return str(flat)
    return None


def read_columns(glob: str) -> List[str]:
    """Read column names from the parquet schema without loading rows."""
    dataset = ds.dataset(_glob_to_paths(glob), format="parquet")
    return [f.name for f in dataset.schema]


def count_rows(glob: str) -> int:
    """Total number of rows across the parquet file(s) (reads only metadata)."""
    total = 0
    for path in _glob_to_paths(glob):
        total += pq.ParquetFile(path).metadata.num_rows
    return total


def _glob_to_paths(glob: str) -> List[str]:
    p = Path(glob)
    if "*" in p.name:
        return sorted(str(x) for x in p.parent.glob(p.name))
    return [str(p)]


def add_dataset_checks(report: StepReport, data_dir: Path, require_concept: bool = True) -> StepReport:
    """Populate ``report`` with checks for the OMOP dataset at ``data_dir``.

    Required tables are error-level; recommended tables and column gaps are
    warning-level (the pipeline can still run, possibly with reduced signal).
    """
    data_dir = Path(data_dir)
    report.add_check("data directory exists", data_dir.exists(), detail=str(data_dir))
    if not data_dir.exists():
        return report

    required = list(REQUIRED_TABLES)
    if not require_concept and "concept" in required:
        required.remove("concept")

    for table in required:
        _check_table(report, data_dir, table, level="error")
    for table in RECOMMENDED_TABLES:
        glob = table_glob(data_dir, table)
        if glob is None:
            report.warn(f"recommended table '{table}'", ok=False, detail="missing (reduces model signal)")
        else:
            _check_table(report, data_dir, table, level="warning", presence_only_level="warning")

    return report


def _check_table(report: StepReport, data_dir: Path, table: str, level: str, presence_only_level: str | None = None) -> None:
    """Check one table: present, readable, non-empty, has expected columns."""
    glob = table_glob(data_dir, table)
    present = glob is not None
    report.add_check(f"table '{table}' present", present, detail="" if present else "no parquet found", level=level)
    if not present:
        return

    # Readable + row count
    try:
        n_rows = count_rows(glob)
    except Exception as exc:  # corrupt/unreadable parquet
        report.add_check(f"table '{table}' readable", False, detail=str(exc), level=level)
        return
    report.add_check(f"table '{table}' non-empty", n_rows > 0, detail=f"{n_rows} rows", level=level)
    report.add_metric(f"{table}_rows", n_rows)

    # Columns
    expected = EXPECTED_COLUMNS.get(table, [])
    if expected:
        try:
            cols = set(read_columns(glob))
        except Exception as exc:
            report.add_check(f"table '{table}' schema readable", False, detail=str(exc), level=level)
            return
        missing = [c for c in expected if c not in cols]
        col_level = presence_only_level or level
        report.add_check(
            f"table '{table}' has required columns",
            ok=not missing,
            detail="ok" if not missing else f"missing: {', '.join(missing)}",
            level=col_level,
        )
