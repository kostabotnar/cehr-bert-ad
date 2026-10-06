#!/usr/bin/env python
"""Step 1b: Fill null *_datetime columns in the OMOP tables from the matching *_date columns.

Some OMOP exports leave every ``*_datetime`` column null while the ``*_date`` columns are
populated. cehrbert_data orders visits by ``visit_start_datetime``, so step 2 needs these
filled. Null datetimes become midnight of the matching date (start from start date, end
from end date); non-null datetimes are never changed. ``person.birth_datetime`` is built
from year_of_birth, month_of_birth, day_of_birth, falling back to January 1 of
year_of_birth when the month is null.

Files are rewritten in place, one at a time (polars streaming, snappy), through a temporary
file and an atomic rename. The step is idempotent. It refuses to run on a path under
~/data/derived (the original source data).

Usage:
    python 01b_fill_datetimes.py --dry-run     # count only, write nothing
    python 01b_fill_datetimes.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import fill_datetimes as fd  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill null *_datetime columns from *_date columns")
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Count nulls and fillable rows; write nothing")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("01b_fill_datetimes")))
    args = parser.parse_args(argv)

    report = StepReport("01b_fill_datetimes", title="Step 1b: Fill null datetime columns")
    data_dir = Path(args.data_dir)
    report.add_metric("data_dir", str(data_dir))
    report.add_metric("dry_run", args.dry_run)

    if not report.add_check("data directory exists", data_dir.exists(), detail=str(data_dir)):
        return _finalize(report, args.report_dir)
    error = fd.guard_data_dir(data_dir)
    if not report.add_check("data directory is not the original source", error is None, detail=error or "ok"):
        return _finalize(report, args.report_dir)

    for table in fd.list_tables(data_dir):
        try:
            result = fd.process_table(data_dir, table, args.dry_run)
        except Exception as exc:
            report.fail(f"table '{table}' processed", detail=f"{type(exc).__name__}: {exc}")
            continue
        for column, reason in result.skipped:
            report.fail(f"{table}.{column} handled", detail=f"not filled: {reason}")
        for c in result.columns.values():
            report.ok(
                f"{table}.{c.column}",
                detail=(
                    f"rows={c.rows} nulls_before={c.nulls_before} "
                    f"{'would_fill' if args.dry_run else 'filled'}={c.filled} "
                    f"nulls_after={c.nulls_after} source={c.source}"
                ),
            )
            report.add_metric(
                f"{table}.{c.column}",
                {"rows": c.rows, "nulls_before": c.nulls_before, "filled": c.filled, "nulls_after": c.nulls_after},
            )
        if result.columns:
            report.note(f"{table}: {len(result.files)} file(s), {result.files_rewritten} rewritten")

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python 02_prepare_pretraining_data.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
