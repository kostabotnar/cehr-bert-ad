#!/usr/bin/env python
"""Step 1: Validate the OMOP dataset — verify all necessary files exist.

Checks that the OMOP source tables required by the pipeline are present, readable,
non-empty, and expose the columns the downstream cehrbert_data tooling and cohort
builder rely on. Required tables are gating; recommended tables are warnings.

Optionally sets up the ``data/omop_data`` entry point used by every later step. This
replaces the old ``01_reorganize_data.py``: pass ``--link /path/to/omop_data`` to point
the pipeline at your data (a symlink is created when possible, otherwise the path is
validated in place).

Usage:
    python 01_validate_dataset.py                      # validate data/omop_data
    python 01_validate_dataset.py --link /path/to/omop_data   # link then validate
    python 01_validate_dataset.py --data-dir /path/to/omop_data  # validate elsewhere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline.omop_validation import add_dataset_checks  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def link_data_folder(source: Path, target: Path, report: StepReport) -> None:
    """Point ``target`` (data/omop_data) at ``source``. Symlink, else record a note."""
    source = source.resolve()
    if not source.exists():
        report.add_check("link source exists", False, detail=str(source))
        return
    report.ok("link source exists", detail=str(source))

    # If the source already *is* data/omop_data (or a link already points there), there is
    # nothing to do — validate it in place instead of trying to link a folder onto itself.
    if target.exists():
        try:
            if source.samefile(target):
                report.ok("data already at data/omop_data (no link needed)", detail=str(source))
                return
        except OSError:
            pass

    if target.is_symlink() or target.exists():
        if target.is_symlink():
            target.unlink()
            report.note(f"removed existing symlink {target}")
        else:
            report.warn("target is a real directory (not relinked)", ok=False, detail=str(target))
            return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        report.ok("data/omop_data symlink created", detail=f"{target} -> {source}")
    except OSError as exc:
        # Windows without privilege: fall back to validating the source path directly.
        report.warn("symlink created", ok=False, detail=f"{exc}; validate --data-dir {source} directly")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the OMOP dataset")
    parser.add_argument("--data-dir", default=str(paths.OMOP_DIR),
                        help=f"OMOP data folder to validate (default: {paths.OMOP_DIR})")
    parser.add_argument("--link", default=None,
                        help="Path to your OMOP data; creates data/omop_data -> this path before validating")
    parser.add_argument("--no-require-concept", action="store_true",
                        help="Treat the concept table as optional (not recommended)")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("01_validate_dataset")))
    args = parser.parse_args(argv)

    report = StepReport("01_validate_dataset", title="Step 1 — Validate OMOP dataset")

    data_dir = Path(args.data_dir)
    if args.link:
        link_data_folder(Path(args.link), paths.OMOP_DIR, report)
        data_dir = paths.OMOP_DIR

    report.add_metric("data_dir", str(data_dir))
    add_dataset_checks(report, data_dir, require_concept=not args.no_require_concept)

    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    if report.passed:
        print("Next: bash 02_generate_pretraining_data.sh  (or python 02_prepare_pretraining_data.py)")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
