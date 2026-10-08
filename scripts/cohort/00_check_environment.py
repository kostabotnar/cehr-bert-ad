#!/usr/bin/env python
"""Step 00: Check the runtime environment of the cohort part.

Verifies the packages needed for the cohort build (polars and numpy; pyarrow for the OMOP table globbing). A missing required package fails the step; a missing optional
package is a warning.

Usage:
    python cohort/00_check_environment.py
    python cohort/00_check_environment.py --report-dir scripts/cohort/reports/00_check_environment
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import env_checks  # noqa: E402

REQUIRED = ["polars", "numpy", "pyarrow"]
OPTIONAL = []


def main(argv: list[str] | None = None) -> int:
    return env_checks.run("cohort", REQUIRED, OPTIONAL, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
