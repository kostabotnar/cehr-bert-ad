#!/usr/bin/env python
"""Step 00: Check the runtime environment of the data_verification part.

Verifies the packages needed for the OMOP validation and datetime fill steps (polars, pyarrow, pandas, yaml, numpy). A missing required package fails the step; a missing optional
package is a warning.

Usage:
    python data_verification/00_check_environment.py
    python data_verification/00_check_environment.py --report-dir scripts/data_verification/reports/00_check_environment
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import env_checks  # noqa: E402

REQUIRED = ["polars", "pandas", "pyarrow", "yaml", "numpy"]
OPTIONAL = []


def main(argv: list[str] | None = None) -> int:
    return env_checks.run("data_verification", REQUIRED, OPTIONAL, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
