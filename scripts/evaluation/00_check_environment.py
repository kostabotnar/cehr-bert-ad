#!/usr/bin/env python
"""Step 00: Check the runtime environment of the evaluation part.

Verifies the packages needed for evaluation metrics and the model comparison. A missing required package fails the step; a missing optional
package is a warning.

Usage:
    python evaluation/00_check_environment.py
    python evaluation/00_check_environment.py --report-dir scripts/evaluation/reports/00_check_environment
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import env_checks  # noqa: E402

REQUIRED = ["polars", "pandas", "numpy", "scipy", "sklearn"]
OPTIONAL = ["matplotlib", "statsmodels"]


def main(argv: list[str] | None = None) -> int:
    return env_checks.run("evaluation", REQUIRED, OPTIONAL, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
