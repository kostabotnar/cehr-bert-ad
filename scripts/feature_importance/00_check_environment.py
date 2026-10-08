#!/usr/bin/env python
"""Step 00: Check the runtime environment of the feature_importance part.

Verifies the packages needed for baseline importance tables and CEHR-BERT attention analysis. A missing required package fails the step; a missing optional
package is a warning.

Usage:
    python feature_importance/00_check_environment.py
    python feature_importance/00_check_environment.py --report-dir scripts/feature_importance/reports/00_check_environment
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import env_checks  # noqa: E402

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "joblib", "torch", "transformers", "datasets", "cehrbert"]
OPTIONAL = ["matplotlib"]


def main(argv: list[str] | None = None) -> int:
    return env_checks.run("feature_importance", REQUIRED, OPTIONAL, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
