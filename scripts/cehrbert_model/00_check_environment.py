#!/usr/bin/env python
"""Step 00: Check the runtime environment of the cehrbert_model part.

Verifies the packages needed for CEHR-BERT sequence generation, pretraining, fine-tuning and prediction. A missing required package fails the step (GPU and Java are advisory warnings); a missing optional
package is a warning.

Usage:
    python cehrbert_model/00_check_environment.py
    python cehrbert_model/00_check_environment.py --report-dir scripts/cehrbert_model/reports/00_check_environment
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import env_checks  # noqa: E402

REQUIRED = ["datasets", "transformers", "torch", "cehrbert", "cehrbert_data"]
OPTIONAL = ["polars", "pandas", "pyarrow", "yaml", "numpy", "sklearn", "scipy", "joblib", "matplotlib", "tqdm"]


def main(argv: list[str] | None = None) -> int:
    return env_checks.run("cehrbert_model", REQUIRED, OPTIONAL, argv=argv, check_gpu=True, check_java=True)


if __name__ == "__main__":
    raise SystemExit(main())
