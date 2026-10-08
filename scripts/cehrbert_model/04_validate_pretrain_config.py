#!/usr/bin/env python
"""Step 04: Evaluate the pretraining config (dry run - no training).

Parses ``configs/ad_pretrain_config.yaml`` exactly as the runner will (via the cehrbert
dataclasses when available), then checks value ranges, required flags, and that the
referenced data/output paths exist. Run this before step 05 to fail fast on a bad config.

Usage:
    python cehrbert_model/04_validate_pretrain_config.py
    python cehrbert_model/04_validate_pretrain_config.py --config cehrbert_model/configs/ad_pretrain_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import paths  # noqa: E402
from pipeline.config_validation import add_pretrain_config_checks  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate pretraining config")
    parser.add_argument("--config", default=str(paths.PRETRAIN_CONFIG))
    parser.add_argument(
        "--report-dir", default=str(paths.report_dir_for("cehrbert_model", "04_validate_pretrain_config"))
    )
    args = parser.parse_args(argv)

    report = StepReport("04_validate_pretrain_config", title="Step 04 - Evaluate pretraining config")
    report.add_metric("config", args.config)
    add_pretrain_config_checks(report, Path(args.config), paths.resolve_under_root)

    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    if report.passed:
        print("Next: bash cehrbert_model/05_pretrain.sh  (or python cehrbert_model/05_pretrain.py)")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
