#!/usr/bin/env python
"""Step 7: Validate the finetuning config (dry run — no training).

Parses ``configs/ad_finetune_config.yaml`` and verifies its upstream dependencies are in
place: the pretrained model + tokenizer (steps 4/5), the tokenized full dataset
(step 4), and the labeled cohort (step 6). Also checks value ranges, observation_window,
and that do_predict is enabled (required to emit test predictions for step 9).

Usage:
    python 07_validate_finetune_config.py
    python 07_validate_finetune_config.py --config configs/ad_finetune_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline.config_validation import add_finetune_config_checks  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate finetuning config")
    parser.add_argument("--config", default=str(paths.FINETUNE_CONFIG))
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("07_validate_finetune_config")))
    args = parser.parse_args(argv)

    report = StepReport("07_validate_finetune_config", title="Step 7 — Validate finetuning config")
    report.add_metric("config", args.config)
    add_finetune_config_checks(report, Path(args.config), paths.resolve_under_root)

    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    if report.passed:
        print("Next: bash 08_finetune.sh  (or python 08_finetune.py)")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
