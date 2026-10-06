#!/usr/bin/env python
"""Step 4: Pretrain the CEHR-BERT model.

Preflight-checks the patient sequences and the pretraining config (reusing step 3's
validation), launches the cehrbert pretraining runner via the thin ``04_pretrain.sh``
wrapper, then sanity-checks that a model directory was produced. Full verification is
step 5.

Usage:
    python 04_pretrain.py
    python 04_pretrain.py --config configs/ad_pretrain_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import launch, paths  # noqa: E402
from pipeline.config_validation import add_pretrain_config_checks, load_yaml, write_config  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

WRAPPER = Path(__file__).resolve().parent / "04_pretrain.sh"


def _apply_ctx_override(config: dict, max_position_embeddings: int | None) -> None:
    """Override the token context in-place when a CLI value is provided.

    Sets ``max_position_embeddings`` and, only if already present, mirrors the value
    into ``sample_packing_max_positions``. A ``None`` value preserves the YAML config.
    """
    if max_position_embeddings is None:
        return
    config["max_position_embeddings"] = int(max_position_embeddings)
    if "sample_packing_max_positions" in config:
        config["sample_packing_max_positions"] = int(max_position_embeddings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain CEHR-BERT (preflight + launch)")
    parser.add_argument("--config", default=str(paths.PRETRAIN_CONFIG))
    parser.add_argument("--skip-launch", action="store_true", help="Run preflight only; do not launch training")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("04_pretrain")))
    parser.add_argument(
        "--max-position-embeddings", "--ctx", type=int, default=None, dest="max_position_embeddings",
        help="Override the token context (max_position_embeddings / sample_packing_max_positions); "
             "default uses the YAML value.")
    args = parser.parse_args(argv)

    report = StepReport("04_pretrain", title="Step 4 — Pretrain CEHR-BERT")
    report.add_metric("config", args.config)

    # --- preflight: sequences present + config valid ----------------------
    seq_dir = paths.PATIENT_SEQUENCE_DIR
    if not report.add_check("patient_sequence present", seq_dir.is_dir() and any(seq_dir.glob("*.parquet")),
                            detail=str(seq_dir)):
        report.note("Run step 2 first to generate patient sequences.")
        return _finalize(report, args.report_dir)

    add_pretrain_config_checks(report, Path(args.config), paths.resolve_under_root)
    if not report.passed:
        report.note("Config preflight failed; not launching training (see step 3).")
        return _finalize(report, args.report_dir)

    paths.PRETRAIN_PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    paths.PRETRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CLI overrides: when provided, write an effective config and launch with that;
    # otherwise launch with the original YAML unchanged (preserves current behavior).
    launch_config_path = Path(args.config)
    if args.max_position_embeddings is not None:
        config = load_yaml(Path(args.config))
        _apply_ctx_override(config, args.max_position_embeddings)
        report.add_metric("max_position_embeddings", config["max_position_embeddings"])
        launch_config_path = write_config(config, Path(args.report_dir) / "effective_pretrain_config.yaml")
        report.add_artifact(launch_config_path)

    # --- launch ------------------------------------------------------------
    if args.skip_launch:
        report.note("Launch skipped (--skip-launch).")
    else:
        rc = launch.run_command(["bash", str(WRAPPER), str(launch_config_path)], cwd=paths.PROJECT_ROOT)
        report.add_metric("launch_returncode", rc)
        if not report.add_check("pretraining process succeeded", rc == 0, detail=f"exit code {rc}"):
            return _finalize(report, args.report_dir)
        # Cheap post-check; step 5 does the thorough verification.
        report.add_check("model directory produced", paths.PRETRAIN_RESULTS_DIR.is_dir(),
                         detail=str(paths.PRETRAIN_RESULTS_DIR))

    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python 05_verify_pretraining.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
