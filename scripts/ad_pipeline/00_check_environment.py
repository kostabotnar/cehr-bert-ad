#!/usr/bin/env python
"""Step 0: Check the runtime environment before running the pipeline.

Verifies the Python packages the heavy steps need (cehrbert, cehrbert_data, datasets,
transformers, torch), reports GPU availability, and checks for Java (required by the
Spark-based pretraining-data generation in step 2). This step is advisory: missing
optional pieces are warnings, but a missing core runtime fails so you find out now
rather than three steps in.

Usage:
    python 00_check_environment.py
    python 00_check_environment.py --report-dir reports/00_check_environment
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

# (module, level): error-level modules are required for pretraining/finetuning.
CORE_MODULES = ["datasets", "transformers", "torch", "cehrbert", "cehrbert_data"]
SUPPORT_MODULES = ["polars", "pandas", "pyarrow", "yaml", "numpy", "sklearn"]


def _version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"<{type(exc).__name__}>"


def build_report() -> StepReport:
    report = StepReport("00_check_environment", title="Step 0 — Environment check")
    report.add_metric("python", sys.version.split()[0])

    for mod in CORE_MODULES:
        ok = importlib.util.find_spec(mod) is not None
        report.add_check(f"core package '{mod}' importable", ok, detail=_version(mod) if ok else "not installed")

    for mod in SUPPORT_MODULES:
        ok = importlib.util.find_spec(mod) is not None
        report.warn(f"support package '{mod}'", ok=ok, detail=_version(mod) if ok else "not installed")

    # GPU availability (warning only — CPU works, just slower).
    try:
        import torch

        cuda = torch.cuda.is_available()
        report.warn("CUDA GPU available", ok=cuda,
                    detail=torch.cuda.get_device_name(0) if cuda else "no CUDA device (training will be slow)")
        report.add_metric("torch_cuda", cuda)
    except Exception:
        report.note("torch not importable; skipped GPU check")

    # Java for Spark (step 2). Warning: only step 2 needs it.
    java = shutil.which("java")
    if java:
        try:
            out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=20)
            ver = (out.stderr or out.stdout).splitlines()[0] if (out.stderr or out.stdout) else "unknown"
        except Exception:
            ver = "present"
        report.warn("Java available (Spark / step 2)", ok=True, detail=ver)
    else:
        report.warn("Java available (Spark / step 2)", ok=False, detail="java not on PATH; step 2 will fail")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check pipeline runtime environment")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("00_check_environment")))
    args = parser.parse_args(argv)

    report = build_report()
    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
