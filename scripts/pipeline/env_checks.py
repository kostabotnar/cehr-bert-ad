"""Environment check shared by the ``00_check_environment.py`` step of every part.

A missing required module fails the report, a missing optional module is a warning. GPU and Java
are advisory (warnings only) and only checked when a part asks for them.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from typing import Sequence

from . import paths
from .reporting import StepReport


def _version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"<{type(exc).__name__}>"


def _installed(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except (ImportError, ValueError):
        return False


def build_report(
    required: Sequence[str],
    optional: Sequence[str] = (),
    *,
    step_name: str = "00_check_environment",
    title: str = "Step 00 - Environment check",
    check_gpu: bool = False,
    check_java: bool = False,
) -> StepReport:
    """Check that ``required`` modules are importable (fail) and ``optional`` ones too (warning)."""
    report = StepReport(step_name, title=title)
    report.add_metric("python", sys.version.split()[0])

    for mod in required:
        ok = _installed(mod)
        report.add_check(f"required package '{mod}' importable", ok, detail=_version(mod) if ok else "not installed")

    for mod in optional:
        ok = _installed(mod)
        report.warn(f"optional package '{mod}'", ok=ok, detail=_version(mod) if ok else "not installed")

    if check_gpu:
        try:
            import torch

            cuda = torch.cuda.is_available()
            report.warn(
                "CUDA GPU available",
                ok=cuda,
                detail=torch.cuda.get_device_name(0) if cuda else "no CUDA device (training will be slow)",
            )
            report.add_metric("torch_cuda", cuda)
        except Exception:
            report.note("torch not importable; skipped GPU check")

    if check_java:
        java = shutil.which("java")
        if java:
            try:
                out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=20)
                ver = (out.stderr or out.stdout).splitlines()[0] if (out.stderr or out.stdout) else "unknown"
            except Exception:
                ver = "present"
            report.warn("Java available (Spark / sequence generation)", ok=True, detail=ver)
        else:
            report.warn(
                "Java available (Spark / sequence generation)",
                ok=False,
                detail="java not on PATH; cehrbert_model step 01 will fail",
            )

    return report


def run(
    part: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
    *,
    argv: list[str] | None = None,
    check_gpu: bool = False,
    check_java: bool = False,
) -> int:
    """Entry point of a part's ``00_check_environment.py``: build, write and print the report."""
    import argparse

    parser = argparse.ArgumentParser(description=f"Check the runtime environment of the {part} part")
    parser.add_argument("--report-dir", default=str(paths.report_dir_for(part, "00_check_environment")))
    args = parser.parse_args(argv)

    report = build_report(
        required,
        optional,
        title=f"Step 00 - Environment check ({part})",
        check_gpu=check_gpu,
        check_java=check_java,
    )
    report.write(args.report_dir)
    report.print_summary()
    print(f"\nReport written to {args.report_dir}")
    return report.exit_code()
