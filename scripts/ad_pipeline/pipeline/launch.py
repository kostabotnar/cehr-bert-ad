"""Subprocess helper for the heavy launcher steps (2, 4, 8).

Kept tiny and separate so tests can monkeypatch :func:`run_command` and exercise the
preflight/report logic without ever invoking Spark or a training runner.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """Run ``cmd``, streaming output to the console. Returns the process exit code."""
    merged_env = dict(os.environ)
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    print("+ " + " ".join(str(c) for c in cmd))
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env)
    return completed.returncode
