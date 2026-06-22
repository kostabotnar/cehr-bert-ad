"""Structured step reporting: collect checks/metrics, emit JSON + Markdown.

A :class:`StepReport` is the single deliverable every pipeline step produces. It is
both machine-readable (``status.json``) and human-readable (``report.md``), and its
overall status drives the process exit code so the orchestrator and CI can gate.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List

# Status precedence: a single failed error-level check fails the whole step.
STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

LEVEL_ERROR = "error"
LEVEL_WARNING = "warning"


@dataclasses.dataclass
class Check:
    """A single named assertion about the step's inputs or outputs."""

    name: str
    ok: bool
    detail: str = ""
    level: str = LEVEL_ERROR  # "error" gates the pipeline; "warning" does not

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class StepReport:
    """Accumulates checks and metrics for one pipeline step.

    Example:
        report = StepReport("01_validate_dataset", title="Validate OMOP dataset")
        report.add_check("person table present", ok=True)
        report.add_metric("person_rows", 1234)
        report.write(report_dir)
        raise SystemExit(report.exit_code())
    """

    def __init__(self, step: str, title: str | None = None) -> None:
        self.step = step
        self.title = title or step
        self.checks: List[Check] = []
        self.metrics: dict[str, Any] = {}
        self.artifacts: List[str] = []
        self.notes: List[str] = []

    # -- recording ----------------------------------------------------------
    def add_check(self, name: str, ok: bool, detail: str = "", level: str = LEVEL_ERROR) -> bool:
        """Record a check. Returns ``ok`` so callers can branch on the result."""
        self.checks.append(Check(name=name, ok=bool(ok), detail=str(detail), level=level))
        return bool(ok)

    def ok(self, name: str, detail: str = "") -> bool:
        return self.add_check(name, True, detail)

    def fail(self, name: str, detail: str = "", level: str = LEVEL_ERROR) -> bool:
        return self.add_check(name, False, detail, level=level)

    def warn(self, name: str, ok: bool, detail: str = "") -> bool:
        """Record a non-blocking (warning-level) check."""
        return self.add_check(name, ok, detail, level=LEVEL_WARNING)

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def add_artifact(self, path: str | Path) -> None:
        self.artifacts.append(str(path))

    def note(self, message: str) -> None:
        self.notes.append(str(message))

    # -- derived state ------------------------------------------------------
    @property
    def status(self) -> str:
        has_error = any((not c.ok) and c.level == LEVEL_ERROR for c in self.checks)
        if has_error:
            return STATUS_FAIL
        has_warning = any((not c.ok) and c.level == LEVEL_WARNING for c in self.checks)
        return STATUS_WARN if has_warning else STATUS_PASS

    @property
    def passed(self) -> bool:
        return self.status != STATUS_FAIL

    def exit_code(self) -> int:
        """0 for pass/warn (pipeline may continue), 1 for fail."""
        return 0 if self.passed else 1

    # -- serialization ------------------------------------------------------
    def to_dict(self, timestamp: str | None = None) -> dict[str, Any]:
        return {
            "step": self.step,
            "title": self.title,
            "status": self.status,
            "timestamp": timestamp if timestamp is not None else datetime.now().isoformat(timespec="seconds"),
            "n_checks": len(self.checks),
            "n_failed": sum(1 for c in self.checks if not c.ok),
            "checks": [c.to_dict() for c in self.checks],
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    def to_markdown(self, timestamp: str | None = None) -> str:
        payload = self.to_dict(timestamp=timestamp)
        badge = {STATUS_PASS: "✅ PASS", STATUS_WARN: "⚠️ WARN", STATUS_FAIL: "❌ FAIL"}[payload["status"]]
        lines: List[str] = [
            f"# {self.title}",
            "",
            f"**Status:** {badge}  ",
            f"**Step:** `{self.step}`  ",
            f"**Time:** {payload['timestamp']}  ",
            f"**Checks:** {payload['n_checks']} ({payload['n_failed']} failed)",
            "",
            "## Checks",
            "",
            "| Result | Level | Check | Detail |",
            "|--------|-------|-------|--------|",
        ]
        for c in self.checks:
            mark = "✅" if c.ok else ("⚠️" if c.level == LEVEL_WARNING else "❌")
            detail = c.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {mark} | {c.level} | {c.name} | {detail} |")

        if self.metrics:
            lines += ["", "## Metrics", "", "| Metric | Value |", "|--------|-------|"]
            for k, v in self.metrics.items():
                lines.append(f"| {k} | {v} |")

        if self.artifacts:
            lines += ["", "## Artifacts", ""]
            lines += [f"- `{a}`" for a in self.artifacts]

        if self.notes:
            lines += ["", "## Notes", ""]
            lines += [f"- {n}" for n in self.notes]

        lines.append("")
        return "\n".join(lines)

    def write(self, report_dir: str | Path, timestamp: str | None = None) -> Path:
        """Write ``status.json`` and ``report.md`` into ``report_dir``."""
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = timestamp if timestamp is not None else datetime.now().isoformat(timespec="seconds")
        (out / "status.json").write_text(
            json.dumps(self.to_dict(timestamp=ts), indent=2, default=str), encoding="utf-8"
        )
        (out / "report.md").write_text(self.to_markdown(timestamp=ts), encoding="utf-8")
        return out

    def print_summary(self) -> None:
        """Print a compact, ASCII-safe human summary to stdout."""
        badge = {STATUS_PASS: "PASS", STATUS_WARN: "WARN", STATUS_FAIL: "FAIL"}[self.status]
        lines = [f"\n[{badge}] {self.title}"]
        for c in self.checks:
            mark = "ok " if c.ok else ("warn" if c.level == LEVEL_WARNING else "FAIL")
            line = f"  [{mark}] {c.name}"
            if c.detail:
                line += f" - {c.detail}"
            lines.append(line)
        if self.metrics:
            lines.append("  metrics: " + ", ".join(f"{k}={v}" for k, v in self.metrics.items()))
        _safe_print("\n".join(lines))


def _safe_print(text: str) -> None:
    """Print text, degrading non-encodable characters instead of raising on cp1252 consoles."""
    import sys

    text = text.replace("—", "-").replace("→", "->")  # em-dash, right-arrow
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))
