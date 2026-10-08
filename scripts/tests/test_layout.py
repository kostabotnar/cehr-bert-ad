"""Layout checks: every part has a runner, a README and an environment check, and its runner lists real steps."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from pipeline import paths

SCRIPTS = Path(__file__).resolve().parents[1]
PARTS = list(paths.PARTS)
STEP_LINE = re.compile(r'^\s*"(?P<id>[0-9A-Za-z_]+):\$\{PY\} (?P<script>\S+\.py)(?: [^"]*)?"\s*$')


def _run_steps(part: str) -> list[tuple[str, str]]:
    """The (step id, script name) pairs listed in the STEPS array of a part's run.sh."""
    steps, inside = [], False
    for line in (SCRIPTS / part / "run.sh").read_text(encoding="utf-8").splitlines():
        if line.startswith("STEPS=("):
            inside = True
        elif inside and line.strip() == ")":
            break
        elif inside:
            match = STEP_LINE.match(line)
            assert match, f"{part}/run.sh: cannot read step line {line!r}"
            steps.append((match["id"], match["script"]))
    return steps


def test_part_list_matches_the_folders_on_disk():
    on_disk = {
        p.name
        for p in SCRIPTS.iterdir()
        if p.is_dir() and (p / "run.sh").exists() and p.name not in {"lib", "tests", "pipeline", "reports"}
    }
    assert on_disk == set(PARTS)
    assert not (SCRIPTS / "ad_pipeline").exists()


@pytest.mark.parametrize("part", PARTS)
def test_part_has_runner_readme_and_environment_check(part):
    folder = SCRIPTS / part
    assert (folder / "run.sh").is_file()
    assert (folder / "README.md").is_file()
    assert (folder / "00_check_environment.py").is_file()


@pytest.mark.parametrize("part", PARTS)
def test_every_step_in_run_sh_exists_in_order(part):
    steps = _run_steps(part)
    assert steps, f"{part}/run.sh lists no steps"
    for _, script in steps:
        assert (SCRIPTS / part / script).is_file(), f"{part}/run.sh lists missing step {script}"
    ids = [sid for sid, _ in steps]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert ids[0] == "00" and steps[0][1] == "00_check_environment.py"
    # The id of a step is the numeric prefix of its script name.
    assert all(script.startswith(sid + "_") for sid, script in steps)


@pytest.mark.parametrize("part", PARTS)
def test_every_numbered_step_of_a_part_is_in_its_runner(part):
    listed = {script for _, script in _run_steps(part)}
    on_disk = {p.name for p in (SCRIPTS / part).glob("[0-9][0-9]_*.py")}
    assert on_disk == listed


def test_run_all_lists_the_parts_in_dependency_order():
    text = (SCRIPTS / "run_all.sh").read_text(encoding="utf-8")
    match = re.search(r"^ALL_PARTS=\((.*)\)$", text, re.M)
    assert match and match.group(1).split() == PARTS


def test_shell_scripts_have_valid_syntax():
    for script in sorted(SCRIPTS.rglob("*.sh")):
        if "reports" in script.parts:
            continue
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"{script}: {result.stderr}"
