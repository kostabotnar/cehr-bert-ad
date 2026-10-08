#!/bin/bash
# Cohort: Build the labeled AD cohort and the canonical train/val/test split from the OMOP tables.
#
# Usage:
#   bash cohort/run.sh
#   bash cohort/run.sh --from 01
#
# Runs the steps below in order from this folder and halts on the first failure.
# Writes reports/summary.md. --from <step id> resumes from that step.

set -uo pipefail

PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PART_DIR}/../lib/run_steps.sh"

parse_args "" "$@"

STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_build_cohort.py"
)

run_steps "Cohort"
exit $?
