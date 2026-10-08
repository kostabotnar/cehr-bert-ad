#!/bin/bash
# Data QC: Descriptive statistics of the raw OMOP data and of the cohort (reads data/omop_data and the splits).
#
# Usage:
#   bash data_qc/run.sh
#   bash data_qc/run.sh --from 02
#
# Runs the steps below in order from this folder and halts on the first failure.
# Writes reports/summary.md. --from <step id> resumes from that step.

set -uo pipefail

PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PART_DIR}/../lib/run_steps.sh"

parse_args "" "$@"

STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_profile_omop.py"
    "02:${PY} 02_history_length_stats.py"
)

run_steps "Data QC"
exit $?
