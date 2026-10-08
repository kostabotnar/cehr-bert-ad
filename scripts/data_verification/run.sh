#!/bin/bash
# Data verification: Check the OMOP dataset and fill null datetime columns.
#
# Usage:
#   bash data_verification/run.sh
#   bash data_verification/run.sh --link /path/to/omop_data   # symlink data/omop_data, then validate
#   bash data_verification/run.sh --from 02
#
# Runs the steps below in order from this folder and halts on the first failure.
# Writes reports/summary.md. --from <step id> resumes from that step.
# --link is passed to step 01.

set -uo pipefail

PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PART_DIR}/../lib/run_steps.sh"

parse_args "link" "$@"

STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_validate_dataset.py ${LINK_ARG}"
    "02:${PY} 02_fill_datetimes.py"
)

run_steps "Data verification"
exit $?
