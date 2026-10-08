#!/bin/bash
# Classical ML: Tabular features from OMOP and the L2 logistic regression and XGBoost baselines.
#
# Usage:
#   bash classical_ml/run.sh
#   bash classical_ml/run.sh --window-days -1
#
# Runs the steps below in order from this folder and halts on the first failure.
# Writes reports/summary.md. --from <step id> resumes from that step.
# --window-days selects the window label (build/classical_ml/<window label>/) and goes to steps 01 and 02.

set -uo pipefail

PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PART_DIR}/../lib/run_steps.sh"

parse_args "window-days" "$@"

STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_prepare_features.py ${WIN_ARG}"
    "02:${PY} 02_train_baselines.py ${WIN_ARG}"
)

run_steps "Classical ML"
exit $?
