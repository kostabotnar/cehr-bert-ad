#!/bin/bash
# Evaluation: Metrics with bootstrap confidence intervals and the model comparison table.
#
# Usage:
#   bash evaluation/run.sh
#   bash evaluation/run.sh --ctx 4096 --window-days -1 --name my_comparison
#
# Runs the steps below in order from this folder and halts on the first failure.
# Writes reports/summary.md. --from <step id> resumes from that step.
# --ctx and --window-days select the default manifests (cehrbert_model run label, classical_ml window label).
# --name sets the comparison folder build/evaluation/<name>/ (default: the cehrbert run label). All go to steps 01 and 02.

set -uo pipefail

PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PART_DIR}/../lib/run_steps.sh"

parse_args "ctx window-days name" "$@"

STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_evaluate.py ${CTX_ARG} ${WIN_ARG} ${NAME_ARG}"
    "02:${PY} 02_compare_models.py ${CTX_ARG} ${WIN_ARG} ${NAME_ARG}"
)

run_steps "Evaluation"
exit $?
