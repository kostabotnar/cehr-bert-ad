#!/bin/bash
# Feature importance: Importance tables for the baselines, attention analysis for CEHR-BERT and the top feature tables.
#
# Usage:
#   bash feature_importance/run.sh
#   bash feature_importance/run.sh --ctx 4096 --window-days -1
#
# Runs the steps below in order from this folder and halts on the first failure.
# Writes reports/summary.md. --from <step id> resumes from that step.
# --ctx and --window-days select the run label (build/<label>/) and go to steps 01 to 03.

set -uo pipefail

PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PART_DIR}/../lib/run_steps.sh"

parse_args "ctx window-days" "$@"

STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_baseline_importance.py ${CTX_ARG} ${WIN_ARG}"
    "02:${PY} 02_attention_analysis.py ${CTX_ARG} ${WIN_ARG}"
    "03:${PY} 03_top_feature_tables.py ${CTX_ARG} ${WIN_ARG}"
)

run_steps "Feature importance"
exit $?
