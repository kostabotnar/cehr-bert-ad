#!/bin/bash
# CEHR-BERT model: Sequence generation, pretraining, fine-tuning, prediction and the frozen-encoder linear probe.
#
# Usage:
#   bash cehrbert_model/run.sh
#   bash cehrbert_model/run.sh --from 05
#   bash cehrbert_model/run.sh --ctx 4096 --observation-window -1
#
# Runs the steps below in order from this folder and halts on the first failure.
# Writes reports/summary.md. --from <step id> resumes from that step.
# --ctx goes to steps 05, 09, 10 and, with --window-days, to 11 to 14.
# --observation-window goes to steps 09 and 10 as --observation-window and to steps 11 to 14 as --window-days. Step 05 (pretraining) has no observation window.

set -uo pipefail

PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PART_DIR}/../lib/run_steps.sh"

parse_args "ctx observation-window" "$@"

STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_generate_sequences.py"
    "02:${PY} 02_prepare_cohort_folders.py"
    "03:${PY} 03_build_pretraining_input.py"
    "04:${PY} 04_validate_pretrain_config.py"
    "05:${PY} 05_pretrain.py ${CTX_ARG}"
    "06:${PY} 06_verify_pretraining.py"
    "07:${PY} 07_tokenize_full_dataset.py"
    "08:${PY} 08_validate_finetune_config.py"
    "09:${PY} 09_finetune.py ${CTX_ARG} ${OBS_ARG}"
    "10:${PY} 10_predict.py ${CTX_ARG} ${OBS_ARG}"
    "11:${PY} 11_validation_predictions.py ${CTX_ARG} ${WIN_ARG}"
    "12:${PY} 12_token_stats.py ${CTX_ARG} ${WIN_ARG}"
    "13:${PY} 13_linear_probe_features.py ${CTX_ARG} ${WIN_ARG}"
    "14:${PY} 14_train_linear_probe.py ${CTX_ARG} ${WIN_ARG}"
)

run_steps "CEHR-BERT model"
exit $?
