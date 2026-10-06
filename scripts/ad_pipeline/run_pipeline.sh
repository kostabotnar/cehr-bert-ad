#!/bin/bash
# Orchestrate the full AD CEHR-BERT pipeline.
#
# Runs each step in order, halting on the first failure (non-zero exit). Each step writes
# its own report under reports/<step>/ (steps 10 onward under build/<run_label>/reports/);
# this script also writes reports/summary.md.
#
# Execution order (not numeric: the cohort and its test split are built before pretraining so
# the test patients can be kept out of it):
#   00 01 01b 02 06 03b 03 04 05 05b 07 08 09 10a 10 11 12 14_attention 14_tokens 15 16 13
# Step 13 runs last because it compares the outputs of steps 10, 12 and 16.
#
# Usage:
#   bash run_pipeline.sh                 # run every step
#   bash run_pipeline.sh --from 07       # resume from step 07 (position in the order above)
#   bash run_pipeline.sh --link /path/to/omop_data   # pass-through to step 01
#   bash run_pipeline.sh --ctx 4096 --observation-window -1  # experiment overrides
#
# Experiment overrides (optional):
#   --ctx <int>                 token context; forwarded to steps 04, 08, 09 as --ctx and to
#                               steps 10 to 16 as --ctx (part of the run label).
#   --observation-window <int>  look-back in days (negative = all history); forwarded
#                               to steps 08 and 09 as --observation-window and to steps 10 to
#                               16 as --window-days. Step 04 (pretraining) has no observation
#                               window, so it is not forwarded there.
#
# Notes:
#   * Steps 02/04/08 are heavy (Spark / model training) and need the full runtime env.
#   * Validation/verification steps (01,03,05,05b,07,09) are gating: a FAIL stops the run.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/reports"
PY="${PYTHON:-python}"
INVOCATION_DIR="$(pwd)"   # captured before we cd into SCRIPT_DIR

FROM="00"
LINK_ARG=""
CTX_ARG=""        # forwarded to steps 04, 08..16 when --ctx is given
OBS_ARG=""        # forwarded to steps 08, 09 when --observation-window is given
WIN_ARG=""        # the same window, forwarded to steps 10..16 as --window-days
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --link)
            # Resolve the link path against the directory the user ran this from, so a
            # relative path keeps its meaning after we cd into SCRIPT_DIR for the steps.
            link_src="$2"
            [[ "${link_src}" != /* ]] && link_src="$(realpath -m "${INVOCATION_DIR}/${link_src}")"
            LINK_ARG="--link ${link_src}"
            shift 2 ;;
        --ctx) CTX_ARG="--ctx $2"; shift 2 ;;
        --observation-window) OBS_ARG="--observation-window $2"; WIN_ARG="--window-days $2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

# step id -> command, in execution order
RUN_ARGS="${CTX_ARG} ${WIN_ARG}"
declare -a STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_validate_dataset.py ${LINK_ARG}"
    "01b:${PY} 01b_fill_datetimes.py"
    "02:${PY} 02_prepare_pretraining_data.py"
    "06:${PY} 06_prepare_finetuning_data.py"
    "03b:${PY} 03b_build_pretraining_input.py"
    "03:${PY} 03_evaluate_pretrain_config.py"
    "04:${PY} 04_pretrain.py ${CTX_ARG}"
    "05:${PY} 05_verify_pretraining.py"
    "05b:${PY} 05b_tokenize_full_dataset.py"
    "07:${PY} 07_validate_finetune_config.py"
    "08:${PY} 08_finetune.py ${CTX_ARG} ${OBS_ARG}"
    "09:${PY} 09_predict.py ${CTX_ARG} ${OBS_ARG}"
    "10a:${PY} 10a_validation_predictions.py ${RUN_ARGS}"
    "10:${PY} 10_evaluate.py ${RUN_ARGS}"
    "11:${PY} 11_prepare_baseline_features.py ${RUN_ARGS}"
    "12:${PY} 12_train_baselines.py ${RUN_ARGS}"
    "14_attention:${PY} 14_attention_analysis.py ${RUN_ARGS}"
    "14_tokens:${PY} 14_token_history_stats.py ${RUN_ARGS}"
    "15:${PY} 15_linear_probe_features.py ${RUN_ARGS}"
    "16:${PY} 16_train_linear_probe.py ${RUN_ARGS}"
    "13:${PY} 13_compare_models.py ${RUN_ARGS}"
)

# --from names a position in STEPS (an id, or its prefix before "_", e.g. 14), not a number
# range, because the execution order is not numeric.
START=-1
for i in "${!STEPS[@]}"; do
    id="${STEPS[$i]%%:*}"
    if [[ "${id}" == "${FROM}" || "${id%%_*}" == "${FROM}" ]]; then START=${i}; break; fi
done
if [[ ${START} -lt 0 ]]; then
    echo "Unknown step for --from: ${FROM}"
    exit 2
fi

mkdir -p "${REPORTS_DIR}"
SUMMARY="${REPORTS_DIR}/summary.md"
echo "# AD Pipeline Run Summary" > "${SUMMARY}"
echo "" >> "${SUMMARY}"
echo "| Step | Command | Exit |" >> "${SUMMARY}"
echo "|------|---------|------|" >> "${SUMMARY}"

cd "${SCRIPT_DIR}"
OVERALL=0
for i in "${!STEPS[@]}"; do
    entry="${STEPS[$i]}"
    num="${entry%%:*}"
    cmd="${entry#*:}"
    if [[ ${i} -lt ${START} ]]; then
        echo "Skipping step ${num} (before --from ${FROM})"
        echo "| ${num} | (skipped) | - |" >> "${SUMMARY}"
        continue
    fi
    echo ""
    echo "=================================================================="
    echo ">>> Step ${num}: ${cmd}"
    echo "=================================================================="
    eval "${cmd}"
    rc=$?
    echo "| ${num} | \`${cmd}\` | ${rc} |" >> "${SUMMARY}"
    if [[ ${rc} -ne 0 ]]; then
        echo ""
        echo "Step ${num} FAILED (exit ${rc}). Halting. See reports/ for details."
        OVERALL=${rc}
        break
    fi
done

echo "" >> "${SUMMARY}"
if [[ ${OVERALL} -eq 0 ]]; then
    echo "**Result: all executed steps passed.**" >> "${SUMMARY}"
    echo ""
    echo "Pipeline finished. Summary: ${SUMMARY}"
else
    echo "**Result: pipeline halted with exit ${OVERALL}.**" >> "${SUMMARY}"
fi
exit ${OVERALL}
