#!/bin/bash
# Orchestrate the full 9-step AD CEHR-BERT pipeline.
#
# Runs each step in order, halting on the first failure (non-zero exit). Each step writes
# its own report under reports/<step>/; this script also writes reports/summary.md.
#
# Usage:
#   bash run_pipeline.sh                 # run steps 00..09
#   bash run_pipeline.sh --from 06       # resume from step 06 (e.g. after pretraining)
#   bash run_pipeline.sh --link /path/to/omop_data   # pass-through to step 01
#   bash run_pipeline.sh --ctx 4096 --observation-window -1  # experiment overrides
#
# Experiment overrides (optional):
#   --ctx <int>                 token context; forwarded to steps 04, 08, 09 as --ctx.
#   --observation-window <int>  look-back in days (negative = all history); forwarded
#                               to steps 08 and 09 as --observation-window. Step 04
#                               (pretraining) has no observation window, so it is not
#                               forwarded there.
#
# Notes:
#   * Steps 02/04/08 are heavy (Spark / model training) and need the full runtime env.
#   * Validation/verification steps (01,03,05,07,09) are gating: a FAIL stops the run.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/reports"
PY="${PYTHON:-python}"
INVOCATION_DIR="$(pwd)"   # captured before we cd into SCRIPT_DIR

FROM="00"
LINK_ARG=""
CTX_ARG=""        # forwarded to steps 04, 08, 09 when --ctx is given
OBS_ARG=""        # forwarded to steps 08, 09 when --observation-window is given
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
        --observation-window) OBS_ARG="--observation-window $2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

# step number -> command
declare -a STEPS=(
    "00:${PY} 00_check_environment.py"
    "01:${PY} 01_validate_dataset.py ${LINK_ARG}"
    "02:${PY} 02_prepare_pretraining_data.py"
    "03:${PY} 03_evaluate_pretrain_config.py"
    "04:${PY} 04_pretrain.py ${CTX_ARG}"
    "05:${PY} 05_verify_pretraining.py"
    "06:${PY} 06_prepare_finetuning_data.py"
    "07:${PY} 07_validate_finetune_config.py"
    "08:${PY} 08_finetune.py ${CTX_ARG} ${OBS_ARG}"
    "09:${PY} 09_predict.py ${CTX_ARG} ${OBS_ARG}"
)

mkdir -p "${REPORTS_DIR}"
SUMMARY="${REPORTS_DIR}/summary.md"
echo "# AD Pipeline Run Summary" > "${SUMMARY}"
echo "" >> "${SUMMARY}"
echo "| Step | Command | Exit |" >> "${SUMMARY}"
echo "|------|---------|------|" >> "${SUMMARY}"

cd "${SCRIPT_DIR}"
OVERALL=0
for entry in "${STEPS[@]}"; do
    num="${entry%%:*}"
    cmd="${entry#*:}"
    if [[ "${num}" < "${FROM}" ]]; then
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
