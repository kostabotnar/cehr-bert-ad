#!/bin/bash
# Run the parts of the AD pipeline in dependency order, halting on the first failing part.
#
# Parts, in the order they run:
#   data_verification cohort data_qc cehrbert_model classical_ml evaluation feature_importance
#
# Usage:
#   bash scripts/run_all.sh                                   # every part
#   bash scripts/run_all.sh --link /path/to/omop_data         # point data_verification at the OMOP data
#   bash scripts/run_all.sh --parts cohort,classical_ml       # a subset (still run in the order above)
#   bash scripts/run_all.sh --ctx 4096 --observation-window -1
#
# Options:
#   --parts a,b,...             parts to run (default: all)
#   --link <path>               OMOP data folder; passed to data_verification
#   --ctx <int>                 token context; passed to cehrbert_model, evaluation, feature_importance
#   --observation-window <int>  look-back in days (negative = all history); passed as --observation-window to
#                               cehrbert_model and as --window-days to classical_ml, evaluation, feature_importance
#   --name <text>               comparison name (build/evaluation/<name>/); passed to evaluation
#
# Each part has its own run.sh (see scripts/<part>/README.md), so a part can also be run on its own.
# Writes scripts/reports/summary.md.

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPTS_DIR}/reports"
SUMMARY="${REPORTS_DIR}/summary.md"
INVOCATION_DIR="$(pwd)"

ALL_PARTS=(data_verification cohort data_qc cehrbert_model classical_ml evaluation feature_importance)
PARTS_ARG=""
LINK_PATH=""
CTX=""
OBS=""
NAME=""

need_value() {
    if [[ $2 -lt 2 ]]; then
        echo "Option $1 needs a value"
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parts) need_value "$1" $#; PARTS_ARG="$2"; shift 2 ;;
        --link)
            need_value "$1" $#
            LINK_PATH="$2"
            [[ "${LINK_PATH}" != /* ]] && LINK_PATH="$(realpath -m "${INVOCATION_DIR}/${LINK_PATH}")"
            shift 2 ;;
        --ctx) need_value "$1" $#; CTX="$2"; shift 2 ;;
        --observation-window) need_value "$1" $#; OBS="$2"; shift 2 ;;
        --name) need_value "$1" $#; NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

# Validate --parts; keep the canonical order whatever order the user gave.
SELECTED=()
if [[ -z "${PARTS_ARG}" ]]; then
    SELECTED=("${ALL_PARTS[@]}")
else
    IFS=',' read -r -a requested <<< "${PARTS_ARG}"
    for name in "${requested[@]}"; do
        found=0
        for part in "${ALL_PARTS[@]}"; do [[ "${part}" == "${name}" ]] && found=1; done
        if [[ ${found} -eq 0 ]]; then
            echo "Unknown part: ${name} (expected one of: ${ALL_PARTS[*]})"
            exit 2
        fi
    done
    for part in "${ALL_PARTS[@]}"; do
        for name in "${requested[@]}"; do
            if [[ "${part}" == "${name}" ]]; then SELECTED+=("${part}"); break; fi
        done
    done
fi

# Arguments each part's run.sh understands.
part_args() {
    case "$1" in
        data_verification) [[ -n "${LINK_PATH}" ]] && echo "--link ${LINK_PATH}" ;;
        cehrbert_model)
            [[ -n "${CTX}" ]] && echo -n "--ctx ${CTX} "
            [[ -n "${OBS}" ]] && echo -n "--observation-window ${OBS}" ;;
        classical_ml) [[ -n "${OBS}" ]] && echo -n "--window-days ${OBS}" ;;
        evaluation)
            [[ -n "${CTX}" ]] && echo -n "--ctx ${CTX} "
            [[ -n "${OBS}" ]] && echo -n "--window-days ${OBS} "
            [[ -n "${NAME}" ]] && echo -n "--name ${NAME}" ;;
        feature_importance)
            [[ -n "${CTX}" ]] && echo -n "--ctx ${CTX} "
            [[ -n "${OBS}" ]] && echo -n "--window-days ${OBS}" ;;
    esac
    return 0
}

mkdir -p "${REPORTS_DIR}"
{
    echo "# AD Pipeline Run Summary"
    echo ""
    echo "| Part | Exit |"
    echo "|------|------|"
} > "${SUMMARY}"

OVERALL=0
for part in "${SELECTED[@]}"; do
    # shellcheck disable=SC2046
    args=$(part_args "${part}")
    echo ""
    echo "##################################################################"
    echo "### Part: ${part} ${args}"
    echo "##################################################################"
    # word splitting of ${args} is intended: it is a list of options
    # shellcheck disable=SC2086
    bash "${SCRIPTS_DIR}/${part}/run.sh" ${args}
    rc=$?
    echo "| ${part} | ${rc} |" >> "${SUMMARY}"
    if [[ ${rc} -ne 0 ]]; then
        echo ""
        echo "Part ${part} FAILED (exit ${rc}). Halting."
        OVERALL=${rc}
        break
    fi
done

echo "" >> "${SUMMARY}"
if [[ ${OVERALL} -eq 0 ]]; then
    echo "**Result: all executed parts passed.**" >> "${SUMMARY}"
    echo ""
    echo "Pipeline finished. Summary: ${SUMMARY}"
else
    echo "**Result: halted with exit ${OVERALL}.**" >> "${SUMMARY}"
fi
exit ${OVERALL}
