#!/bin/bash
# Shared runner logic for scripts/<part>/run.sh. Source this file, then call parse_args and run_steps.
#
# A part's run.sh looks like:
#
#   PART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${PART_DIR}/../lib/run_steps.sh"
#   parse_args "ctx window-days" "$@"
#   STEPS=(
#       "00:${PY} 00_check_environment.py"
#       "01:${PY} 01_step.py ${CTX_ARG}"
#   )
#   run_steps "Part title"
#   exit $?
#
# STEPS holds "<step id>:<command>" entries in execution order; commands run from the part folder.
# run_steps halts on the first non-zero exit and writes <part>/reports/summary.md.
#
# Options handled here (the first argument of parse_args lists the optional ones a part accepts):
#   --from <step id>            resume from that step (always accepted; "5" means "05")
#   --link <path>               OMOP data folder, sets LINK_ARG (relative paths resolve against the caller)
#   --ctx <int>                 token context, sets CTX_ARG
#   --observation-window <int>  look-back in days, sets OBS_ARG (--observation-window) and WIN_ARG (--window-days)
#   --window-days <int>         look-back in days, sets WIN_ARG only

PY="${PYTHON:-python}"
INVOCATION_DIR="$(pwd)"   # captured before run_steps changes into the part folder

FROM=""
LINK_ARG=""
CTX_ARG=""
OBS_ARG=""
WIN_ARG=""

_option_allowed() {
    [[ " ${_ALLOWED_OPTIONS} " == *" $1 "* ]]
}

_need_value() {
    if [[ $2 -lt 2 ]]; then
        echo "Option $1 needs a value"
        exit 2
    fi
}

parse_args() {
    _ALLOWED_OPTIONS="$1"
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from) _need_value "$1" $#; FROM="$2"; shift 2 ;;
            --link)
                _option_allowed link || { echo "Unknown argument: $1"; exit 2; }
                _need_value "$1" $#
                # Resolve against the directory the user ran this from, because run_steps changes directory.
                link_src="$2"
                [[ "${link_src}" != /* ]] && link_src="$(realpath -m "${INVOCATION_DIR}/${link_src}")"
                LINK_ARG="--link ${link_src}"
                shift 2 ;;
            --ctx)
                _option_allowed ctx || { echo "Unknown argument: $1"; exit 2; }
                _need_value "$1" $#
                CTX_ARG="--ctx $2"; shift 2 ;;
            --observation-window)
                _option_allowed observation-window || { echo "Unknown argument: $1"; exit 2; }
                _need_value "$1" $#
                OBS_ARG="--observation-window $2"; WIN_ARG="--window-days $2"; shift 2 ;;
            --window-days)
                _option_allowed window-days || { echo "Unknown argument: $1"; exit 2; }
                _need_value "$1" $#
                WIN_ARG="--window-days $2"; shift 2 ;;
            *) echo "Unknown argument: $1"; exit 2 ;;
        esac
    done
}

run_steps() {
    local title="$1"
    local reports_dir="${PART_DIR}/reports"
    local summary="${reports_dir}/summary.md"
    local from="${FROM}"
    [[ "${from}" =~ ^[0-9]$ ]] && from="0${from}"

    local start=0
    if [[ -n "${from}" ]]; then
        start=-1
        local i
        for i in "${!STEPS[@]}"; do
            if [[ "${STEPS[$i]%%:*}" == "${from}" ]]; then start=${i}; break; fi
        done
        if [[ ${start} -lt 0 ]]; then
            echo "Unknown step for --from: ${FROM}"
            return 2
        fi
    fi

    mkdir -p "${reports_dir}"
    {
        echo "# ${title} Run Summary"
        echo ""
        echo "| Step | Command | Exit |"
        echo "|------|---------|------|"
    } > "${summary}"

    cd "${PART_DIR}" || return 2
    local overall=0 i entry num cmd rc
    for i in "${!STEPS[@]}"; do
        entry="${STEPS[$i]}"
        num="${entry%%:*}"
        cmd="${entry#*:}"
        if [[ ${i} -lt ${start} ]]; then
            echo "Skipping step ${num} (before --from ${FROM})"
            echo "| ${num} | (skipped) | - |" >> "${summary}"
            continue
        fi
        echo ""
        echo "=================================================================="
        echo ">>> ${title} step ${num}: ${cmd}"
        echo "=================================================================="
        eval "${cmd}"
        rc=$?
        echo "| ${num} | \`${cmd}\` | ${rc} |" >> "${summary}"
        if [[ ${rc} -ne 0 ]]; then
            echo ""
            echo "Step ${num} FAILED (exit ${rc}). Halting. See ${reports_dir} for details."
            overall=${rc}
            break
        fi
    done

    echo "" >> "${summary}"
    if [[ ${overall} -eq 0 ]]; then
        echo "**Result: all executed steps passed.**" >> "${summary}"
        echo ""
        echo "${title} finished. Summary: ${summary}"
    else
        echo "**Result: halted with exit ${overall}.**" >> "${summary}"
    fi
    return ${overall}
}
