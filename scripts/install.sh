#!/bin/bash
# Set up the project on Linux with uv: a .venv at the project root, PyTorch, the vendored cehrbert package,
# the plotting extras, and optionally Java 8 inside the project. Uses uv only: never the system Python,
# pip or conda. uv itself is a prerequisite (https://docs.astral.sh/uv/getting-started/installation/).
#
# Usage:
#   bash scripts/install.sh                      # detect the GPU, ask about Java
#   bash scripts/install.sh --cuda cpu --java skip
#   bash scripts/install.sh --dry-run            # print what would change, change nothing
#
# Options:
#   --python <major.minor>       Python version for .venv (default: 3.12)
#   --cuda cu129|cu128|cu126|cpu PyTorch build (default: detect; aarch64 supports cu129 and cpu only)
#   --java ask|install|skip      what to do when Java 8 is missing (default: ask)
#   --dry-run                    print each command that would change something instead of running it
#   --help                       show this text
#
# Safe to rerun: every step detects what is already done and prints "skipped (already present)".
# Java 8 is installed into .tools/jdk8 and added to .venv/bin/activate. Writes build/install/install.log.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
VPY="${VENV}/bin/python"
JDK_DIR="${ROOT}/.tools/jdk8"
CONSTRAINTS="${ROOT}/scripts/install_constraints.txt"
LOG_DIR="${ROOT}/build/install"
TOTAL=8
TORCH_VERSION="2.8.0"
HOOK_MARKER="# cehr-bert-ad: project Java 8"

PYTHON_VERSION="3.12"
CUDA_ARG=""
JAVA_MODE="ask"
DRY_RUN=0

CURRENT_STEP="startup"
CUDA_TAG=""
GPU_FOUND=0
JAVA_SKIPPED=0
JAVA_ADVICE=""
ARCH="$(uname -m)"

usage() {
    sed -n '2,/^set /p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
}

need_value() {
    if [[ $2 -lt 2 ]]; then
        echo "Option $1 needs a value"
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) need_value "$1" $#; PYTHON_VERSION="$2"; shift 2 ;;
        --cuda) need_value "$1" $#; CUDA_ARG="$2"; shift 2 ;;
        --java) need_value "$1" $#; JAVA_MODE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 2 ;;
    esac
done

case "${CUDA_ARG}" in
    ""|cu129|cu128|cu126|cpu) ;;
    *) echo "Invalid --cuda value: ${CUDA_ARG}"; usage; exit 2 ;;
esac
case "${JAVA_MODE}" in
    ask|install|skip) ;;
    *) echo "Invalid --java value: ${JAVA_MODE}"; usage; exit 2 ;;
esac
if [[ ! "${PYTHON_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Invalid --python value: ${PYTHON_VERSION} (expected major.minor, for example 3.12)"
    usage
    exit 2
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/install.log") 2>&1
trap 'echo "ERROR: step \"${CURRENT_STEP}\" failed (line ${LINENO}). See ${LOG_DIR}/install.log" >&2' ERR

step() {
    CURRENT_STEP="[$1/${TOTAL}] $2"
    echo "${CURRENT_STEP}"
}
done_msg() { echo "  done"; }
skip_msg() { echo "  skipped (already present)"; }
warn() { echo "  WARNING: $*"; }
die() { echo "  ERROR: $*" >&2; exit 1; }

# Runs a command that changes something, or only prints it under --dry-run.
run() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

uv_install() {
    run uv pip install --python "${VPY}" --constraint "${CONSTRAINTS}" "$@"
}

torch_version() {
    [[ -x "${VPY}" ]] || return 0
    "${VPY}" -c "import torch; print(torch.__version__)" 2>/dev/null || true
}

java_version() {
    "$1" -version 2>&1 | sed -n 's/.*version "\([^"]*\)".*/\1/p' | head -n 1
}

java_ok() {
    local v
    v="$(java_version "$1" || true)"
    [[ "${v}" == 1.8* || "${v}" == 11* ]]
}

# ---------------------------------------------------------------- 1. Preflight
step 1 "Preflight"
[[ "$(uname -s)" == "Linux" ]] || die "Linux is required (found $(uname -s))."
(( BASH_VERSINFO[0] >= 4 )) || die "bash 4 or newer is required (found ${BASH_VERSION})."
[[ -f "${ROOT}/cehrbert/pyproject.toml" && -d "${ROOT}/scripts/pipeline" ]] \
    || die "Project layout not found under ${ROOT} (expected cehrbert/pyproject.toml and scripts/pipeline/)."
command -v uv >/dev/null 2>&1 \
    || die "uv is required, install it from https://docs.astral.sh/uv/getting-started/installation/"
echo "  architecture: ${ARCH}"
echo "  uv: $(uv --version)"
free_kb="$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')"
free_gb=$(( free_kb / 1024 / 1024 ))
echo "  free disk space: ${free_gb} GB"
(( free_gb >= 20 )) || warn "less than 20 GB free; PyTorch and the pipeline outputs need space."
done_msg

# ---------------------------------------------------------------- 2. GPU
step 2 "GPU and CUDA build"
smi_cuda=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_FOUND=1
    gpu_line="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
    smi_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    echo "  GPU: ${gpu_line%%,*}"
    echo "  driver: ${gpu_line##*, }"
    echo "  CUDA version: ${smi_cuda:-unknown}"
else
    warn "no GPU found (nvidia-smi missing or not working); PyTorch will be the CPU build unless --cuda is given."
fi

if [[ -n "${CUDA_ARG}" ]]; then
    CUDA_TAG="${CUDA_ARG}"
elif [[ ${GPU_FOUND} -eq 0 ]]; then
    CUDA_TAG="cpu"
elif [[ "${ARCH}" == "aarch64" ]]; then
    CUDA_TAG="cu129"
else
    [[ -n "${smi_cuda}" ]] || die "could not read the CUDA version from nvidia-smi. Pass --cuda cu129|cu128|cu126|cpu."
    cuda_num=$(( 10#${smi_cuda%%.*} * 100 + 10#${smi_cuda##*.} ))
    if   (( cuda_num >= 1209 )); then CUDA_TAG="cu129"
    elif (( cuda_num >= 1208 )); then CUDA_TAG="cu128"
    elif (( cuda_num >= 1206 )); then CUDA_TAG="cu126"
    else die "CUDA ${smi_cuda} is older than 12.6, which the PyTorch wheels need. Update the driver or pass --cuda cpu."
    fi
fi
if [[ "${ARCH}" == "aarch64" && "${CUDA_TAG}" != "cu129" && "${CUDA_TAG}" != "cpu" ]]; then
    die "on aarch64 PyTorch ${TORCH_VERSION} has CUDA wheels only for cu129. Use --cuda cu129 or --cuda cpu."
fi
echo "  PyTorch build: ${CUDA_TAG}"
done_msg

# ---------------------------------------------------------------- 3. Environment
step 3 "Python environment (.venv, Python ${PYTHON_VERSION})"
if [[ -x "${VPY}" ]]; then
    have="$("${VPY}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    [[ "${have}" == "${PYTHON_VERSION}" ]] \
        || die ".venv uses Python ${have}, not ${PYTHON_VERSION}. Remove .venv or pass --python ${have}."
    skip_msg
else
    run uv venv --python "${PYTHON_VERSION}" "${VENV}"
    done_msg
fi

# ---------------------------------------------------------------- 4. Java 8
write_activate_hook() {
    [[ -f "${VENV}/bin/activate" ]] || return 0
    grep -qF "${HOOK_MARKER}" "${VENV}/bin/activate" && return 0
    {
        echo ""
        echo "${HOOK_MARKER}"
        echo "export JAVA_HOME=\"${JDK_DIR}\""
        echo "export PATH=\"\$JAVA_HOME/bin:\$PATH\""
    } >> "${VENV}/bin/activate"
}

# curl with a wget fallback. Usage: fetch <url> [output file]; without a file the body goes to stdout.
fetch() {
    if command -v curl >/dev/null 2>&1; then
        if [[ -n "${2:-}" ]]; then curl -fsSL -o "$2" "$1"; else curl -fsSL "$1"; fi
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${2:--}" "$1"
    else
        die "curl or wget is required to download Java."
    fi
}

install_java() {
    local arch tmp url api checksum archive got inner
    case "${ARCH}" in
        x86_64) arch="x64" ;;
        aarch64) arch="aarch64" ;;
        *) die "no Temurin JDK 8 mapping for architecture ${ARCH}. Install OpenJDK 8 yourself." ;;
    esac
    url="https://api.adoptium.net/v3/binary/latest/8/ga/linux/${arch}/jdk/hotspot/normal/eclipse"
    api="https://api.adoptium.net/v3/assets/latest/8/hotspot?architecture=${arch}&image_type=jdk&os=linux&vendor=eclipse"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] download ${url} and verify its SHA-256 from ${api}"
        echo "  [dry-run] unpack into ${JDK_DIR}"
        echo "  [dry-run] append JAVA_HOME and PATH lines to ${VENV}/bin/activate"
        return 0
    fi
    tmp="$(mktemp -d "${LOG_DIR}/jdk.XXXXXX")"
    checksum="$(fetch "${api}" | "${VPY}" -I -c 'import json,sys; print(json.load(sys.stdin)[0]["binary"]["package"]["checksum"])')"
    [[ -n "${checksum}" ]] || die "could not read the JDK checksum from the Adoptium API."
    archive="${tmp}/jdk8.tar.gz"
    echo "  downloading Temurin JDK 8 (${arch})"
    fetch "${url}" "${archive}"
    got="$(sha256sum "${archive}" | awk '{print $1}')"
    [[ "${got}" == "${checksum}" ]] || die "SHA-256 mismatch for the JDK download (expected ${checksum}, got ${got})."
    mkdir "${tmp}/unpacked"
    tar -xzf "${archive}" -C "${tmp}/unpacked"
    inner="$(find "${tmp}/unpacked" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [[ -n "${inner}" ]] || die "the JDK archive was empty."
    mkdir -p "${ROOT}/.tools"
    mv "${inner}" "${JDK_DIR}"
    rm -rf "${tmp}"
    write_activate_hook
    "${JDK_DIR}/bin/java" -version 2>&1 | sed 's/^/  /'
}

step 4 "Java 8"
found_java=""
for candidate in "$(command -v java 2>/dev/null || true)" \
                 "${JAVA_HOME:+${JAVA_HOME}/bin/java}" \
                 "${JDK_DIR}/bin/java"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]] && java_ok "${candidate}"; then
        found_java="${candidate}"
        break
    fi
done

if [[ -n "${found_java}" ]]; then
    echo "  found ${found_java} (version $(java_version "${found_java}"))"
    if [[ "${found_java}" == "${JDK_DIR}/bin/java" && ${DRY_RUN} -eq 0 ]]; then
        write_activate_hook
    fi
    skip_msg
else
    choice="${JAVA_MODE}"
    if [[ "${choice}" == "ask" ]]; then
        if [[ -t 0 ]]; then
            while true; do
                echo "Java 8 is required for sequence generation (Spark) and was not found."
                printf '  [i] install it into the project   [s] skip, I will install it myself\n> '
                read -r answer || answer="s"
                case "${answer}" in
                    i|I) choice="install"; break ;;
                    s|S) choice="skip"; break ;;
                    *) echo "  Please answer i or s." ;;
                esac
            done
        else
            echo "  stdin is not a terminal; treating --java ask as skip."
            choice="skip"
        fi
    fi
    if [[ "${choice}" == "install" ]]; then
        install_java
        done_msg
    else
        JAVA_SKIPPED=1
        if grep -qiE '^ID(_LIKE)?=.*(debian|ubuntu)' /etc/os-release 2>/dev/null; then
            JAVA_ADVICE="sudo apt-get install openjdk-8-jdk-headless"
        else
            JAVA_ADVICE="install OpenJDK 8 with your package manager"
        fi
        warn "Java 8 not found; sequence generation needs it. Run: ${JAVA_ADVICE}"
        done_msg
    fi
fi

# ---------------------------------------------------------------- 5. PyTorch
step 5 "PyTorch ${TORCH_VERSION} (${CUDA_TAG})"
have_torch="$(torch_version)"
if [[ "${have_torch}" == "${TORCH_VERSION}+${CUDA_TAG}" ||
      ( "${CUDA_TAG}" == "cpu" && "${ARCH}" == "aarch64" && "${have_torch}" == "${TORCH_VERSION}" ) ]]; then
    echo "  found torch ${have_torch}"
    skip_msg
else
    uv_install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    done_msg
fi

# ---------------------------------------------------------------- 6. cehrbert
step 6 "cehrbert (editable)"
before_torch="$(torch_version)"
# The relative path "cehrbert[dev]" is resolved against the project root.
cd "${ROOT}"
run env SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+local \
    uv pip install --python "${VPY}" --constraint "${CONSTRAINTS}" -e "cehrbert[dev]"
after_torch="$(torch_version)"
[[ "${before_torch}" == "${after_torch}" ]] \
    || die "installing cehrbert changed torch from ${before_torch:-none} to ${after_torch:-none}. Reinstall torch with the right --cuda."
done_msg

# ---------------------------------------------------------------- 7. Extras
step 7 "Extras (matplotlib, statsmodels)"
uv_install matplotlib statsmodels
done_msg

# ---------------------------------------------------------------- 8. Verify
step 8 "Verify"
failed=0
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  [dry-run] would run scripts/<part>/00_check_environment.py for each part"
else
    for part in data_verification cohort data_qc cehrbert_model classical_ml evaluation feature_importance; do
        if out="$("${VPY}" "${ROOT}/scripts/${part}/00_check_environment.py" 2>&1)"; then
            echo "  ${part}: pass"
        else
            echo "  ${part}: FAIL"
            echo "${out}" | sed 's/^/    /'
            failed=1
        fi
    done
    if [[ ${GPU_FOUND} -eq 1 ]]; then
        echo "  torch.cuda.is_available(): $("${VPY}" -c 'import torch; print(torch.cuda.is_available())')"
    fi
    [[ ${failed} -eq 0 ]] || die "one or more environment checks failed. Fix the packages listed above and rerun."
fi
done_msg

echo ""
echo "Install complete."
echo "  Activate the environment:  source .venv/bin/activate"
if [[ ${JAVA_SKIPPED} -eq 1 ]]; then
    echo "  WARNING: Java 8 is missing. Sequence generation needs it. Run: ${JAVA_ADVICE}"
fi
echo "  Run the pipeline:          bash scripts/run_all.sh --link /path/to/your/omop_data"
