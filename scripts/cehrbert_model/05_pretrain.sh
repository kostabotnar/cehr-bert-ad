#!/bin/bash
# Step 4 (launcher): Pretrain CEHR-BERT on the generated patient sequences.
#
# Thin wrapper: invokes the cehrbert pretraining runner with the given config from the
# project root. Preflight validation and result reporting live in 04_pretrain.py —
# prefer running that:
#
#     python 04_pretrain.py
#
# Args:
#   $1 (optional) path to the pretraining config yaml.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${1:-${SCRIPT_DIR}/configs/ad_pretrain_config.yaml}"

echo "Project root: ${PROJECT_ROOT}"
echo "Using config: ${CONFIG_FILE}"
echo ""

cd "${PROJECT_ROOT}"
python -m cehrbert.runners.hf_cehrbert_pretrain_runner "${CONFIG_FILE}"

echo ""
echo "Done! Model saved to: data/pretrain_results/"
