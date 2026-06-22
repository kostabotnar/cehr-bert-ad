#!/bin/bash
# Step 8 (launcher): Fine-tune CEHR-BERT for Alzheimer's Disease prediction.
#
# Thin wrapper: invokes the cehrbert finetuning runner with the given config from the
# project root. Preflight validation, the test-split fix, and result reporting live in
# 08_finetune.py — prefer running that:
#
#     python 08_finetune.py
#
# Args:
#   $1 (optional) path to the finetuning config yaml.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${1:-${SCRIPT_DIR}/configs/ad_finetune_config.yaml}"

echo "Project root: ${PROJECT_ROOT}"
echo "Using config: ${CONFIG_FILE}"
echo ""

cd "${PROJECT_ROOT}"
python -m cehrbert.runners.hf_cehrbert_finetune_runner "${CONFIG_FILE}"

echo ""
echo "Done! Results saved to: data/finetune_results/"
