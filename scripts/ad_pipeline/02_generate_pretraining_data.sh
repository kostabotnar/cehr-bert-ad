#!/bin/bash
# Step 2 (launcher): Generate CEHR-BERT pretraining sequences from OMOP tables.
#
# Thin wrapper: sets Spark environment and invokes cehrbert_data. Preflight validation
# and result reporting live in 02_prepare_pretraining_data.py — prefer running that:
#
#     python 02_prepare_pretraining_data.py
#
# Running this script directly is supported for debugging the Spark step in isolation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_FOLDER="${INPUT_FOLDER:-${PROJECT_ROOT}/data/omop_data}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${PROJECT_ROOT}/data/cehrbert_data}"
START_DATE="${START_DATE:-1952-01-01}"

# Spark memory/parallelism (override via environment).
export SPARK_WORKER_INSTANCES="${SPARK_WORKER_INSTANCES:-1}"
export SPARK_WORKER_CORES="${SPARK_WORKER_CORES:-8}"
export SPARK_EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-2}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-16g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-16g}"

echo "Project root : ${PROJECT_ROOT}"
echo "Input folder : ${INPUT_FOLDER}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Start date   : ${START_DATE}"
echo "Spark driver/executor memory: ${SPARK_DRIVER_MEMORY}/${SPARK_EXECUTOR_MEMORY}"
echo ""

mkdir -p "${OUTPUT_FOLDER}"
cd "${PROJECT_ROOT}"

echo "Generating concept list..."
python -u -m cehrbert_data.apps.generate_included_concept_list \
    -i "${INPUT_FOLDER}" \
    -o "${OUTPUT_FOLDER}" \
    --min_num_of_patients 100 \
    --ehr_table_list condition_occurrence procedure_occurrence drug_exposure measurement

# Symlink so training-data generation finds qualified_concept_list.
ln -sf "${OUTPUT_FOLDER}/qualified_concept_list" "${INPUT_FOLDER}/qualified_concept_list" 2>/dev/null || true

echo ""
echo "Generating patient sequences..."
python -m cehrbert_data.apps.generate_training_data \
    --input_folder "${INPUT_FOLDER}" \
    --output_folder "${OUTPUT_FOLDER}" \
    -d "${START_DATE}" \
    --att_type day \
    --inpatient_att_type day \
    -iv \
    -ip \
    --include_concept_list \
    --gpt_patient_sequence \
    --domain_table_list condition_occurrence procedure_occurrence drug_exposure measurement

echo ""
echo "Done! Patient sequences saved to: ${OUTPUT_FOLDER}/patient_sequence/"
