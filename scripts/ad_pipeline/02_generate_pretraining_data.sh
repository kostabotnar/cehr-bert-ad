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

# INPUT_FOLDER is the staging folder of symlinks to the OMOP tables (see 02_prepare_pretraining_data.py): cehrbert_data
# writes processed_measurement and reads qualified_concept_list inside it, so it must never be the OMOP folder.
INPUT_FOLDER="${INPUT_FOLDER:-${PROJECT_ROOT}/data/cehrbert_data/omop_input}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${PROJECT_ROOT}/data/cehrbert_data}"
START_DATE="${START_DATE:-1952-01-01}"

# Spark memory/parallelism (override via environment).
export SPARK_WORKER_INSTANCES="${SPARK_WORKER_INSTANCES:-1}"
export SPARK_WORKER_CORES="${SPARK_WORKER_CORES:-8}"
export SPARK_EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-2}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-16g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-16g}"
PYTHON="${PYTHON:-python}"
# Spark starts its Python workers from PYSPARK_PYTHON; without it they use the system python, which lacks cehrbert_data.
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-${PYTHON}}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-${PYTHON}}"
# Birth datetimes of 1900-01-01 trip Spark's legacy-calendar guard on Parquet writes; the output is only read
# by Spark 3 and pyarrow, which both use the proleptic Gregorian calendar, so write values unchanged.
REBASE="spark.sql.legacy.parquet.int96RebaseModeInWrite=CORRECTED"
REBASE_DT="spark.sql.legacy.parquet.datetimeRebaseModeInWrite=CORRECTED"
# Local mode with all 20 cores ran out of memory writing patient_sequence (long patient histories);
# 8 concurrent tasks with a 96g driver completed in about 46 minutes on the DGX Spark.
SPARK_MASTER="${SPARK_MASTER:-local[8]}"
SPARK_SHUFFLE_PARTITIONS="${SPARK_SHUFFLE_PARTITIONS:-1000}"
export PYSPARK_SUBMIT_ARGS="${PYSPARK_SUBMIT_ARGS:---master ${SPARK_MASTER} --driver-memory ${SPARK_DRIVER_MEMORY} --conf spark.sql.shuffle.partitions=${SPARK_SHUFFLE_PARTITIONS} --conf ${REBASE} --conf ${REBASE_DT} pyspark-shell}"

echo "Project root : ${PROJECT_ROOT}"
echo "Input folder : ${INPUT_FOLDER}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Start date   : ${START_DATE}"
echo "Spark driver/executor memory: ${SPARK_DRIVER_MEMORY}/${SPARK_EXECUTOR_MEMORY}"
echo ""

mkdir -p "${OUTPUT_FOLDER}"
cd "${PROJECT_ROOT}"

echo "Generating concept list..."
"${PYTHON}" -u -m cehrbert_data.apps.generate_included_concept_list \
    -i "${INPUT_FOLDER}" \
    -o "${OUTPUT_FOLDER}" \
    --min_num_of_patients 100 \
    --ehr_table_list condition_occurrence procedure_occurrence drug_exposure measurement

# Symlink so training-data generation finds qualified_concept_list (in the staging folder, not the OMOP data).
ln -sfn "${OUTPUT_FOLDER}/qualified_concept_list" "${INPUT_FOLDER}/qualified_concept_list"

echo ""
echo "Generating patient sequences..."
"${PYTHON}" -m cehrbert_data.apps.generate_training_data \
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
