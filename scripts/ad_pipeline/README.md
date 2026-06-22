# Alzheimer's Disease Prediction Pipeline with CEHR-BERT

Trains CEHR-BERT for Alzheimer's Disease (AD) prediction from OMOP data, as a sequence of
**nine self-contained, individually-runnable steps**. Each step validates its inputs,
does its work, and writes a result report (`reports/<step>/status.json` +
`report.md`). Validation/verification steps gate the pipeline: a failure exits non-zero
and the orchestrator halts.

```
00 check environment        (deps / GPU / Java — advisory)
01 validate dataset         verify required OMOP tables/columns exist
02 prepare pretraining data generate CEHR-BERT patient sequences (Spark)
03 evaluate pretrain config dry-run validate the pretraining config
04 pretrain model           train CEHR-BERT
05 verify pretraining       check model + tokenized dataset artifacts
06 prepare finetuning data  build the labeled AD cohort + hold out a test split
07 validate finetune config dry-run validate finetuning config + dependencies
08 finetune                 fine-tune for AD prediction (train/validation)
09 predict & verify         score the held-out test cohort + check ROC-AUC / PR-AUC
```

## Layout

```
scripts/ad_pipeline/
├── run_pipeline.sh                 orchestrator (runs 00..09, halts on first failure)
├── 00_check_environment.py         01_validate_dataset.py
├── 02_prepare_pretraining_data.py  + 02_generate_pretraining_data.sh   (Spark wrapper)
├── 03_evaluate_pretrain_config.py
├── 04_pretrain.py                  + 04_pretrain.sh                     (runner wrapper)
├── 05_verify_pretraining.py        06_prepare_finetuning_data.py
├── 07_validate_finetune_config.py
├── 08_finetune.py                  + 08_finetune.sh                     (runner wrapper)
├── 09_predict.py                   + 09_predict.sh                      (runner wrapper)
├── configs/   ad_pretrain_config.yaml, ad_finetune_config.yaml
├── pipeline/  shared, tested library (reporting, validation, checks)
├── reports/   generated per-step status.json + report.md (+ summary.md)
└── tests/     pytest suite (synthetic fixtures, no real training)
```

The heavy steps (02 Spark, 04/08/09 runners) are split into a Python entrypoint that owns
preflight checks + reporting and a *thin* bash wrapper that only sets environment
variables and invokes the cehrbert / cehrbert_data tool. Run the Python entrypoint; it
calls the wrapper for you.

## Data requirements

Your OMOP folder should contain one subfolder per table (each with `*.parquet`):

```
your_omop_data/
├── person/                  (required)
├── visit_occurrence/        (required)
├── condition_occurrence/    (required)
├── concept/                 (required — AD codes + concept list)
├── drug_exposure/           (recommended)
├── procedure_occurrence/    (recommended)
└── measurement/             (recommended)
```

The cohort (step 6) is built automatically from these tables; you do **not** supply a
cohort file. AD cases are identified from source ICD codes in `concept.concept_code`
(`ICD-9-CM:331.0`, `ICD-10-CM:G30*`).

## Quick start

```bash
cd scripts/ad_pipeline

# Run everything, pointing step 1 at your OMOP data:
bash run_pipeline.sh --link /path/to/your/omop_data

# ...or run steps individually:
python 00_check_environment.py
python 01_validate_dataset.py --link /path/to/your/omop_data
python 02_prepare_pretraining_data.py            # Spark; set SPARK_DRIVER_MEMORY as needed
python 03_evaluate_pretrain_config.py
python 04_pretrain.py
python 05_verify_pretraining.py
python 06_prepare_finetuning_data.py             # cohort -> ad_cohort/finetune + ad_cohort/test
python 07_validate_finetune_config.py
python 08_finetune.py                            # train/validation only (do_predict=false)
python 09_predict.py                             # score the held-out test cohort + verify
```

Resume a partial run (e.g. after pretraining) with `bash run_pipeline.sh --from 06`.

## Reports

Every step writes to `reports/<step>/`:
- `status.json` — machine-readable: `status` (`pass`/`warn`/`fail`), every check,
  metrics (row counts, losses, ROC-AUC, …), artifacts, notes.
- `report.md` — the same as a human-readable summary table.

`run_pipeline.sh` also writes `reports/summary.md` with each step's exit code.

Exit codes: `0` = pass or warn (pipeline may continue), `1` = fail (gating). Warnings
(e.g. a missing recommended table, no GPU) never block the pipeline.

## Outputs

```
data/
├── omop_data -> /path/to/your/data    symlink created by step 1 (--link)
├── cehrbert_data/patient_sequence/     patient sequences            (step 2)
├── pretrain_prepared/                  tokenized DatasetDict         (step 4)
├── pretrain_results/                   pretrained model + tokenizer  (step 4)
├── omop_data/ad_cohort/
│   ├── finetune/cohort.parquet         fine-tuning cohort (~90%)     (step 6)
│   └── test/cohort.parquet             held-out test cohort (~10%)   (step 6)
├── finetune_prepared/                  fine-tuning dataset (train/val)  (step 8)
├── predict_prepared/                   test-cohort dataset           (step 9)
└── finetune_results/
    ├── (fine-tuned model + tokenizer)                                (step 8)
    ├── test_predictions/*.parquet      subject_id, prediction_time,
    │                                    predicted_boolean_probability, boolean_value
    └── test_results.json               roc_auc, pr_auc, test_loss    (step 9)
```

## Configuration

- `configs/ad_pretrain_config.yaml` — pretraining hyperparameters
- `configs/ad_finetune_config.yaml` — finetuning hyperparameters

Steps 3 and 7 dry-run-validate these against the cehrbert runner dataclasses (when
installed) plus value-range and referenced-path checks, so a bad config fails *before*
a long training run starts. Step 9 derives its own predict config from the finetuning
config (see below) and writes it to `reports/09_predict/predict_config.yaml`.

## Tests

```bash
python -m pytest scripts/ad_pipeline/tests -q
```

The suite uses tiny synthetic OMOP parquet / fake model / fake `DatasetDict` fixtures and
mocks the training launchers — **no real training runs**. Tests that need the `datasets`
library skip themselves automatically when it is not installed.

## Train / validation / test split

The held-out **test set is decided once, up front, at the patient level** — not carved out
during training. This keeps fine-tuning simple and prediction independent:

- **Step 6** splits the labeled cohort into `ad_cohort/finetune/` (~90%) and
  `ad_cohort/test/` (~10%), split by `person_id` with a fixed seed (no patient appears in
  both). Tune with `--test-holdout 0.1 --split-seed 42`.
- **Step 8** fine-tunes on `ad_cohort/finetune/` with an internal train/validation split
  (`validation_split_percentage`) and `do_predict: false` — one clean run, no test split
  needed here.
- **Step 9** runs prediction on `ad_cohort/test/` with the fine-tuned model
  (`do_train: false, do_predict: true`) and verifies the metrics. Re-runnable on its own.

This relies on one small edit to the vendored runner
(`cehrbert/src/cehrbert/runners/hf_cehrbert_finetune_runner.py`): in predict-only mode
(`do_predict` without `do_train`), the extracted cohort is treated as the `test` split.
Normal training runs are unaffected.
