# Alzheimer's Disease Prediction Pipeline with CEHR-BERT

Trains CEHR-BERT for Alzheimer's Disease (AD) prediction from OMOP data, as a sequence of
**self-contained, individually-runnable steps**. Each step validates its inputs,
does its work, and writes a result report (`reports/<step>/status.json` +
`report.md`). Validation/verification steps gate the pipeline: a failure exits non-zero
and the orchestrator halts.

Steps in execution order. The order is not numeric: the cohort and its held-out test split
(step 06) are built from the OMOP tables alone, before pretraining, so the test patients are kept
out of pretraining entirely (step 03b), and every patient is tokenized again after pretraining
(step 05b).

```
00  check environment          (deps / GPU / Java, advisory)
01  validate dataset           verify required OMOP tables/columns exist
01b fill datetimes             fill null OMOP *_datetime columns from *_date columns
06  prepare finetuning data    labeled AD cohort + canonical train/val/test split (OMOP only)
02  prepare pretraining data   generate CEHR-BERT patient sequences (Spark)
02b prepare cohort folders     CEHR-BERT finetune/test cohort folders from the splits
03b build pretraining input    patient_sequence minus the test patients
03  evaluate pretrain config   dry-run validate the pretraining config
04  pretrain model             train CEHR-BERT (refuses test patients and stale artifacts)
05  verify pretraining         model + prepared dataset; no test patient was pretrained on
05b tokenize full dataset      every patient, with the pretrained tokenizer
07  validate finetune config   dry-run validate finetuning config + dependencies
08  finetune                   fine-tune for AD prediction (train/validation)
09  predict & verify           score the held-out test cohort + check ROC-AUC / PR-AUC
10a validation predictions     fine-tuned model on the val split (threshold selection)
10  evaluate                   metrics with bootstrap confidence intervals
11  baseline features          tabular features from OMOP
12  train baselines            L2 logistic regression + XGBoost
14  attention / token stats    attention analysis, token history statistics
15  linear probe features      frozen CEHR-BERT features per split
16  train linear probe         logistic regression on the frozen features
13  compare models             parity check + comparison table (needs 10, 12, 16)
```

## Layout

```
scripts/ad_pipeline/
├── run_pipeline.sh                 orchestrator (runs every step in order, halts on first failure)
├── 00_check_environment.py         01_validate_dataset.py         01b_fill_datetimes.py
├── 02_prepare_pretraining_data.py  + 02_generate_pretraining_data.sh   (Spark wrapper)
├── 02b_prepare_cohort_folders.py
├── 03_evaluate_pretrain_config.py  03b_build_pretraining_input.py
├── 04_pretrain.py                  + 04_pretrain.sh                     (runner wrapper)
├── 05_verify_pretraining.py        05b_tokenize_full_dataset.py
├── 06_prepare_finetuning_data.py
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

The cohort (step 6) is built automatically from these tables and from nothing else (no
CEHR-BERT output); you do **not** supply a cohort file. Nothing from the CEHR-BERT steps is ever
written into this folder: step 2 gives cehrbert_data a folder of symlinks to the tables
(`data/cehrbert_data/omop_input`) as its input folder instead.
cehrbert_data caches its measurement events in `omop_input/processed_measurement` and reuses them on
the next run, so delete that folder when the OMOP data changes. Folders from earlier runs that sit in
the OMOP folder (`processed_measurement`, `qualified_concept_list`, `ad_cohort`) are no longer used and can be removed. AD cases are identified from source ICD codes in `concept.concept_code`
(`ICD-9-CM:331.0`, `ICD-10-CM:G30*`).

Cohort definition (step 6, `pipeline/ad_cohort.py`): a case is anchored at its first AD
condition date and a control at one of its own visit dates with at least one year of
observation after it (`--control-followup-days`). The control anchor year is drawn so control
anchor years follow the case anchor years, limited to anchors up to the data cutoff
(`cdm_source.source_release_date`, else the last visit) minus the follow-up; cases after that
date stay in the cohort. Index = anchor minus `--gap-days`. Both groups must satisfy, at the
index: age at least `--min-age`, index on or after `--index-floor`, first visit at least
`--window-days` earlier, and one condition, drug, procedure or measurement in the window
`[index - window, index)`. An event counts only when it would become a CEHR-BERT sequence token: its
`visit_occurrence_id` is not null and its standard concept id (`condition_concept_id`,
`drug_concept_id`, `procedure_concept_id`, `measurement_concept_id`) is not 0 and is used by at least
`--min-concept-patients` (default 100, 0 turns the concept part off) distinct persons, counted over the
visit-linked events of the four tables together. This mirrors `generate_included_concept_list`
(`--min_num_of_patients 100`) and `extract_events_by_domain` of cehrbert_data, so cohort patients do not
end up without a sequence. The step report lists exclusion counts per group, the number of qualified
concepts and the number of persons excluded by the event rule.

## Quick start

```bash
cd scripts/ad_pipeline

# Run everything, pointing step 1 at your OMOP data:
bash run_pipeline.sh --link /path/to/your/omop_data

# ...or run steps individually, in this order:
python 00_check_environment.py
python 01_validate_dataset.py --link /path/to/your/omop_data
python 01b_fill_datetimes.py
python 06_prepare_finetuning_data.py             # cohort + splits.parquet (OMOP tables only)
python 02_prepare_pretraining_data.py            # Spark; set SPARK_DRIVER_MEMORY as needed
python 02b_prepare_cohort_folders.py             # finetune/test cohort folders; every cohort patient needs a sequence
python 03b_build_pretraining_input.py            # patient_sequence_pretrain (no test patients)
python 03_evaluate_pretrain_config.py
python 04_pretrain.py
python 05_verify_pretraining.py
python 05b_tokenize_full_dataset.py              # data/full_tokenized (all patients)
python 07_validate_finetune_config.py
python 08_finetune.py                            # train/validation only (do_predict=false)
python 09_predict.py                             # score the held-out test cohort + verify
python 10a_validation_predictions.py
python 10_evaluate.py
python 11_prepare_baseline_features.py
python 12_train_baselines.py
python 14_attention_analysis.py
python 14_token_history_stats.py
python 15_linear_probe_features.py
python 16_train_linear_probe.py
python 13_compare_models.py
```

Resume a partial run with `bash run_pipeline.sh --from <step>`, e.g. `--from 07` after
pretraining. `--from` names a position in the execution order above, not a number range.

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
├── omop_data -> /path/to/your/data    symlink created by step 1 (--link); never written by CEHR-BERT steps
├── cohort/splits.parquet               person_id, index_date, label, split (train/val/test) (step 6)
├── cehrbert_data/omop_input/           symlinks to the OMOP tables; cehrbert_data's input folder, where it
│                                        writes processed_measurement and qualified_concept_list (step 2)
├── cehrbert_data/patient_sequence/     patient sequences, all patients        (step 2)
├── cehrbert_data/cohort/finetune/cohort.parquet   train + val rows         (step 02b)
├── cehrbert_data/cohort/test/cohort.parquet       held-out test rows       (step 02b)
├── cehrbert_data/patient_sequence_pretrain/  same, without test patients    (step 3b)
├── pretrain_prepared/<name>_<hash>/    pretraining DatasetDict, no test patients (step 4)
├── pretrain_results/                   pretrained model + tokenizer           (step 4)
├── full_tokenized/                     all patients tokenized; validation =
│                                        canonical val, train = the rest        (step 5b)
├── finetune_prepared/                  fine-tuning dataset (train/val)  (step 8)
├── predict_prepared/                   test-cohort dataset           (step 9)
└── finetune_results/
    ├── (fine-tuned model + tokenizer)                                (step 8)
    ├── test_predictions/*.parquet      subject_id, prediction_time,
    │                                    predicted_boolean_probability, boolean_value
    └── test_results.json               roc_auc, pr_auc, test_loss    (step 9)
```

## Raw OMOP profile (step 18)

`python 18_profile_omop.py` profiles the raw OMOP tables for AD cases (index = first AD
diagnosis) versus controls under two index rules (`control_last_visit` and a seeded
`control_matched_index` drawn from each control's own visit dates). It reads only the OMOP
tables, writes `build/omop_profile/` (`profile.json`, `table1.csv`/`.md` with SMDs versus AD,
`exclusion_flow.csv`, `figures/`) and a report, and changes no cohort file. Concept groups live in
`configs/omop_profile_concepts.yaml`. Useful options: `--person-sample-fraction 0.02` for a fast
run, `--skip-sections measurement,duplicates` to drop the heaviest parts, `--date-floor`.

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

- **Step 6** splits the labeled cohort into a fine-tuning part (~90%, later train and val) and a
  test part (~10%), split by `person_id` with a fixed seed (no patient appears in both), and writes
  the canonical `data/cohort/splits.parquet` (train / val / test). Tune with
  `--test-holdout 0.1 --split-seed 42`. It reads only the OMOP tables, so the classical baselines
  (steps 11 and 12) need nothing from CEHR-BERT.
- **Step 02b** writes the folders the finetune runner reads from the splits: `cohort/finetune`
  (train + val) and `cohort/test` under `data/cehrbert_data`. It fails, listing up to 20 person_ids,
  when a cohort patient has no non-empty sequence in `patient_sequence`.
- **Step 3b** removes every test patient from the pretraining input, so the pretrained
  encoder never sees them (neither in its train nor its validation split). Step 4 refuses an
  input that still holds a test patient, and step 5 checks the prepared dataset again.
- **Step 5b** tokenizes every patient with the pretrained tokenizer into `data/full_tokenized`,
  which fine-tuning, prediction and the linear probe read (`tokenized_full_dataset_path`).
  Its `validation` split is the canonical val split, so step 8 validates on exactly that.
- **Step 8** fine-tunes on `data/cehrbert_data/cohort/finetune/` with `do_predict: false`. The train/validation
  assignment comes from `data/full_tokenized` (canonical train / val), not from
  `validation_split_percentage`, which this flow does not read.
- **Step 9** runs prediction on `data/cehrbert_data/cohort/test/` with the fine-tuned model
  (`do_train: false, do_predict: true`) and verifies the metrics. Re-runnable on its own.

This relies on one small edit to the vendored runner
(`cehrbert/src/cehrbert/runners/hf_cehrbert_finetune_runner.py`): in predict-only mode
(`do_predict` without `do_train`), the extracted cohort is treated as the `test` split.
Normal training runs are unaffected.
