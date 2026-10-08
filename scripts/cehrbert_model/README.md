# cehrbert_model

Pretrains CEHR-BERT, fine-tunes it for AD prediction, scores the held-out test cohort and fits a
linear probe on the frozen encoder. It needs `data/omop_data` (data_verification) and
`data/cohort/splits.parquet` (cohort).

## Steps

Steps run in numeric order.

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required: datasets, transformers, torch, cehrbert, cehrbert_data; GPU and Java warnings |
| 01 | `01_generate_sequences.py` (+ `.sh`) | CEHR-BERT patient sequences from the OMOP tables (Spark) |
| 02 | `02_prepare_cohort_folders.py` | Finetune and test cohort folders from the splits; every cohort patient needs a sequence |
| 03 | `03_build_pretraining_input.py` | `patient_sequence` minus the test patients |
| 04 | `04_validate_pretrain_config.py` | Dry-run validation of the pretraining config |
| 05 | `05_pretrain.py` (+ `.sh`) | Pretrain CEHR-BERT (refuses test patients and stale artifacts) |
| 06 | `06_verify_pretraining.py` | Model and prepared dataset; no test patient was pretrained on |
| 07 | `07_tokenize_full_dataset.py` | Every patient tokenized with the pretrained tokenizer |
| 08 | `08_validate_finetune_config.py` | Dry-run validation of the fine-tuning config and its dependencies |
| 09 | `09_finetune.py` (+ `.sh`) | Fine-tune for AD prediction (train and validation) |
| 10 | `10_predict.py` (+ `.sh`) | Score the held-out test cohort and check ROC-AUC and PR-AUC |
| 11 | `11_validation_predictions.py` | Fine-tuned model on the validation split (threshold selection) |
| 12 | `12_token_stats.py` | Token counts per patient sequence, full history and windowed |
| 13 | `13_linear_probe_features.py` | Frozen CEHR-BERT features per split |
| 14 | `14_train_linear_probe.py` | Logistic regression on the frozen features |

The heavy steps (01 Spark, 05, 09, 10 runners) are a Python entrypoint that owns preflight checks
and reporting plus a thin bash wrapper that sets environment variables and calls the cehrbert or
cehrbert_data tool. Run the Python entrypoint; it calls the wrapper.

```bash
bash cehrbert_model/run.sh                                  # all steps
bash cehrbert_model/run.sh --from 05                        # resume
bash cehrbert_model/run.sh --ctx 4096 --observation-window -1

python cehrbert_model/01_generate_sequences.py              # SPARK_DRIVER_MEMORY or --spark-driver-memory
python cehrbert_model/09_finetune.py --ctx 4096 --observation-window -1
python cehrbert_model/12_token_stats.py --window-days 365 --ctx 512
```

## Options of `run.sh`

- `--from <step id>` resume from a step (`--from 7` means `07`).
- `--ctx <int>` token context: passed to steps 05, 09, 10 and to steps 11 to 14 (part of the run label).
- `--observation-window <int>` look-back in days (negative means all history): passed to steps 09
  and 10 as `--observation-window` and to steps 11 to 14 as `--window-days`. Step 05 (pretraining)
  has no observation window.

## Configuration

- `configs/ad_pretrain_config.yaml` pretraining hyperparameters.
- `configs/ad_finetune_config.yaml` fine-tuning hyperparameters.

Steps 04 and 08 dry-run-validate these against the cehrbert runner dataclasses (when installed)
plus value-range and referenced-path checks, so a bad config fails before a long training run
starts. Step 10 derives its own predict config from the fine-tuning config and writes it to
`scripts/cehrbert_model/reports/10_predict/predict_config.yaml`.

## Train, validation and test

The held-out test set is decided once, at the patient level, in the cohort part. This part never
re-splits:

- Step 02 writes the folders the finetune runner reads from the splits: `cohort/finetune` (train
  and val) and `cohort/test` under `data/cehrbert_data`. It fails, listing up to 20 person_ids,
  when a cohort patient has no non-empty sequence in `patient_sequence`.
- Step 03 removes every test patient from the pretraining input, so the pretrained encoder never
  sees them. Step 05 refuses an input that still holds a test patient, and step 06 checks the
  prepared dataset again.
- Step 07 tokenizes every patient into `data/full_tokenized`, which fine-tuning, prediction and the
  linear probe read (`tokenized_full_dataset_path`). Its `validation` split is the canonical val
  split.
- Step 09 fine-tunes on `data/cehrbert_data/cohort/finetune/` with `do_predict: false`. Train and
  validation are the canonical splits of `data/full_tokenized`; `validation_split_percentage` does
  not split the data in this flow.
- Step 10 runs prediction on `data/cehrbert_data/cohort/test/` with the fine-tuned model
  (`do_train: false, do_predict: true`) and verifies the metrics. It can be rerun on its own.
- Step 11 scores the validation split of `data/finetune_prepared/<hash>`, so thresholds are chosen
  without touching the test set.

This relies on one small edit to the vendored runner
(`cehrbert/src/cehrbert/runners/hf_cehrbert_finetune_runner.py`): in predict-only mode (`do_predict`
without `do_train`), the extracted cohort is treated as the `test` split. Normal training runs are
unaffected.

## Inputs and outputs

Reads `data/omop_data` and `data/cohort/splits.parquet`.

```
data/
├── cehrbert_data/omop_input/           symlinks to the OMOP tables; cehrbert_data's input folder, where it
│                                        writes processed_measurement and qualified_concept_list (step 01)
├── cehrbert_data/patient_sequence/     patient sequences, all patients                       (step 01)
├── cehrbert_data/cohort/finetune/cohort.parquet   train + val rows                           (step 02)
├── cehrbert_data/cohort/test/cohort.parquet       held-out test rows                         (step 02)
├── cehrbert_data/patient_sequence_pretrain/  same, without test patients                     (step 03)
├── pretrain_prepared/<name>_<hash>/    pretraining DatasetDict, no test patients             (step 05)
├── pretrain_results/                   pretrained model + tokenizer                          (step 05)
├── full_tokenized/                     all patients tokenized; validation = canonical val    (step 07)
├── finetune_prepared/                  fine-tuning dataset (train/validation)                (step 09)
├── predict_prepared/                   test-cohort dataset                                   (step 10)
└── finetune_results/
    ├── (fine-tuned model + tokenizer)                                                        (step 09)
    ├── test_predictions/*.parquet      subject_id, prediction_time,
    │                                    predicted_boolean_probability, boolean_value         (step 10)
    ├── test_results.json               roc_auc, pr_auc, test_loss                            (step 10)
    ├── validation_predictions/*.parquet                                                      (step 11)
    └── validation_results.json         roc_auc, pr_auc, val_loss                             (step 11)
build/cehrbert_model/<run_label>/       run label: w_<window>_ctx<ctx>
├── token_stats.json, figures/token_length_hist.png                                           (step 12)
├── linear_probe/                       features/<split>/ (step 13), <kind>/ predictions (step 14)
└── predictions_manifest.json           cehrbert_finetuned (step 11), linear_probe_cls and
                                         linear_probe_mean (step 14); read by evaluation
```

The part only produces predictions. Step 11 adds `cehrbert_finetuned` (the step 10 test predictions
and the step 11 validation predictions in `data/finetune_results/`) to the manifest, and skips that
entry with a warning when the test predictions are missing. Step 14 adds both linear probe
kinds; the kind with the lower validation AUROC is marked `secondary`. Scoring and comparison are in the
[evaluation](../evaluation/README.md) part.

cehrbert_data caches its measurement events in `omop_input/processed_measurement` and reuses them
on the next run, so delete that folder when the OMOP data changes. Step reports go to
`scripts/cehrbert_model/reports/<step>/` for steps 00 to 10 and to
`build/cehrbert_model/<run_label>/reports/<step>/` for steps 11 to 14.
