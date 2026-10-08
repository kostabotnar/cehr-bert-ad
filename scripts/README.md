# Alzheimer's Disease Prediction Pipeline with CEHR-BERT

Predicts Alzheimer's Disease (AD) from OMOP data with CEHR-BERT and with classical baselines. The
work is split into independent **parts**, each in its own folder with its own numbered steps, a
`run.sh` runner and a README. A step validates its inputs, does its work and writes a report
(`status.json` + `report.md`). Validation steps gate their part: a failure exits non-zero and the
runner halts.

| Part | What it does |
|------|--------------|
| [data_verification](data_verification/README.md) | Checks the OMOP tables and fills null datetime columns |
| [cohort](cohort/README.md) | Builds the labeled AD cohort and the canonical train/val/test split |
| [data_qc](data_qc/README.md) | Profiles the raw OMOP data and the patient history length |
| [cehrbert_model](cehrbert_model/README.md) | Sequences, pretraining, fine-tuning, prediction, linear probe |
| [classical_ml](classical_ml/README.md) | Tabular features, L2 logistic regression and XGBoost baselines |
| [evaluation](evaluation/README.md) | Metrics with confidence intervals and the model comparison |
| [feature_importance](feature_importance/README.md) | Baseline importance, attention analysis, top feature tables |

## Layout

```
scripts/
├── run_all.sh              runs the parts in the order below
├── lib/run_steps.sh        runner logic sourced by every part's run.sh
├── pipeline/               shared, tested library (reporting, validation, paths, env_checks, ...)
├── tests/                  pytest suite (tests/<part>/ for steps, tests/pipeline/ for the library)
└── <part>/                 00_check_environment.py, numbered steps, run.sh, README.md, reports/
```

Step reports go to `scripts/<part>/reports/<step>/`, and each runner writes
`scripts/<part>/reports/summary.md`. Reports of the steps that depend on the run label (token
context and look-back window) go to `build/<run_label>/reports/<step>/`. All `reports/` folders are
git-ignored.

## Dependencies between parts

```
data_verification -> cohort -> data_qc
                            -> cehrbert_model ----+-> evaluation
                            -> classical_ml ------+-> feature_importance
```

`run_all.sh` runs data_verification, cohort, data_qc, cehrbert_model, classical_ml, evaluation,
feature_importance, in that order. Every part only needs the files named under Contracts, so a part
can be rerun on its own once its inputs exist.

## Contracts

The files that one part writes and another reads.

| File or folder | Written by | Read by |
|----------------|------------|---------|
| `data/omop_data/<table>/*.parquet` (symlink to your OMOP export, datetimes filled) | data_verification | cohort, data_qc, cehrbert_model, classical_ml |
| `data/cohort/splits.parquet` with `person_id, index_date, label, split` (train/val/test) | cohort | data_qc, cehrbert_model, classical_ml, evaluation |
| `data/cehrbert_data/` (sequences, cohort folders, `patient_sequence_pretrain`) | cehrbert_model | cehrbert_model |
| `data/pretrain_results/`, `data/full_tokenized/`, `data/finetune_results/` (fine-tuned model) | cehrbert_model | cehrbert_model, feature_importance |
| `data/finetune_results/{test,validation}_predictions/*.parquet` | cehrbert_model | evaluation |
| `build/<label>/linear_probe/<kind>/{test,validation}_predictions/` and `summary.json` | cehrbert_model | evaluation |
| `build/<label>/baselines/features/` | classical_ml | classical_ml |
| `build/<label>/baselines/{lr,xgboost}/{test,validation}_predictions/`, `model.joblib` | classical_ml | evaluation, feature_importance |
| `build/<label>/metrics.json` and `build/<label>/<model>/metrics.json` | evaluation | evaluation |
| `build/<label>/baselines/{lr,xgboost}/feature_importance.csv` | feature_importance | feature_importance |
| `build/<label>/attention/concept_attribution_*.csv` | feature_importance | feature_importance |
| `build/<label>/feature_tables/` | feature_importance | (final tables) |
| `build/omop_profile/`, `build/data_qc/` | data_qc | (QC output) |

Prediction folders use the CEHR-BERT schema: `subject_id, prediction_time,
predicted_boolean_probability, predicted_boolean_value, boolean_value`. The run label is
`w_<window>_ctx<ctx>` (default `w_365_ctx512`).

## Quick start

```bash
# Everything, pointing data_verification at your OMOP export:
bash scripts/run_all.sh --link /path/to/your/omop_data

# A subset (always run in the dependency order above):
bash scripts/run_all.sh --parts cohort,classical_ml,evaluation

# Experiment overrides:
bash scripts/run_all.sh --ctx 4096 --observation-window -1

# One part, or resume a part from a step:
bash scripts/cehrbert_model/run.sh --from 07
```

`run_all.sh` options:

- `--parts a,b,...` parts to run (default all).
- `--link <path>` OMOP data folder, passed to data_verification.
- `--ctx <int>` token context, passed to cehrbert_model, classical_ml, evaluation, feature_importance.
- `--observation-window <int>` look-back in days (negative means all history). Passed as
  `--observation-window` to cehrbert_model and as `--window-days` to classical_ml, evaluation and
  feature_importance.

It halts on the first failing part and writes `scripts/reports/summary.md` (git-ignored). The heavy
steps (Spark sequence generation, pretraining, fine-tuning) need the full runtime environment; each
part starts with a `00_check_environment.py` that lists what its steps import.

Exit codes of every step: `0` for pass or warn (the run may continue), `1` for fail (gating).

## Tests

```bash
python -m pytest scripts/tests -q
```

The suite uses tiny synthetic OMOP parquet files, fake model directories and fake `DatasetDict`
objects, and mocks the training launchers, so no real training runs. Tests that need the `datasets`
library skip themselves when it is not installed. Step tests load a script with
`load_step("<part>/<script>.py")` (a path relative to `scripts/`).
