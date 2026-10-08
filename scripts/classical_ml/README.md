# classical_ml

Classical baselines for the same AD task: L2 logistic regression and XGBoost on tabular features
built from the OMOP tables. They use the canonical split from the cohort part, so the comparison
with CEHR-BERT uses the same patients. Nothing here needs CEHR-BERT output.

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required: polars, pandas, numpy, scipy, sklearn, xgboost, joblib |
| 01 | `01_prepare_features.py` | Feature matrices frozen before each patient's index date |
| 02 | `02_train_baselines.py` | Tune and train L2 logistic regression and XGBoost |

```bash
bash classical_ml/run.sh
bash classical_ml/run.sh --ctx 4096 --window-days -1
python classical_ml/01_prepare_features.py --window-days 365 --min-prevalence 0.01
python classical_ml/02_train_baselines.py --cv-folds 5 --xgb-iter 30 --seed 42
```

Options of `run.sh`: `--ctx` and `--window-days` select the run label (`w_<window>_ctx<ctx>`, default
`w_365_ctx512`) and go to both steps; `--from <step id>`.

## Features (step 01)

Per-concept occurrence counts (condition, drug, procedure) plus demographics (age at index, one-hot
gender and race) within the look-back window. The vocabulary is fit on train only: concepts present
in at least `--min-prevalence` of the training patients are kept and applied unchanged to val and
test. The step fails when `data/cohort/splits.parquet` is missing.

## Models (step 02)

- L2 logistic regression: log1p(counts), MaxAbsScaler, tuned over C and class_weight with a
  cross-validated grid search.
- XGBoost: `hist` tree method with a fixed `scale_pos_weight`, randomized search, then a refit on
  train with early stopping on the validation split.

The AUROC and AUPRC printed here are sanity numbers; the evaluation part computes the headline
metrics.

## Outputs (under `build/<run_label>/baselines/`)

```
features/{train,val,test}_X.npz, {train,val,test}_meta.npz, feature_spec.json, feature_names.json
lr/{test,validation}_predictions/predictions.parquet, lr/model.joblib, lr/best_params.json
xgboost/{test,validation}_predictions/predictions.parquet, xgboost/model.joblib, xgboost/best_params.json
```

Prediction parquets use the CEHR-BERT schema, so they are inputs to `evaluation/01_evaluate.py`.
Reports go to `build/<run_label>/reports/<step>/`; `scripts/classical_ml/reports/` holds the
environment check and the runner summary.
