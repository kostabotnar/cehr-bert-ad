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
bash classical_ml/run.sh --window-days -1
python classical_ml/01_prepare_features.py --window-days 365
python classical_ml/02_train_baselines.py --cv-folds 5 --xgb-iter 30 --seed 42
```

Options of `run.sh`: `--window-days` selects the window label (`w_<window>`, default `w_365`; `w_all` for
all history) and goes to both steps; `--from <step id>`. The token context does not affect these
models, so it is not part of the label.

## Features (step 01)

Per-concept features within the look-back window (conditions as presence/absence, drugs,
procedures and measurements as occurrence counts) plus demographics (age at index, one-hot gender and race). Only visit-linked events count, the same
rows the cohort rule and CEHR-BERT use (`ad_cohort._event_rows`); `history_length_days` (index date minus the earliest earlier visit, over all history regardless of the look-back window, imputed with the train median when a patient has no earlier visit) is also a feature; there is no visit count. The
vocabulary is fit on train only and applied unchanged to val and test. A concept is kept when it is in
the qualified set that CEHR-BERT and the cohort also use (standard concepts used by at least
`MIN_CONCEPT_PATIENTS` persons, from `pipeline/omop_common.py`, pooled over condition, drug, procedure
and measurement events) and occurs in at least one training patient's window. The report records the threshold, the qualified concept count and the kept size per domain. The step fails when `data/cohort/splits.parquet` is missing.

## Models (step 02)

- L2 logistic regression: log1p(counts), MaxAbsScaler, tuned over C and class_weight with a
  cross-validated grid search.
- XGBoost: `hist` tree method with a fixed `scale_pos_weight`, randomized search, then a refit on
  train with early stopping on the validation split.

The AUROC and AUPRC printed here are sanity numbers; the evaluation part computes the headline
metrics.

## Outputs (under `build/classical_ml/<window_label>/`)

```
features/{train,val,test}_X.npz, {train,val,test}_meta.npz, feature_spec.json, feature_names.json
predictions_manifest.json     models lr and xgboost, read by evaluation
lr/{test,validation}_predictions/predictions.parquet, lr/model.joblib, lr/best_params.json
xgboost/{test,validation}_predictions/predictions.parquet, xgboost/model.joblib, xgboost/best_params.json
```

Prediction parquets use the CEHR-BERT schema and are listed in the manifest (display labels
"L2 Logistic Regression" and "XGBoost"); the evaluation part scores them.
Reports go to `build/classical_ml/<window_label>/reports/<step>/`; `scripts/classical_ml/reports/` holds the
environment check and the runner summary.
