# feature_importance

Which features drive the predictions: ranked importances for the baselines, attention-based
attribution for CEHR-BERT, and annotated top-10 tables. Needs the trained baselines (classical_ml)
and the fine-tuned model (cehrbert_model).

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required: numpy, pandas, scipy, sklearn, xgboost, joblib, torch, transformers, datasets, cehrbert |
| 01 | `01_baseline_importance.py` | Ranked importance tables for the LR and XGBoost baselines |
| 02 | `02_attention_analysis.py` | Attention attribution of the fine-tuned CEHR-BERT on the test cohort |
| 03 | `03_top_feature_tables.py` | Annotated top-10 tables for the LR, XGBoost and attention models |

```bash
bash feature_importance/run.sh
bash feature_importance/run.sh --ctx 4096 --window-days -1
python feature_importance/02_attention_analysis.py --max-patients 300 --top-k 20 --device cpu
python feature_importance/03_top_feature_tables.py --top 10
```

Options of `run.sh`: `--ctx` and `--window-days` select the run label (`w_<window>_ctx<ctx>`) and
go to all three steps; `--from <step id>`.

## Step 01

Loads the fitted estimators (`model.joblib`, classical_ml step 02) and the column-aligned
`feature_names.json` (classical_ml step 01). LR is ranked by signed coefficient (features are
log1p-scaled and MaxAbs-scaled, so magnitudes are roughly comparable), XGBoost by
`feature_importances_`. OMOP codes are mapped to labels through the CONCEPT table in
`data/omop_data/concept/`; in a de-identified dataset the name is the source vocabulary code, so
pass an Athena CONCEPT table with `--concept-table` for real descriptions.

Models and feature names are read from `build/classical_ml/<window_label>/`. Outputs in
`build/feature_importance/<run_label>/`: `lr/feature_importance.csv`, `xgboost/feature_importance.csv`.

## Step 02

Runs the fine-tuned classifier (on the GPU when CUDA is available, otherwise the CPU; override with
`--device`) over the held-out test cohort with attention output (eager
attention, no sample packing) and attributes each prediction to concept tokens with three methods:
raw last-layer [CLS] attention, attention rollout and gradient-weighted attention rollout. Outputs
in `build/feature_importance/<run_label>/attention/`: `concept_attribution_<group>.csv`,
`method_rank_correlation.json`; the figure is
`build/feature_importance/<run_label>/figures/attention_top_concepts.png`.
The fine-tuned model comes from `data/finetune_results/` and the test data from `data/predict_prepared/`.

## Step 03

Combines the rankings above with source codes and descriptions (curated in the script, from RxNorm
and standard code references). It reads the rankings from `build/feature_importance/<run_label>/`. Outputs in
`build/feature_importance/<run_label>/feature_tables/`: `lr_top10.csv`,
`xgboost_top10.csv`, `attention_top10.csv`, `top_features.md`.
