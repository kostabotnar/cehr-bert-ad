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
python feature_importance/02_attention_analysis.py --batch-size 64
python feature_importance/02_attention_analysis.py --max-patients 300 --top-k 20 --device cpu --batch-size 8
python feature_importance/03_top_feature_tables.py --top 10
```

Options of `run.sh`: `--ctx` and `--window-days` select the run label (`w_<window>_ctx<ctx>`) and
go to all three steps; `--from <step id>`.

## Step 01

Loads the fitted estimators (`model.joblib`, classical_ml step 02) and the column-aligned
`feature_names.json` (classical_ml step 01). LR is ranked by signed coefficient (features are
log1p-scaled and MaxAbs-scaled, so magnitudes are roughly comparable), XGBoost by
`feature_importances_`. The `label` column comes from the OMOP CONCEPT table in
`data/omop_data/concept/` (`--concept-table` selects another one): `cond: Amnesia`, `race: White`,
`gender: MALE`; id 0 reads `race: unknown` or `gender: unknown`; `age` and `history_length_days`
get plain-language labels. The lookup is shared by steps 01 to 03 in `pipeline/concept_labels.py`.

Models and feature names are read from `build/classical_ml/<window_label>/`. Outputs in
`build/feature_importance/<run_label>/`: `lr/feature_importance.csv`, `xgboost/feature_importance.csv`.

## Step 02

Runs the fine-tuned classifier (on the GPU when CUDA is available, otherwise the CPU; override with
`--device`) over the held-out test cohort with attention output (eager
attention, no sample packing) and attributes each prediction to concept tokens with three methods:
raw last-layer [CLS] attention, attention rollout and gradient-weighted attention rollout.
Patients are processed `--batch-size` at a time (default 32; `--batch-size 1` processes one at a
time): one padded forward and one backward per batch, then each patient's attention and gradient
are cut back to its real length, so scores match the one-at-a-time run up to floating-point
rounding. Patients are sorted by length within chunks of 16 batches to reduce padding, and the
rollouts run on the same device as the model. Lower `--batch-size` if the GPU runs out of memory.
The report records `batch_size`, and progress lines show patients per minute. Outputs
in `build/feature_importance/<run_label>/attention/`: `concept_attribution_<group>.csv`,
`method_rank_correlation.json`. `concept_id` keeps the raw token; `concept_name` is the OMOP concept name
for numeric ids and a decoded label for structural tokens: `D<n>` and `i-D<n>` (days between visits, or
between events inside an inpatient visit), `LT` and `i-LT` (long-term gap, 1080 days or more), `year:<YYYY>`
and `age:<n>` (year and age at the first visit), `Gender/<x>`, `Race/<x>`, `Visit/<x>`, `Discharge/<x>` (`0`
means unknown). `--concept-table` selects another CONCEPT table. The figure, labelled by concept name only, is
`build/feature_importance/<run_label>/figures/attention_top_concepts.png`.
The fine-tuned model comes from `data/finetune_results/` and the test data from `data/predict_prepared/`.

## Step 03

Combines the rankings above with source codes and descriptions. `source_code` is
`<vocabulary_id>:<concept_code>` from the CONCEPT table (e.g. `SNOMED:48167000`); structural tokens and
derived features keep their raw name. `description` is the CONCEPT name or decoded token label, unless a
curated entry in the script applies (RxNorm drug names, a few code references, age, history length, gender).
`--concept-table` selects another CONCEPT table. It reads the rankings from `build/feature_importance/<run_label>/`. Outputs in
`build/feature_importance/<run_label>/feature_tables/`: `lr_top10.csv`,
`xgboost_top10.csv`, `attention_top10.csv`, `top_features.md`.
