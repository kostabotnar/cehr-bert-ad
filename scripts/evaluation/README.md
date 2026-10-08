# evaluation

Scores the models' predictions with one evaluation code path and compares them. The model parts
(cehrbert_model, classical_ml) only write predictions and a `predictions_manifest.json`; all scoring and
comparison happens here, so run those parts first.

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required: polars, pandas, numpy, scipy, sklearn; optional: matplotlib, statsmodels |
| 01 | `01_evaluate.py` | Metrics with bootstrap confidence intervals for every model in the manifests |
| 02 | `02_compare_models.py` | Parity check and comparison table for the same models |

```bash
bash evaluation/run.sh
bash evaluation/run.sh --ctx 4096 --window-days -1 --name my_comparison
python evaluation/01_evaluate.py --manifest build/classical_ml/w_365/predictions_manifest.json --name classical
python evaluation/01_evaluate.py --predictions data/finetune_results/test_predictions   # one folder, no manifest
```

Options of `run.sh`: `--ctx` and `--window-days` select the default manifests; `--name` sets the
comparison name; `--from <step id>`. Both steps accept the same selection options:

- `--manifest PATH` (repeatable) names the manifests to use. Without it the defaults are
  `build/cehrbert_model/<run_label>/predictions_manifest.json` (`w_<window>_ctx<ctx>`) and
  `build/classical_ml/<window_label>/predictions_manifest.json` (`w_<window>`). A missing default
  manifest is a warning and that family is skipped; no model at all fails step 01.
- `--name` is the comparison name and the folder under `build/evaluation/`. The default is the
  cehrbert run label.

## Step 01

Scores every model of the manifests into `build/evaluation/<name>/models/<model name>/`: `metrics.json`,
`figures/` (ROC, precision-recall, calibration, confusion matrix) and `reports/01_evaluate/`. The
operating threshold is chosen on the model's validation predictions, never on the test set. The
bootstrap size (2000), seed (42) and calibration bins (10) are the same for every model; they are
constants in `pipeline/evaluation.py` and can be changed with `--n-bootstrap`, `--seed`, `--cal-bins`. One
failing model fails the step, the other models still run. The overall result is in
`build/evaluation/<name>/reports/01_evaluate/`.

Single-model mode: with `--predictions` (and `--val-predictions`) one prediction folder is scored; the
output goes to `models/<--model-name>/` or to `--results-dir`, `--figures-dir`, `--report-dir`.

## Step 02

Compares the models of the same manifests. It starts no scoring: each model's `metrics.json` comes from
step 01 (run it first, with the same `--name`); a missing one fails the check for that model. Before
comparing, it checks that every model's test predictions match the canonical test split and its
validation predictions the canonical val split (person ids, labels, row count); any difference fails the
step.

Inputs: `data/cohort/splits.parquet`, the manifests, `build/evaluation/<name>/models/*/metrics.json`.

Outputs in `build/evaluation/<name>/`: `comparison.json`, `comparison.md` (metrics with 95% CIs and
paired AUROC and AUPRC differences), `figures/compare_roc.png`, `figures/compare_pr.png`,
`reports/02_compare_models/`.

## Prediction manifest

`predictions_manifest.json`, written by the model parts, is

```json
{"models": [{"name": "lr", "display_label": "L2 Logistic Regression", "family": "classical_ml",
             "test_predictions": "build/classical_ml/w_365/lr/test_predictions",
             "validation_predictions": "build/classical_ml/w_365/lr/validation_predictions",
             "secondary": false}]}
```

Paths are relative to the project root when the folder is inside the project, otherwise absolute.
`name` is unique and names the model folder; `display_label` is the row label in the comparison table;
`secondary: true` marks a result not chosen by validation AUROC (the weaker linear probe kind). Each
prediction folder holds parquet files with `subject_id, prediction_time, predicted_boolean_probability,
boolean_value`. See `pipeline/manifest.py`.
