# evaluation

Scores prediction folders with one evaluation code path and compares the models. It reads
predictions written by cehrbert_model and classical_ml, so run those parts first.

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required: polars, pandas, numpy, scipy, sklearn; optional: matplotlib, statsmodels |
| 01 | `01_evaluate.py` | Metrics with bootstrap confidence intervals for the fine-tuned CEHR-BERT |
| 02 | `02_compare_models.py` | Parity check and comparison table for all models |

```bash
bash evaluation/run.sh
bash evaluation/run.sh --ctx 4096 --window-days -1
python evaluation/01_evaluate.py --predictions data/finetune_results/test_predictions
python evaluation/02_compare_models.py --skip-eval      # reuse existing metrics.json
```

Options of `run.sh`: `--ctx` and `--window-days` select the run label (`w_<window>_ctx<ctx>`) and
go to both steps; `--from <step id>`.

## Step 01

Reads `data/finetune_results/test_predictions` and, when present, `validation_predictions`. The
operating threshold is chosen on the validation predictions, never on the test set. Options include
`--n-bootstrap`, `--cal-bins`, `--seed`, `--predictions`, `--val-predictions`, `--results-dir`,
`--figures-dir`. Outputs in `build/<run_label>/`: `metrics.json`, `figures/` (ROC, precision-recall,
calibration, confusion matrix), `reports/01_evaluate/`.

## Step 02

Compares the models, each evaluated with the same `01_evaluate.py` arguments: the fine-tuned
CEHR-BERT (when its metrics exist), the linear probe (primary feature kind from
`linear_probe/summary.json`, the other kind as secondary), and the L2 logistic regression and
XGBoost baselines. Before scoring it checks that every model's test predictions match the canonical
test split and its validation predictions the canonical val split (person ids, labels, row count);
any difference fails the step.

Inputs: `data/cohort/splits.parquet`, `build/<run_label>/linear_probe/`,
`build/<run_label>/baselines/<model>/{test,validation}_predictions`, and optionally
`build/<run_label>/metrics.json` with `data/finetune_results/`.

Outputs: `build/<run_label>/comparison.json`, `comparison.md` (metrics with 95% CIs and paired
AUROC and AUPRC differences), `figures/compare_roc.png`, `figures/compare_pr.png`.
