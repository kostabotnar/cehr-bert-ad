# data_qc

Descriptive statistics of the raw OMOP data and of the cohort. Nothing here changes a cohort file
or feeds a later part; use it to choose cohort rules and to describe the data.

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required modules: polars, numpy, pyarrow, yaml, matplotlib |
| 01 | `01_profile_omop.py` | Raw OMOP profile of AD cases versus controls |
| 02 | `02_history_length_stats.py` | Patient history length (index date minus first visit) |

```bash
bash data_qc/run.sh
python data_qc/01_profile_omop.py --person-sample-fraction 0.02 --skip-sections measurement,duplicates
python data_qc/02_history_length_stats.py
```

## Step 01: OMOP profile

Profiles the raw OMOP tables for AD cases (index = first AD diagnosis) versus controls under two
index rules (`control_last_visit` and a seeded `control_matched_index` drawn from each control's own
visit dates). It reads only the OMOP tables. Concept groups live in
`configs/omop_profile_concepts.yaml` (`--concepts`). Useful options: `--person-sample-fraction 0.02`
for a fast run, `--skip-sections measurement,duplicates` to drop the heaviest parts, `--date-floor`,
`--seed`, `--out-dir`.

Outputs in `build/omop_profile/`: `profile.json`, `table1.csv` and `table1.md` (with standardized
mean differences versus AD), `exclusion_flow.csv`, `figures/`.

## Step 02: history length

`history_days = index_date - first_visit_date`, with the cohort from `data/cohort/splits.parquet`
and the first visit per person from `visit_occurrence`. Reported overall and by label (cases versus
controls). Options: `--data-dir`, `--splits-file`, `--results-dir`, `--figures-dir`.

Outputs in `build/data_qc/`: `history_length_stats.json`, `figures/history_length_hist.png`.

Reports go to `scripts/data_qc/reports/`.
