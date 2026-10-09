# cohort

Builds the labeled AD cohort and the canonical train/val/test split from the OMOP tables alone (no
CEHR-BERT output), so every other part uses the same patients.

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required modules: polars, numpy, pyarrow |
| 01 | `01_build_cohort.py` | Cohort, index dates, labels and the patient-level split |

```bash
bash cohort/run.sh
python cohort/01_build_cohort.py --gap-days 730
```

## Cohort definition (`pipeline/ad_cohort.py`)

AD cases are identified from source ICD codes in `condition_occurrence.condition_source_value`
(`ICD-9-CM:331.0`, `ICD-10-CM:G30*`). A case is anchored at its first AD condition date and a
control at one of its own visit dates with at least one year of observation after it
(`--control-followup-days`). The control anchor year is drawn so control anchor years follow the
case anchor years, limited to anchors up to the data cutoff (`cdm_source.source_release_date`, else
the last visit) minus the follow-up; cases after that date stay in the cohort. Index = anchor minus
`--gap-days`.

Both groups must satisfy, at the index: age at least `--min-age` (calendar year of the index minus `year_of_birth`, as in the classical baselines), index on or after `--index-floor`,
first visit at least `--window-days` earlier, and one condition, drug, procedure or measurement in
the window `[index - window, index)`. An event counts only when it would become a CEHR-BERT sequence
token: its `visit_occurrence_id` is not null and its standard concept id is not 0 and is used by at
least `MIN_CONCEPT_PATIENTS` (100, set in `pipeline/omop_common.py`, shared with cehrbert_model step 01) distinct persons, counted
over the visit-linked events of the four tables together. This mirrors
`generate_included_concept_list` (`--min_num_of_patients` from the same constant) and `extract_events_by_domain` of
cehrbert_data, so cohort patients do not end up without a sequence. The report lists exclusion
counts per group, the number of qualified concepts and the persons excluded by the event rule.

## Split

The held-out test set is decided once, at the patient level, with a fixed seed: about 10% of the
patients (`--test-holdout 0.1 --split-seed 42`) form `test`, the rest is split into `train` and
`val` (`--val-fraction`, `--val-seed`). No patient appears in two splits.

## Outputs

`data/cohort/splits.parquet` with the columns `person_id, index_date, label, split`
(`split` in train, val, test). Reports go to `scripts/cohort/reports/`.
