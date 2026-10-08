# data_verification

Checks that the OMOP export is usable and fills null datetime columns. Run it first; every other
part reads `data/omop_data`.

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 00 | `00_check_environment.py` | Required modules: polars, pandas, pyarrow, yaml, numpy |
| 01 | `01_validate_dataset.py` | Verifies the required OMOP tables and columns exist |
| 02 | `02_fill_datetimes.py` | Fills null `*_datetime` columns from the matching `*_date` columns |

```bash
bash data_verification/run.sh --link /path/to/your/omop_data   # all steps
bash data_verification/run.sh --from 02                         # resume
python data_verification/01_validate_dataset.py --data-dir /path/to/omop_data   # validate elsewhere
python data_verification/02_fill_datetimes.py --dry-run         # count only, write nothing
```

Options of `run.sh`: `--link <path>` (passed to step 01), `--from <step id>`.

## Data requirements

The OMOP folder holds one subfolder per table, each with `*.parquet` files:

```
your_omop_data/
├── person/                  (required)
├── visit_occurrence/        (required)
├── condition_occurrence/    (required)
├── concept/                 (required: AD codes and the concept list)
├── drug_exposure/           (recommended)
├── procedure_occurrence/    (recommended)
└── measurement/             (recommended)
```

Step 01 with `--link` symlinks the folder to `data/omop_data` (a link that already points there is
validated in place). Nothing from the CEHR-BERT steps is ever written into the OMOP folder.

Step 02 rewrites the files in place, one at a time, through a temporary file and an atomic rename.
Null datetimes become midnight of the matching date (`person.birth_datetime` is built from year,
month and day of birth). It is idempotent and refuses to run on a path under `~/data/derived`.
cehrbert_data orders visits by `visit_start_datetime`, so cehrbert_model step 01 needs this.

## Outputs

- `data/omop_data` (symlink, step 01 with `--link`) and the datetime-filled tables.
- Reports in `scripts/data_verification/reports/<step>/` and `reports/summary.md`.
