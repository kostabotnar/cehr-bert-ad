# CEHR-BERT Alzheimer's Disease Prediction

Predicts Alzheimer's Disease (AD) from Observational Medical Outcomes Partnership (OMOP) data with
CEHR-BERT and with classical baselines (logistic regression, XGBoost). The pipeline lives in
[`scripts/`](scripts/README.md). [`cehrbert/`](cehrbert/README.md) is a vendored copy of the upstream
[CEHR-BERT](https://github.com/cumc-dbmi/cehr-bert) package, installed in editable mode.

## Requirements

- Linux with an NVIDIA GPU and driver for pretraining and fine-tuning. The tests and the classical
  baselines run without a GPU.
- [uv](https://docs.astral.sh/uv/getting-started/installation/). It creates the Python environment (3.12
  by default), so no system Python, pip or conda is needed.
- Java 8 for sequence generation (`cehrbert_model` step 01), which runs on PySpark 3.1.2. The install
  script can install it into the project.
- Your OMOP export as one folder per table with parquet files. Patient data is not in this repository.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and run
the install script:

```bash
git clone https://github.com/kostabotnar/cehr-bert-ad.git
cd cehr-bert-ad
bash scripts/install.sh
```

The script creates `.venv` at the project root, installs PyTorch, the vendored CEHR-BERT package and
the plotting extras with the versions pinned in `scripts/install_constraints.txt`, and runs the
environment check of every pipeline part. It is safe to rerun: finished steps print
`skipped (already present)`. Its log is `build/install/install.log`.

Options:

- `--python 3.12`: Python version for `.venv`.
- `--cuda cu129|cu128|cu126|cpu`: PyTorch build. By default it is detected from `nvidia-smi`, and falls
  back to `cpu` without a GPU. On aarch64 only `cu129` and `cpu` exist.
- `--java ask|install|skip`: what to do when Java 8 is missing (default `ask`).
- `--dry-run`: print the commands that would change something instead of running them.

If Java 8 (or 11) is not found, the script asks whether to install Eclipse Temurin JDK 8 into
`.tools/jdk8` (checksum verified, added to `.venv/bin/activate`) or to skip, in which case you install
it yourself. Without a terminal, `ask` means skip.

Activate the environment before running the pipeline:

```bash
source .venv/bin/activate
```

### Manual installation

The same steps by hand, from the project root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --constraint scripts/install_constraints.txt torch \
    --index-url https://download.pytorch.org/whl/cu129
```

Pick the PyTorch index that matches your driver (`cu129`, `cu128`, `cu126` or `cpu`; aarch64 has CUDA
wheels for `cu129` only). The vendored package takes its version from git tags through `setuptools_scm`,
and `cehrbert/` is not a git repository of its own, so the version has to be set explicitly:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+local uv pip install --python .venv/bin/python \
    --constraint scripts/install_constraints.txt -e "cehrbert[dev]"
uv pip install --python .venv/bin/python --constraint scripts/install_constraints.txt matplotlib statsmodels
```

Java 8 must be on `PATH`, or at `.tools/jdk8/bin/java`.

## Verify the installation

Each pipeline part has an environment check that lists the packages its steps import, and checks the
GPU and Java where the part needs them:

```bash
python scripts/cehrbert_model/00_check_environment.py
python -m pytest scripts/tests -q
```

## Usage

Point the pipeline at your OMOP export and run all parts:

```bash
bash scripts/run_all.sh --link /path/to/your/omop_data
```

See [`scripts/README.md`](scripts/README.md) for the parts, their outputs, the options and how to rerun
a single part.
