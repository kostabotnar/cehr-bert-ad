# CEHR-BERT Alzheimer's Disease Prediction

Predicts Alzheimer's Disease (AD) from Observational Medical Outcomes Partnership (OMOP) data with
CEHR-BERT and with classical baselines (logistic regression, XGBoost). The pipeline lives in
[`scripts/`](scripts/README.md). [`cehrbert/`](cehrbert/README.md) is a vendored copy of the upstream
[CEHR-BERT](https://github.com/cumc-dbmi/cehr-bert) package, installed in editable mode.

## Requirements

- Linux with an NVIDIA GPU and driver for pretraining and fine-tuning. The tests and the classical
  baselines run without a GPU.
- Python 3.10 or newer (the project is run with 3.12).
- Java 8 on `PATH`. Sequence generation (`cehrbert_model` step 01) runs on PySpark 3.1.2, which needs it.
- Your OMOP export as one folder per table with parquet files. Patient data is not in this repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/kostabotnar/cehr-bert-ad.git
cd cehr-bert-ad
```

Create a virtual environment in `.venv` at the project root and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install PyTorch for your CUDA version first, so the CPU-only wheel is not pulled in as a dependency.
The project runs with `torch==2.8.0` built for CUDA 12.9; pick the index URL that matches your driver
from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu129
```

Install the vendored CEHR-BERT package with its development extras (pytest and friends). It takes its
version from git tags through `setuptools_scm`, and `cehrbert/` is not a git repository of its own, so
the version has to be set explicitly:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+local pip install -e "cehrbert[dev]"
```

This also installs `cehrbert_data`, PySpark, XGBoost, scikit-learn, polars and the other pipeline
dependencies. Install the packages the evaluation and feature importance plots use:

```bash
pip install matplotlib statsmodels
```

With [uv](https://docs.astral.sh/uv/), replace `python3 -m venv .venv` with `uv venv .venv` and
`pip install` with `uv pip install`.

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
