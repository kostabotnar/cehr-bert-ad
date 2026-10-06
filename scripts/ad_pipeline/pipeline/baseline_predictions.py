"""Write baseline model predictions in CEHR-BERT's prediction-parquet schema.

The evaluation code (``pipeline.evaluation`` / ``10_evaluate.py``) loads a folder of
``*.parquet`` and reads two columns: ``predicted_boolean_probability`` and
``boolean_value``. We emit the full schema the finetune runner produces so the baseline
folders are drop-in replacements for ``data/finetune_results/{test,validation}_predictions``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def write_predictions(
    out_dir: str | Path,
    person_ids: np.ndarray,
    index_dates: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Path:
    """Write a single ``predictions.parquet`` into ``out_dir`` (clearing it first)."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "subject_id": np.asarray(person_ids).astype("int64"),
            "prediction_time": pd.to_datetime(np.asarray(index_dates)),
            "predicted_boolean_probability": np.asarray(y_prob, dtype=float),
            "predicted_boolean_value": (np.asarray(y_prob, dtype=float) >= 0.5),
            "boolean_value": np.asarray(y_true).astype(bool),
        }
    )
    path = out / "predictions.parquet"
    df.to_parquet(path, index=False)
    return path
