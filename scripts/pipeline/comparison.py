"""Comparability checks and paired-difference tables for step 13.

``check_parity`` compares one model's prediction folder against a canonical split (the
person_ids and labels of the test or val split) and returns per-model counts. Any non-zero
discrepancy count, a row count that differs from the split, or an absent folder is a failure.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import pandas as pd

from pipeline import evaluation as ev


@dataclasses.dataclass
class ModelSpec:
    """One model to compare: its label and where its predictions and metrics live."""

    label: str
    test_dir: Path
    val_dir: Path
    metrics_path: Path
    evaluate_dir: Path | None = None  # folder holding test_/validation_predictions; None = do not re-evaluate
    secondary: bool = False


@dataclasses.dataclass
class ParityResult:
    """Counts describing how a prediction folder differs from a canonical split."""

    split: str
    present: bool
    n_expected: int
    n_rows: int = 0
    n_missing: int = 0
    n_extra: int = 0
    n_duplicate_rows: int = 0
    n_label_mismatch: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.present
            and not self.error
            and self.n_rows == self.n_expected
            and self.n_missing == 0
            and self.n_extra == 0
            and self.n_duplicate_rows == 0
            and self.n_label_mismatch == 0
        )

    def detail(self) -> str:
        if not self.present:
            return f"{self.split}: no predictions found"
        if self.error:
            return f"{self.split}: unreadable predictions ({self.error})"
        return (
            f"{self.split}: rows={self.n_rows} expected={self.n_expected} missing={self.n_missing} "
            f"extra={self.n_extra} duplicate_rows={self.n_duplicate_rows} label_mismatch={self.n_label_mismatch}"
        )

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "ok": self.ok}


def has_predictions(pred_dir: Path) -> bool:
    return pred_dir.is_file() or (pred_dir.is_dir() and any(pred_dir.glob("*.parquet")))


def check_parity(pred_dir: Path, expected: pd.DataFrame, split: str) -> ParityResult:
    """Compare ``pred_dir`` with ``expected`` (columns person_id, label) for the named split."""
    result = ParityResult(split=split, present=has_predictions(pred_dir), n_expected=len(expected))
    if not result.present:
        return result
    try:
        df = pd.read_parquet(pred_dir, columns=[ev.SUBJECT_COLUMN, ev.LABEL_COLUMN])
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)[:200]
        return result

    result.n_rows = len(df)
    result.n_duplicate_rows = int(df[ev.SUBJECT_COLUMN].duplicated().sum())
    got_ids = set(df[ev.SUBJECT_COLUMN].astype("int64"))
    want = dict(zip(expected["person_id"].astype("int64"), expected["label"].astype("int64")))
    result.n_missing = len(set(want) - got_ids)
    result.n_extra = len(got_ids - set(want))

    labels = df[ev.LABEL_COLUMN]
    mismatch = labels.isna().sum()
    known = df[ev.SUBJECT_COLUMN].astype("int64").isin(want.keys()) & labels.notna()
    sub = df[known]
    expected_labels = sub[ev.SUBJECT_COLUMN].astype("int64").map(want)
    mismatch += int((sub[ev.LABEL_COLUMN].astype("int64") != expected_labels).sum())
    result.n_label_mismatch = int(mismatch)
    return result


def paired_difference_rows(frames: dict[str, pd.DataFrame], n_resamples: int, seed: int) -> list[dict]:
    """Paired AUROC and AUPRC differences for every ordered-by-listing pair of models."""
    rows = []
    for (name_a, frame_a), (name_b, frame_b) in itertools.combinations(frames.items(), 2):
        res = ev.paired_bootstrap_differences(frame_a, frame_b, n_resamples=n_resamples, seed=seed)
        rows.append({"model_a": name_a, "model_b": name_b, "n": len(frame_a), **res})
    return rows
