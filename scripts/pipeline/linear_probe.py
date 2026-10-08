"""Shared, testable code for the frozen CEHR-BERT linear probe (cehrbert_model steps 13 and 14).

cehrbert_model step 13 turns each patient's pre-index history into a fixed-size vector with the frozen
pretrained encoder; cehrbert_model step 14 fits an L2 logistic regression on those vectors. Two feature
kinds are produced from ``last_hidden_state``:

  * ``cls``  : the hidden state at position 0, where the collator prepends the [CLS] token
  * ``mean`` : the mean over positions with ``attention_mask == 1`` (CLS included)

``pooler_output`` is never used: the pretraining objective (masked tokens) does not train
the pooler. cehrbert's own ``linear_prob/compute_cehrbert_features.py`` is not reused because
its CLS path returns a (B, B, H) block for batch sizes above 1.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_KINDS = ("cls", "mean")
SPLIT_NAMES = ("train", "val", "test")

# Columns the collator emits that the encoder does not accept.
NON_MODEL_COLUMNS = ("person_id", "index_date", "age_at_index", "classifier_label")

C_GRID = tuple(float(c) for c in np.logspace(-4, 2, 7))
CLASS_WEIGHT_GRID = (None, "balanced")


# --------------------------------------------------------------------------- #
# cehrbert_model step 13: feature extraction
# --------------------------------------------------------------------------- #
def pool_features(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    cls_token_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cls, mean)`` features, each of shape (B, H), from a (B, L, H) hidden state."""
    if not bool((input_ids[:, 0] == cls_token_index).all()):
        raise AssertionError("position 0 is not the [CLS] token in every row of the batch")
    hidden = last_hidden_state.float()
    cls = hidden[:, 0, :]
    mask = (attention_mask == 1).to(hidden.dtype).unsqueeze(-1)
    mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return cls, mean


def load_cohort_args(config_path: str | Path, cohort_folder: str | Path, tokenized_path: str | Path, window_days: int):
    """Build the ``(CehrBertArguments, DataTrainingArguments)`` that ``extract_cohort_sequences`` reads.

    Preprocessing settings (workers, batch size, MEDS flag) come from the fine-tune config;
    the cohort folder, tokenized dataset and observation window are overridden.
    """
    import yaml
    from cehrbert.runners.hf_runner_argument_dataclass import CehrBertArguments, DataTrainingArguments
    from transformers import HfArgumentParser

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cfg["cohort_folder"] = str(cohort_folder)
    cfg["tokenized_full_dataset_path"] = str(tokenized_path)
    cfg["observation_window"] = int(window_days) if int(window_days) > 0 else 0
    parser = HfArgumentParser((CehrBertArguments, DataTrainingArguments))
    cehrbert_args, data_args = parser.parse_dict(cfg, allow_extra_keys=True)
    return cehrbert_args, data_args


def keep_demographic_prompt_from_config(config_path: str | Path) -> bool:
    """The ``keep_demographic_prompt`` flag of the fine-tune config (False when absent)."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return bool(cfg.get("keep_demographic_prompt", False))


def shift_index_dates_for_exclusive_end(cohort):
    """Return ``cohort`` with ``index_date`` as a datetime one second before midnight of the index date.

    cehrbert keeps events with ``start <= epoch_time <= index_date`` and ``epoch_times`` are
    midnight UTC, so a date-valued index_date would include events on the index date. The
    comparability contract (and classical_ml step 01) use ``[index_date - window, index_date)``. Ending the
    window one second early excludes index-day events and, with the window length unchanged,
    keeps events on index - window and drops index - window - 1 day. Applies to the
    unbounded window too (the end is shifted, the start stays at 0).
    """
    import polars as pl

    return cohort.with_columns(
        (pl.col("index_date").cast(pl.Datetime("us")) - pl.duration(seconds=1)).alias("index_date")
    )


def build_split_dataset(split_df, work_dir: str | Path, config_path: str | Path, tokenized_path, window_days: int):
    """Window and label one canonical split with cehrbert's ``extract_cohort_sequences``.

    The split is written to its own cohort folder (person_id, index_date, label). The
    function keeps the train/validation assignment of each patient in the tokenized dataset, so the
    returned splits are concatenated into one dataset, as the fine-tune runner does for
    predict-only runs.
    """
    from cehrbert.runners.data_utils import extract_cohort_sequences
    from datasets import concatenate_datasets

    cohort_folder = Path(work_dir)
    cohort_folder.mkdir(parents=True, exist_ok=True)
    shifted = shift_index_dates_for_exclusive_end(split_df.select(["person_id", "index_date", "label"]))
    shifted.write_parquet(cohort_folder / "cohort.parquet")
    cehrbert_args, data_args = load_cohort_args(config_path, cohort_folder, tokenized_path, window_days)
    processed = extract_cohort_sequences(data_args, cehrbert_args)
    return concatenate_datasets(list(processed.values()))


def load_frozen_model(model_dir: str | Path, device: str = "cuda"):
    """Load ``CehrBertForPreTraining`` with sdpa attention in eval mode on ``device``."""
    from cehrbert.models.hf_models.hf_cehrbert import CehrBertForPreTraining

    model = CehrBertForPreTraining.from_pretrained(str(model_dir), attn_implementation="sdpa")
    return model.to(device).eval()


@dataclasses.dataclass
class ExtractedFeatures:
    cls: np.ndarray
    mean: np.ndarray
    person_id: np.ndarray
    label: np.ndarray
    n_tokens_before_truncation: np.ndarray
    truncated: np.ndarray
    # In-window event tokens, not counting the demographic prompt. Zero means an empty window.
    n_event_tokens: np.ndarray
    # Demographic prompt tokens in the model input after truncation: 4, 2 (gender and race only) or 0.
    demographic_prompt_tokens: np.ndarray


def _batches(dataset, batch_size: int) -> Iterable[list[dict]]:
    """Yield lists of per-patient records, in dataset order."""
    for chunk in dataset.iter(batch_size=batch_size):
        keys = list(chunk.keys())
        n = len(chunk[keys[0]])
        yield [{k: chunk[k][i] for k in keys} for i in range(n)]


def cls_only_batch(n: int, cls_token_index: int) -> dict[str, torch.Tensor]:
    """Batch of ``n`` sequences holding only [CLS], with the dtypes and CLS values the collator uses."""
    return {
        "input_ids": torch.full((n, 1), cls_token_index, dtype=torch.long),
        "attention_mask": torch.ones((n, 1), dtype=torch.float32),
        "ages": torch.zeros((n, 1), dtype=torch.long),
        "dates": torch.zeros((n, 1), dtype=torch.long),
        "visit_concept_orders": torch.zeros((n, 1), dtype=torch.long),
        "concept_values": torch.full((n, 1), -1.0, dtype=torch.float32),
        "concept_value_masks": torch.zeros((n, 1), dtype=torch.float32),
        "visit_segments": torch.zeros((n, 1), dtype=torch.long),
    }


def _encode(model, batch, cls_token_index: int, device: str, amp_dtype) -> tuple[np.ndarray, np.ndarray]:
    batch = {k: v.to(device) for k, v in batch.items()}
    use_amp = amp_dtype is not None and torch.device(device).type == "cuda"
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    with ctx:
        hidden = model.bert(**batch).last_hidden_state
    cls, mean = pool_features(hidden, batch["attention_mask"], batch["input_ids"], cls_token_index)
    return cls.cpu().numpy().astype(np.float32), mean.cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_features(
    model,
    dataset,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 128,
    device: str = "cuda",
    amp_dtype: torch.dtype | None = torch.bfloat16,
    progress_every: int = 0,
    keep_demographic_prompt: bool = False,
) -> ExtractedFeatures:
    """Run the frozen encoder over ``dataset`` (rows in, rows out, same order).

    ``n_tokens_before_truncation`` is ``len(input_ids)`` of the record as it leaves
    ``extract_cohort_sequences`` (the 365-day window plus, with ``keep_demographic_prompt``, the
    four prompt tokens, before the collator adds [CLS] and keeps the most recent tokens). The
    collator keeps ``max_length - 1`` record tokens, prompt included, so a patient is
    ``truncated`` when ``n_tokens_before_truncation > max_length - 1``.

    ``n_event_tokens`` excludes the prompt tokens (4, or 2 when the first visit is after the index
    date), so an empty window (no in-window events) is ``n_event_tokens == 0`` even though such a
    patient is still encoded as [CLS] plus the prompt.
    Only a record with no tokens at all (no prompt either) is encoded as a [CLS]-only sequence.
    """
    from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator

    from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import (
        DEMOGRAPHIC_PROMPT_LENGTH,
        demographic_prompt_length_from_ages,
    )

    collator = CehrBertDataCollator(
        tokenizer, max_length=max_length, is_pretraining=False, keep_demographic_prompt=keep_demographic_prompt
    )
    model.eval()
    cls_parts, mean_parts = [], []
    person_ids, labels, n_tokens, n_events, prompt_counts = [], [], [], [], []
    hidden_size = model.config.hidden_size
    done = 0
    n_batches = 0
    for records in _batches(dataset, batch_size):
        person_ids.extend(int(r["person_id"]) for r in records)
        labels.extend(int(r["classifier_label"]) for r in records)
        n_tokens.extend(len(r["input_ids"]) for r in records)
        n_events.extend(
            len(r["input_ids"]) - (demographic_prompt_length_from_ages(r["ages"]) if keep_demographic_prompt else 0)
            for r in records
        )
        # Rows with no tokens at all never reach the collator: torch.tensor([]) is float32, which
        # makes the whole padded batch's input_ids float (and an all-empty batch has no CLS column).
        # They are encoded as a CLS-only sequence instead. Rows that kept only the demographic
        # prompt have tokens and go through the collator.
        nonempty = [i for i, r in enumerate(records) if len(r["input_ids"]) > 0]
        empty = [i for i, r in enumerate(records) if len(r["input_ids"]) == 0]
        cls = np.zeros((len(records), hidden_size), dtype=np.float32)
        prompt_tokens = np.zeros(len(records), dtype=np.int64)
        mean = np.zeros((len(records), hidden_size), dtype=np.float32)
        cls_idx = tokenizer.cls_token_index
        if nonempty:
            batch = collator([records[i] for i in nonempty])
            for col in NON_MODEL_COLUMNS:
                batch.pop(col, None)
            if batch["input_ids"].dtype != torch.long:
                raise AssertionError(f"collator produced input_ids of dtype {batch['input_ids'].dtype}")
            # Position 0 is [CLS]; the prompt tokens are the only ones with age -1.
            ages = batch["ages"][:, 1 : 1 + DEMOGRAPHIC_PROMPT_LENGTH]
            four = (ages == -1).all(dim=1)
            two = (ages[:, :2] == -1).all(dim=1)
            prompt_tokens[nonempty] = torch.where(four, 4, torch.where(two, 2, 0)).numpy()
            cls[nonempty], mean[nonempty] = _encode(model, batch, cls_idx, device, amp_dtype)
        if empty:
            batch = cls_only_batch(len(empty), cls_idx)
            cls[empty], mean[empty] = _encode(model, batch, cls_idx, device, amp_dtype)
        prompt_counts.append(prompt_tokens if keep_demographic_prompt else prompt_tokens * 0)
        cls_parts.append(cls)
        mean_parts.append(mean)
        done += len(records)
        n_batches += 1
        if progress_every and n_batches % progress_every == 0:
            print(f"  {done} rows", flush=True)

    n_tokens_arr = np.asarray(n_tokens, dtype=np.int64)
    return ExtractedFeatures(
        cls=np.concatenate(cls_parts) if cls_parts else np.zeros((0, hidden_size), dtype=np.float32),
        mean=np.concatenate(mean_parts) if mean_parts else np.zeros((0, hidden_size), dtype=np.float32),
        person_id=np.asarray(person_ids, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        n_tokens_before_truncation=n_tokens_arr,
        truncated=n_tokens_arr > (max_length - 1),
        n_event_tokens=np.asarray(n_events, dtype=np.int64),
        demographic_prompt_tokens=np.concatenate(prompt_counts) if prompt_counts else np.zeros(0, dtype=np.int64),
    )


def save_split_features(out_dir: str | Path, feats: ExtractedFeatures, index_dates: np.ndarray) -> list[Path]:
    """Write features_<kind>.npy and meta.parquet for one split; returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for kind in FEATURE_KINDS:
        path = out / f"features_{kind}.npy"
        np.save(path, getattr(feats, kind).astype(np.float32))
        written.append(path)
    meta = pd.DataFrame(
        {
            "person_id": feats.person_id,
            "index_date": pd.to_datetime(np.asarray(index_dates)),
            "label": feats.label,
            "n_tokens_before_truncation": feats.n_tokens_before_truncation,
            "truncated": feats.truncated.astype(bool),
            "n_event_tokens": feats.n_event_tokens,
            "demographic_prompt_tokens": feats.demographic_prompt_tokens,
        }
    )
    meta_path = out / "meta.parquet"
    meta.to_parquet(meta_path, index=False)
    written.append(meta_path)
    return written


def load_split_features(features_dir: str | Path, split: str, kind: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load ``(X, meta)`` for one split and feature kind, checking that the rows line up."""
    d = Path(features_dir) / split
    X = np.load(d / f"features_{kind}.npy")
    meta = pd.read_parquet(d / "meta.parquet")
    if len(X) != len(meta):
        raise ValueError(f"{d}: {len(X)} feature rows but {len(meta)} meta rows")
    return X, meta


# --------------------------------------------------------------------------- #
# cehrbert_model step 14: tuning
# --------------------------------------------------------------------------- #
def build_search(cv_folds: int = 5, seed: int = 42, n_jobs: int = -1) -> GridSearchCV:
    """StandardScaler + L2 logistic regression, tuned over C and class_weight by CV AUROC."""
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)),
        ]
    )
    grid = {"clf__C": list(C_GRID), "clf__class_weight": list(CLASS_WEIGHT_GRID)}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=n_jobs, refit=True)


def fit_on_train(X_train: np.ndarray, y_train: np.ndarray, cv_folds: int, seed: int, n_jobs: int) -> GridSearchCV:
    """Tune on the training rows only; the returned search holds the refit best estimator."""
    search = build_search(cv_folds=cv_folds, seed=seed, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search


def choose_primary_kind(val_auroc: dict[str, float]) -> str:
    """Feature kind with the highest validation AUROC (ties go to the first kind in FEATURE_KINDS)."""
    return max(FEATURE_KINDS, key=lambda k: (val_auroc.get(k, float("-inf")), -FEATURE_KINDS.index(k)))


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path
