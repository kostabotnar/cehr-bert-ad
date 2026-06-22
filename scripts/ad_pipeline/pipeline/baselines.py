"""Shared library for the classical-ML baselines (L2 logistic regression, XGBoost).

These baselines predict the *same* AD task as CEHR-BERT and are scored by the *same*
evaluation code (``pipeline.evaluation`` / ``10_evaluate.py``). To make the comparison
fair, they reuse the identical patient partition:

  * **test**  : ``data/omop_data/ad_cohort/test``         (held out up front, step 6)
  * **val**   : the patients CEHR-BERT used for internal validation, recovered from the
                ``subject_id`` column of ``data/finetune_results/validation_predictions``
                (so all models pick their operating threshold on the same patients)
  * **train** : the finetune cohort minus the val patients

CEHR-BERT consumes token sequences; the baselines need a tabular design matrix, which
this module builds directly from the OMOP source tables, **frozen strictly before each
patient's index_date** and limited to a fixed look-back window (365 days, matching
``observation_window`` in ``ad_finetune_config.yaml``):

  * per-concept occurrence **counts** for condition / drug / procedure domains
  * demographics: age at index, one-hot gender, one-hot race

The vocabulary (which concepts become columns) and the demographic categories are fit on
**train only** — concepts present in at least ``min_prevalence`` of training patients are
kept — then applied unchanged to val/test, so there is no train/test leakage.

The module is deliberately model-free and depends only on polars / numpy / scipy so it is
cheap to unit-test with synthetic fixtures.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import paths
from .omop_validation import table_glob

# Domains contributing count features: (feature-prefix, table, concept column, date column).
DOMAINS: List[Tuple[str, str, str, str]] = [
    ("cond", "condition_occurrence", "condition_concept_id", "condition_start_date"),
    ("drug", "drug_exposure", "drug_concept_id", "drug_exposure_start_date"),
    ("proc", "procedure_occurrence", "procedure_concept_id", "procedure_date"),
]

DEFAULT_WINDOW_DAYS = 365
DEFAULT_MIN_PREVALENCE = 0.01

# Per-domain feature encoding. Conditions become presence/absence indicators (binary),
# while drugs and procedures keep their raw occurrence counts. ``assemble_matrix`` maps a
# feature's "<domain>:<concept_id>" prefix through this table (default "count" if unknown).
DOMAIN_ENCODING: Dict[str, str] = {"cond": "binary", "drug": "count", "proc": "count"}


# --------------------------------------------------------------------------- #
# Cohort / split recovery
# --------------------------------------------------------------------------- #
def load_cohort(folder: str | Path) -> pl.DataFrame:
    """Read a cohort folder (``*.parquet`` with person_id, index_date, label)."""
    folder = Path(folder)
    df = pl.read_parquet(str(folder / "*.parquet"))
    return df.select(
        pl.col("person_id").cast(pl.Int64),
        pl.col("index_date").cast(pl.Date),
        pl.col("label").cast(pl.Int64),
    ).unique(subset="person_id").sort("person_id")


def _val_person_ids_from_prepared(prepared_dir: str | Path) -> List[int]:
    """Recover the validation patients from CEHR-BERT's tokenized finetune dataset.

    The fine-tuning dataset is saved as a DatasetDict under a hashed subdirectory with
    ``train`` / ``validation`` splits, each carrying the real OMOP ``person_id``. This is
    the authoritative record of which patients CEHR-BERT validated on. (The
    ``validation_predictions`` parquet is *not* usable for this: its ``subject_id`` is a
    hashed identifier that does not join back to ``person_id``.)
    """
    try:
        from .dataset_checks import datasets_available, resolve_prepared_path

        if not datasets_available():
            return []
        resolved = resolve_prepared_path(prepared_dir)
        if resolved is None:
            return []
        from datasets import load_from_disk

        dataset = load_from_disk(resolved)
        if hasattr(dataset, "keys") and "validation" in dataset and "person_id" in dataset["validation"].column_names:
            return [int(p) for p in dataset["validation"]["person_id"]]
    except Exception:  # noqa: BLE001
        return []
    return []


def _deterministic_val_ids(finetune: pl.DataFrame, val_fraction: float, seed: int) -> List[int]:
    """Fallback: a deterministic patient-level validation split (same hashing style as step 6)."""
    import hashlib

    def _bucket(pid: int) -> int:
        h = hashlib.md5(f"{seed}:{pid}".encode(), usedforsecurity=False).hexdigest()
        return int(h[:8], 16) % 1_000_000

    threshold = int(val_fraction * 1_000_000)
    with_bucket = finetune.with_columns(
        pl.col("person_id").map_elements(_bucket, return_dtype=pl.Int64).alias("_b")
    )
    return with_bucket.filter(pl.col("_b") < threshold).get_column("person_id").to_list()


@dataclasses.dataclass
class Splits:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    val_source: str  # "cehrbert_prepared_validation_split" or "deterministic_fallback"


def recover_splits(
    finetune_dir: str | Path = paths.COHORT_FINETUNE_DIR,
    test_dir: str | Path = paths.COHORT_TEST_DIR,
    prepared_dir: str | Path = paths.FINETUNE_PREPARED_DIR,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Splits:
    """Build the train/val/test cohort frames shared by all baselines.

    Val patients are taken from CEHR-BERT's tokenized validation split when available
    (exact parity); otherwise a deterministic 10% split of the finetune cohort is used.
    """
    finetune = load_cohort(finetune_dir)
    test = load_cohort(test_dir)
    ft_ids = set(finetune.get_column("person_id").to_list())

    val_ids = [pid for pid in _val_person_ids_from_prepared(prepared_dir) if pid in ft_ids]
    source = "cehrbert_prepared_validation_split"
    if not val_ids:
        val_ids = _deterministic_val_ids(finetune, val_fraction, seed)
        source = "deterministic_fallback"
    val_set = set(val_ids)

    val = finetune.filter(pl.col("person_id").is_in(list(val_set))).sort("person_id")
    train = finetune.filter(~pl.col("person_id").is_in(list(val_set))).sort("person_id")
    return Splits(train=train, val=val, test=test, val_source=source)


# --------------------------------------------------------------------------- #
# Feature extraction (frozen before index_date, fixed look-back window)
# --------------------------------------------------------------------------- #
def _require_glob(data_dir: Path, table: str) -> str | None:
    return table_glob(data_dir, table)


def build_counts(cohort: pl.DataFrame, data_dir: str | Path, window_days: int) -> pl.DataFrame:
    """Per-(person, concept) occurrence counts strictly before each ``index_date``.

    When ``window_days`` is ``None`` or negative the look-back is unbounded ("all history"):
    only the upper bound ``event_date < index_date`` is applied. Otherwise events are limited
    to ``[index_date - window_days, index_date)``.

    Returns a long frame: ``person_id, feature, count`` where ``feature`` is
    ``"<domain>:<concept_id>"``. Only patients in ``cohort`` are kept.
    """
    data_dir = Path(data_dir)
    unbounded = window_days is None or window_days < 0
    index = cohort.lazy().select("person_id", "index_date")
    parts: List[pl.LazyFrame] = []
    for prefix, table, concept_col, date_col in DOMAINS:
        glob = _require_glob(data_dir, table)
        if glob is None:
            continue
        date_filter = pl.col("event_date") < pl.col("index_date")
        if not unbounded:
            date_filter = date_filter & (
                pl.col("event_date") >= pl.col("index_date") - pl.duration(days=window_days)
            )
        events = (
            pl.scan_parquet(glob)
            .select(
                pl.col("person_id").cast(pl.Int64),
                pl.col(concept_col).cast(pl.Int64).alias("concept_id"),
                pl.col(date_col).cast(pl.Date).alias("event_date"),
            )
            .filter(pl.col("concept_id").is_not_null() & (pl.col("concept_id") != 0))
            .join(index, on="person_id", how="inner")
            .filter(date_filter)
            .group_by("person_id", "concept_id")
            .agg(count=pl.len())
            .with_columns(
                feature=pl.format("{}:{}", pl.lit(prefix), pl.col("concept_id"))
            )
            .select("person_id", "feature", pl.col("count").cast(pl.Int64))
        )
        parts.append(events)

    if not parts:
        return pl.DataFrame(schema={"person_id": pl.Int64, "feature": pl.Utf8, "count": pl.Int64})
    return pl.concat(parts).collect()


def build_demographics(cohort: pl.DataFrame, data_dir: str | Path) -> pl.DataFrame:
    """Per-person demographics: age at index, gender_concept_id, race_concept_id."""
    data_dir = Path(data_dir)
    glob = _require_glob(data_dir, "person")
    person = (
        pl.scan_parquet(glob)
        .select(
            pl.col("person_id").cast(pl.Int64),
            pl.col("gender_concept_id").cast(pl.Int64),
            pl.col("race_concept_id").cast(pl.Int64),
            pl.col("year_of_birth").cast(pl.Int64),
        )
        .collect()
    )
    return (
        cohort.select("person_id", "index_date")
        .join(person, on="person_id", how="left")
        .with_columns(age=(pl.col("index_date").dt.year() - pl.col("year_of_birth")).cast(pl.Float64))
        .select("person_id", "age", "gender_concept_id", "race_concept_id")
    )


# --------------------------------------------------------------------------- #
# Vocabulary / encoder fit on TRAIN, applied to every split
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class FeatureSpec:
    """Everything needed to turn raw counts+demographics into a fixed-width matrix."""

    code_features: List[str]          # ordered "<domain>:<concept_id>" columns
    gender_categories: List[int]
    race_categories: List[int]
    age_median: float
    min_prevalence: float
    window_days: int
    domain_encoding: Dict[str, str]   # domain prefix -> "binary" | "count"

    @property
    def feature_names(self) -> List[str]:
        names = list(self.code_features)
        names.append("age")
        names += [f"gender:{g}" for g in self.gender_categories]
        names += [f"race:{r}" for r in self.race_categories]
        return names

    def to_json(self) -> dict:
        return {
            "code_features": self.code_features,
            "gender_categories": self.gender_categories,
            "race_categories": self.race_categories,
            "age_median": self.age_median,
            "min_prevalence": self.min_prevalence,
            "window_days": self.window_days,
            "domain_encoding": dict(self.domain_encoding),
            "n_features": len(self.feature_names),
        }

    @classmethod
    def from_json(cls, d: dict) -> "FeatureSpec":
        return cls(
            code_features=list(d["code_features"]),
            gender_categories=list(d["gender_categories"]),
            race_categories=list(d["race_categories"]),
            age_median=float(d["age_median"]),
            min_prevalence=float(d["min_prevalence"]),
            window_days=int(d["window_days"]),
            domain_encoding=dict(d.get("domain_encoding", DOMAIN_ENCODING)),
        )


def fit_feature_spec(
    train_counts: pl.DataFrame,
    train_demo: pl.DataFrame,
    n_train: int,
    min_prevalence: float,
    window_days: int,
    domain_encoding: Dict[str, str] = DOMAIN_ENCODING,
) -> FeatureSpec:
    """Select the kept concept vocabulary + demographic categories from training data only."""
    if train_counts.height:
        prevalence = (
            train_counts.select("person_id", "feature")
            .unique()
            .group_by("feature")
            .agg(n=pl.len())
            .with_columns(prev=pl.col("n") / max(n_train, 1))
            .filter(pl.col("prev") >= min_prevalence)
            .sort("feature")
        )
        code_features = prevalence.get_column("feature").to_list()
    else:
        code_features = []

    gender = (
        train_demo.filter(pl.col("gender_concept_id").is_not_null())
        .get_column("gender_concept_id").unique().sort().to_list()
    )
    race = (
        train_demo.filter(pl.col("race_concept_id").is_not_null())
        .get_column("race_concept_id").unique().sort().to_list()
    )
    age_median = float(train_demo.get_column("age").median() or 0.0)
    return FeatureSpec(
        code_features=code_features,
        gender_categories=[int(g) for g in gender],
        race_categories=[int(r) for r in race],
        age_median=age_median,
        min_prevalence=min_prevalence,
        window_days=window_days,
        domain_encoding=dict(domain_encoding),
    )


def assemble_matrix(
    cohort: pl.DataFrame,
    counts: pl.DataFrame,
    demo: pl.DataFrame,
    spec: FeatureSpec,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Build the sparse design matrix for ``cohort`` using a fitted ``spec``.

    Returns (X, y, person_ids, index_dates). Row order follows ``cohort`` (sorted by
    person_id). Columns follow ``spec.feature_names``: code counts, then age, gender
    one-hots, race one-hots.
    """
    cohort = cohort.sort("person_id")
    person_ids = cohort.get_column("person_id").to_numpy()
    y = cohort.get_column("label").to_numpy()
    index_dates = cohort.get_column("index_date").to_numpy()

    n_rows = len(person_ids)
    pid_to_row = {int(p): i for i, p in enumerate(person_ids)}

    # --- code-count block --------------------------------------------------
    feat_to_col = {f: j for j, f in enumerate(spec.code_features)}
    n_code = len(spec.code_features)
    if counts.height and n_code:
        c = counts.filter(pl.col("feature").is_in(spec.code_features))
        features = c.get_column("feature").to_list()
        rows = np.fromiter((pid_to_row[int(p)] for p in c.get_column("person_id")), dtype=np.int64, count=c.height)
        cols = np.fromiter((feat_to_col[f] for f in features), dtype=np.int64, count=c.height)
        raw = c.get_column("count").to_numpy().astype(np.float64)
        # Per-domain encoding: "binary" domains (e.g. conditions) become 1.0 presence
        # indicators; all others keep their raw occurrence count.
        is_binary = np.fromiter(
            (spec.domain_encoding.get(f.split(":", 1)[0], "count") == "binary" for f in features),
            dtype=bool,
            count=c.height,
        )
        data = np.where(is_binary, 1.0, raw)
        code_block = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_code)).tocsr()
    else:
        code_block = sp.csr_matrix((n_rows, n_code))

    # --- demographic block (dense, then sparsified) ------------------------
    demo = cohort.select("person_id").join(demo, on="person_id", how="left")
    age = demo.get_column("age").to_numpy().astype(np.float64)
    age = np.where(np.isnan(age), spec.age_median, age).reshape(-1, 1)

    def _one_hot(col: str, categories: Sequence[int]) -> np.ndarray:
        vals = demo.get_column(col).to_numpy()
        mat = np.zeros((n_rows, len(categories)), dtype=np.float64)
        cat_to_idx = {int(c): k for k, c in enumerate(categories)}
        for i, v in enumerate(vals):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                k = cat_to_idx.get(int(v))
                if k is not None:
                    mat[i, k] = 1.0
        return mat

    gender_oh = _one_hot("gender_concept_id", spec.gender_categories)
    race_oh = _one_hot("race_concept_id", spec.race_categories)
    demo_block = sp.csr_matrix(np.hstack([age, gender_oh, race_oh]))

    X = sp.hstack([code_block, demo_block], format="csr")
    return X, y.astype(np.int64), person_ids, index_dates


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_split(out_dir: str | Path, name: str, X: sp.csr_matrix, y: np.ndarray,
               person_ids: np.ndarray, index_dates: np.ndarray) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / f"{name}_X.npz", X)
    np.savez(out / f"{name}_meta.npz", y=y, person_ids=person_ids,
             index_dates=index_dates.astype("datetime64[D]").astype("int64"))


def load_split(out_dir: str | Path, name: str) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    out = Path(out_dir)
    X = sp.load_npz(out / f"{name}_X.npz")
    meta = np.load(out / f"{name}_meta.npz", allow_pickle=True)
    index_dates = meta["index_dates"].astype("datetime64[D]")
    return X, meta["y"], meta["person_ids"], index_dates


def save_feature_spec(out_dir: str | Path, spec: FeatureSpec) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "feature_spec.json"
    path.write_text(json.dumps(spec.to_json(), indent=2), encoding="utf-8")
    (out / "feature_names.json").write_text(json.dumps(spec.feature_names), encoding="utf-8")
    return path


def load_feature_spec(out_dir: str | Path) -> FeatureSpec:
    out = Path(out_dir)
    return FeatureSpec.from_json(json.loads((out / "feature_spec.json").read_text(encoding="utf-8")))
