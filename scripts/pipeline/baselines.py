"""Shared library for the classical-ML baselines (L2 logistic regression, XGBoost).

These baselines predict the *same* AD task as CEHR-BERT and are scored by the *same*
evaluation code (``pipeline.evaluation`` / ``evaluation/01_evaluate.py``). To make the comparison
fair, they reuse the identical patient partition, read from the canonical split file
written by cohort step 01 (``data/cohort/splits.parquet``, one row per patient with
``split`` in {train, val, test}). :func:`recover_splits` requires that file and raises
when it is missing.

CEHR-BERT consumes token sequences; the baselines need a tabular design matrix, which
this module builds directly from the OMOP source tables, **frozen strictly before each
patient's index_date** and limited to a fixed look-back window (365 days, matching
``observation_window`` in ``ad_finetune_config.yaml``):

  * per-concept occurrence **counts** for condition / drug / procedure / measurement domains,
    limited to visit-linked events (the rows ``ad_cohort._event_rows`` returns)
  * demographics: age at index, one-hot gender, one-hot race

The vocabulary (which concepts become columns) and the demographic categories are fit on
**train only**, then applied unchanged to val/test, so there is no train/test leakage. A concept
becomes a column when it is in the qualified set shared with CEHR-BERT and the cohort
(``ad_cohort.qualified_concepts`` with ``omop_common.MIN_CONCEPT_PATIENTS``) and occurs in at least
one training patient's window.

The module is deliberately model-free and depends only on polars / numpy / scipy so it is
cheap to unit-test with synthetic fixtures.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import paths
from .ad_cohort import EVENT_TABLES, _event_rows
from .omop_validation import table_glob

logger = logging.getLogger(__name__)

# Feature-name prefix of each event table. The tables, columns and the event rule come from
# ``ad_cohort.EVENT_TABLES`` / ``ad_cohort._event_rows``, the definition CEHR-BERT also uses.
DOMAIN_PREFIX: Dict[str, str] = {
    "condition_occurrence": "cond",
    "drug_exposure": "drug",
    "procedure_occurrence": "proc",
    "measurement": "meas",
}

# Domains contributing count features: (feature-prefix, table, concept column, date column).
DOMAINS: List[Tuple[str, str, str, str]] = [
    (DOMAIN_PREFIX[table], table, concept_col, date_col) for table, date_col, concept_col in EVENT_TABLES
]

DEFAULT_WINDOW_DAYS = 365

# Per-domain feature encoding. Conditions become presence/absence indicators (binary),
# while drugs and procedures keep their raw occurrence counts. ``assemble_matrix`` maps a
# feature's "<domain>:<concept_id>" prefix through this table (default "count" if unknown).
DOMAIN_ENCODING: Dict[str, str] = {"cond": "binary", "drug": "count", "proc": "count", "meas": "count"}


# --------------------------------------------------------------------------- #
# Cohort / split recovery
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Splits:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def load_canonical_splits(splits_file: str | Path) -> Splits:
    """Build ``Splits`` from the cohort step 01 split file (person_id, index_date, label, split)."""
    df = pl.read_parquet(str(splits_file)).select(
        pl.col("person_id").cast(pl.Int64),
        pl.col("index_date").cast(pl.Date),
        pl.col("label").cast(pl.Int64),
        pl.col("split").cast(pl.Utf8),
    )
    unknown = set(df.get_column("split").unique().to_list()) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"{splits_file}: unexpected split values {sorted(unknown)}")
    if df.get_column("person_id").n_unique() != df.height:
        raise ValueError(f"{splits_file}: a person_id appears more than once")

    def _part(name: str) -> pl.DataFrame:
        return df.filter(pl.col("split") == name).drop("split").sort("person_id")

    return Splits(train=_part("train"), val=_part("val"), test=_part("test"))


def recover_splits(splits_file: str | Path = paths.COHORT_SPLITS_FILE) -> Splits:
    """Train/val/test cohort frames shared by all baselines, from the canonical split file of cohort step 01."""
    if not Path(splits_file).is_file():
        raise FileNotFoundError(f"Canonical splits file {splits_file} not found; run cohort step 01 first.")
    return load_canonical_splits(splits_file)


# --------------------------------------------------------------------------- #
# Feature extraction (frozen before index_date, fixed look-back window)
# --------------------------------------------------------------------------- #
def _require_glob(data_dir: Path, table: str) -> str | None:
    return table_glob(data_dir, table)


def build_counts(cohort: pl.DataFrame, data_dir: str | Path, window_days: int) -> pl.DataFrame:
    """Per-(person, concept) occurrence counts strictly before each ``index_date``.

    The events are exactly the rows ``ad_cohort._event_rows`` returns for the tables of
    ``ad_cohort.EVENT_TABLES`` (visit linked, concept present and not 0, date present except for
    measurement), so the baselines and CEHR-BERT share one definition of an event.

    When ``window_days`` is ``None`` or negative the look-back is unbounded ("all history"):
    only the upper bound ``event_date < index_date`` is applied. Otherwise events are limited
    to ``[index_date - window_days, index_date)``.

    Returns a long frame: ``person_id, feature, count`` where ``feature`` is
    ``"<domain>:<concept_id>"``. Only patients in ``cohort`` are kept.
    """
    data_dir = Path(data_dir)
    unbounded = window_days is None or window_days < 0
    index = cohort.lazy().select(pl.col("person_id").cast(pl.Int64), "index_date")
    parts: List[pl.LazyFrame] = []
    for prefix, table, concept_col, date_col in DOMAINS:
        rows = _event_rows(data_dir, table, date_col, concept_col)
        if rows is None:
            continue
        date_filter = pl.col("date") < pl.col("index_date")
        if not unbounded:
            date_filter = date_filter & (pl.col("date") >= pl.col("index_date") - pl.duration(days=window_days))
        events = (
            rows.select(pl.col("person_id").cast(pl.Int64), pl.col("date").cast(pl.Date), "concept_id")
            .join(index, on="person_id", how="inner")
            .filter(date_filter)
            .group_by("person_id", "concept_id")
            .agg(count=pl.len())
            .with_columns(feature=pl.format("{}:{}", pl.lit(prefix), pl.col("concept_id")))
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

    code_features: List[str]  # ordered "<domain>:<concept_id>" columns
    gender_categories: List[int]
    race_categories: List[int]
    age_median: float
    min_concept_patients: int  # threshold that produced the qualified concept set
    window_days: int
    domain_encoding: Dict[str, str]  # domain prefix -> "binary" | "count"

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
            "min_concept_patients": self.min_concept_patients,
            "window_days": self.window_days,
            "domain_encoding": dict(self.domain_encoding),
            "n_features": len(self.feature_names),
        }

    @classmethod
    def from_json(cls, d: dict) -> "FeatureSpec":
        if "min_prevalence" in d or "min_concept_patients" not in d:
            raise ValueError(
                "feature_spec.json was written by an older step 01 (min_prevalence vocabulary); "
                "rerun classical_ml/01_prepare_features.py to rebuild the features"
            )
        if "history_median" in d or "nvisits_median" in d:
            raise ValueError(
                "feature_spec.json still has the history_length_days / n_visits features, which were removed; "
                "rerun classical_ml/01_prepare_features.py to rebuild the features"
            )
        return cls(
            code_features=list(d["code_features"]),
            gender_categories=list(d["gender_categories"]),
            race_categories=list(d["race_categories"]),
            age_median=float(d["age_median"]),
            min_concept_patients=int(d["min_concept_patients"]),
            window_days=int(d["window_days"]),
            domain_encoding=dict(d.get("domain_encoding", DOMAIN_ENCODING)),
        )


def fit_feature_spec(
    train_counts: pl.DataFrame,
    train_demo: pl.DataFrame,
    qualified: pl.Series,
    min_concept_patients: int,
    window_days: int,
    domain_encoding: Dict[str, str] = DOMAIN_ENCODING,
) -> FeatureSpec:
    """Select the kept concept vocabulary + demographic categories from training data only.

    A concept is kept when it is in ``qualified`` (concept ids from ``ad_cohort.qualified_concepts``)
    and occurs in at least one training window; ``min_concept_patients`` is only recorded on the spec.
    """
    if train_counts.height:
        concept = pl.col("feature").str.split(":").list.last().cast(pl.Int64)
        code_features = (
            train_counts.select("feature")
            .unique()
            .filter(concept.is_in(qualified.cast(pl.Int64)))
            .sort("feature")
            .get_column("feature")
            .to_list()
        )
    else:
        code_features = []

    gender = (
        train_demo.filter(pl.col("gender_concept_id").is_not_null())
        .get_column("gender_concept_id")
        .unique()
        .sort()
        .to_list()
    )
    race = (
        train_demo.filter(pl.col("race_concept_id").is_not_null())
        .get_column("race_concept_id")
        .unique()
        .sort()
        .to_list()
    )
    age_median = float(train_demo.get_column("age").median() or 0.0)

    return FeatureSpec(
        code_features=code_features,
        gender_categories=[int(g) for g in gender],
        race_categories=[int(r) for r in race],
        age_median=age_median,
        min_concept_patients=min_concept_patients,
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
def save_split(
    out_dir: str | Path, name: str, X: sp.csr_matrix, y: np.ndarray, person_ids: np.ndarray, index_dates: np.ndarray
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / f"{name}_X.npz", X)
    np.savez(
        out / f"{name}_meta.npz",
        y=y,
        person_ids=person_ids,
        index_dates=index_dates.astype("datetime64[D]").astype("int64"),
    )


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
