"""Human-readable labels for OMOP concept ids and CEHR-BERT structural tokens.

One lookup shared by the feature importance steps, so the baseline tables, the attention tables and the
top-feature tables agree. Names come from the OMOP CONCEPT table (``concept.parquet``); structural tokens
that are not OMOP concepts are decoded by pattern.

Token meanings, from the code that writes them:
  ``D<n>``        days between consecutive visits (``cehrbert_data`` ``time_day_token``, used for the
                  ``day`` attention-token type; ``W<n>``/``M<n>``/``Y<n>`` are the week/month/year forms of
                  the other types, see ``cehrbert.cehrbert_utils.extract_time_interval_in_days``).
  ``LT``          the long-term gap token that replaces ``D<n>`` once the gap reaches 1080 days
                  (``time_day_token``); ``cehrbert_utils`` treats it as three years.
  ``i-D<n>``/``i-LT``  the same gap tokens between events inside an inpatient or ER visit
                  (``hf_dataset_mapping`` prefixes them with ``i-``).
  ``year:<YYYY>`` and ``age:<n>``  year and age (completed years) at the first visit, the first two tokens of
                  the demographic prompt (``hf_dataset_mapping``), followed by the gender and race tokens.
  ``Gender/<x>``, ``Race/<x>``, ``Visit/<x>``, ``Discharge/<x>``  concept id ``x`` of that kind; ``0`` is the
                  placeholder for a missing value (``cehrbert_data.const.artificial_tokens``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from . import paths

CONCEPT_COLUMNS = ("concept_id", "concept_name", "vocabulary_id", "concept_code")

# Feature prefixes in the classical_ml feature names whose value is an OMOP concept id.
CONCEPT_FEATURE_PREFIXES = ("cond", "drug", "proc", "meas", "race", "gender")

# Visit-type tokens written as an OMOP Visit vocabulary code instead of a concept id.
VISIT_CODE_NAMES = {
    "IP": "Inpatient Visit",
    "ER": "Emergency Room Visit",
    "OP": "Outpatient Visit",
    "ERIP": "Emergency Room and Inpatient Visit",
}

STRUCTURAL_FEATURE_LABELS = {
    "age": "age at index date (years)",
    "history_length_days": "record length before index date (days)",
}

_GAP_UNITS = {"D": "day", "W": "week", "M": "month", "Y": "year"}
_GAP_RE = re.compile(r"^([DWMY])(\d+)$")
_SLASH_KINDS = {"Gender": "gender", "Race": "race", "Visit": "visit type", "Discharge": "discharge destination"}


def default_concept_table() -> Path:
    return paths.OMOP_DIR / "concept" / "concept.parquet"


def numeric_ids(keys: Iterable[str]) -> set[int]:
    """Concept ids referenced by feature names (``cond:123``), tokens (``Race/8527``) or bare ids."""
    ids: set[int] = set()
    for key in keys:
        parts = re.split(r"[:/]", str(key))
        if parts[-1].isdigit() and parts[0] not in ("year", "age"):
            ids.add(int(parts[-1]))
    return ids


class ConceptLabels:
    """Concept id lookups over a (usually id-filtered) CONCEPT table."""

    def __init__(self, table: pd.DataFrame | None = None):
        self._names: dict[str, str] = {}
        self._codes: dict[str, str] = {}
        if table is not None and len(table):
            ids = table["concept_id"].astype(str)
            if "concept_name" in table:
                self._names = dict(zip(ids, table["concept_name"].astype(str)))
            if "vocabulary_id" in table and "concept_code" in table:
                for cid, vocab, code in zip(ids, table["vocabulary_id"], table["concept_code"]):
                    if cid != "0" and isinstance(vocab, str) and isinstance(code, str):
                        self._codes[cid] = f"{vocab}:{code}"

    @classmethod
    def load(cls, concept_table: Path | str | None = None, ids: Iterable[int] | None = None) -> ConceptLabels:
        """Read the CONCEPT parquet, keeping only ``ids`` when given. A missing file gives empty labels."""
        path = Path(concept_table) if concept_table else default_concept_table()
        if not path.is_file():
            return cls()
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
        columns = [c for c in CONCEPT_COLUMNS if c in available]
        if "concept_id" not in columns:
            return cls()
        filters = None
        if ids is not None:
            ids = sorted({int(i) for i in ids})
            if not ids:
                return cls()
            filters = [("concept_id", "in", ids)]
        return cls(pd.read_parquet(path, columns=columns, filters=filters))

    def __bool__(self) -> bool:
        return bool(self._names or self._codes)

    def __len__(self) -> int:
        return len(self._names)

    def concept_name(self, concept_id: str | int) -> str | None:
        return self._names.get(str(concept_id))

    def source_code(self, concept_id: str | int) -> str | None:
        """``<vocabulary_id>:<concept_code>`` for a concept id, or None when unknown or unmapped."""
        return self._codes.get(str(concept_id))

    def _id_label(self, concept_id: str) -> str | None:
        """Concept name for an id; ``unknown`` for the 0 placeholder, None when unresolved."""
        if concept_id == "0":
            return "unknown"
        return self.concept_name(concept_id)

    def feature_label(self, feature: str) -> str:
        """Label for a classical_ml feature name such as ``cond:439147``, ``race:8527`` or ``age``."""
        if feature in STRUCTURAL_FEATURE_LABELS:
            return STRUCTURAL_FEATURE_LABELS[feature]
        prefix, sep, value = feature.partition(":")
        if sep and value.isdigit():
            name = self._id_label(value)
            if name is not None:
                return f"{prefix}: {name}"
        return feature

    def token_label(self, token: str) -> str:
        """Label for a vocabulary token: an OMOP concept id or a structural token. Unknown tokens pass through."""
        token = str(token)
        if token.isdigit():
            return self.concept_name(token) or token
        inpatient = token.startswith("i-")
        body = token[2:] if inpatient else token
        where = "within an inpatient visit" if inpatient else "between visits"
        match = _GAP_RE.match(body)
        if match:
            unit, n = _GAP_UNITS[match.group(1)], int(match.group(2))
            return f"time gap: {n} {unit}{'' if n == 1 else 's'} {where}"
        if body == "LT":
            return f"time gap: long-term (1080 days or more) {where}"
        if token.startswith("year:") and token[5:].isdigit():
            return f"year of first visit: {token[5:]}"
        if token.startswith("age:") and token[4:].isdigit():
            return f"age at first visit: {token[4:]}"
        kind, sep, value = token.partition("/")
        if sep and kind in _SLASH_KINDS:
            label = _SLASH_KINDS[kind]
            if value.isdigit():
                return f"{label}: {self._id_label(value) or value}"
            if kind == "Visit" and value in VISIT_CODE_NAMES:
                return f"{label}: {VISIT_CODE_NAMES[value]}"
            return f"{label}: {value}"
        return token

    def describe(self, key: str) -> str | None:
        """Plain description of a feature name or token (no ``prefix:`` part), or None when none is known."""
        prefix, _, value = key.partition(":")
        if prefix in CONCEPT_FEATURE_PREFIXES and value.isdigit():
            if value == "0":
                return f"unknown {prefix}"
            return self.concept_name(value)
        if key in STRUCTURAL_FEATURE_LABELS:
            return STRUCTURAL_FEATURE_LABELS[key]
        label = self.token_label(key)
        return None if label == key else label

    def feature_source_code(self, key: str) -> str:
        """``<vocabulary_id>:<concept_code>`` for a feature name or bare id; other keys are returned as is."""
        value = key.partition(":")[2] if key.partition(":")[0] in CONCEPT_FEATURE_PREFIXES else key
        return self.source_code(value) or key
