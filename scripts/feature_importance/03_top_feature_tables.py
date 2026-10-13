#!/usr/bin/env python
"""Build annotated top-10 feature tables for the LR, XGBoost, and attention models.

Combines the per-model importance/attribution rankings (already on disk) with the source code
(``<vocabulary_id>:<concept_code>``, e.g. ``SNOMED:48167000``) and the concept name from the OMOP CONCEPT table
(``data/omop_data/concept/concept.parquet``, or ``--concept-table``). Structural tokens (time gaps, year, age,
gender, race, visit type, discharge) are decoded by ``pipeline.concept_labels``. DESCRIPTIONS below holds curated
descriptions that take precedence where they apply: drug names resolved from the NLM RxNorm API (RXCUI ->
ingredient), a few ICD/CPT/HCPCS/LOINC references, and the demographic and derived features.

Outputs, under ``build/feature_importance/<label>/feature_tables/``. The rankings are read from the same folder
(``lr/``, ``xgboost/``, ``attention/``):
  lr_top10.csv, xgboost_top10.csv, attention_top10.csv
  top_features.md   (all three tables, manuscript-ready)

Usage
  python feature_importance/03_top_feature_tables.py
  python feature_importance/03_top_feature_tables.py --window-days -1 --ctx 4096 --top 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import paths  # noqa: E402
from pipeline.concept_labels import ConceptLabels, numeric_ids  # noqa: E402
from pipeline.reporting import markdown_table  # noqa: E402

# Curated descriptions that override the CONCEPT table name. Keys are a source code (``<vocabulary_id>:<code>``)
# or a feature name. Drugs: RxNorm RXCUI -> ingredient (verified via https://rxnav.nlm.nih.gov RxNorm API).
# Others: standard ICD-9-CM / ICD-10-CM / CPT / HCPCS / LOINC references.
DESCRIPTIONS = {
    # anti-dementia and dementia-related psychotropic drugs (RxNorm)
    "RxNorm:6719": "Memantine (NMDA receptor antagonist; anti-dementia)",
    "RxNorm:135447": "Donepezil (cholinesterase inhibitor; anti-dementia)",
    "RxNorm:183379": "Rivastigmine (cholinesterase inhibitor; anti-dementia)",
    "RxNorm:4637": "Galantamine (cholinesterase inhibitor; anti-dementia)",
    "RxNorm:15996": "Mirtazapine (antidepressant; used in dementia)",
    "RxNorm:51272": "Quetiapine (atypical antipsychotic; behavioral/psychological symptoms of dementia)",
    # conditions
    "ICD9CM:780.93": "Memory loss",
    "ICD9CM:294.20": "Dementia, unspecified, without behavioral disturbance",
    "ICD10CM:F03.90": "Unspecified dementia, without behavioral disturbance",
    "ICD10CM:R41.3": "Other amnesia",
    "ICD10CM:I10": "Essential (primary) hypertension",
    "ICD9CM:401.9": "Essential hypertension, unspecified",
    "ICD9CM:753.9": "Congenital anomaly of urinary system, unspecified (likely dataset artifact)",
    # procedures
    "HCPCS:G2211": "Office/outpatient E/M visit-complexity add-on (continuity of care)",
    "CPT4:99213": "Office/outpatient visit, established patient (low-to-moderate complexity)",
    "CPT4:99214": "Office/outpatient visit, established patient (moderate-to-high complexity)",
    "CPT4:86803": "Hepatitis C antibody test",
    # measurements (LOINC)
    "LOINC:2345-7": "Glucose [Mass/volume] in serum or plasma",
    # demographic and derived features
    "age": "Age at index date",
    "history_length_days": "Record length: days from the earliest earlier visit to the index date",
    "gender:8507": "Male",
    "gender:8532": "Female",
}


def _describe(source_code: str, raw_feature: str, labels: ConceptLabels) -> str:
    """Curated description first, then the CONCEPT name or decoded token label."""
    return (
        DESCRIPTIONS.get(source_code)
        or DESCRIPTIONS.get(raw_feature)
        or labels.describe(raw_feature)
        or "(no description available)"
    )


def _top_rows(csv_path: Path, score_col: str, top: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"feature": str, "concept_id": str})
    if score_col == "gradxatt_mean":
        return df.sort_values(score_col, ascending=False).head(top)
    return df.reindex(df[score_col].abs().sort_values(ascending=False).index).head(top)


def _baseline_table(df: pd.DataFrame, score_col: str, labels: ConceptLabels) -> pd.DataFrame:
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        src = labels.feature_source_code(r["feature"])
        rows.append(
            {
                "rank": rank,
                "feature": r["feature"],
                "source_code": src,
                "description": _describe(src, r["feature"], labels),
                score_col: round(float(r[score_col]), 5),
            }
        )
    return pd.DataFrame(rows)


def _attention_table(df: pd.DataFrame, labels: ConceptLabels) -> pd.DataFrame:
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        token = str(r["concept_id"])
        src = labels.feature_source_code(token)
        rows.append(
            {
                "rank": rank,
                "concept_id": token,
                "source_code": src,
                "description": _describe(src, token, labels),
                "gradxatt_mean": round(float(r["gradxatt_mean"]), 5),
                "pct_patients_present": round(float(r["pct_patients_present"]), 1),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build annotated top-N feature tables")
    parser.add_argument("--window-days", type=int, default=paths.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--concept-table",
        default=None,
        help="Path to an OMOP CONCEPT parquet (default: data/omop_data/concept/concept.parquet)",
    )
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    run = paths.feature_importance_dir(label)
    out_dir = run / "feature_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    lr = _top_rows(run / "lr" / "feature_importance.csv", "coefficient", args.top)
    xgb = _top_rows(run / "xgboost" / "feature_importance.csv", "importance", args.top)
    att = _top_rows(run / "attention" / "concept_attribution_pred_pos.csv", "gradxatt_mean", args.top)
    shown = [*lr["feature"], *xgb["feature"], *att["concept_id"]]
    labels = ConceptLabels.load(args.concept_table, numeric_ids(shown))

    lr = _baseline_table(lr, "coefficient", labels)
    xgb = _baseline_table(xgb, "importance", labels)
    att = _attention_table(att, labels)

    lr.to_csv(out_dir / "lr_top10.csv", index=False)
    xgb.to_csv(out_dir / "xgboost_top10.csv", index=False)
    att.to_csv(out_dir / "attention_top10.csv", index=False)

    md = [
        f"# Top {args.top} features per model ({label})",
        "",
        "Source codes are `<vocabulary_id>:<concept_code>` from the OMOP CONCEPT table; descriptions are the "
        "CONCEPT names, except curated entries (RxNorm drug names, a few ICD/CPT/HCPCS/LOINC references, "
        "demographic features).",
        "",
        "## Logistic Regression (signed coefficient; + pushes toward AD)",
        "",
        markdown_table(lr),
        "",
        "## XGBoost (feature importance; unsigned)",
        "",
        markdown_table(xgb),
        "",
        "## Attention (gradient x attention, predicted-positive patients)",
        "",
        markdown_table(att),
        "",
    ]
    md_path = out_dir / "top_features.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote tables to {out_dir}")
    for p in ("lr_top10.csv", "xgboost_top10.csv", "attention_top10.csv", "top_features.md"):
        print(f"  {out_dir / p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
