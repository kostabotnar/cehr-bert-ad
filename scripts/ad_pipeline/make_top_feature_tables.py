#!/usr/bin/env python
"""Build annotated top-10 feature tables for the LR, XGBoost, and attention models.

Combines the per-model importance/attribution rankings (already on disk) with real source
codes (via the OMOP CONCEPT parquet) and human-readable descriptions. Drug descriptions were
resolved from the NLM RxNorm API (RXCUI -> ingredient name); ICD/CPT/HCPCS/LOINC descriptions
are from standard code references. These external lookups are captured in DESCRIPTIONS below so
the tables are reproducible without re-querying.

Outputs, under ``build/<label>/feature_tables/``:
  lr_top10.csv, xgboost_top10.csv, attention_top10.csv
  top_features.md   (all three tables, manuscript-ready)

Usage
  python make_top_feature_tables.py
  python make_top_feature_tables.py --window-days -1 --ctx 4096 --top 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402

# Curated descriptions keyed by resolved source code / token / feature name.
# Drugs: RxNorm RXCUI -> ingredient (verified via https://rxnav.nlm.nih.gov RxNorm API).
# Conditions/procedures/measurements: standard ICD-9-CM / ICD-10-CM / CPT / HCPCS / LOINC refs.
DESCRIPTIONS = {
    # anti-dementia and dementia-related psychotropic drugs (RxNorm)
    "6719": "Memantine (NMDA receptor antagonist; anti-dementia)",
    "135447": "Donepezil (cholinesterase inhibitor; anti-dementia)",
    "183379": "Rivastigmine (cholinesterase inhibitor; anti-dementia)",
    "4637": "Galantamine (cholinesterase inhibitor; anti-dementia)",
    "15996": "Mirtazapine (antidepressant; used in dementia)",
    "51272": "Quetiapine (atypical antipsychotic; behavioral/psychological symptoms of dementia)",
    # conditions
    "ICD-9-CM:780.93": "Memory loss",
    "ICD-9-CM:294.20": "Dementia, unspecified, without behavioral disturbance",
    "ICD-10-CM:F03.90": "Unspecified dementia, without behavioral disturbance",
    "ICD-10-CM:R41.3": "Other amnesia",
    "ICD-10-CM:I10": "Essential (primary) hypertension",
    "ICD-9-CM:401.9": "Essential hypertension, unspecified",
    "ICD-9-CM:753.9": "Congenital anomaly of urinary system, unspecified (likely dataset artifact)",
    # procedures
    "HCPCS:G2211": "Office/outpatient E/M visit-complexity add-on (continuity of care)",
    "CPT:99213": "Office/outpatient visit, established patient (low-to-moderate complexity)",
    "CPT:99214": "Office/outpatient visit, established patient (moderate-to-high complexity)",
    "CPT:86803": "Hepatitis C antibody test",
    # measurements (LOINC)
    "2345-7": "Glucose [Mass/volume] in serum or plasma",
    # structural / demographic features
    "D0": "Artificial time-interval token (structural)",
    "LT": "Artificial time-interval token (structural)",
    "Visit/0": "Visit-type token (structural)",
    "Discharge/0": "Discharge token (structural)",
    "age": "Age at index date",
    "n_visits": "Number of visits in look-back window",
    "gender:8507": "Male",
    "gender:8532": "Female",
}


def _concept_lookup(concept_table: Path) -> dict[str, str]:
    df = pd.read_parquet(concept_table, columns=["concept_id", "concept_name"])
    return dict(zip(df["concept_id"].astype(str), df["concept_name"].astype(str)))


def _source_code(feature_or_id: str, codes: dict[str, str]) -> str:
    """Resolve a baseline feature ('cond:1845') or attention concept_id to its source code."""
    s = str(feature_or_id)
    if ":" in s and s.split(":")[0] in ("cond", "drug", "proc"):
        prefix, _, cid = s.partition(":")
        return codes.get(cid, cid)
    return codes.get(s, s)  # attention concept_id may be a bare OMOP id or a structural token


def _describe(source_code: str, raw_feature: str) -> str:
    return DESCRIPTIONS.get(source_code, DESCRIPTIONS.get(raw_feature, "(no description available)"))


def _baseline_table(csv_path: Path, score_col: str, codes: dict[str, str], top: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.reindex(df[score_col].abs().sort_values(ascending=False).index).head(top)
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        src = _source_code(r["feature"], codes)
        rows.append({
            "rank": rank,
            "feature": r["feature"],
            "source_code": src,
            "description": _describe(src, r["feature"]),
            score_col: round(float(r[score_col]), 5),
        })
    return pd.DataFrame(rows)


def _attention_table(csv_path: Path, codes: dict[str, str], top: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path).sort_values("gradxatt_mean", ascending=False).head(top)
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        src = _source_code(r["concept_id"], codes)
        rows.append({
            "rank": rank,
            "concept_id": r["concept_id"],
            "source_code": src,
            "description": _describe(src, str(r["concept_id"])),
            "gradxatt_mean": round(float(r["gradxatt_mean"]), 5),
            "pct_patients_present": round(float(r["pct_patients_present"]), 1),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build annotated top-N feature tables")
    parser.add_argument("--window-days", type=int, default=paths.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--concept-table", default=None)
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    run = paths.run_dir(label)
    out_dir = run / "feature_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_table = Path(args.concept_table) if args.concept_table else (
        paths.OMOP_DIR / "concept" / "concept.parquet")
    codes = _concept_lookup(concept_table)

    lr = _baseline_table(run / "baselines" / "lr" / "feature_importance.csv", "coefficient", codes, args.top)
    xgb = _baseline_table(run / "baselines" / "xgboost" / "feature_importance.csv", "importance", codes, args.top)
    att = _attention_table(run / "attention" / "concept_attribution_pred_pos.csv", codes, args.top)

    lr.to_csv(out_dir / "lr_top10.csv", index=False)
    xgb.to_csv(out_dir / "xgboost_top10.csv", index=False)
    att.to_csv(out_dir / "attention_top10.csv", index=False)

    md = [
        f"# Top {args.top} features per model ({label})",
        "",
        "Drug names resolved via the NLM RxNorm API (RXCUI -> ingredient); ICD/CPT/HCPCS/LOINC "
        "from standard code references. Source codes are the de-identified dataset's vocabulary "
        "codes (no free-text names are stored in the data).",
        "",
        "## Logistic Regression (signed coefficient; + pushes toward AD)",
        "",
        lr.to_markdown(index=False),
        "",
        "## XGBoost (feature importance; unsigned)",
        "",
        xgb.to_markdown(index=False),
        "",
        "## Attention (gradient x attention, predicted-positive patients)",
        "",
        att.to_markdown(index=False),
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
