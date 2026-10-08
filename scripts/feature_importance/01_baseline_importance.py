#!/usr/bin/env python
"""Extract and rank feature importance for the trained LR and XGBoost baselines.

classical_ml step 02 saves the fitted estimators (``model.joblib``) and classical_ml step 01 saves a column-aligned
``feature_names.json``, but neither persists ranked importances. This helper loads both and
writes ranked tables:

  * LR       -- signed coefficients from the fitted Pipeline's LogisticRegression step. The
                pipeline scales each feature with MaxAbsScaler (to [-1, 1]) after log1p, so
                coefficient magnitudes are roughly comparable across features. Sign = direction
                (positive pushes the prediction toward the positive/AD class).
  * XGBoost  -- ``feature_importances_`` (the booster's importance, gain-based by default).

OMOP code features (``cond:<id>`` / ``drug:<id>`` / ``proc:<id>``) are mapped to a label via the
OMOP CONCEPT table (``data/omop_data/concept/concept.parquet``, column ``concept_name``). NOTE:
in this de-identified dataset ``concept_name`` is the *source vocabulary code* (e.g.
``ICD-10-CM:M17.11``, ``CPT:93892``, or a bare RxNorm/NDC number for drugs), not free text.
True descriptions require an external OMOP vocabulary (Athena) CONCEPT table -- supply one with
``--concept-table`` and it is used verbatim.

Models and feature names are read from ``build/classical_ml/<window_label>/`` (the window label is
``w_<days>`` or ``w_all``). Outputs, under ``build/feature_importance/<run_label>/``:
  lr/feature_importance.csv        all features by |coefficient|
  xgboost/feature_importance.csv   all features by importance

Usage
  python feature_importance/01_baseline_importance.py                       # default run label
  python feature_importance/01_baseline_importance.py --window-days 365 --ctx 512
  python feature_importance/01_baseline_importance.py --classical-dir build/classical_ml/w_365 --top 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import paths  # noqa: E402

# Standard OMOP gender concept ids (kept un-obfuscated in the demographic features).
GENDER_LABELS = {"8507": "Male", "8532": "Female"}


def _load_concept_names(concept_table: Path | None) -> dict[str, str]:
    """Load ``concept_id -> concept_name`` from the OMOP CONCEPT parquet, if present.

    ``concept_table`` overrides the default ``data/omop_data/concept/concept.parquet``. In the
    de-identified dataset the name is the source vocabulary code; an external Athena CONCEPT
    table here would supply real descriptions instead.
    """
    path = concept_table or (paths.OMOP_DIR / "concept" / "concept.parquet")
    if not Path(path).exists():
        return {}
    df = pd.read_parquet(path, columns=["concept_id", "concept_name"])
    return dict(zip(df["concept_id"].astype(str), df["concept_name"].astype(str)))


def _readable(feature: str, names: dict[str, str]) -> str:
    """Map a feature name (e.g. 'cond:10054') to a label via the CONCEPT table."""
    if ":" not in feature:
        return feature
    prefix, _, concept_id = feature.partition(":")
    if prefix in ("cond", "drug", "proc") and concept_id in names:
        return f"{prefix}: {names[concept_id]}"
    if prefix == "gender":
        return f"gender: {GENDER_LABELS.get(concept_id, concept_id)}"
    return feature


def _rank_lr(model_dir: Path, feature_names: list[str], names: dict[str, str]) -> pd.DataFrame:
    pipeline = joblib.load(model_dir / "model.joblib")
    coef = np.asarray(pipeline.named_steps["clf"].coef_).ravel()
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "label": [_readable(f, names) for f in feature_names],
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    )
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _rank_xgb(model_dir: Path, feature_names: list[str], names: dict[str, str]) -> pd.DataFrame:
    model = joblib.load(model_dir / "model.joblib")
    importance = np.asarray(model.feature_importances_).ravel()
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "label": [_readable(f, names) for f in feature_names],
            "importance": importance,
        }
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank LR/XGBoost baseline feature importance")
    parser.add_argument("--window-days", type=int, default=paths.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX)
    parser.add_argument(
        "--classical-dir",
        default=None,
        help="Override the classical_ml folder with lr/, xgboost/ and features/ (default: build/classical_ml/<window_label>)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output folder (default: build/feature_importance/<run_label>)",
    )
    parser.add_argument("--top", type=int, default=25, help="Rows to print to stdout per model")
    parser.add_argument(
        "--concept-table",
        default=None,
        help="Path to an OMOP CONCEPT parquet (concept_id, concept_name). "
        "Default: data/omop_data/concept/concept.parquet. Supply an Athena "
        "vocabulary table here for true descriptions.",
    )
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    classical_dir = (
        Path(args.classical_dir) if args.classical_dir else paths.classical_ml_dir(paths.window_label(args.window_days))
    )
    output_dir = Path(args.output_dir) if args.output_dir else paths.feature_importance_dir(label)
    feature_names_path = classical_dir / "features" / "feature_names.json"

    if not feature_names_path.is_file():
        available = sorted(str(p) for p in paths.CLASSICAL_ML_BUILD_DIR.glob("*/features") if p.is_dir())
        print(f"feature_names.json not found at {feature_names_path}")
        print(
            "Available classical_ml feature folders (use --window-days to select one):\n  "
            + ("\n  ".join(available) or "(none)")
        )
        return 1

    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    concept_table = Path(args.concept_table) if args.concept_table else None
    names = _load_concept_names(concept_table)
    print(
        f"Classical ML: {classical_dir}  |  {len(feature_names)} features  |  "
        f"CONCEPT table: {'yes (' + str(len(names)) + ' codes)' if names else 'unavailable'}\n"
    )

    for model_name, ranker in (("lr", _rank_lr), ("xgboost", _rank_xgb)):
        model_dir = classical_dir / model_name
        if not (model_dir / "model.joblib").is_file():
            print(f"[skip] {model_name}: no model.joblib in {model_dir}\n")
            continue
        df = ranker(model_dir, feature_names, names)
        out_path = output_dir / model_name / "feature_importance.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        score_col = "coefficient" if model_name == "lr" else "importance"
        print(
            f"=== {model_name.upper()} top {args.top} (by "
            f"{'|coefficient|' if model_name == 'lr' else 'importance'}) ==="
        )
        cols = ["feature", "label", score_col]
        print(df.head(args.top)[cols].to_string(index=False))
        print(f"-> full ranking written to {out_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
