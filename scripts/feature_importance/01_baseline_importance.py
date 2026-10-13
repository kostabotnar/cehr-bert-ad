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

Features that carry an OMOP concept id (``cond:<id>``, ``drug:<id>``, ``proc:<id>``, ``meas:<id>``,
``race:<id>``, ``gender:<id>``) are labelled with the ``concept_name`` from the OMOP CONCEPT table
(``data/omop_data/concept/concept.parquet``), e.g. ``cond: Amnesia``. Id 0 reads ``race: unknown`` or
``gender: unknown``. ``age`` and ``history_length_days`` get plain-language labels. Another CONCEPT table can be
supplied with ``--concept-table``; ``pipeline.concept_labels`` holds the shared lookup.

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
from pipeline.concept_labels import ConceptLabels, numeric_ids  # noqa: E402


def _rank_lr(model_dir: Path, feature_names: list[str], labels: ConceptLabels) -> pd.DataFrame:
    pipeline = joblib.load(model_dir / "model.joblib")
    coef = np.asarray(pipeline.named_steps["clf"].coef_).ravel()
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "label": [labels.feature_label(f) for f in feature_names],
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    )
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _rank_xgb(model_dir: Path, feature_names: list[str], labels: ConceptLabels) -> pd.DataFrame:
    model = joblib.load(model_dir / "model.joblib")
    importance = np.asarray(model.feature_importances_).ravel()
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "label": [labels.feature_label(f) for f in feature_names],
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
        "Default: data/omop_data/concept/concept.parquet.",
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
    labels = ConceptLabels.load(args.concept_table, numeric_ids(feature_names))
    print(
        f"Classical ML: {classical_dir}  |  {len(feature_names)} features  |  "
        f"CONCEPT table: {'yes (' + str(len(labels)) + ' labelled ids)' if labels else 'unavailable'}\n"
    )

    for model_name, ranker in (("lr", _rank_lr), ("xgboost", _rank_xgb)):
        model_dir = classical_dir / model_name
        if not (model_dir / "model.joblib").is_file():
            print(f"[skip] {model_name}: no model.joblib in {model_dir}\n")
            continue
        df = ranker(model_dir, feature_names, labels)
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
