#!/usr/bin/env python
"""Extract and rank feature importance for the trained LR and XGBoost baselines.

Step 12 saves the fitted estimators (``model.joblib``) and step 11 saves a column-aligned
``feature_names.json``, but neither persists ranked importances. This helper loads both and
writes ranked tables:

  * LR       -- signed coefficients from the fitted Pipeline's LogisticRegression step. The
                pipeline scales each feature with MaxAbsScaler (to [-1, 1]) after log1p, so
                coefficient magnitudes are roughly comparable across features. Sign = direction
                (positive pushes the prediction toward the positive/AD class).
  * XGBoost  -- ``feature_importances_`` (the booster's importance, gain-based by default).

OMOP code features (``cond:<id>`` / ``drug:<id>`` / ``proc:<id>``) are mapped to human-readable
names via the tokenizer's ``concept_name_mapping.json`` when available.

Outputs, under ``build/<label>/baselines/``:
  lr/feature_importance.csv        all features by |coefficient|
  xgboost/feature_importance.csv   all features by importance

Usage
  python baseline_feature_importance.py                       # default run label
  python baseline_feature_importance.py --window-days 365 --ctx 512
  python baseline_feature_importance.py --baselines-dir build/w_365_ctx512/baselines --top 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths  # noqa: E402


def _load_concept_names() -> dict[str, str]:
    """Load concept_id -> name from the pretraining tokenizer dir, if present."""
    path = paths.PRETRAIN_RESULTS_DIR / "concept_name_mapping.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _readable(feature: str, names: dict[str, str]) -> str:
    """Map a feature name (e.g. 'cond:10054') to a human-readable label."""
    if ":" in feature:
        prefix, _, concept_id = feature.partition(":")
        if prefix in ("cond", "drug", "proc") and concept_id in names:
            return f"{prefix}: {names[concept_id]}"
    return feature


def _rank_lr(model_dir: Path, feature_names: list[str], names: dict[str, str]) -> pd.DataFrame:
    pipeline = joblib.load(model_dir / "model.joblib")
    coef = np.asarray(pipeline.named_steps["clf"].coef_).ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "concept_name": [_readable(f, names) for f in feature_names],
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    })
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _rank_xgb(model_dir: Path, feature_names: list[str], names: dict[str, str]) -> pd.DataFrame:
    model = joblib.load(model_dir / "model.joblib")
    importance = np.asarray(model.feature_importances_).ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "concept_name": [_readable(f, names) for f in feature_names],
        "importance": importance,
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank LR/XGBoost baseline feature importance")
    parser.add_argument("--window-days", type=int, default=paths.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX)
    parser.add_argument("--baselines-dir", default=None,
                        help="Override the baselines folder (default: per-run baselines_dir)")
    parser.add_argument("--top", type=int, default=25, help="Rows to print to stdout per model")
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else paths.baselines_dir(label)
    feature_names_path = baselines_dir / "features" / "feature_names.json"

    if not feature_names_path.is_file():
        available = sorted(str(p) for p in paths.BUILD_DIR.glob("*/baselines") if p.is_dir())
        print(f"feature_names.json not found at {feature_names_path}")
        print("Available baselines folders:\n  " + ("\n  ".join(available) or "(none)"))
        return 1

    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    names = _load_concept_names()
    print(f"Run label: {label}  |  {len(feature_names)} features  |  "
          f"concept names: {'yes' if names else 'unavailable'}\n")

    for model_name, ranker in (("lr", _rank_lr), ("xgboost", _rank_xgb)):
        model_dir = baselines_dir / model_name
        if not (model_dir / "model.joblib").is_file():
            print(f"[skip] {model_name}: no model.joblib in {model_dir}\n")
            continue
        df = ranker(model_dir, feature_names, names)
        out_path = model_dir / "feature_importance.csv"
        df.to_csv(out_path, index=False)
        score_col = "coefficient" if model_name == "lr" else "importance"
        print(f"=== {model_name.upper()} top {args.top} (by "
              f"{'|coefficient|' if model_name == 'lr' else 'importance'}) ===")
        cols = ["concept_name", score_col]
        print(df.head(args.top)[cols].to_string(index=False))
        print(f"-> full ranking written to {out_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
