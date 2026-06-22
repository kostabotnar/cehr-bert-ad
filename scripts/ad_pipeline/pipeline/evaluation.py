"""Evaluation metrics for the fine-tuned AD classifier.

Consumes the prediction parquet folders written by the finetune runner
(``hf_cehrbert_finetune_runner.do_predict``), whose schema is:

    subject_id, prediction_time, predicted_boolean_probability,
    predicted_boolean_value, boolean_value

This module is intentionally model-free: it operates only on the saved
``predicted_boolean_probability`` (continuous score in [0, 1]) and the true
``boolean_value`` label, so it runs with just numpy/pandas/sklearn/scipy/matplotlib
(no torch/transformers needed).

Functions are split into:
  * pure metric computation (discrimination, threshold-based, calibration, bootstrap CIs)
  * matplotlib plot helpers (ROC, PR, calibration, confusion matrix) saved at 300 DPI

Threshold selection (``select_threshold``) is meant to be run on a *validation*
prediction set and the resulting cutoff applied to the test set, to avoid the
optimistic bias of choosing the operating point on the same data used to report
sensitivity/specificity/PPV.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROBABILITY_COLUMN = "predicted_boolean_probability"
LABEL_COLUMN = "boolean_value"

# Clip probabilities away from {0, 1} before logit transforms (calibration slope).
_EPS = 1e-7


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_predictions(pred_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load (y_true, y_prob) from a test_predictions-style parquet folder.

    Accepts either a directory of ``*.parquet`` shards (pandas reads them as one
    table) or a single parquet file. Returns float label array and probability
    array, with any rows missing either value dropped.
    """
    df = pd.read_parquet(pred_dir)
    if PROBABILITY_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"prediction frame must contain '{PROBABILITY_COLUMN}' and '{LABEL_COLUMN}'; "
            f"found columns: {list(df.columns)}"
        )
    y_prob = pd.to_numeric(df[PROBABILITY_COLUMN], errors="coerce").to_numpy(dtype=float)
    # boolean_value may be bool, 0/1, or nullable boolean -> coerce to float {0,1}
    y_true = pd.to_numeric(df[LABEL_COLUMN].astype("boolean").astype("float"), errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(y_prob) | np.isnan(y_true))
    return y_true[valid], y_prob[valid]


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #
def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """AUROC, AUPRC (average precision), and the positive-class prevalence.

    Prevalence is the no-skill baseline for AUPRC, so it is reported alongside.
    """
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "prevalence": float(np.mean(y_true)),
    }


# --------------------------------------------------------------------------- #
# Threshold selection + threshold-dependent metrics
# --------------------------------------------------------------------------- #
def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, method: str = "youden",
                     target_sensitivity: float = 0.90) -> float:
    """Pick an operating threshold.

    method="youden": maximize Youden's J = sensitivity + specificity - 1.
    method="high_sensitivity": smallest threshold achieving sensitivity >= target
        (screening operating point); falls back to the Youden point if unreachable.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if method == "youden":
        j = tpr - fpr
        return float(thresholds[int(np.argmax(j))])
    if method == "high_sensitivity":
        ok = np.where(tpr >= target_sensitivity)[0]
        if len(ok):
            # tpr is monotincreasing along roc_curve; the largest threshold meeting
            # the target sensitivity is at the first qualifying index.
            return float(thresholds[ok[0]])
        return select_threshold(y_true, y_prob, method="youden")
    raise ValueError(f"unknown threshold method: {method}")


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Confusion-matrix-derived metrics at a fixed probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    yt = y_true.astype(int)
    tp = int(np.sum((y_pred == 1) & (yt == 1)))
    fp = int(np.sum((y_pred == 1) & (yt == 0)))
    tn = int(np.sum((y_pred == 0) & (yt == 0)))
    fn = int(np.sum((y_pred == 0) & (yt == 1)))

    def _safe(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    sensitivity = _safe(tp, tp + fn)  # recall / TPR
    specificity = _safe(tn, tn + fp)  # TNR
    ppv = _safe(tp, tp + fp)          # precision
    npv = _safe(tn, tn + fn)
    f1 = _safe(2 * ppv * sensitivity, ppv + sensitivity) if (ppv + sensitivity) else float("nan")
    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "accuracy": _safe(tp + tn, tp + fp + tn + fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class CalibrationResult:
    brier: float
    ece: float
    slope: float
    intercept: float
    bin_centers: List[float]      # mean predicted prob per bin
    bin_true: List[float]         # observed event rate per bin
    bin_counts: List[int]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
                        strategy: str = "quantile") -> CalibrationResult:
    """Brier score, Expected Calibration Error, calibration slope/intercept, reliability curve.

    Calibration slope/intercept come from a logistic regression of the outcome on the
    logit of the predicted probability (slope=1, intercept=0 is perfect). Computed with
    statsmodels if available, else a small Newton-IRLS fallback.
    """
    brier = float(brier_score_loss(y_true, y_prob))

    # Reliability curve + ECE
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize bins; clip indices into [0, len(edges)-2]
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)
    centers: List[float] = []
    trues: List[float] = []
    counts: List[int] = []
    ece = 0.0
    n = len(y_true)
    for b in range(len(edges) - 1):
        mask = idx == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        mean_pred = float(np.mean(y_prob[mask]))
        obs_rate = float(np.mean(y_true[mask]))
        centers.append(mean_pred)
        trues.append(obs_rate)
        counts.append(cnt)
        ece += (cnt / n) * abs(obs_rate - mean_pred)

    slope, intercept = _calibration_slope_intercept(y_true, y_prob)
    return CalibrationResult(
        brier=brier, ece=float(ece), slope=slope, intercept=intercept,
        bin_centers=centers, bin_true=trues, bin_counts=counts,
    )


def _calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Logistic regression y ~ logit(p). Returns (slope, intercept)."""
    x = _logit(y_prob)
    # Degenerate cases (single class) -> undefined fit
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    try:
        import statsmodels.api as sm

        model = sm.Logit(y_true, sm.add_constant(x, has_constant="add"))
        res = model.fit(disp=0)
        intercept, slope = float(res.params[0]), float(res.params[1])
        return slope, intercept
    except Exception:
        return _irls_logit(x, y_true)


def _irls_logit(x: np.ndarray, y: np.ndarray, n_iter: int = 50) -> Tuple[float, float]:
    """Minimal Newton-IRLS logistic fit with intercept; fallback for slope/intercept."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(n_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        wx = X * w[:, None]
        try:
            beta_new = np.linalg.solve(X.T @ wx, X.T @ (w * z))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[1]), float(beta[0])


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 n_resamples: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar metric.

    Returns (point_estimate, ci_low, ci_high). Resampling is done with replacement
    over the full prediction set. Resamples whose draw is single-class (metric undefined)
    are skipped.
    """
    point = float(metric_fn(y_true, y_prob))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: List[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            val = float(metric_fn(yt, yp))
        except Exception:
            continue
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


# --------------------------------------------------------------------------- #
# Plot helpers (each saves a 300 DPI figure and returns the path)
# --------------------------------------------------------------------------- #
def _new_ax(figsize=(6, 5)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path,
             auroc: Optional[float] = None) -> Path:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if auroc is None:
        auroc = roc_auc_score(y_true, y_prob)
    fig, ax = _new_ax()
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}", color="#1f77b4")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return _save(fig, out_path)


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path,
            auprc: Optional[float] = None) -> Path:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    if auprc is None:
        auprc = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))
    fig, ax = _new_ax()
    ax.plot(recall, precision, label=f"AUPRC = {auprc:.3f}", color="#d62728")
    ax.axhline(prevalence, linestyle="--", color="grey", label=f"baseline (prev={prevalence:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    return _save(fig, out_path)


def plot_calibration(cal: CalibrationResult, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = _new_ax()
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.plot(cal.bin_centers, cal.bin_true, marker="o", color="#2ca02c",
            label=f"model (slope={cal.slope:.2f}, intercept={cal.intercept:.2f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"Calibration (Brier={cal.brier:.3f}, ECE={cal.ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return _save(fig, out_path)


def plot_confusion(tm: Dict[str, float], out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    cm = np.array([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]], dtype=int)
    fig, ax = _new_ax(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title(f"Confusion matrix @ threshold={tm['threshold']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out
