"""Tests for pipeline.evaluation (pure metric functions).

All synthetic: no dependency on the real ``test_predictions`` parquet. Uses
``numpy.random.default_rng`` with fixed seeds so every case is deterministic.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pipeline import evaluation as ev


# --------------------------------------------------------------------------
# load_predictions
# --------------------------------------------------------------------------
def _write_pred_parquet(folder, probs, labels, with_extra=True):
    folder.mkdir(parents=True, exist_ok=True)
    cols = {
        "subject_id": list(range(len(probs))),
        "prediction_time": [dt.datetime(2020, 1, 1)] * len(probs),
        "predicted_boolean_probability": probs,
        "boolean_value": labels,
    }
    if with_extra:
        cols["predicted_boolean_value"] = [None] * len(probs)
    pd.DataFrame(cols).to_parquet(folder / "0.parquet")


def test_load_predictions_shapes_and_nan_drop(tmp_path):
    pred = tmp_path / "test_predictions"
    # one row has a NaN probability and must be dropped
    probs = [0.1, 0.9, np.nan, 0.7, 0.3]
    labels = [False, True, True, True, False]  # real schema: bool dtype
    _write_pred_parquet(pred, probs, labels)

    y_true, y_prob = ev.load_predictions(pred)

    assert y_true.shape == y_prob.shape == (4,)  # NaN-prob row dropped
    assert y_true.dtype == float and y_prob.dtype == float
    assert not np.isnan(y_prob).any()
    # labels coerced from bool -> {0.0, 1.0}
    assert set(np.unique(y_true)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(y_true, np.array([0.0, 1.0, 1.0, 0.0]))


def test_load_predictions_reads_directory_of_shards(tmp_path):
    pred = tmp_path / "test_predictions"
    pred.mkdir(parents=True)
    _write_pred_parquet(pred, [0.2, 0.8], [False, True])
    # second shard
    pd.DataFrame(
        {
            "subject_id": [10, 11],
            "prediction_time": [dt.datetime(2020, 1, 1)] * 2,
            "predicted_boolean_probability": [0.3, 0.6],
            "predicted_boolean_value": [None, None],
            "boolean_value": [True, False],
        }
    ).to_parquet(pred / "1.parquet")

    y_true, y_prob = ev.load_predictions(pred)
    assert y_true.shape == (4,)
    assert y_prob.shape == (4,)


def test_load_predictions_missing_column_raises(tmp_path):
    pred = tmp_path / "test_predictions"
    pred.mkdir(parents=True)
    pd.DataFrame({"subject_id": [1, 2], "predicted_boolean_probability": [0.5, 0.6]}).to_parquet(
        pred / "0.parquet"
    )
    with pytest.raises(ValueError, match="boolean_value"):
        ev.load_predictions(pred)


# --------------------------------------------------------------------------
# discrimination_metrics
# --------------------------------------------------------------------------
def test_discrimination_perfectly_separable():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    m = ev.discrimination_metrics(y_true, y_prob)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["prevalence"] == pytest.approx(0.5)


def test_discrimination_prevalence():
    y_true = np.array([0, 0, 0, 0, 1], dtype=float)
    y_prob = np.array([0.2, 0.3, 0.1, 0.4, 0.9])
    m = ev.discrimination_metrics(y_true, y_prob)
    assert m["prevalence"] == pytest.approx(0.2)
    assert 0.0 <= m["auroc"] <= 1.0


# --------------------------------------------------------------------------
# select_threshold
# --------------------------------------------------------------------------
def test_select_threshold_youden_separates_easy_case():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = ev.select_threshold(y_true, y_prob, method="youden")
    assert isinstance(thr, float)
    y_pred = (y_prob >= thr).astype(int)
    # perfect separation at the chosen threshold
    np.testing.assert_array_equal(y_pred, y_true.astype(int))


def test_select_threshold_high_sensitivity_meets_target():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n).astype(float)
    # noisy-but-informative scores
    y_prob = np.clip(0.5 * y_true + rng.normal(0.3, 0.2, size=n), 0.0, 1.0)
    target = 0.90
    thr = ev.select_threshold(y_true, y_prob, method="high_sensitivity", target_sensitivity=target)
    tm = ev.threshold_metrics(y_true, y_prob, thr)
    assert tm["sensitivity"] >= target


def test_select_threshold_high_sensitivity_unreachable_falls_back_to_youden():
    # Make a case where sensitivity 1.0 + tiny eps cannot be exceeded; request > 1.0.
    y_true = np.array([0, 0, 1, 1], dtype=float)
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    youden = ev.select_threshold(y_true, y_prob, method="youden")
    # target_sensitivity > 1.0 is never reachable -> youden fallback
    fallback = ev.select_threshold(y_true, y_prob, method="high_sensitivity", target_sensitivity=1.5)
    assert fallback == youden


def test_select_threshold_unknown_method_raises():
    y_true = np.array([0, 1], dtype=float)
    y_prob = np.array([0.2, 0.8])
    with pytest.raises(ValueError, match="unknown threshold method"):
        ev.select_threshold(y_true, y_prob, method="nope")


# --------------------------------------------------------------------------
# threshold_metrics
# --------------------------------------------------------------------------
def test_threshold_metrics_hand_computed():
    # threshold 0.5; pred = prob >= 0.5
    #   true:  1   0   1   0   1   0
    #   prob: 0.9 0.8 0.4 0.3 0.6 0.1
    #   pred:  1   1   0   0   1   0
    # tp: idx0,idx4 -> 2 ; fp: idx1 -> 1 ; fn: idx2 -> 1 ; tn: idx3,idx5 -> 2
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.1])
    tm = ev.threshold_metrics(y_true, y_prob, 0.5)
    assert tm["tp"] == 2
    assert tm["fp"] == 1
    assert tm["fn"] == 1
    assert tm["tn"] == 2
    assert tm["sensitivity"] == pytest.approx(2 / 3)  # tp/(tp+fn)
    assert tm["specificity"] == pytest.approx(2 / 3)  # tn/(tn+fp)
    assert tm["ppv"] == pytest.approx(2 / 3)  # tp/(tp+fp)
    assert tm["npv"] == pytest.approx(2 / 3)  # tn/(tn+fn)
    assert tm["accuracy"] == pytest.approx(4 / 6)
    # f1 = 2*ppv*sens/(ppv+sens)
    assert tm["f1"] == pytest.approx(2 / 3)
    assert tm["threshold"] == pytest.approx(0.5)


def test_threshold_metrics_divide_by_zero_safe():
    # threshold above every score -> all predicted negative: tp=fp=0
    y_true = np.array([1, 1, 0, 0], dtype=float)
    y_prob = np.array([0.1, 0.2, 0.05, 0.15])
    tm = ev.threshold_metrics(y_true, y_prob, 0.9)
    assert tm["tp"] == 0 and tm["fp"] == 0
    # ppv = tp/(tp+fp) -> 0/0 -> nan
    assert np.isnan(tm["ppv"])
    # sensitivity = tp/(tp+fn) = 0/2 = 0 (defined)
    assert tm["sensitivity"] == 0.0
    # specificity = tn/(tn+fp) = 2/2 = 1
    assert tm["specificity"] == 1.0


# --------------------------------------------------------------------------
# calibration_metrics
# --------------------------------------------------------------------------
def test_calibration_well_calibrated_slope_near_one():
    rng = np.random.default_rng(123)
    n = 5000
    # draw true probabilities, then Bernoulli outcomes -> predictions are calibrated
    p = rng.uniform(0.02, 0.98, size=n)
    y = (rng.uniform(size=n) < p).astype(float)
    cal = ev.calibration_metrics(y, p, n_bins=10)

    assert 0.6 <= cal.slope <= 1.6
    assert 0.0 <= cal.brier <= 1.0
    assert cal.ece >= 0.0
    # reliability arrays must be mutually consistent in length
    assert len(cal.bin_centers) == len(cal.bin_true) == len(cal.bin_counts)
    assert len(cal.bin_centers) >= 1
    assert sum(cal.bin_counts) == n  # every sample lands in exactly one bin


def test_calibration_slope_intercept_single_class_is_nan():
    y = np.zeros(50, dtype=float)  # one class only
    p = np.random.default_rng(1).uniform(0, 1, size=50)
    slope, intercept = ev._calibration_slope_intercept(y, p)
    assert np.isnan(slope)
    assert np.isnan(intercept)


def test_calibration_uniform_strategy_bins():
    rng = np.random.default_rng(7)
    n = 1000
    p = rng.uniform(0, 1, size=n)
    y = (rng.uniform(size=n) < p).astype(float)
    cal = ev.calibration_metrics(y, p, n_bins=5, strategy="uniform")
    assert len(cal.bin_centers) == len(cal.bin_true) == len(cal.bin_counts)
    assert sum(cal.bin_counts) == n


# --------------------------------------------------------------------------
# bootstrap_ci
# --------------------------------------------------------------------------
def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(2024)
    n = 300
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_prob = np.clip(0.5 * y_true + rng.normal(0.25, 0.18, size=n), 0.0, 1.0)

    point, lo, hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.roc_auc_score(t, p), n_resamples=200, seed=11
    )
    assert 0.0 <= point <= 1.0
    assert lo <= point <= hi
    assert lo <= hi

    # deterministic given the same seed
    point2, lo2, hi2 = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: ev.roc_auc_score(t, p), n_resamples=200, seed=11
    )
    assert (point, lo, hi) == (point2, lo2, hi2)


def test_bootstrap_ci_all_single_class_returns_nan_ci():
    # all positives -> every resample is single-class -> no valid stats
    y_true = np.ones(20, dtype=float)
    y_prob = np.random.default_rng(0).uniform(0, 1, size=20)
    point, lo, hi = ev.bootstrap_ci(
        y_true, y_prob, lambda t, p: float(np.mean(p)), n_resamples=50, seed=0
    )
    assert np.isnan(lo) and np.isnan(hi)
