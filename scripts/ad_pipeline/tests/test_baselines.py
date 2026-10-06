"""Tests for the classical-ML baseline code merged from add-baseline-comparison."""

from __future__ import annotations

from pipeline import paths


def test_run_label_formats():
    assert paths.run_label(365, 512) == "w_365_ctx512"
    assert paths.run_label(-1, 4096) == "w_all_ctx4096"
    assert paths.run_label(None, 512) == "w_all_ctx512"


def test_default_run_label_matches_trained_model():
    assert paths.run_label(paths.DEFAULT_WINDOW_DAYS, paths.DEFAULT_CTX) == "w_365_ctx512"
