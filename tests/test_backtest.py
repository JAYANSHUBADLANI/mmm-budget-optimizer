"""Tests for the evaluation harness.

The split logic gets its own tests because a leaky split is the failure mode that makes a
model look good and be useless, and it is invisible in the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm.backtest import (
    evaluate,
    mape,
    naive_seasonal_baseline,
    r_squared,
    rmse,
    rolling_origin_splits,
    smape,
)


def test_mape_is_zero_for_a_perfect_forecast():
    y = np.array([100.0, 200.0, 300.0])
    assert mape(y, y) == pytest.approx(0.0)


def test_mape_ignores_zero_actuals_rather_than_dividing_by_zero():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([50.0, 110.0])
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_returns_nan_when_every_actual_is_zero():
    assert np.isnan(mape(np.zeros(4), np.ones(4)))


def test_smape_is_symmetric():
    a, b = np.array([100.0]), np.array([120.0])
    assert smape(a, b) == pytest.approx(smape(b, a))


def test_rmse_and_r2_behave():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == pytest.approx(0.0)
    assert r_squared(y, y) == pytest.approx(1.0)
    # Predicting the mean everywhere gives an R squared of zero by construction.
    assert r_squared(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_evaluate_returns_only_the_requested_metrics():
    y = np.array([1.0, 2.0, 3.0])
    out = evaluate(y, y, metrics=["mape", "rmse"])
    assert set(out) == {"mape", "rmse"}


def test_splits_are_ordered_and_never_overlap_the_training_window():
    splits = rolling_origin_splits(n_periods=113, holdout_weeks=13, n_folds=3, step_weeks=4)
    assert len(splits) == 3
    for train_end, holdout_end in splits:
        assert holdout_end - train_end == 13
        assert train_end > 0
        assert holdout_end <= 113
    # Every fold trains on strictly more data than the one before it.
    assert [s[0] for s in splits] == sorted(s[0] for s in splits)


def test_splits_stop_when_the_training_window_gets_too_short():
    splits = rolling_origin_splits(n_periods=40, holdout_weeks=13, n_folds=10, step_weeks=4)
    assert len(splits) < 10
    for train_end, _ in splits:
        assert train_end >= 26


def test_the_holdout_never_starts_before_the_training_data_ends():
    """This is the leakage test. A holdout that begins inside training is the classic bug."""
    for train_end, holdout_end in rolling_origin_splits(113, 13, 3, 4):
        assert holdout_end > train_end


def test_seasonal_baseline_uses_the_same_week_last_year():
    y = np.zeros((110, 2))
    y[57, :] = 42.0  # week 57 in the series
    preds = naive_seasonal_baseline(y, train_end=100, season=52)
    # Predicting week 109 should look back to week 57.
    assert preds[9, 0] == pytest.approx(42.0)


def test_seasonal_baseline_falls_back_when_history_is_short():
    y = np.arange(30, dtype=float).reshape(-1, 1)
    preds = naive_seasonal_baseline(y, train_end=25, season=52)
    assert np.all(preds == 24.0)  # last observed training value
