"""Tests for the constrained budget optimiser.

These use a synthetic posterior rather than a fitted model, so they run in milliseconds and
test the optimiser rather than the sampler. Constructing the posterior by hand also means I
know the right answer in advance for the cases where an analytical answer exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm.config import DotDict
from mmm.models.bayesian_mmm import ModelData, PosteriorParams
from mmm.optimizer import (
    build_constraints,
    check_feasibility,
    make_objective,
    naive_proportional_reallocation,
    optimise_budget,
    project_to_feasible,
)


@pytest.fixture
def fake_setup():
    """Three channels with deliberately different response curves.

    Channel 0 has a high coefficient and low half saturation, so it saturates early.
    Channel 1 has a moderate coefficient and high half saturation, so it has headroom.
    Channel 2 is weak. A correct optimiser should move budget toward channel 1.
    """
    rng = np.random.default_rng(3)
    n_t, n_d, n_c, n_s = 40, 3, 3, 25
    X = rng.uniform(0.2, 0.9, size=(n_t, n_d, n_c))

    data = ModelData(
        X=X,
        y=rng.uniform(0.4, 0.8, size=(n_t, n_d)),
        controls=np.zeros((n_t, n_d, 0)),
        seasonality=np.zeros((n_t, 2)),
        trend=np.linspace(0, 1, n_t),
        entities=["A", "B", "C"],
        dates=list(range(n_t)),
        channels=["fast_saturating", "headroom", "weak"],
        control_names=[],
        media_scalers={"fast_saturating": 1e6, "headroom": 1e6, "weak": 1e6},
        target_scale=1e6,
        train_mask=np.ones(n_t, dtype=bool),
    )
    params = PosteriorParams(
        decay=np.tile(np.array([0.3, 0.4, 0.5]), (n_s, 1)),
        half_sat=np.tile(np.array([0.15, 0.90, 0.50]), (n_s, 1)),
        slope=np.ones((n_s, 3)),
        beta=np.tile(np.array([[0.30], [0.25], [0.05]]), (n_s, 1, n_d)),
        alpha=np.full((n_s, n_d), 0.4),
        trend_coef=np.zeros(n_s),
        seas_coef=np.zeros((n_s, 2)),
        control_coef=np.zeros((n_s, 0)),
        sigma=np.full(n_s, 0.05),
    )
    current_spend = np.array([300_000.0, 300_000.0, 400_000.0])
    return params, data, current_spend


def test_budget_is_conserved(fake_setup, cfg):
    params, data, current = fake_setup
    result = optimise_budget(params, data, current, cfg)
    assert result.optimal_spend.sum() == pytest.approx(current.sum(), rel=1e-6)


def test_per_channel_bounds_are_respected(fake_setup, cfg):
    params, data, current = fake_setup
    result = optimise_budget(params, data, current, cfg)
    lo = float(cfg["optimizer"]["min_spend_pct_of_current"]) * current
    hi = float(cfg["optimizer"]["max_spend_pct_of_current"]) * current
    assert np.all(result.optimal_spend >= lo - 1e-6)
    assert np.all(result.optimal_spend <= hi + 1e-6)


def test_total_reallocation_cap_is_respected(fake_setup, cfg):
    params, data, current = fake_setup
    result = optimise_budget(params, data, current, cfg)
    moved = np.abs(result.optimal_spend - current).sum()
    cap = 2.0 * float(cfg["optimizer"]["max_total_reallocation_pct"]) * current.sum()
    assert moved <= cap + 1e-4


def test_optimiser_does_not_reduce_predicted_sales(fake_setup, cfg):
    """The current allocation is always feasible, so the optimum cannot be worse."""
    params, data, current = fake_setup
    result = optimise_budget(params, data, current, cfg)
    assert result.predicted_sales_optimal >= result.predicted_sales_current - 1e-6


def test_budget_moves_toward_the_channel_with_headroom(fake_setup, cfg):
    params, data, current = fake_setup
    result = optimise_budget(params, data, current, cfg)
    frame = result.to_frame().set_index("channel")
    assert frame.loc["headroom", "change_absolute"] > 0
    assert frame.loc["weak", "change_absolute"] < 0


def test_tighter_reallocation_cap_produces_a_smaller_uplift(fake_setup, cfg):
    """Constraints cost money. The relationship should be monotone and visible."""
    params, data, current = fake_setup
    loose = DotDict({**cfg, "optimizer": {**cfg["optimizer"], "max_total_reallocation_pct": 0.40}})
    tight = DotDict({**cfg, "optimizer": {**cfg["optimizer"], "max_total_reallocation_pct": 0.05}})
    assert (
        optimise_budget(params, data, current, tight).uplift
        <= optimise_budget(params, data, current, loose).uplift + 1e-9
    )


def test_infeasible_floors_raise_a_clear_error(fake_setup, cfg):
    params, data, current = fake_setup
    bad = DotDict({**cfg, "optimizer": {**cfg["optimizer"], "min_spend_pct_of_current": 1.2}})
    with pytest.raises(ValueError, match="floors already exceed"):
        optimise_budget(params, data, current, bad)


def test_infeasible_caps_raise_a_clear_error(fake_setup, cfg):
    params, data, current = fake_setup
    bad = DotDict({**cfg, "optimizer": {**cfg["optimizer"], "max_spend_pct_of_current": 0.8}})
    with pytest.raises(ValueError, match="caps cannot absorb"):
        optimise_budget(params, data, current, bad)


def test_constraint_matrix_shapes(fake_setup):
    _, _, current = fake_setup
    constraints, bounds = build_constraints(current, current.sum(), 0.6, 1.6, 0.25)
    assert len(constraints) == 4
    assert len(bounds) == 2 * len(current)
    for con in constraints:
        assert con.A.shape[1] == 2 * len(current)


def test_objective_is_monotone_in_total_budget(fake_setup):
    """More money cannot buy less, given non negative coefficients."""
    params, data, current = fake_setup
    objective = make_objective(params, data, current)
    assert objective(current * 1.2) > objective(current)
    assert objective(current * 0.8) < objective(current)


def test_naive_reallocation_conserves_budget_and_follows_roi(fake_setup):
    import pandas as pd

    _, data, current = fake_setup
    roi = pd.DataFrame(
        {"channel": data.channels, "roi_mean": [3.0, 1.0, 0.5]}
    )
    out = naive_proportional_reallocation(roi, current, data.channels)
    assert out.sum() == pytest.approx(current.sum())
    assert out[0] > out[1] > out[2]


def test_uncertainty_run_returns_shares_that_sum_to_one(fake_setup, cfg):
    from mmm.optimizer import optimise_with_uncertainty

    params, data, current = fake_setup
    summary, uplift = optimise_with_uncertainty(params, data, current, cfg, n_draws=5)
    assert summary["recommended_share_mean"].sum() == pytest.approx(1.0, rel=1e-6)
    assert set(["mean", "p05", "p95", "prob_positive"]).issubset(uplift.columns)
    assert 0.0 <= float(uplift["prob_positive"].iloc[0]) <= 1.0


def test_feasibility_check_flags_an_unconstrained_plan(fake_setup, cfg):
    """The naive proportional plan usually violates the constraints, and must be seen to."""
    import pandas as pd

    _, data, current = fake_setup
    roi = pd.DataFrame({"channel": data.channels, "roi_mean": [10.0, 1.0, 0.2]})
    naive = naive_proportional_reallocation(roi, current, data.channels)
    check = check_feasibility(naive, current, float(current.sum()), cfg)
    assert check["feasible"] is False
    assert check["violations"]


def test_projection_returns_a_feasible_plan(fake_setup, cfg):
    import pandas as pd

    _, data, current = fake_setup
    roi = pd.DataFrame({"channel": data.channels, "roi_mean": [10.0, 1.0, 0.2]})
    naive = naive_proportional_reallocation(roi, current, data.channels)
    projected = project_to_feasible(naive, current, float(current.sum()), cfg)
    check = check_feasibility(projected, current, float(current.sum()), cfg, tol=1e-4)
    assert check["feasible"] is True, check["violations"]


def test_optimiser_beats_the_naive_plan_on_equal_terms(fake_setup, cfg):
    """The comparison that matters: both plans obeying the same constraints.

    Comparing a constrained optimum against an unconstrained heuristic would be comparing
    solutions to two different problems, and the heuristic could win for the wrong reason.
    """
    import pandas as pd

    params, data, current = fake_setup
    roi = pd.DataFrame({"channel": data.channels, "roi_mean": [10.0, 1.0, 0.2]})
    naive = project_to_feasible(
        naive_proportional_reallocation(roi, current, data.channels),
        current,
        float(current.sum()),
        cfg,
    )
    objective = make_objective(params, data, current)
    optimal = optimise_budget(params, data, current, cfg)
    assert optimal.predicted_sales_optimal >= objective(naive) - 1e-6
