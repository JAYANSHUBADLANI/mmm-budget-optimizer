"""Tests for the media transforms.

The most valuable test in this file is the one asserting that the NumPy adstock and the
PyTensor adstock produce the same numbers. The model is fitted with the PyTensor version
and the optimiser searches over the NumPy version, so if they ever diverge the optimiser
would be maximising a function the model never fitted, and nothing else in the test suite
would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm.features import (
    carryover_half_life,
    fourier_terms,
    geometric_adstock,
    geometric_adstock_panel,
    hill_saturation,
    scale_media,
)
from mmm.models.bayesian_mmm import adstock_draws, hill_draws


def test_adstock_with_zero_decay_is_the_identity():
    x = np.array([1.0, 5.0, 0.0, 3.0])
    np.testing.assert_allclose(geometric_adstock(x, decay=0.0), x)


def test_adstock_preserves_total_volume_when_normalised():
    """Normalised weights sum to one, so the transform is a weighted average, not a gain.

    Total is not exactly preserved at the start of the series because the first weeks have
    no history to carry over from, so I compare well inside the series.
    """
    x = np.full(50, 10.0)
    out = geometric_adstock(x, decay=0.6, max_lag=8, normalise=True)
    np.testing.assert_allclose(out[20:], 10.0, rtol=1e-9)


def test_adstock_spreads_a_single_spike_forward():
    x = np.zeros(10)
    x[2] = 100.0
    out = geometric_adstock(x, decay=0.5, max_lag=4)
    assert out[2] > out[3] > out[4] > out[5]
    assert out[1] == 0.0  # nothing leaks backwards in time
    assert out[2:].sum() == pytest.approx(100.0, rel=1e-9)


def test_adstock_rejects_out_of_range_decay():
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(5), decay=1.0)
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(5), decay=-0.1)


def test_adstock_never_crosses_entity_boundaries(tiny_panel):
    """A lag applied to a stacked panel without grouping would leak across divisions."""
    decays = {"Google_Impressions": 0.8}
    out = geometric_adstock_panel(
        tiny_panel, ["Google_Impressions"], "Division", "Calendar_Week", decays
    )
    for div in out["Division"].unique():
        sub = out[out["Division"] == div].sort_values("Calendar_Week")
        raw = sub["Google_Impressions"].to_numpy()
        expected = geometric_adstock(raw, 0.8, max_lag=8)
        np.testing.assert_allclose(sub["Google_Impressions_adstock"].to_numpy(), expected)


def test_numpy_and_pytensor_adstock_agree():
    pytensor = pytest.importorskip("pytensor")
    import pytensor.tensor as pt

    from mmm.features import pt_geometric_adstock

    rng = np.random.default_rng(0)
    X = rng.random((25, 3, 2))
    decay = np.array([0.25, 0.75])

    reference = np.zeros_like(X)
    for d in range(X.shape[1]):
        for c in range(X.shape[2]):
            reference[:, d, c] = geometric_adstock(X[:, d, c], decay[c], max_lag=8)

    x_sym, d_sym = pt.tensor3("x"), pt.vector("d")
    fn = pytensor.function([x_sym, d_sym], pt_geometric_adstock(x_sym, d_sym, max_lag=8))
    np.testing.assert_allclose(fn(X, decay), reference, atol=1e-12)


def test_vectorised_adstock_matches_the_scalar_version():
    rng = np.random.default_rng(1)
    X = rng.random((30, 4, 3))
    decay = np.array([[0.1, 0.5, 0.9]])
    out = adstock_draws(X, decay, max_lag=8)[0]
    for d in range(X.shape[1]):
        for c in range(X.shape[2]):
            expected = geometric_adstock(X[:, d, c], decay[0, c], max_lag=8)
            np.testing.assert_allclose(out[:, d, c], expected, atol=1e-12)


def test_hill_is_one_half_at_the_half_saturation_point():
    assert hill_saturation(np.array([0.4]), half_sat=0.4, slope=1.0)[0] == pytest.approx(0.5)
    assert hill_saturation(np.array([0.4]), half_sat=0.4, slope=2.5)[0] == pytest.approx(0.5)


def test_hill_is_monotone_increasing_and_bounded():
    x = np.linspace(0, 100, 500)
    out = hill_saturation(x, half_sat=1.0, slope=1.3)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.0
    assert out.max() < 1.0


def test_hill_exhibits_diminishing_returns():
    """The increment from doubling spend must shrink as spend grows.

    This is the property the whole budget argument rests on, so it is worth asserting
    rather than assuming.
    """
    first = hill_saturation(np.array([1.0]), 1.0, 1.0) - hill_saturation(np.array([0.5]), 1.0, 1.0)
    second = hill_saturation(np.array([2.0]), 1.0, 1.0) - hill_saturation(np.array([1.0]), 1.0, 1.0)
    assert second < first


def test_hill_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        hill_saturation(np.ones(3), half_sat=0.0)
    with pytest.raises(ValueError):
        hill_saturation(np.ones(3), half_sat=1.0, slope=0.0)


def test_numpy_and_pytensor_hill_agree():
    pytensor = pytest.importorskip("pytensor")
    import pytensor.tensor as pt

    from mmm.features import pt_hill_saturation

    rng = np.random.default_rng(2)
    X = rng.random((15, 2, 3))
    half_sat = np.array([0.3, 0.5, 0.7])
    slope = np.array([0.8, 1.0, 1.6])

    reference = np.zeros_like(X)
    for c in range(X.shape[2]):
        reference[:, :, c] = hill_saturation(X[:, :, c], half_sat[c], slope[c])

    x_sym = pt.tensor3("x")
    fn = pytensor.function(
        [x_sym],
        pt_hill_saturation(x_sym, pt.as_tensor_variable(half_sat), pt.as_tensor_variable(slope)),
    )
    np.testing.assert_allclose(fn(X), reference, atol=1e-12)
    np.testing.assert_allclose(
        hill_draws(X[None, ...], half_sat[None, :], slope[None, :])[0], reference, atol=1e-12
    )


def test_fourier_terms_are_periodic():
    t = np.arange(0, 105)
    terms = fourier_terms(t, period=52.0, order=2)
    assert terms.shape == (105, 4)
    np.testing.assert_allclose(
        terms.iloc[0].to_numpy(), terms.iloc[52].to_numpy(), atol=1e-9
    )


def test_scaling_maps_zero_to_zero(tiny_panel):
    """Media scaling must not shift the origin, a dark week has to stay dark."""
    frame = tiny_panel.copy()
    frame.loc[0, "Google_Impressions"] = 0.0
    scaled, scalers = scale_media(frame, ["Google_Impressions"])
    assert scaled.loc[0, "Google_Impressions"] == 0.0
    assert scaled["Google_Impressions"].max() == pytest.approx(1.0)
    assert scalers["Google_Impressions"] == pytest.approx(frame["Google_Impressions"].max())


def test_half_life_matches_the_decay_rate():
    assert carryover_half_life(0.5) == pytest.approx(1.0)
    assert carryover_half_life(0.25) == pytest.approx(0.5)
    assert carryover_half_life(0.0) == 0.0
