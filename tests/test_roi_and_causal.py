"""Tests for the money translation layer and the causal checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm.causal import detect_spend_shocks, select_holdout_divisions
from mmm.roi import (
    check_spend_plausibility,
    hdi,
    impressions_to_spend,
    observed_spend_by_channel,
    spend_frame,
    suggested_cpm_scaling,
)


def test_cpm_conversion_is_per_thousand():
    """A CPM of 10 over 1000 impressions must cost exactly 10."""
    assert impressions_to_spend(1000, 10.0) == pytest.approx(10.0)
    assert impressions_to_spend(2_500_000, 7.2) == pytest.approx(18_000.0)


def test_spend_frame_adds_a_column_per_channel(tiny_panel, cfg):
    channels = ["Google_Impressions", "Facebook_Impressions"]
    out = spend_frame(tiny_panel, channels, cfg)
    for channel in channels:
        assert f"{channel}_spend" in out.columns
    assert out["total_spend"].gt(0).all()


def test_observed_spend_matches_a_hand_calculation(tiny_panel, cfg):
    rate = float(cfg["cpm"]["Facebook_Impressions"]["value"])
    expected = tiny_panel["Facebook_Impressions"].sum() / 1000.0 * rate
    got = observed_spend_by_channel(tiny_panel, ["Facebook_Impressions"], cfg)
    assert float(got["Facebook_Impressions"]) == pytest.approx(expected)


def test_organic_carries_no_media_cost(cfg):
    assert float(cfg["cpm"]["Organic_Views"]["value"]) == 0.0


def test_spend_plausibility_flags_an_implausible_share(tiny_panel, cfg):
    """The check exists to catch CPM assumptions that are off by orders of magnitude."""
    channels = ["Google_Impressions", "Facebook_Impressions"]
    inflated = tiny_panel.copy()
    inflated["Sales"] = inflated["Sales"] * 10_000  # spend becomes a rounding error
    result = check_spend_plausibility(inflated, channels, "Sales", cfg)
    assert result["plausible"] is False
    assert result["spend_share_of_revenue"] < 0.02


def test_suggested_scaling_hits_the_target_share(tiny_panel, cfg):
    channels = ["Google_Impressions", "Facebook_Impressions"]
    factor = suggested_cpm_scaling(tiny_panel, channels, "Sales", cfg, target_share=0.10)
    base = float(observed_spend_by_channel(tiny_panel, channels, cfg).sum())
    revenue = float(tiny_panel["Sales"].sum())
    assert (base * factor) / revenue == pytest.approx(0.10)


def test_hdi_covers_the_requested_mass():
    rng = np.random.default_rng(11)
    samples = rng.normal(0, 1, 20_000)
    lo, hi = hdi(samples, prob=0.94)
    covered = float(((samples >= lo) & (samples <= hi)).mean())
    assert covered == pytest.approx(0.94, abs=0.02)
    assert lo < 0 < hi


def test_hdi_is_narrower_than_the_full_range():
    samples = np.concatenate([np.zeros(1000), np.array([1000.0])])
    lo, hi = hdi(samples, prob=0.90)
    assert hi < 1000.0  # the outlier sits outside the highest density region


def test_shock_detection_finds_an_injected_step():
    dates = pd.date_range("2018-01-06", periods=60, freq="7D")
    rows = []
    for div in ["A", "B"]:
        base = np.full(60, 1000.0)
        if div == "A":
            base[30:] = 5000.0  # a clear step up
        for i, week in enumerate(dates):
            rows.append(
                {
                    "Division": div,
                    "Calendar_Week": week,
                    "Google_Impressions": base[i],
                    "Sales": 100000.0,
                }
            )
    df = pd.DataFrame(rows)
    shocks = detect_spend_shocks(df, "Google_Impressions", threshold_sd=1.0)
    assert "A" in set(shocks["entity"])
    row = shocks[shocks["entity"] == "A"].iloc[0]
    assert row["direction"] == "increase"
    assert 25 <= row["shock_week_index"] <= 35


def test_shock_detection_ignores_a_flat_series():
    dates = pd.date_range("2018-01-06", periods=60, freq="7D")
    df = pd.DataFrame(
        {
            "Division": "A",
            "Calendar_Week": dates,
            "Google_Impressions": 1000.0,
            "Sales": 100000.0,
        }
    )
    assert detect_spend_shocks(df, "Google_Impressions").empty


def test_holdout_divisions_are_selected_deterministically(tiny_panel):
    channels = ["Google_Impressions", "Facebook_Impressions"]
    first = select_holdout_divisions(tiny_panel, channels, n_holdout=2)
    second = select_holdout_divisions(tiny_panel, channels, n_holdout=2)
    assert first == second
    assert len(first) == 2
    assert set(first) <= set(tiny_panel["Division"].astype(str))
