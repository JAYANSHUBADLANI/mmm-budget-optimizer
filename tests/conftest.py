"""Shared fixtures. The synthetic panel here is tiny on purpose so tests stay fast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHANNELS = [
    "Google_Impressions",
    "Facebook_Impressions",
    "Email_Impressions",
    "Affiliate_Impressions",
    "Paid_Views",
]


@pytest.fixture(scope="session")
def cfg():
    from mmm.config import load_config

    return load_config()


@pytest.fixture
def tiny_panel() -> pd.DataFrame:
    """Four divisions, thirty weeks, clean and balanced."""
    rng = np.random.default_rng(7)
    divisions = list("ABCD")
    dates = pd.date_range("2018-01-06", periods=30, freq="7D")
    rows = []
    for div in divisions:
        scale = rng.uniform(0.8, 1.4)
        for week in dates:
            media = {c: float(rng.lognormal(10, 0.4) * scale) for c in CHANNELS}
            organic = float(rng.lognormal(9.5, 0.3) * scale)
            sales = 500_000 * scale + 0.02 * sum(media.values()) + rng.normal(0, 5_000)
            rows.append(
                {
                    "Division": div,
                    "Calendar_Week": week,
                    "Organic_Views": organic,
                    **media,
                    "Overall_Views": media["Paid_Views"] + organic,
                    "Sales": float(max(sales, 1.0)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def panel_with_duplicate_division(tiny_panel: pd.DataFrame) -> pd.DataFrame:
    """The Division Z bug in miniature: one division repeated in full."""
    dupe = tiny_panel[tiny_panel["Division"] == "D"].copy()
    return pd.concat([tiny_panel, dupe], ignore_index=True)


@pytest.fixture
def panel_with_conflicting_rows(tiny_panel: pd.DataFrame) -> pd.DataFrame:
    """A key collision where the measures disagree, which must not be silently deduped."""
    conflict = tiny_panel[tiny_panel["Division"] == "D"].head(3).copy()
    conflict["Sales"] = conflict["Sales"] * 1.5
    return pd.concat([tiny_panel, conflict], ignore_index=True)
