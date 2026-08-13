"""Media transforms: adstock, saturation, seasonality, scaling.

Two things every marketing mix model has to represent, and the reason a plain regression
on raw impressions is the wrong tool:

1. Carryover. An impression served this week keeps working next week. I model that with
   geometric adstock, a one parameter decay per channel.
2. Diminishing returns. The tenth thousand impressions is worth less than the first. I
   model that with a Hill curve, which is monotone increasing, saturating, and has an
   interpretable half saturation point.

Both transforms appear twice in this file: a NumPy version used for feature engineering,
diagnostics and the optimiser, and a PyTensor version used inside the PyMC model so the
transform parameters are learned rather than fixed. I test the two against each other in
tests/test_features.py, because a silent divergence between them would mean the optimiser
is maximising a different function from the one I fitted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NumPy implementations
# ---------------------------------------------------------------------------
def geometric_adstock(
    x: np.ndarray, decay: float, max_lag: int = 8, normalise: bool = True
) -> np.ndarray:
    """Geometric carryover with a finite lag window.

    adstocked[t] = sum over l of w[l] * x[t - l], with w[l] proportional to decay ** l.

    normalise divides the weights by their sum. With normalisation the transform preserves
    the scale of the input, so the saturation and coefficient parameters do not have to
    absorb a scale shift when decay moves. That decoupling is what stops the sampler from
    trading decay against the coefficient along a ridge.
    """
    x = np.asarray(x, dtype=float)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), received {decay}")
    weights = decay ** np.arange(max_lag + 1, dtype=float)
    if normalise:
        weights = weights / weights.sum()
    out = np.zeros_like(x)
    for lag, w in enumerate(weights):
        if w == 0.0:
            continue
        shifted = np.concatenate([np.zeros(lag), x[: len(x) - lag]]) if lag else x
        out = out + w * shifted
    return out


def geometric_adstock_panel(
    df: pd.DataFrame,
    columns: list[str],
    entity_col: str,
    date_col: str,
    decays: dict[str, float],
    max_lag: int = 8,
    suffix: str = "_adstock",
) -> pd.DataFrame:
    """Apply adstock within each entity, never across the boundary between entities.

    Applying a lag operator to a stacked panel without grouping is one of the easiest ways
    to produce a model that looks fine and is wrong, because the first weeks of one
    division inherit carryover from the last weeks of another.
    """
    out = df.sort_values([entity_col, date_col]).copy()
    for col in columns:
        decay = float(decays.get(col, 0.5))
        out[f"{col}{suffix}"] = (
            out.groupby(entity_col, sort=False)[col]
            .transform(lambda s, d=decay: geometric_adstock(s.to_numpy(), d, max_lag))
        )
    return out


def hill_saturation(
    x: np.ndarray, half_sat: float, slope: float = 1.0
) -> np.ndarray:
    """Hill curve, returns a value in [0, 1).

    f(x) = x ** slope / (half_sat ** slope + x ** slope)

    At x equal to half_sat the response is exactly 0.5, which makes half_sat readable as
    "the volume at which this channel has delivered half of everything it can deliver".
    slope above 1 gives an S shape with a slow start, slope below 1 saturates immediately.
    """
    x = np.asarray(x, dtype=float)
    if half_sat <= 0:
        raise ValueError(f"half_sat must be positive, received {half_sat}")
    if slope <= 0:
        raise ValueError(f"slope must be positive, received {slope}")
    xs = np.clip(x, 0.0, None) ** slope
    return xs / (half_sat**slope + xs)


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    """Alternative saturation used as a robustness check in the sensitivity analysis."""
    x = np.asarray(x, dtype=float)
    return (1.0 - np.exp(-lam * x)) / (1.0 + np.exp(-lam * x))


def fourier_terms(
    t: np.ndarray, period: float = 52.18, order: int = 3
) -> pd.DataFrame:
    """Fourier basis for annual seasonality.

    Three harmonics on a weekly series is enough to capture a smooth annual shape without
    the model absorbing genuine media effects into seasonality, which is the usual failure
    mode when you throw 52 week dummies at two years of data.
    """
    t = np.asarray(t, dtype=float)
    cols: dict[str, np.ndarray] = {}
    for k in range(1, order + 1):
        cols[f"seas_sin_{k}"] = np.sin(2.0 * np.pi * k * t / period)
        cols[f"seas_cos_{k}"] = np.cos(2.0 * np.pi * k * t / period)
    return pd.DataFrame(cols)


def scale_media(
    df: pd.DataFrame, columns: list[str], method: str = "max"
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scale media columns to a comparable range and return the divisors.

    I scale by the column maximum rather than standardising, because the transforms need a
    non negative input. Zero has to stay zero: a week with no spend must map to no effect,
    and subtracting a mean would break that.
    """
    out = df.copy()
    scalers: dict[str, float] = {}
    for col in columns:
        if method == "max":
            denom = float(np.nanmax(df[col].to_numpy(dtype=float)))
        elif method == "mean":
            denom = float(np.nanmean(df[col].to_numpy(dtype=float)))
        else:
            raise ValueError(f"Unknown scaling method {method}")
        denom = denom if denom > 0 else 1.0
        scalers[col] = denom
        out[col] = df[col].astype(float) / denom
    return out, scalers


def build_feature_frame(
    df: pd.DataFrame,
    media_cols: list[str],
    entity_col: str,
    date_col: str,
    fourier_order: int = 3,
    period_weeks: float = 52.18,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assemble the modelling frame: scaled media, seasonality, trend, indices.

    Note what is deliberately absent. I do not adstock or saturate here. Those transforms
    have parameters, and fixing them before fitting would mean assuming the answer to one
    of the questions the model is supposed to answer. The Bayesian model learns them from
    the data. The NumPy transforms above are for the optimiser and for diagnostics, where
    the parameters are already known because they came out of the posterior.
    """
    out = df.sort_values([entity_col, date_col]).reset_index(drop=True)
    out, scalers = scale_media(out, media_cols, method="max")

    if "time_idx" not in out.columns:
        weeks = sorted(out[date_col].unique())
        out["time_idx"] = out[date_col].map({w: i for i, w in enumerate(weeks)})

    seas = fourier_terms(out["time_idx"].to_numpy(), period=period_weeks, order=fourier_order)
    out = pd.concat([out, seas], axis=1)

    n_periods = max(int(out["time_idx"].max()), 1)
    out["trend"] = out["time_idx"] / n_periods

    return out, scalers


def carryover_half_life(decay: float) -> float:
    """Weeks until the carryover of a single impression falls to half its initial value.

    Half life communicates better than a decay rate in a stakeholder conversation, so I
    report both.
    """
    if decay <= 0:
        return 0.0
    if decay >= 1:
        return float("inf")
    return float(np.log(0.5) / np.log(decay))


# ---------------------------------------------------------------------------
# PyTensor implementations, used inside the PyMC model
# ---------------------------------------------------------------------------
def pt_geometric_adstock(x, decay, max_lag: int = 8, normalise: bool = True):
    """Adstock over the time axis of a (time, entity, channel) tensor.

    Implemented as an explicit weighted sum of shifted copies rather than a scan, because
    the lag window is short and fixed, and the unrolled version compiles to a much faster
    graph than a scan does for max_lag around eight.
    """
    import pytensor.tensor as pt

    lags = pt.arange(max_lag + 1)
    weights = decay ** lags.dimshuffle(0, "x")  # (lag, 1) broadcast over channels
    if normalise:
        weights = weights / pt.sum(weights, axis=0, keepdims=True)

    padded = pt.concatenate([pt.zeros_like(x[:max_lag]), x], axis=0)
    n_time = x.shape[0]
    terms = []
    for lag in range(max_lag + 1):
        start = max_lag - lag
        terms.append(padded[start : start + n_time] * weights[lag])
    return sum(terms)


def pt_hill_saturation(x, half_sat, slope):
    """Hill curve in PyTensor, matching hill_saturation above."""
    import pytensor.tensor as pt

    xs = pt.power(pt.clip(x, 1e-9, np.inf), slope)
    return xs / (pt.power(half_sat, slope) + xs)
