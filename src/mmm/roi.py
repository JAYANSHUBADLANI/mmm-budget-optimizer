"""Turning model output into money.

The dataset ships impressions and views. It does not ship spend. Every currency figure in
this project therefore rests on an assumed cost per thousand impressions per channel,
which I set in config/config.yaml and justify in docs/cpm_assumptions.md.

I want to be direct about what this does and does not affect, because it is the first
thing anyone relying on these numbers should push on.

What the CPM assumption does not change: which channel has the steepest response curve,
the ranking of channels by incremental effect per impression, the shape of the saturation
curves, and the model fit. Those come out of the data.

What the CPM assumption does change: every ROI number expressed per dollar, and therefore
the optimiser's answer. If my assumed Google CPM is twice the real one, the optimiser
will under invest in search by roughly that factor. The ranking by ROI is only as good as
the relative CPMs across channels, so those relative values are the assumption that
actually carries weight, not the absolute levels.

The mitigation is in scripts/run_phase3.py, which reruns the optimiser across a range of
CPM values and reports how much of the recommendation survives.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DotDict, cpm_rates, load_config
from .models.bayesian_mmm import (
    ModelData,
    PosteriorParams,
    adstock_draws,
    hill_draws,
    media_contributions,
)

logger = logging.getLogger(__name__)


def impressions_to_spend(
    impressions: np.ndarray | pd.Series, cpm: float
) -> np.ndarray | pd.Series:
    """Convert an impression count into currency at an assumed CPM.

    CPM is cost per thousand, so spend equals impressions divided by 1000 times CPM.
    """
    return impressions / 1000.0 * cpm


def spend_frame(
    df: pd.DataFrame, channels: list[str], cfg: DotDict | None = None
) -> pd.DataFrame:
    """Add a _spend column per channel alongside the impression column."""
    cfg = cfg or load_config()
    rates = cpm_rates(cfg)
    out = df.copy()
    for channel in channels:
        rate = rates.get(channel)
        if rate is None:
            logger.warning("No CPM configured for %s, treating spend as zero", channel)
            rate = 0.0
        out[f"{channel}_spend"] = impressions_to_spend(out[channel], rate)
    out["total_spend"] = out[[f"{c}_spend" for c in channels]].sum(axis=1)
    return out


def observed_spend_by_channel(
    df: pd.DataFrame, channels: list[str], cfg: DotDict | None = None
) -> pd.Series:
    """Total assumed spend per channel across the whole panel."""
    cfg = cfg or load_config()
    rates = cpm_rates(cfg)
    return pd.Series(
        {c: float(impressions_to_spend(df[c].sum(), rates.get(c, 0.0))) for c in channels},
        name="spend",
    )


def check_spend_plausibility(
    df: pd.DataFrame,
    channels: list[str],
    target_col: str = "Sales",
    cfg: DotDict | None = None,
    lower: float = 0.02,
    upper: float = 0.25,
) -> dict[str, float | bool | str]:
    """Sanity check the CPM assumptions against total revenue.

    I added this after the first run, where the assumed rates implied a media budget worth
    well under one percent of revenue. Real advertisers spend somewhere between about two
    and twenty five percent of revenue on media depending on the sector, so a figure far
    outside that band is not a finding about the business, it is evidence that my assumed
    CPMs are wrong by an order of magnitude.

    This matters because every ROI number scales inversely with assumed spend. If the
    denominator is a hundred times too small, the ROI is a hundred times too large, and a
    return of six hundred to one will be dismissed on sight by anyone who has worked in
    media, correctly.
    """
    cfg = cfg or load_config()
    total_spend = float(observed_spend_by_channel(df, channels, cfg).sum())
    total_revenue = float(df[target_col].sum())
    share = total_spend / total_revenue if total_revenue else np.nan
    plausible = bool(lower <= share <= upper)
    if not plausible:
        logger.warning(
            "Assumed media spend is %.2f percent of revenue, outside the %0.0f to %0.0f "
            "percent range typical of real advertisers. Scale the CPM values in "
            "config/config.yaml before quoting any ROI figure.",
            share * 100,
            lower * 100,
            upper * 100,
        )
    return {
        "total_assumed_spend": total_spend,
        "total_revenue": total_revenue,
        "spend_share_of_revenue": share,
        "plausible": plausible,
        "guidance": (
            "ROI scales inversely with assumed spend. A share outside the typical band "
            "means the CPM assumptions, not the model, are driving the headline number."
        ),
    }


def suggested_cpm_scaling(
    df: pd.DataFrame,
    channels: list[str],
    target_col: str = "Sales",
    cfg: DotDict | None = None,
    target_share: float = 0.10,
) -> float:
    """Return the factor that would put total assumed spend at target_share of revenue.

    I report this rather than applying it silently. Rescaling every CPM by a common factor
    leaves the relative costs across channels untouched, so the channel ranking by ROI does
    not move, only the absolute level does. That is worth saying out loud, because it means
    the reallocation recommendation is robust to getting the overall level wrong, and
    sensitive only to getting the relative costs wrong.
    """
    cfg = cfg or load_config()
    total_spend = float(observed_spend_by_channel(df, channels, cfg).sum())
    total_revenue = float(df[target_col].sum())
    if total_spend <= 0:
        return 1.0
    return float(target_share * total_revenue / total_spend)


def hdi(samples: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    """Highest density interval of a 1d sample, computed without an ArviZ dependency."""
    x = np.sort(np.asarray(samples).ravel())
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    interval_idx = int(np.floor(prob * n))
    if interval_idx < 1:
        return (float(x[0]), float(x[-1]))
    widths = x[interval_idx:] - x[: n - interval_idx]
    lower_pos = int(np.argmin(widths))
    return (float(x[lower_pos]), float(x[lower_pos + interval_idx]))


def contribution_table(
    params: PosteriorParams,
    data: ModelData,
    max_lag: int = 8,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Total incremental sales attributed to each channel, with a credible interval.

    Contribution is summed over weeks and divisions and put back into original sales units
    by multiplying through the target scaler.
    """
    contrib = media_contributions(params, data.X, max_lag)  # (S, T, D, C)
    totals = contrib.sum(axis=(1, 2)) * data.target_scale  # (S, C)

    rows = []
    for k, channel in enumerate(data.channels):
        samples = totals[:, k]
        lo, hi = hdi(samples, hdi_prob)
        rows.append(
            {
                "channel": channel,
                "contribution_mean": float(samples.mean()),
                "contribution_median": float(np.median(samples)),
                "contribution_hdi_lower": lo,
                "contribution_hdi_upper": hi,
            }
        )
    out = pd.DataFrame(rows)
    total_sales = float(np.nansum(data.y) * data.target_scale)
    out["share_of_total_sales"] = out["contribution_mean"] / total_sales
    out["total_observed_sales"] = total_sales
    return out


def roi_table(
    params: PosteriorParams,
    data: ModelData,
    df: pd.DataFrame,
    cfg: DotDict | None = None,
    max_lag: int = 8,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """ROI per channel, defined as incremental sales per unit of assumed spend.

    The interval comes from dividing every posterior draw of contribution by the same
    fixed spend figure, so it reflects model uncertainty only. It does not reflect
    uncertainty in the CPM assumption, which is a separate and, here, unquantified source
    of error. I handle that with the sensitivity sweep rather than by pretending the
    interval covers it.
    """
    cfg = cfg or load_config()
    contrib = media_contributions(params, data.X, max_lag)
    totals = contrib.sum(axis=(1, 2)) * data.target_scale  # (S, C)
    spend = observed_spend_by_channel(df, data.channels, cfg)

    rows = []
    for k, channel in enumerate(data.channels):
        channel_spend = float(spend.get(channel, 0.0))
        if channel_spend <= 0:
            continue
        roi_samples = totals[:, k] / channel_spend
        lo, hi = hdi(roi_samples, hdi_prob)
        rows.append(
            {
                "channel": channel,
                "assumed_spend": channel_spend,
                "contribution_mean": float(totals[:, k].mean()),
                "roi_mean": float(roi_samples.mean()),
                "roi_median": float(np.median(roi_samples)),
                "roi_hdi_lower": lo,
                "roi_hdi_upper": hi,
                "prob_roi_above_1": float((roi_samples > 1.0).mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("roi_mean", ascending=False).reset_index(drop=True)
    out["spend_share"] = out["assumed_spend"] / out["assumed_spend"].sum()
    return out


def marginal_roi(
    params: PosteriorParams,
    data: ModelData,
    df: pd.DataFrame,
    cfg: DotDict | None = None,
    max_lag: int = 8,
    bump_pct: float = 0.01,
) -> pd.DataFrame:
    """Return on the next dollar, not the average dollar.

    This is the number that should drive reallocation. Average ROI tells you what a channel
    has delivered across its whole spend range, including the cheap early impressions.
    Marginal ROI tells you what one more dollar buys at the current operating point, which
    on a saturating curve is a much smaller number. Reallocating on average ROI is the most
    common way a marketing mix model produces a recommendation that fails when executed.
    """
    cfg = cfg or load_config()
    rates = cpm_rates(cfg)
    base = media_contributions(params, data.X, max_lag).sum(axis=(1, 2)) * data.target_scale

    rows = []
    for k, channel in enumerate(data.channels):
        bumped = data.X.copy()
        bumped[:, :, k] *= 1.0 + bump_pct
        new = media_contributions(params, bumped, max_lag).sum(axis=(1, 2)) * data.target_scale
        delta_sales = new[:, k] - base[:, k]

        current_impressions = float(df[channel].sum())
        delta_spend = float(
            impressions_to_spend(current_impressions * bump_pct, rates.get(channel, 0.0))
        )
        if delta_spend <= 0:
            continue
        mroi_samples = delta_sales / delta_spend
        lo, hi = hdi(mroi_samples)
        rows.append(
            {
                "channel": channel,
                "marginal_roi_mean": float(mroi_samples.mean()),
                "marginal_roi_hdi_lower": lo,
                "marginal_roi_hdi_upper": hi,
                "prob_marginal_roi_above_1": float((mroi_samples > 1.0).mean()),
                "bump_pct": bump_pct,
            }
        )
    return pd.DataFrame(rows).sort_values("marginal_roi_mean", ascending=False).reset_index(
        drop=True
    )


def saturation_curve(
    params: PosteriorParams,
    data: ModelData,
    channel: str,
    cfg: DotDict | None = None,
    multipliers: np.ndarray | None = None,
    max_lag: int = 8,
) -> pd.DataFrame:
    """Response curve for one channel, sweeping its volume while holding the others fixed.

    Useful in the write up because a stakeholder can read the point where the curve flattens
    directly off the chart, which is a far more actionable statement than a single ROI number.
    """
    cfg = cfg or load_config()
    rates = cpm_rates(cfg)
    k = data.channels.index(channel)
    multipliers = (
        np.linspace(0.0, 2.0, 21) if multipliers is None else np.asarray(multipliers)
    )

    base_impressions = float(
        (data.X[:, :, k] * data.media_scalers[channel]).sum()
    )
    rows = []
    for m in multipliers:
        scaled = data.X.copy()
        scaled[:, :, k] *= m
        adstocked = adstock_draws(scaled, params.decay, max_lag)
        saturated = hill_draws(adstocked, params.half_sat, params.slope)
        contrib = (
            saturated[:, :, :, k] * params.beta[:, k, :][:, None, :]
        ).sum(axis=(1, 2)) * data.target_scale
        lo, hi = hdi(contrib)
        rows.append(
            {
                "channel": channel,
                "multiplier": float(m),
                "impressions": base_impressions * float(m),
                "assumed_spend": float(
                    impressions_to_spend(base_impressions * float(m), rates.get(channel, 0.0))
                ),
                "contribution_mean": float(contrib.mean()),
                "contribution_hdi_lower": lo,
                "contribution_hdi_upper": hi,
            }
        )
    return pd.DataFrame(rows)
