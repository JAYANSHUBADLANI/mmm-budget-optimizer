"""Generalisation check on the secondary dataset.

The secondary Kaggle dataset has 300 rows and six channels that barely overlap with the
primary set: TV, billboards, Google Ads, social media, influencer marketing and affiliate
marketing. Running the same pipeline on it tests whether the methodology travels, or
whether it was quietly tuned to one file.

I want to be clear about what this can support. Three hundred rows and six channels is a
small, single entity dataset. The hierarchy has nothing to pool over, the saturation
parameters are weakly identified, and the credible intervals should come out wide. If they
come out narrow, that is a red flag about my priors rather than a strong result.

So the claim I make from this is narrow and specific: the code runs end to end on a
different channel mix without special casing, and the estimates it produces are not
absurd. That is a software generalisation claim, not a scientific one, and the write up
says so.

The two datasets are never merged. Different products, different channels, different time
ranges, no shared key. Stacking them would create a table describing no real process.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DotDict, load_config

logger = logging.getLogger(__name__)

KNOWN_SPEND_HINTS = ("tv", "billboard", "google", "social", "influencer", "affiliate", "ads")
KNOWN_TARGET_HINTS = ("product_sold", "sales", "revenue", "units")


def resolve_secondary_columns(df: pd.DataFrame) -> tuple[list[str], str, str | None]:
    """Infer the channel columns, the target and the date column by name and dtype."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    target = next(
        (c for c in df.columns if any(h in c.lower() for h in KNOWN_TARGET_HINTS)), None
    )
    if target is None:
        # Fall back to the numeric column with the highest mean, which in an advertising
        # table is almost always the outcome rather than a channel budget.
        target = max(numeric, key=lambda c: df[c].mean())
        logger.warning("Guessed %s as the target column for the secondary dataset", target)

    date_col = next(
        (c for c in df.columns if "date" in c.lower() or "day" in c.lower() or "week" in c.lower()),
        None,
    )

    channels = [
        c
        for c in numeric
        if c != target and any(h in c.lower() for h in KNOWN_SPEND_HINTS)
    ]
    if not channels:
        channels = [c for c in numeric if c != target]
    return channels, target, date_col


def prepare_secondary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Shape the secondary dataset into the single entity panel the model expects."""
    channels, target, date_col = resolve_secondary_columns(df)
    out = df.copy()

    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        if out[date_col].isna().all():
            date_col = None

    if date_col is None:
        # No usable date column. I create a synthetic weekly index rather than dropping the
        # time structure entirely, and I record that this is synthetic in the notes the
        # phase script writes out, because the seasonality term then means nothing.
        out = out.reset_index(drop=True)
        out["Calendar_Week"] = pd.date_range("2020-01-06", periods=len(out), freq="W-MON")
        logger.warning(
            "No date column found in the secondary dataset. Assigned a synthetic weekly "
            "index. Seasonality estimates from this run are not interpretable."
        )
    else:
        out = out.rename(columns={date_col: "Calendar_Week"}).sort_values("Calendar_Week")

    out = out.rename(columns={target: "Sales"})
    out["Division"] = "ALL"
    keep = ["Division", "Calendar_Week", "Sales"] + channels
    out = out[[c for c in keep if c in out.columns]].reset_index(drop=True)
    for c in channels:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out, channels


def run_generalization(
    cfg: DotDict | None = None, smoke: bool = True
) -> dict[str, pd.DataFrame | dict]:
    """Run the full pipeline on the secondary dataset and report what came out."""
    from .backtest import evaluate
    from .io import load_secondary_raw
    from .models.bayesian_mmm import HierarchicalMMM, prepare_model_data
    from .roi import contribution_table

    cfg = cfg or load_config()
    raw = load_secondary_raw(cfg)
    panel, channels = prepare_secondary(raw)
    logger.info("Secondary panel: %d rows, channels %s", len(panel), channels)

    n_weeks = panel["Calendar_Week"].nunique()
    holdout = min(int(cfg["backtest"]["holdout_weeks"]), max(n_weeks // 5, 4))

    data = prepare_model_data(
        panel,
        media_cols=channels,
        control_cols=[],
        fourier_order=int(cfg["features"]["seasonality"]["fourier_order"]),
        period_weeks=float(cfg["features"]["seasonality"]["period_weeks"]),
        train_weeks=n_weeks - holdout,
    )
    model = HierarchicalMMM(cfg=cfg, smoke=smoke)
    model.fit(data)
    params = model.posterior_params(max_draws=300)

    preds = model.predict(data).mean(axis=0)
    actual = data.y * data.target_scale
    train_end = n_weeks - holdout
    metrics = {
        "train": evaluate(actual[:train_end], preds[:train_end]),
        "holdout": evaluate(actual[train_end:], preds[train_end:]),
    }
    contrib = contribution_table(params, data, max_lag=int(cfg["features"]["adstock"]["max_lag"]))

    # The interval width is the number I care about here. On 300 rows with six channels I
    # expect wide intervals, and I would distrust a result that came out tight.
    contrib["hdi_width_relative"] = (
        contrib["contribution_hdi_upper"] - contrib["contribution_hdi_lower"]
    ) / contrib["contribution_mean"].abs().replace(0, np.nan)

    checks = {
        "pipeline_ran_end_to_end": True,
        "n_channels": len(channels),
        "n_observations": int(len(panel)),
        "holdout_mape": metrics["holdout"].get("mape"),
        "all_contributions_positive": bool((contrib["contribution_mean"] > 0).all()),
        "median_relative_hdi_width": float(contrib["hdi_width_relative"].median()),
        "intervals_appropriately_wide": bool(
            float(contrib["hdi_width_relative"].median()) > 0.2
        ),
        "n_divergences": model.divergences(),
        "smoke_mode": smoke,
        "interpretation": (
            "The claim supported by this run is that the same code fits a different channel "
            "mix without modification. With this few observations the parameter estimates "
            "are weakly identified and should not be quoted as findings."
        ),
    }
    return {
        "contributions": contrib,
        "metrics": pd.DataFrame([{"split": k, **v} for k, v in metrics.items()]),
        "checks": checks,
        "channels": pd.DataFrame({"channel": channels}),
    }
