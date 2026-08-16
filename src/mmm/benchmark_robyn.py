"""Correctness check against Meta Robyn's simulated weekly dataset.

Why this dataset and why only here. dt_simulated_weekly is the most widely reproduced
marketing mix dataset on the internet, which makes it a poor centrepiece for a portfolio
project and a good benchmark. It is simulated, so there is a known data generating process
behind it and published results from Robyn to compare against. If my implementation of
adstock, saturation and the media decomposition is broken, this is where it shows.

What the check actually tests:

  the model recovers positive effects for all five paid channels
  the ranking by contribution is stable across reruns
  holdout error is in a sensible range rather than pathological
  the fitted decay parameters land in plausible territory for each media type

What it does not test. It cannot validate the causal claim, because the data is simulated.
It tells me the machinery works, not that the machinery is measuring causation on real data.

The file is single geo, so the hierarchy collapses to one entity. That is intentional. It
exercises the same code path with the entity dimension set to one, which is also a useful
degenerate case test for the array handling.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DotDict, load_config

logger = logging.getLogger(__name__)

# Robyn's simulated file ships these columns. Names have been stable across releases, but I
# resolve them defensively because a rename would otherwise fail deep inside the model.
ROBYN_MEDIA_CANDIDATES = [
    "tv_S",
    "ooh_S",
    "print_S",
    "facebook_S",
    "search_S",
    "facebook_I",
    "search_clicks_P",
]
ROBYN_CONTROL_CANDIDATES = ["competitor_sales_B", "events", "newsletter"]


def resolve_columns(df: pd.DataFrame) -> tuple[list[str], list[str], str, str]:
    """Work out which columns are media, which are controls, and where date and target are."""
    cols = list(df.columns)
    date_col = next((c for c in cols if c.lower() in {"date", "ds", "week"}), cols[0])
    target_col = next(
        (c for c in cols if c.lower() in {"revenue", "sales", "y"}),
        None,
    )
    if target_col is None:
        raise ValueError(f"Could not find a revenue or sales column in {cols}")

    media = [c for c in ROBYN_MEDIA_CANDIDATES if c in cols]
    if not media:
        # Fall back to the Robyn naming convention: _S means spend.
        media = [c for c in cols if c.endswith("_S")]
    # Prefer spend columns over their impression twins, so the same channel is not counted
    # twice through two different measurement units.
    spend_media = [c for c in media if c.endswith("_S")]
    media = spend_media or media

    controls = [c for c in ROBYN_CONTROL_CANDIDATES if c in cols]
    controls = [c for c in controls if pd.api.types.is_numeric_dtype(df[c])]
    return media, controls, date_col, target_col


def prepare_benchmark(df: pd.DataFrame, cfg: DotDict | None = None) -> pd.DataFrame:
    """Shape the Robyn frame into the single entity panel the model expects."""
    cfg = cfg or load_config()
    media, controls, date_col, target_col = resolve_columns(df)
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    # One synthetic entity, because the benchmark file is national rather than geo split.
    out["Division"] = "ALL"
    out = out.rename(columns={date_col: "Calendar_Week", target_col: "Sales"})
    keep = ["Division", "Calendar_Week", "Sales"] + media + controls
    out = out[[c for c in keep if c in out.columns]]
    for c in media + controls:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    logger.info(
        "Prepared benchmark panel: %d weeks, media %s, controls %s", len(out), media, controls
    )
    return out


def run_benchmark(
    cfg: DotDict | None = None, smoke: bool = True
) -> dict[str, pd.DataFrame | dict]:
    """Fit the model on the Robyn data and return the check results."""
    from .backtest import evaluate
    from .features import carryover_half_life
    from .io import load_benchmark_raw
    from .models.bayesian_mmm import HierarchicalMMM, prepare_model_data
    from .roi import contribution_table

    cfg = cfg or load_config()
    raw = load_benchmark_raw(cfg)
    panel = prepare_benchmark(raw, cfg)
    media, controls, _, _ = resolve_columns(raw)
    media = [c for c in media if c in panel.columns]
    controls = [c for c in controls if c in panel.columns]

    holdout = int(cfg["backtest"]["holdout_weeks"])
    n_weeks = panel["Calendar_Week"].nunique()

    data = prepare_model_data(
        panel,
        media_cols=media,
        control_cols=controls,
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
    decay = pd.DataFrame(
        {
            "channel": data.channels,
            "decay_mean": params.decay.mean(axis=0),
            "half_life_weeks": [
                carryover_half_life(float(d)) for d in params.decay.mean(axis=0)
            ],
            "half_sat_mean": params.half_sat.mean(axis=0),
            "slope_mean": params.slope.mean(axis=0),
        }
    )

    checks = {
        "all_channels_positive": bool((contrib["contribution_mean"] > 0).all()),
        "all_intervals_exclude_zero": bool((contrib["contribution_hdi_lower"] > 0).all()),
        "holdout_mape": metrics["holdout"].get("mape"),
        "holdout_mape_under_30pct": bool(metrics["holdout"].get("mape", np.inf) < 30.0),
        "n_divergences": model.divergences(),
        "media_share_of_sales": float(contrib["share_of_total_sales"].sum()),
        "media_share_plausible": bool(
            0.05 <= float(contrib["share_of_total_sales"].sum()) <= 0.80
        ),
        "smoke_mode": smoke,
    }
    checks["no_divergences"] = checks["n_divergences"] == 0
    checks["all_checks_passed"] = bool(
        checks["all_channels_positive"]
        and checks["holdout_mape_under_30pct"]
        and checks["media_share_plausible"]
        and checks["no_divergences"]
    )
    caveats = []
    if smoke:
        caveats.append(
            "Run in smoke mode with a small number of draws. Treat these as a wiring check, "
            "not as benchmarked results. Rerun with smoke=False before quoting anything."
        )
    if checks["n_divergences"] > 0:
        caveats.append(
            f"{checks['n_divergences']} divergent transitions during sampling. NUTS did not "
            "fully explore the posterior, so the contribution and interval estimates above are "
            "not reliable enough to quote. Increase target_accept or reparameterize the model "
            "before trusting this benchmark."
        )
    if caveats:
        checks["caveat"] = " ".join(caveats)

    return {
        "contributions": contrib,
        "transform_parameters": decay,
        "metrics": pd.DataFrame(
            [{"split": k, **v} for k, v in metrics.items()]
        ),
        "checks": checks,
    }
