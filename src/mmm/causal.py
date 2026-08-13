"""Causal validation.

The standard and correct criticism of marketing mix modelling is that it is a regression on
observational data. Media budgets are not randomly assigned. They are set by planners who
already expect certain weeks and certain regions to sell well, so spend and sales are
correlated partly because spend causes sales and partly because expected sales cause spend.
A model that ignores this will attribute the planner's foresight to the media.

I cannot randomise anything after the fact. What I can do is run checks that would fail if
the estimates were mostly reverse causality or confounding, and report the results either
way. Three checks live here.

1. Geo holdout. Fit on a subset of divisions, predict divisions the model never saw using
   only the pooled channel level parameters. If media effects were division specific noise
   or reverse causality, the pooled parameters would not transfer and holdout error on the
   unseen geos would collapse toward the baseline.

2. Difference in differences. Find divisions with a sharp step change in a channel's
   volume, match them to divisions without one, and compare the change in sales before and
   after against the change in the controls. This estimates the effect of the shock without
   relying on the functional form of the marketing mix model at all, which is the point.

3. Placebo. Run the same difference in differences on randomly assigned fake treatment
   dates. The estimated effect should be indistinguishable from zero. If a placebo produces
   effects the same size as the real one, the design is picking up trend, not treatment.

None of this makes the model an experiment. The honest summary, which is in the executive
summary and in docs/limitations.md, is that these checks constrain how wrong the estimates
can be, and a geo lift test would settle what they cannot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DotDict, load_config

logger = logging.getLogger(__name__)


@dataclass
class GeoHoldoutResult:
    holdout_divisions: list[str]
    metrics_seen: dict[str, float]
    metrics_unseen: dict[str, float]
    baseline_metrics_unseen: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for label, m in [
            ("divisions_used_in_fitting", self.metrics_seen),
            ("held_out_divisions", self.metrics_unseen),
            ("held_out_divisions_baseline", self.baseline_metrics_unseen),
        ]:
            row = {"group": label}
            row.update(m)
            rows.append(row)
        return pd.DataFrame(rows)


def select_holdout_divisions(
    df: pd.DataFrame,
    media_cols: list[str],
    entity_col: str = "Division",
    n_holdout: int = 5,
    seed: int = 42,
) -> list[str]:
    """Pick holdout divisions by media variation rank rather than by hand.

    I take a spread across the distribution of within division media variability, so the
    holdout is not accidentally all easy divisions or all hard ones. Choosing them by hand
    after seeing the results would be the exact thing this check is supposed to rule out.
    """
    variability = (
        df.groupby(entity_col)[media_cols]
        .apply(lambda g: (g.std() / g.mean().replace(0, np.nan)).mean())
        .sort_values()
    )
    ranked = variability.index.tolist()
    if n_holdout >= len(ranked):
        return [str(x) for x in ranked]
    positions = np.linspace(0, len(ranked) - 1, n_holdout).round().astype(int)
    return [str(ranked[p]) for p in sorted(set(positions.tolist()))]


def geo_holdout_validation(
    df: pd.DataFrame,
    media_cols: list[str],
    control_cols: list[str] | None = None,
    cfg: DotDict | None = None,
    smoke: bool = False,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
) -> GeoHoldoutResult:
    """Fit without a set of divisions, then predict them from the pooled parameters."""
    from .backtest import evaluate
    from .models.bayesian_mmm import (
        HierarchicalMMM,
        PosteriorParams,
        forward,
        prepare_model_data,
    )

    cfg = cfg or load_config()
    n_holdout = int(cfg["causal"]["n_holdout_divisions"])
    holdout = select_holdout_divisions(
        df, media_cols, entity_col, n_holdout, int(cfg["project"]["random_seed"])
    )
    logger.info("Holding out divisions %s", holdout)

    train_df = df[~df[entity_col].isin(holdout)].copy()
    data_train = prepare_model_data(
        train_df,
        media_cols=media_cols,
        control_cols=control_cols,
        entity_col=entity_col,
        date_col=date_col,
        target_col=target_col,
        fourier_order=int(cfg["features"]["seasonality"]["fourier_order"]),
        period_weeks=float(cfg["features"]["seasonality"]["period_weeks"]),
    )
    model = HierarchicalMMM(cfg=cfg, smoke=smoke)
    model.fit(data_train)
    params = model.posterior_params(max_draws=300)

    preds_seen = model.predict(data_train).mean(axis=0)
    actual_seen = data_train.y * data_train.target_scale
    metrics_seen = evaluate(actual_seen, preds_seen)

    # For unseen divisions I have no division specific coefficient, which is the whole
    # point. I use the channel level population mean, exp(mu_beta + sigma_beta^2 / 2),
    # which is the mean of the lognormal the division effects are drawn from. That is what
    # a planner would have to use for a region with no history.
    post = model.idata.posterior
    mu_beta = post["mu_beta"].to_numpy().reshape(-1, len(media_cols))
    sigma_beta = post["sigma_beta"].to_numpy().reshape(-1, len(media_cols))
    mu_alpha = post["mu_alpha"].to_numpy().reshape(-1)
    pooled_beta = np.exp(mu_beta + 0.5 * sigma_beta**2)

    holdout_df = df[df[entity_col].isin(holdout)].copy()
    data_unseen = prepare_model_data(
        holdout_df,
        media_cols=media_cols,
        control_cols=control_cols,
        entity_col=entity_col,
        date_col=date_col,
        target_col=target_col,
        fourier_order=int(cfg["features"]["seasonality"]["fourier_order"]),
        period_weeks=float(cfg["features"]["seasonality"]["period_weeks"]),
    )
    n_draws = min(params.n_draws, pooled_beta.shape[0])
    n_unseen = data_unseen.n_entity
    pooled_params = PosteriorParams(
        decay=params.decay[:n_draws],
        half_sat=params.half_sat[:n_draws],
        slope=params.slope[:n_draws],
        beta=np.repeat(pooled_beta[:n_draws, :, None], n_unseen, axis=2),
        alpha=np.repeat(mu_alpha[:n_draws, None], n_unseen, axis=1),
        trend_coef=params.trend_coef[:n_draws],
        seas_coef=params.seas_coef[:n_draws],
        control_coef=params.control_coef[:n_draws],
        sigma=params.sigma[:n_draws],
    )
    preds_unseen = (
        forward(pooled_params, data_unseen, max_lag=int(cfg["features"]["adstock"]["max_lag"]))
        .mean(axis=0)
        * data_unseen.target_scale
    )
    actual_unseen = data_unseen.y * data_unseen.target_scale
    metrics_unseen = evaluate(actual_unseen, preds_unseen)

    # Baseline for the unseen divisions: the pooled intercept alone, media set to zero.
    # If the media terms are not carrying real information, the full model will not beat it.
    zero_media = np.zeros_like(data_unseen.X)
    preds_baseline = (
        forward(pooled_params, data_unseen, X=zero_media,
                max_lag=int(cfg["features"]["adstock"]["max_lag"])).mean(axis=0)
        * data_unseen.target_scale
    )
    baseline_metrics = evaluate(actual_unseen, preds_baseline)

    notes = [
        f"Fitted on {data_train.n_entity} divisions, predicted {n_unseen} unseen divisions "
        "using only the pooled channel level parameters.",
        "The comparison that matters is held_out_divisions against "
        "held_out_divisions_baseline. If the media terms transfer, the full model beats "
        "the media free baseline on divisions it never saw.",
    ]
    improvement = baseline_metrics.get("mape", np.nan) - metrics_unseen.get("mape", np.nan)
    notes.append(
        f"MAPE improvement over the media free baseline on unseen divisions: "
        f"{improvement:.2f} percentage points."
    )
    return GeoHoldoutResult(holdout, metrics_seen, metrics_unseen, baseline_metrics, notes)


# ---------------------------------------------------------------------------
# Difference in differences
# ---------------------------------------------------------------------------
def detect_spend_shocks(
    df: pd.DataFrame,
    channel: str,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    threshold_sd: float = 1.5,
    pre_weeks: int = 12,
    post_weeks: int = 12,
) -> pd.DataFrame:
    """Find step changes in a channel's volume within each division.

    A shock is a week where the mean of the following post_weeks differs from the mean of
    the preceding pre_weeks by more than threshold_sd division level standard deviations.
    I only keep the largest shock per division, so one division cannot dominate through
    repeated overlapping events.
    """
    rows = []
    for entity, grp in df.sort_values(date_col).groupby(entity_col):
        series = grp[channel].to_numpy(dtype=float)
        dates = grp[date_col].to_numpy()
        sd = series.std()
        if sd <= 0:
            continue
        best: dict[str, Any] | None = None
        for t in range(pre_weeks, len(series) - post_weeks):
            pre = series[t - pre_weeks : t].mean()
            post = series[t : t + post_weeks].mean()
            z = (post - pre) / sd
            if abs(z) >= threshold_sd and (best is None or abs(z) > abs(best["z"])):
                best = {
                    "entity": str(entity),
                    "shock_week_index": int(t),
                    "shock_date": dates[t],
                    "pre_mean": float(pre),
                    "post_mean": float(post),
                    "z": float(z),
                    "direction": "increase" if z > 0 else "decrease",
                }
        if best:
            rows.append(best)
    return pd.DataFrame(rows)


def difference_in_differences(
    df: pd.DataFrame,
    channel: str,
    cfg: DotDict | None = None,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
) -> dict[str, Any]:
    """Two way fixed effects difference in differences around a media shock.

    The estimate is the coefficient on treated times post in a regression of log sales on
    entity fixed effects, week fixed effects, and the interaction. Standard errors are
    clustered at the division level, because sales within a division are serially
    correlated and unclustered errors would be far too small.

    Interpretation, stated carefully: this recovers the effect of whatever happened at the
    shock, which includes the media change and anything correlated with it. It is not the
    same estimand as the model's channel coefficient, and I would not expect the two to be
    numerically identical. What I want is a sign and an order of magnitude that do not
    contradict the model.
    """
    cfg = cfg or load_config()
    did_cfg = cfg["causal"]["did"]
    pre_weeks = int(did_cfg["pre_weeks"])
    post_weeks = int(did_cfg["post_weeks"])

    shocks = detect_spend_shocks(
        df,
        channel,
        entity_col,
        date_col,
        float(did_cfg["shock_threshold_sd"]),
        pre_weeks,
        post_weeks,
    )
    increases = shocks[shocks["direction"] == "increase"] if not shocks.empty else shocks
    if len(increases) < int(did_cfg["min_treated"]):
        return {
            "channel": channel,
            "status": "insufficient_treated_units",
            "n_treated_found": int(len(increases)),
            "message": (
                f"Only {len(increases)} divisions show an upward shock of at least "
                f"{did_cfg['shock_threshold_sd']} standard deviations in {channel}. "
                "I do not run the estimator below the configured minimum, because a "
                "difference in differences on two units is not an estimate, it is an anecdote."
            ),
        }

    treated = set(increases["entity"])
    # Align every treated unit on its own shock week, and use the median shock week as the
    # event time for controls, which is the usual approach when treatment timing varies.
    event_index = int(increases["shock_week_index"].median())
    controls = [e for e in df[entity_col].unique() if str(e) not in treated]

    dates = sorted(df[date_col].unique())
    window = dates[max(event_index - pre_weeks, 0) : min(event_index + post_weeks, len(dates))]
    panel = df[df[date_col].isin(window)].copy()

    panel["treated"] = panel[entity_col].astype(str).isin(treated).astype(float)
    shock_map = dict(zip(increases["entity"], increases["shock_week_index"], strict=False))
    date_pos = {d: i for i, d in enumerate(dates)}
    panel["week_index"] = panel[date_col].map(date_pos)
    panel["post"] = panel.apply(
        lambda r: float(r["week_index"] >= shock_map.get(str(r[entity_col]), event_index)),
        axis=1,
    )
    panel["did"] = panel["treated"] * panel["post"]
    panel["log_sales"] = np.log(panel[target_col].clip(lower=1e-6))

    ent_dummies = pd.get_dummies(panel[entity_col], prefix="e", drop_first=True).astype(float)
    week_dummies = pd.get_dummies(panel["week_index"], prefix="w", drop_first=True).astype(float)
    design = pd.concat(
        [
            pd.Series(1.0, index=panel.index, name="intercept"),
            panel[["did"]],
            ent_dummies,
            week_dummies,
        ],
        axis=1,
    )
    X = design.to_numpy(dtype=float)
    y = panel["log_sales"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef

    # Cluster robust variance at the division level.
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros_like(xtx_inv)
    for _, idx in panel.groupby(entity_col).indices.items():
        Xg = X[idx]
        ug = resid[idx]
        score = Xg.T @ ug
        meat += np.outer(score, score)
    n_clusters = panel[entity_col].nunique()
    scale = n_clusters / max(n_clusters - 1, 1)
    vcov = xtx_inv @ meat @ xtx_inv * scale
    did_pos = list(design.columns).index("did")
    se = float(np.sqrt(max(vcov[did_pos, did_pos], 0.0)))
    estimate = float(coef[did_pos])

    return {
        "channel": channel,
        "status": "ok",
        "n_treated": len(treated),
        "n_control": len(controls),
        "event_week_index": event_index,
        "did_estimate_log_points": estimate,
        "did_estimate_pct": float((np.exp(estimate) - 1.0) * 100.0),
        "cluster_robust_se": se,
        "t_stat": estimate / se if se > 0 else np.nan,
        "ci_lower_pct": float((np.exp(estimate - 1.96 * se) - 1.0) * 100.0),
        "ci_upper_pct": float((np.exp(estimate + 1.96 * se) - 1.0) * 100.0),
        "treated_divisions": sorted(treated),
    }


def parallel_trends_test(
    df: pd.DataFrame,
    channel: str,
    cfg: DotDict | None = None,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
) -> dict[str, Any]:
    """Test the assumption the whole design rests on, using pre period data only.

    I regress log sales on treated times a linear pre period time trend, restricted to weeks
    before the shock. A coefficient indistinguishable from zero is consistent with parallel
    trends. A large one means the treated divisions were already diverging and the
    difference in differences estimate is picking that up rather than the media change.

    This is a check that can fail, which is the point of including it.
    """
    cfg = cfg or load_config()
    did_cfg = cfg["causal"]["did"]
    pre_weeks = int(did_cfg["pre_weeks"])

    shocks = detect_spend_shocks(
        df, channel, entity_col, date_col, float(did_cfg["shock_threshold_sd"]),
        pre_weeks, int(did_cfg["post_weeks"]),
    )
    increases = shocks[shocks["direction"] == "increase"] if not shocks.empty else shocks
    if len(increases) < int(did_cfg["min_treated"]):
        return {"channel": channel, "status": "insufficient_treated_units"}

    treated = set(increases["entity"])
    event_index = int(increases["shock_week_index"].median())
    dates = sorted(df[date_col].unique())
    pre_window = dates[max(event_index - pre_weeks, 0) : event_index]
    panel = df[df[date_col].isin(pre_window)].copy()
    if panel.empty:
        return {"channel": channel, "status": "empty_pre_window"}

    date_pos = {d: i for i, d in enumerate(pre_window)}
    panel["t"] = panel[date_col].map(date_pos).astype(float)
    panel["treated"] = panel[entity_col].astype(str).isin(treated).astype(float)
    panel["treated_x_t"] = panel["treated"] * panel["t"]
    panel["log_sales"] = np.log(panel[target_col].clip(lower=1e-6))

    ent_dummies = pd.get_dummies(panel[entity_col], prefix="e", drop_first=True).astype(float)
    design = pd.concat(
        [
            pd.Series(1.0, index=panel.index, name="intercept"),
            panel[["t", "treated_x_t"]],
            ent_dummies,
        ],
        axis=1,
    )
    X = design.to_numpy(dtype=float)
    y = panel["log_sales"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    pos = list(design.columns).index("treated_x_t")
    se = float(np.sqrt(max(xtx_inv[pos, pos] * sigma2, 0.0)))
    est = float(coef[pos])
    return {
        "channel": channel,
        "status": "ok",
        "pre_trend_differential_per_week": est,
        "std_error": se,
        "t_stat": est / se if se > 0 else np.nan,
        "passes_at_5pct": bool(abs(est / se) < 1.96) if se > 0 else False,
        "interpretation": (
            "A t statistic below about 2 in absolute value is consistent with parallel "
            "pre trends. Above that, the treated divisions were already moving differently "
            "and the difference in differences estimate should not be read causally."
        ),
    }


def placebo_test(
    df: pd.DataFrame,
    channel: str,
    cfg: DotDict | None = None,
    n_placebos: int = 50,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
) -> dict[str, Any]:
    """Randomly assign fake treatment and re-estimate, many times.

    The real estimate should sit in the tail of the placebo distribution. If it sits in the
    middle, the design is measuring noise. This gives a randomisation inference p value that
    does not depend on the regression standard errors being right.
    """
    cfg = cfg or load_config()
    real = difference_in_differences(df, channel, cfg, entity_col, date_col, target_col)
    if real.get("status") != "ok":
        return {"channel": channel, "status": real.get("status", "unavailable")}

    rng = np.random.default_rng(int(cfg["project"]["random_seed"]))
    entities = df[entity_col].unique().tolist()
    n_treated = int(real["n_treated"])
    dates = sorted(df[date_col].unique())
    date_pos = {d: i for i, d in enumerate(dates)}

    estimates = []
    for _ in range(n_placebos):
        fake_treated = set(map(str, rng.choice(entities, size=n_treated, replace=False)))
        fake_event = int(rng.integers(int(cfg["causal"]["did"]["pre_weeks"]),
                                      len(dates) - int(cfg["causal"]["did"]["post_weeks"])))
        window = dates[
            fake_event - int(cfg["causal"]["did"]["pre_weeks"]) : fake_event
            + int(cfg["causal"]["did"]["post_weeks"])
        ]
        panel = df[df[date_col].isin(window)].copy()
        if panel.empty:
            continue
        panel["week_index"] = panel[date_col].map(date_pos)
        panel["treated"] = panel[entity_col].astype(str).isin(fake_treated).astype(float)
        panel["post"] = (panel["week_index"] >= fake_event).astype(float)
        panel["did"] = panel["treated"] * panel["post"]
        panel["log_sales"] = np.log(panel[target_col].clip(lower=1e-6))

        ent_d = pd.get_dummies(panel[entity_col], prefix="e", drop_first=True).astype(float)
        week_d = pd.get_dummies(panel["week_index"], prefix="w", drop_first=True).astype(float)
        design = pd.concat(
            [pd.Series(1.0, index=panel.index, name="intercept"), panel[["did"]], ent_d, week_d],
            axis=1,
        )
        coef, *_ = np.linalg.lstsq(design.to_numpy(dtype=float),
                                   panel["log_sales"].to_numpy(dtype=float), rcond=None)
        estimates.append(float(coef[list(design.columns).index("did")]))

    estimates_arr = np.asarray(estimates)
    real_est = float(real["did_estimate_log_points"])
    p_value = float(np.mean(np.abs(estimates_arr) >= abs(real_est))) if len(estimates_arr) else np.nan
    return {
        "channel": channel,
        "status": "ok",
        "real_estimate_log_points": real_est,
        "n_placebos": int(len(estimates_arr)),
        "placebo_mean": float(estimates_arr.mean()) if len(estimates_arr) else np.nan,
        "placebo_sd": float(estimates_arr.std()) if len(estimates_arr) else np.nan,
        "randomisation_p_value": p_value,
        "interpretation": (
            "The p value is the share of random placebo assignments producing an effect at "
            "least as large in absolute value as the real one. Small means the real shock "
            "stands out from noise."
        ),
    }
