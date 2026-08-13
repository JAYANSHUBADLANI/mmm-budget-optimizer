"""Linear marketing mix model, fitted as the comparison case.

The point of this file is not to build a straw man. It is the model most marketing mix
work actually ships: media columns, a couple of controls, fixed effects, ordinary least
squares. I fit it properly, with adstock and saturation applied at grid searched values so
it is not handicapped on functional form, and then I compare it against the Bayesian model
on three things.

1. Point estimates. Do the two approaches even agree on which channel is strongest.
2. Uncertainty. OLS standard errors assume the adstock and saturation parameters were
   known in advance. They were not, I searched over them on the same data, so those
   intervals are too narrow by an amount the standard errors cannot see. The Bayesian
   intervals propagate that uncertainty.
3. Stability. Media regressors move together. I report the variance inflation factors and
   the sign pattern, because a negative coefficient on a channel that plainly works is the
   classic symptom, and it is exactly what the lognormal prior in the Bayesian model rules
   out on substantive grounds.

If the two models agree, that is evidence the finding is robust to method. If they
disagree, the disagreement is the finding.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import DotDict, load_config
from ..features import geometric_adstock, hill_saturation

logger = logging.getLogger(__name__)


@dataclass
class LinearMMMResult:
    coefficients: pd.DataFrame
    decays: dict[str, float]
    half_sats: dict[str, float]
    r2: float
    adj_r2: float
    vif: pd.DataFrame
    n_obs: int
    n_params: int
    fitted: np.ndarray
    residuals: np.ndarray
    design_columns: list[str]
    grid_size: int
    notes: list[str]

    def to_frame(self) -> pd.DataFrame:
        return self.coefficients.copy()


def variance_inflation(design: np.ndarray, names: list[str]) -> pd.DataFrame:
    """VIF per regressor, computed by regressing each column on the others.

    I report this because it is the honest way to show that the media block is collinear,
    which is the structural reason single point estimates from OLS are fragile here.
    """
    rows = []
    for j, name in enumerate(names):
        others = np.delete(design, j, axis=1)
        target = design[:, j]
        if np.allclose(target.std(), 0):
            rows.append({"regressor": name, "vif": np.nan})
            continue
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
        rows.append({"regressor": name, "vif": float(vif)})
    return pd.DataFrame(rows)


def _build_design(
    df: pd.DataFrame,
    media_cols: list[str],
    control_cols: list[str],
    entity_col: str,
    date_col: str,
    decays: dict[str, float],
    half_sats: dict[str, float],
    max_lag: int,
    fourier_order: int,
    period_weeks: float,
) -> tuple[np.ndarray, list[str]]:
    from ..features import fourier_terms

    frame = df.sort_values([entity_col, date_col]).reset_index(drop=True)
    blocks: list[np.ndarray] = []
    names: list[str] = []

    for col in media_cols:
        scale = float(frame[col].max()) or 1.0
        transformed = (
            frame.groupby(entity_col, sort=False)[col]
            .transform(
                lambda s, d=decays[col], sc=scale: geometric_adstock(
                    s.to_numpy() / sc, d, max_lag
                )
            )
            .to_numpy()
        )
        blocks.append(hill_saturation(transformed, half_sats[col], slope=1.0).reshape(-1, 1))
        names.append(col)

    for col in control_cols:
        values = frame[col].to_numpy(dtype=float)
        std = values.std() or 1.0
        blocks.append(((values - values.mean()) / std).reshape(-1, 1))
        names.append(col)

    time_idx = frame.groupby(entity_col, sort=False).cumcount().to_numpy(dtype=float)
    seas = fourier_terms(time_idx, period=period_weeks, order=fourier_order)
    blocks.append(seas.to_numpy())
    names.extend(seas.columns.tolist())

    blocks.append((time_idx / max(time_idx.max(), 1.0)).reshape(-1, 1))
    names.append("trend")

    # Division fixed effects, dropping one level to keep the design full rank alongside
    # the intercept.
    dummies = pd.get_dummies(frame[entity_col], prefix="div", drop_first=True).astype(float)
    blocks.append(dummies.to_numpy())
    names.extend(dummies.columns.tolist())

    blocks.append(np.ones((len(frame), 1)))
    names.append("intercept")

    return np.hstack(blocks), names


def fit_linear_mmm(
    df: pd.DataFrame,
    media_cols: list[str],
    control_cols: list[str] | None = None,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
    cfg: DotDict | None = None,
    decay_grid: tuple[float, ...] = (0.0, 0.3, 0.6),
    half_sat_grid: tuple[float, ...] = (0.2, 0.5, 0.8),
    max_grid_combinations: int = 2000,
) -> LinearMMMResult:
    """Fit OLS with a grid search over shared adstock and saturation parameters.

    To keep the search tractable I search one decay and one half saturation value applied
    to every channel, rather than the full per channel product. That is a real limitation
    of the frequentist route and I state it as such: doing it per channel means fitting
    len(grid) raised to the number of channels models, which is exactly the combinatorial
    problem the Bayesian version sidesteps by estimating those parameters jointly with
    everything else.
    """
    cfg = cfg or load_config()
    control_cols = list(control_cols or [])
    max_lag = int(cfg["features"]["adstock"]["max_lag"])
    fourier_order = int(cfg["features"]["seasonality"]["fourier_order"])
    period_weeks = float(cfg["features"]["seasonality"]["period_weeks"])
    notes: list[str] = []

    frame = df.sort_values([entity_col, date_col]).reset_index(drop=True)
    y = frame[target_col].to_numpy(dtype=float)

    combos = list(itertools.product(decay_grid, half_sat_grid))
    if len(combos) > max_grid_combinations:
        combos = combos[:max_grid_combinations]
        notes.append(f"Grid truncated to {max_grid_combinations} combinations.")

    best: dict[str, Any] | None = None
    for decay, half_sat in combos:
        decays = dict.fromkeys(media_cols, float(decay))
        half_sats = dict.fromkeys(media_cols, float(half_sat))
        design, names = _build_design(
            frame, media_cols, control_cols, entity_col, date_col,
            decays, half_sats, max_lag, fourier_order, period_weeks,
        )
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        ss_res = float(np.sum(resid**2))
        if best is None or ss_res < best["ss_res"]:
            best = {
                "ss_res": ss_res,
                "decay": float(decay),
                "half_sat": float(half_sat),
                "coef": coef,
                "design": design,
                "names": names,
                "resid": resid,
            }

    assert best is not None
    design, names, coef, resid = best["design"], best["names"], best["coef"], best["resid"]
    n, p = design.shape
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - best["ss_res"] / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else np.nan

    # Standard errors under the usual OLS assumptions. I record in the notes that these
    # are conditional on the grid searched transform parameters, which the formula treats
    # as if they had been known before seeing the data.
    dof = max(n - p, 1)
    sigma2 = best["ss_res"] / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, coef / se, np.nan)

    coefficients = pd.DataFrame(
        {
            "regressor": names,
            "coefficient": coef,
            "std_error": se,
            "t_stat": t_stat,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
        }
    )
    coefficients["is_media"] = coefficients["regressor"].isin(media_cols)

    media_idx = [names.index(c) for c in media_cols if c in names]
    vif = variance_inflation(design[:, media_idx], [names[i] for i in media_idx])

    notes.append(
        "Standard errors are conditional on the grid searched adstock and saturation "
        "values. Because I selected those on the same data, the reported intervals are "
        "narrower than the true sampling variability."
    )
    negative = coefficients.loc[
        coefficients["is_media"] & (coefficients["coefficient"] < 0), "regressor"
    ].tolist()
    if negative:
        notes.append(
            f"Channels with a negative fitted coefficient: {negative}. I read this as a "
            "collinearity artefact rather than evidence that advertising reduced sales."
        )

    return LinearMMMResult(
        coefficients=coefficients,
        decays=dict.fromkeys(media_cols, best["decay"]),
        half_sats=dict.fromkeys(media_cols, best["half_sat"]),
        r2=float(r2),
        adj_r2=float(adj_r2),
        vif=vif,
        n_obs=int(n),
        n_params=int(p),
        fitted=y - resid,
        residuals=resid,
        design_columns=names,
        grid_size=len(combos),
        notes=notes,
    )


def compare_models(
    linear: LinearMMMResult,
    bayes_roi: pd.DataFrame,
    media_cols: list[str],
) -> pd.DataFrame:
    """Side by side comparison table on the media channels.

    interval_width_ratio is the number I actually want a reader to look at. It is how much
    wider the Bayesian credible interval is than the OLS confidence interval for the same
    channel. A ratio well above one is the quantitative version of the claim that the
    simple model is overconfident.
    """
    lin = (
        linear.coefficients[linear.coefficients["is_media"]]
        .set_index("regressor")
        .loc[[c for c in media_cols if c in linear.coefficients["regressor"].values]]
    )
    rows = []
    for channel in media_cols:
        b = bayes_roi[bayes_roi["channel"] == channel]
        if channel not in lin.index or b.empty:
            continue
        lin_width = float(lin.loc[channel, "ci_upper"] - lin.loc[channel, "ci_lower"])
        bay_width = float(b["roi_hdi_upper"].iloc[0] - b["roi_hdi_lower"].iloc[0])
        lin_rel = lin_width / abs(float(lin.loc[channel, "coefficient"])) if float(
            lin.loc[channel, "coefficient"]
        ) != 0 else np.nan
        bay_rel = bay_width / abs(float(b["roi_mean"].iloc[0])) if float(
            b["roi_mean"].iloc[0]
        ) != 0 else np.nan
        rows.append(
            {
                "channel": channel,
                "ols_coefficient": float(lin.loc[channel, "coefficient"]),
                "ols_t_stat": float(lin.loc[channel, "t_stat"]),
                "ols_relative_ci_width": lin_rel,
                "bayes_roi_mean": float(b["roi_mean"].iloc[0]),
                "bayes_roi_hdi_lower": float(b["roi_hdi_lower"].iloc[0]),
                "bayes_roi_hdi_upper": float(b["roi_hdi_upper"].iloc[0]),
                "bayes_relative_hdi_width": bay_rel,
                "interval_width_ratio": bay_rel / lin_rel
                if lin_rel and not np.isnan(lin_rel)
                else np.nan,
                "ols_sign_negative": bool(float(lin.loc[channel, "coefficient"]) < 0),
            }
        )
    return pd.DataFrame(rows)
