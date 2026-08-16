"""Hierarchical Bayesian marketing mix model.

Structure of the model, and why each piece is there:

  sales[t, d] = base[d] + trend + seasonality + sum over channels of
                beta[c, d] * Hill(Adstock(media[t, d, c])) + controls + noise

Pooling. Each of the 26 divisions gets its own channel coefficient, but those
coefficients are drawn from a shared distribution per channel. Divisions with thin or
uninformative media variation are pulled toward the channel level mean instead of
producing a wild estimate, and divisions with strong signal stay close to their own data.
This is the reason I did not fit 26 separate models, and it is also the reason I did not
pool everything into one flat time series: the first throws away the shared structure, the
second throws away the heterogeneity.

Transform parameters. Adstock decay, Hill half saturation and Hill slope are estimated at
the channel level and shared across divisions. With 113 weeks per division there is not
enough within division variation to identify a separate decay per division per channel,
and pretending otherwise produces confident nonsense.

Positivity. Channel coefficients are lognormal, so media effects cannot come out negative.
A negative media coefficient is almost never a real finding, it is a symptom of
collinearity, and letting the sampler explore that region produces ROI tables that no
media team will believe.

Non centred parameterisation. The hierarchical offsets are sampled as standard normals and
then rescaled. The centred version funnels badly when a group variance is small, which is
exactly the situation in the divisions with low media variation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import DotDict, load_config

logger = logging.getLogger(__name__)


@dataclass
class ModelData:
    """Arrays the model consumes, plus everything needed to invert the scaling."""

    X: np.ndarray  # (time, entity, channel), media scaled to [0, 1]
    y: np.ndarray  # (time, entity), target scaled by target_scale
    controls: np.ndarray  # (time, entity, n_control)
    seasonality: np.ndarray  # (time, n_fourier)
    trend: np.ndarray  # (time,)
    entities: list[str]
    dates: list[Any]
    channels: list[str]
    control_names: list[str]
    media_scalers: dict[str, float]
    target_scale: float
    train_mask: np.ndarray = field(default=None)  # (time,) boolean

    @property
    def n_time(self) -> int:
        return self.X.shape[0]

    @property
    def n_entity(self) -> int:
        return self.X.shape[1]

    @property
    def n_channel(self) -> int:
        return self.X.shape[2]


def prepare_model_data(
    df: pd.DataFrame,
    media_cols: list[str],
    control_cols: list[str] | None = None,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
    fourier_order: int = 3,
    period_weeks: float = 52.18,
    train_weeks: int | None = None,
) -> ModelData:
    """Pivot the long panel into the (time, entity, channel) arrays the model needs."""
    from ..features import fourier_terms

    control_cols = list(control_cols or [])
    frame = df.sort_values([date_col, entity_col]).copy()

    entities = sorted(frame[entity_col].unique().tolist())
    dates = sorted(frame[date_col].unique().tolist())
    n_t, n_d = len(dates), len(entities)

    date_pos = {d: i for i, d in enumerate(dates)}
    ent_pos = {e: i for i, e in enumerate(entities)}
    ti = frame[date_col].map(date_pos).to_numpy()
    di = frame[entity_col].map(ent_pos).to_numpy()

    def to_grid(values: np.ndarray) -> np.ndarray:
        grid = np.full((n_t, n_d), np.nan)
        grid[ti, di] = values
        if np.isnan(grid).any():
            raise ValueError(
                "The panel has holes after pivoting. Every division must be present in "
                "every week. Run the data quality checks before building model data."
            )
        return grid

    media_scalers: dict[str, float] = {}
    X = np.zeros((n_t, n_d, len(media_cols)))
    for k, col in enumerate(media_cols):
        raw = frame[col].to_numpy(dtype=float)
        scale = float(np.nanmax(raw)) or 1.0
        media_scalers[col] = scale
        X[:, :, k] = to_grid(raw) / scale

    controls = np.zeros((n_t, n_d, len(control_cols)))
    for k, col in enumerate(control_cols):
        raw = frame[col].to_numpy(dtype=float)
        std = float(np.nanstd(raw)) or 1.0
        mean = float(np.nanmean(raw))
        controls[:, :, k] = (to_grid(raw) - mean) / std

    y_raw = to_grid(frame[target_col].to_numpy(dtype=float))
    target_scale = float(np.nanmax(y_raw)) or 1.0
    y = y_raw / target_scale

    time_idx = np.arange(n_t, dtype=float)
    seas = fourier_terms(time_idx, period=period_weeks, order=fourier_order).to_numpy()
    trend = time_idx / max(n_t - 1, 1)

    train_mask = np.ones(n_t, dtype=bool)
    if train_weeks is not None:
        train_mask[train_weeks:] = False

    return ModelData(
        X=X,
        y=y,
        controls=controls,
        seasonality=seas,
        trend=trend,
        entities=[str(e) for e in entities],
        dates=dates,
        channels=list(media_cols),
        control_names=control_cols,
        media_scalers=media_scalers,
        target_scale=target_scale,
        train_mask=train_mask,
    )


# ---------------------------------------------------------------------------
# NumPy forward pass. Shared by prediction, decomposition and the optimiser.
# ---------------------------------------------------------------------------
def adstock_draws(X: np.ndarray, decay: np.ndarray, max_lag: int = 8) -> np.ndarray:
    """Vectorised adstock across posterior draws.

    X is (T, D, C), decay is (S, C), result is (S, T, D, C).
    """
    n_s, n_c = decay.shape
    lags = np.arange(max_lag + 1)
    weights = decay[:, None, :] ** lags[None, :, None]  # (S, L+1, C)
    weights = weights / weights.sum(axis=1, keepdims=True)

    padded = np.concatenate([np.zeros((max_lag, X.shape[1], X.shape[2])), X], axis=0)
    out = np.zeros((n_s, X.shape[0], X.shape[1], n_c))
    for lag in range(max_lag + 1):
        start = max_lag - lag
        shifted = padded[start : start + X.shape[0]]  # (T, D, C)
        out += weights[:, lag, :][:, None, None, :] * shifted[None, ...]
    return out


def hill_draws(x: np.ndarray, half_sat: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Hill saturation across draws. x is (S, T, D, C), half_sat and slope are (S, C)."""
    s = slope[:, None, None, :]
    h = half_sat[:, None, None, :]
    xs = np.clip(x, 1e-12, None) ** s
    return xs / (h**s + xs)


@dataclass
class PosteriorParams:
    """Posterior draws flattened into plain arrays, so downstream code never needs ArviZ."""

    decay: np.ndarray  # (S, C)
    half_sat: np.ndarray  # (S, C)
    slope: np.ndarray  # (S, C)
    beta: np.ndarray  # (S, C, D)
    alpha: np.ndarray  # (S, D)
    trend_coef: np.ndarray  # (S,)
    seas_coef: np.ndarray  # (S, K)
    control_coef: np.ndarray  # (S, K2)
    sigma: np.ndarray  # (S,)

    @property
    def n_draws(self) -> int:
        return self.decay.shape[0]

    def subsample(self, n: int, seed: int = 42) -> PosteriorParams:
        if n >= self.n_draws:
            return self
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.n_draws, size=n, replace=False)
        return PosteriorParams(
            decay=self.decay[idx],
            half_sat=self.half_sat[idx],
            slope=self.slope[idx],
            beta=self.beta[idx],
            alpha=self.alpha[idx],
            trend_coef=self.trend_coef[idx],
            seas_coef=self.seas_coef[idx],
            control_coef=self.control_coef[idx],
            sigma=self.sigma[idx],
        )


def media_contributions(
    params: PosteriorParams, X: np.ndarray, max_lag: int = 8
) -> np.ndarray:
    """Per channel contribution in scaled target units. Result is (S, T, D, C)."""
    saturated = hill_draws(adstock_draws(X, params.decay, max_lag), params.half_sat, params.slope)
    return saturated * params.beta.transpose(0, 2, 1)[:, None, :, :]


def forward(
    params: PosteriorParams,
    data: ModelData,
    X: np.ndarray | None = None,
    max_lag: int = 8,
) -> np.ndarray:
    """Expected scaled sales for every draw. Result is (S, T, D).

    Passing an alternative X is how the optimiser evaluates a counterfactual media plan
    against the fitted posterior without refitting anything.
    """
    X = data.X if X is None else X
    contrib = media_contributions(params, X, max_lag).sum(axis=3)  # (S, T, D)
    mu = params.alpha[:, None, :] + contrib
    mu = mu + (params.trend_coef[:, None] * data.trend[None, :])[:, :, None]
    mu = mu + (params.seas_coef @ data.seasonality.T)[:, :, None]
    if data.controls.shape[2] > 0 and params.control_coef.shape[1] > 0:
        mu = mu + np.einsum("sk,tdk->std", params.control_coef, data.controls)
    return mu


class HierarchicalMMM:
    """PyMC implementation, with a NumPy forward pass for everything after fitting."""

    def __init__(self, cfg: DotDict | None = None, smoke: bool = False) -> None:
        self.cfg = cfg or load_config()
        self.smoke = smoke
        self.max_lag = int(self.cfg["features"]["adstock"]["max_lag"])
        self.data: ModelData | None = None
        self.model = None
        self.idata = None

    # -- model construction -------------------------------------------------
    def build(self, data: ModelData):
        import pymc as pm

        from ..features import pt_geometric_adstock, pt_hill_saturation

        self.data = data
        ad_cfg = self.cfg["features"]["adstock"]
        sat_cfg = self.cfg["features"]["saturation"]

        coords = {
            "channel": data.channels,
            "entity": data.entities,
            "fourier": [f"f{i}" for i in range(data.seasonality.shape[1])],
            "control": data.control_names or ["_none"],
        }

        train_t = int(data.train_mask.sum())
        X_train = data.X[:train_t]
        y_train = data.y[:train_t]
        seas_train = data.seasonality[:train_t]
        trend_train = data.trend[:train_t]
        ctrl_train = data.controls[:train_t]

        with pm.Model(coords=coords) as model:
            # Adstock decay per channel. Beta(2, 2) is weakly informative and keeps the
            # mass away from both 0 and 1, then I rescale onto the configured bounds so
            # decay cannot approach 1 and turn carryover into an unidentified random walk.
            decay_raw = pm.Beta("decay_raw", alpha=2.0, beta=2.0, dims="channel")
            decay = pm.Deterministic(
                "decay",
                ad_cfg["decay_lower"]
                + (ad_cfg["decay_upper"] - ad_cfg["decay_lower"]) * decay_raw,
                dims="channel",
            )

            # Half saturation on the scaled media input, which lives in [0, 1].
            # Beta(2, 2) puts the prior mass in the interior, saying I expect the bend in
            # the response curve somewhere inside the observed spend range rather than far
            # outside it, which is what makes the curve identifiable at all.
            half_sat = pm.Beta("half_sat", alpha=2.0, beta=2.0, dims="channel")
            slope = pm.TruncatedNormal(
                "slope",
                mu=1.0,
                sigma=0.5,
                lower=sat_cfg["slope_lower"],
                upper=sat_cfg["slope_upper"],
                dims="channel",
            )

            # Hierarchical channel coefficients, lognormal so effects stay positive,
            # non centred so a small between division variance does not create a funnel.
            mu_beta = pm.Normal("mu_beta", mu=-2.0, sigma=1.0, dims="channel")
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5, dims="channel")
            z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, dims=("channel", "entity"))
            beta = pm.Deterministic(
                "beta",
                pm.math.exp(mu_beta[:, None] + sigma_beta[:, None] * z_beta),
                dims=("channel", "entity"),
            )

            # Division baselines, also non centred.
            mu_alpha = pm.Normal("mu_alpha", mu=0.3, sigma=0.5)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
            z_alpha = pm.Normal("z_alpha", mu=0.0, sigma=1.0, dims="entity")
            alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * z_alpha, dims="entity")

            trend_coef = pm.Normal("trend_coef", mu=0.0, sigma=0.5)
            seas_coef = pm.Normal("seas_coef", mu=0.0, sigma=0.3, dims="fourier")

            X_ = pm.Data("X", X_train)
            seas_ = pm.Data("seasonality", seas_train)
            trend_ = pm.Data("trend", trend_train)

            adstocked = pt_geometric_adstock(X_, decay, max_lag=self.max_lag)
            saturated = pt_hill_saturation(adstocked, half_sat, slope)
            media_effect = pm.math.sum(saturated * beta.T[None, :, :], axis=2)

            mu = (
                alpha[None, :]
                + media_effect
                + (trend_coef * trend_)[:, None]
                + pm.math.dot(seas_, seas_coef)[:, None]
            )

            if ctrl_train.shape[2] > 0:
                control_coef = pm.Normal("control_coef", mu=0.0, sigma=0.3, dims="control")
                ctrl_ = pm.Data("controls", ctrl_train)
                mu = mu + pm.math.sum(ctrl_ * control_coef[None, None, :], axis=2)

            mu = pm.Deterministic("mu", mu)
            sigma = pm.HalfNormal("sigma", sigma=0.1)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        self.model = model
        return model

    # -- fitting ------------------------------------------------------------
    def fit(self, data: ModelData | None = None, **kwargs):
        import pymc as pm

        if data is not None or self.model is None:
            self.build(data if data is not None else self.data)

        settings = dict(
            self.cfg["model"]["bayesian"]["smoke"]
            if self.smoke
            else {
                k: v
                for k, v in self.cfg["model"]["bayesian"].items()
                if k != "smoke"
            }
        )
        settings.update(kwargs)
        seed = int(self.cfg["project"]["random_seed"])

        logger.info("Sampling with %s", settings)
        with self.model:
            self.idata = pm.sample(
                draws=int(settings["draws"]),
                tune=int(settings["tune"]),
                chains=int(settings["chains"]),
                cores=int(settings.get("cores", settings["chains"])),
                target_accept=float(settings["target_accept"]),
                random_seed=seed,
                progressbar=True,
            )
        return self.idata

    # -- posterior extraction ----------------------------------------------
    def posterior_params(self, max_draws: int | None = None) -> PosteriorParams:
        """Flatten chain and draw dimensions into a single sample axis."""
        if self.idata is None:
            raise RuntimeError("Fit the model before extracting posterior parameters.")
        post = self.idata.posterior

        def flat(name: str) -> np.ndarray:
            arr = post[name].to_numpy()
            return arr.reshape((-1,) + arr.shape[2:])

        n_ctrl = len(self.data.control_names) if self.data else 0
        control_coef = (
            flat("control_coef") if n_ctrl and "control_coef" in post else None
        )
        n_samples = flat("decay").shape[0]
        if control_coef is None:
            control_coef = np.zeros((n_samples, 0))

        params = PosteriorParams(
            decay=flat("decay"),
            half_sat=flat("half_sat"),
            slope=flat("slope"),
            beta=flat("beta"),
            alpha=flat("alpha"),
            trend_coef=flat("trend_coef"),
            seas_coef=flat("seas_coef"),
            control_coef=control_coef,
            sigma=flat("sigma"),
        )
        return params.subsample(max_draws) if max_draws else params

    # -- prediction ---------------------------------------------------------
    def predict(
        self, data: ModelData | None = None, X: np.ndarray | None = None, max_draws: int = 400
    ) -> np.ndarray:
        """Posterior mean predictions in original sales units. Result is (S, T, D).

        Prediction runs over the full time axis, including the holdout weeks. Adstock is
        applied to the complete media series and then sliced, so carryover flows from the
        training period into the holdout period exactly as it would in production. Slicing
        first and adstocking second would zero out the carryover at the boundary and
        flatter the holdout error.
        """
        data = data or self.data
        params = self.posterior_params(max_draws=max_draws)
        mu = forward(params, data, X=X, max_lag=self.max_lag)
        return mu * data.target_scale

    # -- diagnostics --------------------------------------------------------
    def diagnostics(self) -> pd.DataFrame:
        """Convergence summary for the parameters that carry the interpretation."""
        import arviz as az

        var_names = [
            "decay",
            "half_sat",
            "slope",
            "mu_beta",
            "sigma_beta",
            "mu_alpha",
            "sigma_alpha",
            "trend_coef",
            "sigma",
        ]
        available = [v for v in var_names if v in self.idata.posterior]
        summary = az.summary(self.idata, var_names=available, round_to=4)
        return summary.reset_index().rename(columns={"index": "parameter"})

    def divergences(self) -> int:
        if self.idata is None or "sample_stats" not in self.idata:
            return 0
        if "diverging" not in self.idata.sample_stats:
            return 0
        return int(self.idata.sample_stats["diverging"].to_numpy().sum())

    def save(self, path: str | Path) -> Path:
        # The module level az.to_netcdf helper was dropped in arviz 1.x. The method on the
        # inference object itself exists in both 0.x and 1.x, so it is the portable call.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.idata.to_netcdf(str(path))
        logger.info("Saved inference data to %s", path)
        return path

    def load(self, path: str | Path) -> None:
        import arviz as az

        self.idata = az.from_netcdf(str(path))
