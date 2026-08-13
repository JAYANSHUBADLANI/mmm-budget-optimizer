"""Out of sample evaluation.

In sample fit on a marketing mix model is close to meaningless. With division fixed
effects, a trend, a Fourier seasonality basis and five saturating media terms, the model
has enough freedom to track almost any well behaved sales series. A high in sample R
squared tells me the model can describe the past, not that it can predict.

So the headline number in this project is holdout MAPE from a rolling origin backtest,
which is the closest thing available to asking "if I had fitted this in week N, how wrong
would the next quarter have been".

Two details that matter and are easy to get wrong:

1. The split is by time, never by row. Shuffling rows leaks future weeks into training
   through the adstock lag and through the shared seasonality basis, and the resulting
   error is optimistic by a wide margin.
2. Adstock is applied over the full media series and then sliced, so carryover from the
   training weeks flows into the holdout weeks. Transforming the holdout in isolation
   would start it with zero carryover and quietly understate the error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DotDict, load_config

logger = logging.getLogger(__name__)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring zero actuals which would divide by zero."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE, reported alongside MAPE because MAPE punishes over prediction less."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


METRICS = {"mape": mape, "smape": smape, "rmse": rmse, "r2": r_squared}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str] | None = None) -> dict:
    metrics = metrics or list(METRICS)
    return {m: METRICS[m](y_true, y_pred) for m in metrics if m in METRICS}


@dataclass
class FoldResult:
    fold: int
    train_end_week: int
    holdout_weeks: int
    train_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    n_divergences: int = 0
    notes: list[str] = field(default_factory=list)


def rolling_origin_splits(
    n_periods: int, holdout_weeks: int, n_folds: int, step_weeks: int
) -> list[tuple[int, int]]:
    """Expanding window splits, returned as (train_end, holdout_end) index pairs.

    The last fold ends at the final week, and earlier folds step backwards. Expanding
    rather than sliding, because in a real planning cycle you would refit on everything you
    have, not throw away old weeks.
    """
    splits: list[tuple[int, int]] = []
    for i in range(n_folds):
        holdout_end = n_periods - i * step_weeks
        train_end = holdout_end - holdout_weeks
        if train_end < holdout_weeks * 2:
            logger.warning(
                "Stopping at %d folds, the training window would be too short to fit", i
            )
            break
        splits.append((train_end, holdout_end))
    return sorted(splits)


def naive_seasonal_baseline(y: np.ndarray, train_end: int, season: int = 52) -> np.ndarray:
    """Last year same week, or last observed value when there is no full year of history.

    Every backtest needs a baseline. A model that cannot beat "same week last year" has
    not earned the complexity, and quoting a MAPE with nothing to compare it against is the
    single most common way a backtest overstates its results.
    """
    n_t = y.shape[0]
    preds = np.zeros_like(y[train_end:])
    for offset in range(n_t - train_end):
        t = train_end + offset
        source = t - season
        preds[offset] = y[source] if source >= 0 else y[train_end - 1]
    return preds


def backtest_bayesian(
    df: pd.DataFrame,
    media_cols: list[str],
    control_cols: list[str] | None = None,
    cfg: DotDict | None = None,
    smoke: bool = False,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Rolling origin backtest of the hierarchical model.

    Each fold refits from scratch on the training weeks only. Refitting is the slow and
    correct choice: reusing one fit and only re-slicing would let the holdout weeks
    influence the posterior through the shared seasonality and hierarchy.
    """
    from .models.bayesian_mmm import HierarchicalMMM, prepare_model_data

    cfg = cfg or load_config()
    bt = cfg["backtest"]
    holdout_weeks = int(bt["holdout_weeks"])
    n_folds = int(bt["n_folds"])
    step_weeks = int(bt["step_weeks"])
    metrics = list(bt["metrics"])

    n_periods = df[date_col].nunique()
    splits = rolling_origin_splits(n_periods, holdout_weeks, n_folds, step_weeks)
    logger.info("Backtest splits (train_end, holdout_end): %s", splits)

    results: list[FoldResult] = []
    for i, (train_end, holdout_end) in enumerate(splits):
        dates = sorted(df[date_col].unique())
        sub = df[df[date_col].isin(dates[:holdout_end])]

        data = prepare_model_data(
            sub,
            media_cols=media_cols,
            control_cols=control_cols,
            entity_col=entity_col,
            date_col=date_col,
            target_col=target_col,
            fourier_order=int(cfg["features"]["seasonality"]["fourier_order"]),
            period_weeks=float(cfg["features"]["seasonality"]["period_weeks"]),
            train_weeks=train_end,
        )

        model = HierarchicalMMM(cfg=cfg, smoke=smoke)
        model.fit(data)
        preds = model.predict(data).mean(axis=0)  # (T, D) posterior mean
        actual = data.y * data.target_scale

        train_slice = slice(0, train_end)
        hold_slice = slice(train_end, holdout_end)

        results.append(
            FoldResult(
                fold=i,
                train_end_week=train_end,
                holdout_weeks=holdout_end - train_end,
                train_metrics=evaluate(actual[train_slice], preds[train_slice], metrics),
                holdout_metrics=evaluate(actual[hold_slice], preds[hold_slice], metrics),
                n_divergences=model.divergences(),
            )
        )
        logger.info(
            "Fold %d holdout metrics: %s", i, results[-1].holdout_metrics
        )

    rows: list[dict[str, Any]] = []
    for r in results:
        row = {
            "fold": r.fold,
            "train_end_week": r.train_end_week,
            "holdout_weeks": r.holdout_weeks,
            "n_divergences": r.n_divergences,
        }
        row.update({f"train_{k}": v for k, v in r.train_metrics.items()})
        row.update({f"holdout_{k}": v for k, v in r.holdout_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows), results


def backtest_linear(
    df: pd.DataFrame,
    media_cols: list[str],
    control_cols: list[str] | None = None,
    cfg: DotDict | None = None,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
) -> pd.DataFrame:
    """Same splits, same metrics, applied to the linear model so the comparison is fair.

    Running both models through an identical harness is the part that makes the comparison
    an argument rather than an assertion.
    """
    from .models.linear_mmm import _build_design, fit_linear_mmm

    cfg = cfg or load_config()
    bt = cfg["backtest"]
    n_periods = df[date_col].nunique()
    splits = rolling_origin_splits(
        n_periods, int(bt["holdout_weeks"]), int(bt["n_folds"]), int(bt["step_weeks"])
    )
    dates = sorted(df[date_col].unique())
    metrics = list(bt["metrics"])

    rows = []
    for i, (train_end, holdout_end) in enumerate(splits):
        train = df[df[date_col].isin(dates[:train_end])]
        full = df[df[date_col].isin(dates[:holdout_end])]
        fit = fit_linear_mmm(
            train,
            media_cols=media_cols,
            control_cols=control_cols,
            entity_col=entity_col,
            date_col=date_col,
            target_col=target_col,
            cfg=cfg,
        )
        design, names = _build_design(
            full.sort_values([entity_col, date_col]).reset_index(drop=True),
            media_cols,
            list(control_cols or []),
            entity_col,
            date_col,
            fit.decays,
            fit.half_sats,
            int(cfg["features"]["adstock"]["max_lag"]),
            int(cfg["features"]["seasonality"]["fourier_order"]),
            float(cfg["features"]["seasonality"]["period_weeks"]),
        )
        coef_map = dict(zip(fit.design_columns, fit.coefficients["coefficient"].to_numpy(), strict=False))
        coef = np.array([coef_map.get(n, 0.0) for n in names])
        preds = design @ coef

        ordered = full.sort_values([entity_col, date_col]).reset_index(drop=True)
        is_holdout = ordered[date_col].isin(dates[train_end:holdout_end]).to_numpy()
        y = ordered[target_col].to_numpy(dtype=float)

        row = {"fold": i, "train_end_week": train_end, "holdout_weeks": holdout_end - train_end}
        row.update({f"train_{k}": v for k, v in evaluate(y[~is_holdout], preds[~is_holdout], metrics).items()})
        row.update({f"holdout_{k}": v for k, v in evaluate(y[is_holdout], preds[is_holdout], metrics).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_backtest(
    df: pd.DataFrame,
    cfg: DotDict | None = None,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
) -> pd.DataFrame:
    """Seasonal naive baseline on the same splits, so the model has something to beat."""
    cfg = cfg or load_config()
    bt = cfg["backtest"]
    dates = sorted(df[date_col].unique())
    n_periods = len(dates)
    splits = rolling_origin_splits(
        n_periods, int(bt["holdout_weeks"]), int(bt["n_folds"]), int(bt["step_weeks"])
    )
    pivot = df.pivot_table(index=date_col, columns=entity_col, values=target_col).sort_index()
    y = pivot.to_numpy(dtype=float)

    rows = []
    for i, (train_end, holdout_end) in enumerate(splits):
        preds = naive_seasonal_baseline(y[:holdout_end], train_end, season=52)
        actual = y[train_end:holdout_end]
        row = {"fold": i, "train_end_week": train_end, "holdout_weeks": holdout_end - train_end}
        row.update({f"holdout_{k}": v for k, v in evaluate(actual, preds, list(bt["metrics"])).items()})
        rows.append(row)
    return pd.DataFrame(rows)
