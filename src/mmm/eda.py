"""Exploratory analysis, written to produce tables rather than a wall of notebook output.

Everything here returns a DataFrame that the phase scripts save to reports/tables, so the
findings end up in files I can cite in the README instead of screenshots of a notebook.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DotDict, load_config

logger = logging.getLogger(__name__)


def dataset_overview(
    df: pd.DataFrame, entity_col: str = "Division", date_col: str = "Calendar_Week"
) -> pd.DataFrame:
    """Shape, coverage and key counts in one small table."""
    rows = [
        {"metric": "rows", "value": len(df)},
        {"metric": "columns", "value": df.shape[1]},
        {"metric": "divisions", "value": df[entity_col].nunique()},
        {"metric": "weeks", "value": df[date_col].nunique()},
        {"metric": "first_week", "value": str(pd.to_datetime(df[date_col]).min().date())},
        {"metric": "last_week", "value": str(pd.to_datetime(df[date_col]).max().date())},
        {"metric": "missing_cells", "value": int(df.isna().sum().sum())},
        {"metric": "duplicate_rows", "value": int(df.duplicated().sum())},
        {
            "metric": "rows_per_division_min",
            "value": int(df.groupby(entity_col).size().min()),
        },
        {
            "metric": "rows_per_division_max",
            "value": int(df.groupby(entity_col).size().max()),
        },
    ]
    return pd.DataFrame(rows)


def channel_summary(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Distribution of each channel, including the share of weeks with zero volume.

    The zero share matters more than it looks. A channel that is dark in most weeks has
    very little information about its own saturation curve, and the posterior for its half
    saturation parameter will be close to the prior. Knowing that in advance stops me from
    over reading a confident looking result later.
    """
    rows = []
    for col in channels:
        values = df[col].astype(float)
        rows.append(
            {
                "channel": col,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "coefficient_of_variation": float(values.std() / values.mean())
                if values.mean()
                else np.nan,
                "share_zero_weeks": float((values == 0).mean()),
                "skew": float(values.skew()),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].astype(float).corr().reset_index().rename(columns={"index": "variable"})


def media_target_correlation(
    df: pd.DataFrame, channels: list[str], target_col: str = "Sales", max_lag: int = 8
) -> pd.DataFrame:
    """Cross correlation of each channel with sales at several lags.

    This is a cheap first read on carryover. If a channel correlates more strongly with
    sales at a lag of two weeks than at zero, that is a hint the adstock decay for that
    channel will come out high. It is only a hint, because these correlations are
    confounded by seasonality and by the other channels moving together.
    """
    rows = []
    for col in channels:
        for lag in range(max_lag + 1):
            shifted = df.groupby("Division")[col].shift(lag)
            mask = shifted.notna()
            if mask.sum() < 10:
                continue
            corr = float(
                np.corrcoef(shifted[mask].astype(float), df.loc[mask, target_col].astype(float))[
                    0, 1
                ]
            )
            rows.append({"channel": col, "lag_weeks": lag, "correlation": corr})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best = out.loc[out.groupby("channel")["correlation"].idxmax()]
    best = best.rename(columns={"lag_weeks": "peak_lag_weeks", "correlation": "peak_correlation"})
    return out.merge(best[["channel", "peak_lag_weeks", "peak_correlation"]], on="channel")


def division_profile(
    df: pd.DataFrame,
    channels: list[str],
    entity_col: str = "Division",
    target_col: str = "Sales",
) -> pd.DataFrame:
    """Per division scale and media mix, which is what justifies the hierarchy.

    If every division had the same mix and the same scale, pooling would be pointless and a
    single aggregate model would do. The spread in this table is the empirical case for the
    hierarchical structure.
    """
    grouped = df.groupby(entity_col)
    out = pd.DataFrame(
        {
            "total_sales": grouped[target_col].sum(),
            "mean_weekly_sales": grouped[target_col].mean(),
            "sales_cv": grouped[target_col].std() / grouped[target_col].mean(),
        }
    )
    total_media = df[channels].sum(axis=1)
    tmp = df.assign(_total_media=total_media)
    for col in channels:
        out[f"{col}_share"] = tmp.groupby(entity_col).apply(
            lambda g, c=col: g[c].sum() / g["_total_media"].sum()
            if g["_total_media"].sum()
            else np.nan,
            include_groups=False,
        )
    return out.reset_index()


def seasonality_profile(
    df: pd.DataFrame, date_col: str = "Calendar_Week", target_col: str = "Sales"
) -> pd.DataFrame:
    """Average sales by ISO week number, aggregated across divisions."""
    tmp = df.copy()
    tmp["week_of_year"] = pd.to_datetime(tmp[date_col]).dt.isocalendar().week.astype(int)
    grouped = tmp.groupby("week_of_year")[target_col]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["index_vs_average"] = out["mean"] / out["mean"].mean() * 100.0
    return out


def zero_inflation_report(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Where and how often channels go dark, by division."""
    rows = []
    for col in channels:
        by_div = df.groupby("Division")[col].apply(lambda s: float((s == 0).mean()))
        rows.append(
            {
                "channel": col,
                "overall_zero_share": float((df[col] == 0).mean()),
                "max_division_zero_share": float(by_div.max()),
                "n_divisions_always_zero": int((by_div == 1.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_eda(
    df: pd.DataFrame, cfg: DotDict | None = None
) -> dict[str, pd.DataFrame]:
    """Run the full EDA suite and return named tables."""
    cfg = cfg or load_config()
    channels = list(cfg["channels"]["paid"]) + list(cfg["channels"]["organic_controls"])
    channels = [c for c in channels if c in df.columns]
    entity = cfg["data"]["primary"]["entity_col"]
    date_col = cfg["data"]["primary"]["date_col"]
    target = cfg["data"]["primary"]["target_col"]

    tables = {
        "eda_overview": dataset_overview(df, entity, date_col),
        "eda_channel_summary": channel_summary(df, channels),
        "eda_correlation": correlation_matrix(df, channels + [target]),
        "eda_media_target_lag_correlation": media_target_correlation(
            df, channels, target, int(cfg["features"]["adstock"]["max_lag"])
        ),
        "eda_division_profile": division_profile(df, channels, entity, target),
        "eda_seasonality": seasonality_profile(df, date_col, target),
        "eda_zero_inflation": zero_inflation_report(df, channels),
    }
    logger.info("EDA produced %d tables", len(tables))
    return tables
