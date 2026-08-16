"""Figures. Matplotlib only, no seaborn styling dependency at import time.

Every function saves to reports/figures and returns the path, so the phase scripts can log
exactly what was produced.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless. The pipeline runs in Docker and in CI without a display.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DotDict, load_config, repo_root  # noqa: E402

logger = logging.getLogger(__name__)

FIGSIZE = (10, 6)
DPI = 130


def _fig_path(name: str, cfg: DotDict | None = None) -> Path:
    cfg = cfg or load_config()
    folder = repo_root() / cfg["paths"]["figures"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.png"


def _save(fig, name: str, cfg: DotDict | None = None) -> Path:
    path = _fig_path(name, cfg)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path)
    return path


def plot_sales_over_time(
    df: pd.DataFrame,
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
    entity_col: str = "Division",
    cfg: DotDict | None = None,
) -> Path:
    agg = df.groupby(date_col)[target_col].sum()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(agg.index, agg.to_numpy(), linewidth=1.6)
    ax.set_title("Total weekly sales across all divisions")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.grid(alpha=0.3)
    return _save(fig, "sales_over_time", cfg)


def plot_channel_volumes(
    df: pd.DataFrame,
    channels: list[str],
    date_col: str = "Calendar_Week",
    cfg: DotDict | None = None,
) -> Path:
    agg = df.groupby(date_col)[channels].sum()
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.2 * len(channels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, channels, strict=False):
        ax.plot(agg.index, agg[col].to_numpy(), linewidth=1.3)
        ax.set_ylabel(col.replace("_", " "), fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("Weekly media volume by channel, summed across divisions")
    axes[-1].set_xlabel("Week")
    return _save(fig, "channel_volumes", cfg)


def plot_division_panel(
    df: pd.DataFrame,
    entity_col: str = "Division",
    date_col: str = "Calendar_Week",
    target_col: str = "Sales",
    max_panels: int = 26,
    cfg: DotDict | None = None,
) -> Path:
    entities = sorted(df[entity_col].unique())[:max_panels]
    n_cols = 5
    n_rows = int(np.ceil(len(entities) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.0 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, entity in zip(axes, entities, strict=False):
        sub = df[df[entity_col] == entity].sort_values(date_col)
        ax.plot(sub[date_col], sub[target_col], linewidth=1.0)
        ax.set_title(str(entity), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    for ax in axes[len(entities):]:
        ax.axis("off")
    fig.suptitle("Weekly sales by division, the heterogeneity the hierarchy pools over", y=1.0)
    return _save(fig, "division_panel", cfg)


def plot_roi_comparison(roi: pd.DataFrame, cfg: DotDict | None = None) -> Path:
    """ROI point estimates with credible intervals, sorted."""
    data = roi.sort_values("roi_mean")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    y = np.arange(len(data))
    ax.errorbar(
        data["roi_mean"],
        y,
        xerr=[
            data["roi_mean"] - data["roi_hdi_lower"],
            data["roi_hdi_upper"] - data["roi_mean"],
        ],
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(1.0, linestyle="--", linewidth=1.0, color="grey")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in data["channel"]])
    ax.set_xlabel("Return per unit of assumed spend")
    ax.set_title("Channel ROI with 94 percent credible intervals")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, "roi_with_intervals", cfg)


def plot_method_comparison(comparison: pd.DataFrame, cfg: DotDict | None = None) -> Path:
    """Interval width, Bayesian against OLS, the visual form of the overconfidence point."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(comparison))
    width = 0.38
    ax.bar(x - width / 2, comparison["ols_relative_ci_width"], width, label="OLS interval")
    ax.bar(x + width / 2, comparison["bayes_relative_hdi_width"], width, label="Bayesian interval")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in comparison["channel"]], fontsize=8)
    ax.set_ylabel("Interval width relative to the point estimate")
    ax.set_title("Uncertainty reported by each method for the same channels")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "method_comparison_intervals", cfg)


def plot_saturation_curves(
    curves: dict[str, pd.DataFrame], cfg: DotDict | None = None
) -> Path:
    """Response curves with the current operating point marked.

    The marker is the part a stakeholder reads first. It answers whether a channel is on
    the steep part of its curve, where more budget still buys volume, or on the flat part,
    where it does not.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for channel, curve in curves.items():
        ax.plot(curve["assumed_spend"], curve["contribution_mean"], label=channel.replace("_", " "))
        ax.fill_between(
            curve["assumed_spend"],
            curve["contribution_hdi_lower"],
            curve["contribution_hdi_upper"],
            alpha=0.15,
        )
        current = curve[np.isclose(curve["multiplier"], 1.0)]
        if not current.empty:
            ax.scatter(
                current["assumed_spend"], current["contribution_mean"], marker="o", s=45, zorder=5
            )
    ax.set_xlabel("Assumed spend")
    ax.set_ylabel("Incremental sales contribution")
    ax.set_title("Saturation curves, markers show the current operating point")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "saturation_curves", cfg)


def plot_backtest(
    bayes: pd.DataFrame,
    linear: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    metric: str = "holdout_mape",
    cfg: DotDict | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(bayes["fold"], bayes[metric], marker="o", label="Hierarchical Bayesian")
    if linear is not None and metric in linear:
        ax.plot(linear["fold"], linear[metric], marker="s", label="Linear MMM")
    if baseline is not None and metric in baseline:
        ax.plot(baseline["fold"], baseline[metric], marker="^", label="Seasonal naive baseline")
    ax.set_xlabel("Backtest fold")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("Out of sample error by fold")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "backtest_comparison", cfg)


def plot_budget_allocation(result_frame: pd.DataFrame, cfg: DotDict | None = None) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(result_frame))
    width = 0.38
    ax.bar(x - width / 2, result_frame["current_spend"], width, label="Current")
    ax.bar(x + width / 2, result_frame["recommended_spend"], width, label="Recommended")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in result_frame["channel"]], fontsize=8)
    ax.set_ylabel("Assumed spend")
    ax.set_title("Current against recommended allocation under the stated constraints")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "budget_allocation", cfg)
