"""External enrichment, joined by week.

Three sources, all joined on the week key and only on the week key:

  holidays  from the holidays package, a per week count of public holidays
  trends    from Google Trends via pytrends, category level search interest
  cpi       from FRED, a macro control

Why by week, and why nothing else. The primary dataset is a division by week panel. These
sources are national and weekly, so they vary over time and not across divisions. Joining
them broadcasts the same value to all 26 divisions in a week, which is correct: a public
holiday in a given week applies everywhere in the country.

What I deliberately did not do. I did not merge the secondary or benchmark datasets into
this panel. They have different channels, different time ranges, different units and no
shared key. Concatenating them would produce a wider table that no longer describes any
real process, and any model fitted to it would be estimating a relationship that does not
exist. They stay in their own pipelines as separate checks.

A caution on the trends control specifically. Search interest for a product category is
partly caused by the advertising being measured. If a Google Trends index is partly a
downstream consequence of the media in the model, controlling for it absorbs some of the
media effect and biases the channel coefficients toward zero. That makes it a bad control
in the technical sense. I fit the model both with and without it and report the difference,
rather than silently including it.

Every fetcher writes its result to data/external and reads from that cache on the next run,
so the pipeline is reproducible offline and does not hammer a rate limited endpoint.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DotDict, load_config, repo_root

logger = logging.getLogger(__name__)


def _external_dir(cfg: DotDict) -> Path:
    path = repo_root() / cfg["paths"]["external"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def week_start(dates: pd.Series) -> pd.Series:
    """Normalise any date to the Monday of its week, so joins line up exactly."""
    d = pd.to_datetime(dates)
    return d - pd.to_timedelta(d.dt.dayofweek, unit="D")


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------
def build_holiday_features(
    weeks: pd.Series, cfg: DotDict | None = None
) -> pd.DataFrame:
    """Per week holiday count and named flags for the retail relevant dates.

    I keep a small number of named flags rather than one dummy per holiday. With 113 weeks
    of data, a dummy per holiday would put several regressors on a single observation each,
    which fits noise perfectly and generalises not at all.
    """
    cfg = cfg or load_config()
    hol_cfg = cfg["enrichment"]["holidays"]
    country = str(hol_cfg["country"])
    years = [int(y) for y in hol_cfg["years"]]

    weeks_norm = pd.Series(sorted(pd.unique(week_start(weeks))), name="week")
    out = pd.DataFrame({"week": weeks_norm})

    try:
        import holidays as holidays_pkg
    except ImportError:
        logger.warning(
            "holidays package not installed, returning zero filled holiday features. "
            "Install it with 'pip install holidays' to enable this control."
        )
        out["n_holidays"] = 0.0
        out["is_major_holiday_week"] = 0.0
        out["is_black_friday_week"] = 0.0
        return out

    calendar = holidays_pkg.country_holidays(country, years=years)
    hol_df = pd.DataFrame(
        {"date": pd.to_datetime(list(calendar.keys())), "name": list(calendar.values())}
    )
    hol_df["week"] = week_start(hol_df["date"])

    counts = hol_df.groupby("week").size().rename("n_holidays")
    out = out.merge(counts, on="week", how="left")
    out["n_holidays"] = out["n_holidays"].fillna(0.0)

    major_terms = ["Christmas", "Thanksgiving", "New Year", "Independence", "Easter"]
    major_weeks = set(
        hol_df.loc[
            hol_df["name"].str.contains("|".join(major_terms), case=False, na=False), "week"
        ]
    )
    out["is_major_holiday_week"] = out["week"].isin(major_weeks).astype(float)

    # Black Friday is not a public holiday but drives more retail volume than most that are.
    thanksgiving_weeks = set(
        hol_df.loc[hol_df["name"].str.contains("Thanksgiving", case=False, na=False), "week"]
    )
    out["is_black_friday_week"] = out["week"].isin(thanksgiving_weeks).astype(float)
    return out


# ---------------------------------------------------------------------------
# Google Trends
# ---------------------------------------------------------------------------
def fetch_google_trends(
    cfg: DotDict | None = None, force_refresh: bool = False
) -> pd.DataFrame:
    """Weekly Google Trends interest, cached to data/external.

    pytrends is an unofficial client for an endpoint Google rate limits aggressively. It
    fails often, and a control this peripheral is not worth a hard dependency, so a failure
    here is logged and returns an empty frame rather than stopping the pipeline.
    """
    cfg = cfg or load_config()
    tr_cfg = cfg["enrichment"]["google_trends"]
    cache = _external_dir(cfg) / str(tr_cfg["cache_file"])

    if cache.exists() and not force_refresh:
        logger.info("Using cached Google Trends data at %s", cache)
        df = pd.read_csv(cache, parse_dates=["week"])
        return df

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload(
            [str(tr_cfg["keyword"])],
            timeframe=str(tr_cfg["timeframe"]),
            geo=str(tr_cfg["geo"]),
        )
        raw = pytrends.interest_over_time()
        if raw.empty:
            raise RuntimeError("Google Trends returned an empty frame")
        out = raw.reset_index().rename(
            columns={"date": "week", str(tr_cfg["keyword"]): "trends_index"}
        )
        out = out[["week", "trends_index"]]
        out["week"] = week_start(out["week"])
        out.to_csv(cache, index=False)
        logger.info("Fetched and cached Google Trends data to %s", cache)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Google Trends fetch failed (%s). Continuing without the trends control. "
            "The model is fitted with and without this control anyway, see docs/methodology.md.",
            exc,
        )
        return pd.DataFrame(columns=["week", "trends_index"])


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def fetch_fred_series(
    cfg: DotDict | None = None, force_refresh: bool = False
) -> pd.DataFrame:
    """Monthly CPI from FRED, forward filled to weekly, cached to data/external.

    CPI is published monthly. Forward filling it to weekly creates a step function, which
    is fine for a slow moving macro control but would be wrong for anything with real weekly
    variation. I use it in levels relative to the series start rather than raw index points,
    so the coefficient reads as a response to cumulative price level change.
    """
    cfg = cfg or load_config()
    fred_cfg = cfg["enrichment"]["fred"]
    cache = _external_dir(cfg) / str(fred_cfg["cache_file"])

    if cache.exists() and not force_refresh:
        logger.info("Using cached FRED data at %s", cache)
        return pd.read_csv(cache, parse_dates=["week"])

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "FRED_API_KEY is not set, skipping the CPI control. Copy .env.example to .env "
            "and add a free key from fred.stlouisfed.org to enable it."
        )
        return pd.DataFrame(columns=["week", "cpi", "cpi_index"])

    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        series = fred.get_series(str(fred_cfg["series_id"]))
        out = series.reset_index()
        out.columns = ["date", "cpi"]
        out["date"] = pd.to_datetime(out["date"])
        weekly = (
            out.set_index("date")
            .resample("W-MON")
            .ffill()
            .reset_index()
            .rename(columns={"date": "week"})
        )
        weekly["week"] = week_start(weekly["week"])
        weekly["cpi_index"] = weekly["cpi"] / weekly["cpi"].iloc[0]
        weekly.to_csv(cache, index=False)
        logger.info("Fetched and cached FRED series to %s", cache)
        return weekly[["week", "cpi", "cpi_index"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED fetch failed (%s). Continuing without the CPI control.", exc)
        return pd.DataFrame(columns=["week", "cpi", "cpi_index"])


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------
def enrich_panel(
    df: pd.DataFrame,
    cfg: DotDict | None = None,
    date_col: str = "Calendar_Week",
    use_trends: bool = True,
    use_cpi: bool = True,
    use_holidays: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Left join the weekly external series onto the panel and return the new columns.

    Left join, always. An inner join would silently drop panel weeks whenever an external
    source has a shorter range, and losing weeks from the middle of a time series would
    break the adstock lag without any visible error.
    """
    cfg = cfg or load_config()
    out = df.copy()
    out["_week_key"] = week_start(out[date_col])
    added: list[str] = []

    if use_holidays and cfg["enrichment"]["holidays"]["enabled"]:
        hol = build_holiday_features(out[date_col], cfg)
        out = out.merge(hol, left_on="_week_key", right_on="week", how="left").drop(
            columns=["week"]
        )
        cols = ["n_holidays", "is_major_holiday_week", "is_black_friday_week"]
        for c in cols:
            out[c] = out[c].fillna(0.0)
        added.extend(cols)

    if use_trends and cfg["enrichment"]["google_trends"]["enabled"]:
        trends = fetch_google_trends(cfg)
        if not trends.empty:
            trends["week"] = week_start(trends["week"])
            out = out.merge(trends, left_on="_week_key", right_on="week", how="left").drop(
                columns=["week"]
            )
            # Interpolate rather than fill with zero: a missing trends value means the
            # series did not cover that week, not that search interest was zero.
            out["trends_index"] = out["trends_index"].interpolate().bfill().ffill()
            added.append("trends_index")

    if use_cpi and cfg["enrichment"]["fred"]["enabled"]:
        cpi = fetch_fred_series(cfg)
        if not cpi.empty:
            cpi["week"] = week_start(cpi["week"])
            out = out.merge(
                cpi[["week", "cpi_index"]], left_on="_week_key", right_on="week", how="left"
            ).drop(columns=["week"])
            out["cpi_index"] = out["cpi_index"].interpolate().bfill().ffill()
            added.append("cpi_index")

    out = out.drop(columns=["_week_key"])
    missing = [c for c in added if out[c].isna().any()]
    if missing:
        raise ValueError(
            f"Enrichment left missing values in {missing}. Fix the join before modelling, "
            "do not fill them blindly."
        )
    logger.info("Enrichment added columns: %s", added)
    return out, added


def collinearity_with_media(
    df: pd.DataFrame, control_cols: list[str], media_cols: list[str]
) -> pd.DataFrame:
    """Correlation of each control with each media channel.

    This is the diagnostic behind the bad control warning at the top of this module. A
    control strongly correlated with a media channel will take credit that belongs to the
    channel, and the coefficient the model reports for that channel will be too small.
    """
    rows = []
    for control in control_cols:
        for media in media_cols:
            if control not in df.columns or media not in df.columns:
                continue
            corr = float(np.corrcoef(df[control].astype(float), df[media].astype(float))[0, 1])
            rows.append(
                {
                    "control": control,
                    "channel": media,
                    "correlation": corr,
                    "flag": "possible bad control" if abs(corr) > 0.5 else "",
                }
            )
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)
