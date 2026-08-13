"""Data quality checks.

This module exists because of a real bug I found in the primary dataset, not a
hypothetical one. Division Z ships 226 rows where every other division ships 113. The
extra rows are exact duplicates of the first 113. If I had loaded the file and gone
straight to modelling, Division Z would have carried twice the weight of every other
division in the likelihood, and the pooled channel coefficients would have been pulled
toward whatever Z happens to do.

The checks below are written generically, so they catch the same class of problem in any
panel, and the pipeline fails loudly rather than continuing on data it does not trust.
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
class CheckResult:
    name: str
    passed: bool
    severity: str  # "error" blocks the pipeline, "warning" is recorded and allowed through
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
        }


class DataQualityReport:
    """Collects check results and decides whether the pipeline may continue."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        log = logger.info if result.passed else (
            logger.error if result.severity == "error" else logger.warning
        )
        log("[%s] %s", result.name, result.message)
        return result

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.as_row() for r in self.results])

    def raise_if_failed(self) -> None:
        if self.failures:
            lines = "\n".join(f"  - {r.name}: {r.message}" for r in self.failures)
            raise ValueError(f"Data quality checks failed:\n{lines}")


def check_exact_duplicates(df: pd.DataFrame, report: DataQualityReport) -> CheckResult:
    """Fully duplicated rows across every column."""
    dup_mask = df.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    return report.add(
        CheckResult(
            name="exact_duplicate_rows",
            passed=n_dup == 0,
            severity="warning",
            message=(
                f"Found {n_dup} rows that are exact duplicates of another row"
                if n_dup
                else "No exact duplicate rows"
            ),
            detail={"n_duplicate_rows": n_dup},
        )
    )


def check_primary_key(
    df: pd.DataFrame, keys: list[str], report: DataQualityReport
) -> CheckResult:
    """The pair (Division, Calendar_Week) should uniquely identify a row.

    This is the check that catches the Division Z problem directly. A panel with a
    duplicated entity week is not a panel, it is a panel plus noise with extra weight.
    """
    dup = df.duplicated(subset=keys, keep=False)
    offenders = (
        df.loc[dup, keys[0]].value_counts().to_dict() if dup.any() and keys else {}
    )
    n = int(dup.sum())
    return report.add(
        CheckResult(
            name="primary_key_unique",
            passed=n == 0,
            severity="error",
            message=(
                f"{n} rows violate uniqueness on {keys}. Offending entities: {offenders}"
                if n
                else f"{keys} uniquely identifies every row"
            ),
            detail={"n_violating_rows": n, "offenders": offenders},
        )
    )


def check_balanced_panel(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    report: DataQualityReport,
    expected_periods: int | None = None,
) -> CheckResult:
    """Every entity should be observed for the same number of periods."""
    counts = df.groupby(entity_col)[date_col].nunique()
    modal = int(counts.mode().iloc[0]) if len(counts) else 0
    target = expected_periods if expected_periods is not None else modal
    offenders = counts[counts != target].to_dict()
    return report.add(
        CheckResult(
            name="balanced_panel",
            passed=len(offenders) == 0,
            severity="error",
            message=(
                f"Entities with a period count other than {target}: {offenders}"
                if offenders
                else f"Panel is balanced, every entity has {target} distinct periods"
            ),
            detail={"expected_periods": target, "offenders": offenders},
        )
    )


def check_no_missing(df: pd.DataFrame, report: DataQualityReport) -> CheckResult:
    missing = df.isna().sum()
    offenders = {k: int(v) for k, v in missing.items() if v > 0}
    return report.add(
        CheckResult(
            name="no_missing_values",
            passed=len(offenders) == 0,
            severity="error",
            message=(
                f"Missing values found: {offenders}" if offenders else "No missing values"
            ),
            detail={"missing_by_column": offenders},
        )
    )


def check_non_negative(
    df: pd.DataFrame, columns: list[str], report: DataQualityReport
) -> CheckResult:
    """Impressions, views and sales cannot be negative."""
    present = [c for c in columns if c in df.columns]
    offenders = {c: int((df[c] < 0).sum()) for c in present if (df[c] < 0).any()}
    return report.add(
        CheckResult(
            name="non_negative_measures",
            passed=len(offenders) == 0,
            severity="error",
            message=(
                f"Negative values in measure columns: {offenders}"
                if offenders
                else "All measure columns are non negative"
            ),
            detail={"negative_counts": offenders},
        )
    )


def check_derived_aggregate(
    df: pd.DataFrame,
    aggregate_col: str,
    component_cols: list[str],
    report: DataQualityReport,
    rel_tol: float = 1e-6,
) -> CheckResult:
    """Test whether a column is just the sum of other columns.

    I ran this because Overall_Views sits next to Paid_Views and Organic_Views and looked
    suspicious. If it is the exact sum, it is a linear combination of two regressors
    already in the design matrix. Including it would make the media block rank deficient,
    which shows up as an unstable OLS fit and as divergent or non identified posteriors in
    the Bayesian model. Either way the individual channel effects stop meaning anything.
    """
    if aggregate_col not in df.columns:
        return report.add(
            CheckResult(
                name=f"derived_check_{aggregate_col}",
                passed=True,
                severity="warning",
                message=f"{aggregate_col} not present, nothing to test",
            )
        )
    present = [c for c in component_cols if c in df.columns]
    if not present:
        return report.add(
            CheckResult(
                name=f"derived_check_{aggregate_col}",
                passed=True,
                severity="warning",
                message="No component columns available for the derived column test",
            )
        )
    total = df[present].sum(axis=1)
    agg = df[aggregate_col]
    denom = agg.abs().clip(lower=1.0)
    max_rel_diff = float(((agg - total).abs() / denom).max())
    is_derived = max_rel_diff <= rel_tol
    return report.add(
        CheckResult(
            name=f"derived_check_{aggregate_col}",
            passed=not is_derived,
            severity="warning",
            message=(
                f"{aggregate_col} equals the sum of {present} to within {rel_tol}. "
                "I exclude it from the design matrix to avoid perfect collinearity."
                if is_derived
                else f"{aggregate_col} is not the exact sum of {present}, "
                f"max relative difference {max_rel_diff:.4g}"
            ),
            detail={
                "is_derived": bool(is_derived),
                "max_relative_difference": max_rel_diff,
                "components": present,
            },
        )
    )


def check_date_continuity(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    report: DataQualityReport,
    freq_days: int = 7,
) -> CheckResult:
    """Weekly data should step by exactly seven days inside every entity.

    Adstock is a lag operator over rows. If a week is missing, the lag silently spans a
    gap and the carryover estimate is wrong, so this check matters more than it looks.
    """
    offenders: dict[str, int] = {}
    for entity, grp in df.sort_values(date_col).groupby(entity_col):
        deltas = grp[date_col].diff().dropna().dt.days
        bad = int((deltas != freq_days).sum())
        if bad:
            offenders[str(entity)] = bad
    return report.add(
        CheckResult(
            name="weekly_date_continuity",
            passed=len(offenders) == 0,
            severity="error",
            message=(
                f"Entities with irregular week spacing: {offenders}"
                if offenders
                else f"Every entity steps by exactly {freq_days} days"
            ),
            detail={"offenders": offenders},
        )
    )


def check_zero_variance(
    df: pd.DataFrame, columns: list[str], report: DataQualityReport
) -> CheckResult:
    """A channel with no variation carries no information about its own effect."""
    present = [c for c in columns if c in df.columns]
    offenders = [c for c in present if float(np.nanstd(df[c].to_numpy(dtype=float))) == 0.0]
    return report.add(
        CheckResult(
            name="channel_has_variance",
            passed=len(offenders) == 0,
            severity="error",
            message=(
                f"Zero variance channels, their coefficients are not identified: {offenders}"
                if offenders
                else "Every channel varies"
            ),
            detail={"zero_variance_columns": offenders},
        )
    )


def run_primary_checks(
    df: pd.DataFrame,
    cfg: DotDict | None = None,
    stage: str = "raw",
) -> DataQualityReport:
    """Run the full check suite against the primary panel.

    I run this twice, once on the raw file and once after cleaning, so the report shows
    exactly which checks the cleaning step fixed.
    """
    cfg = cfg or load_config()
    primary = cfg["data"]["primary"]
    entity, date_col = primary["entity_col"], primary["date_col"]
    measures = (
        list(cfg["channels"]["paid"])
        + list(cfg["channels"]["organic_controls"])
        + list(cfg["channels"]["suspected_derived"])
        + [primary["target_col"]]
    )

    report = DataQualityReport()
    logger.info("Running primary data quality checks at stage '%s'", stage)

    check_no_missing(df, report)
    check_exact_duplicates(df, report)
    check_primary_key(df, [entity, date_col], report)
    check_balanced_panel(
        df, entity, date_col, report, expected_periods=int(primary["expected_weeks"])
    )
    check_non_negative(df, measures, report)
    check_zero_variance(df, list(cfg["channels"]["paid"]), report)
    for agg in cfg["channels"]["suspected_derived"]:
        check_derived_aggregate(
            df, agg, list(cfg["channels"]["organic_controls"]) + ["Paid_Views"], report
        )
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        check_date_continuity(df, entity, date_col, report)

    return report
