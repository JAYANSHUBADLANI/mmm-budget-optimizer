"""Cleaning the primary panel.

The centrepiece here is the Division Z fix. I want to be precise about what I did and
why, because "I dropped duplicates" says nothing on its own unless it also says what the
duplicates were and how they got there.

What I found: Division Z has 226 rows. Every other division has 113. The extra 113 rows
are byte for byte repeats of the first 113, same weeks, same impressions, same sales.

Why it matters: the Bayesian model pools information across divisions. A division that
appears twice contributes twice as many likelihood terms, so it gets roughly double the
influence on the pooled channel coefficients and on the group level hyperparameters. It
also breaks the balanced panel assumption the adstock lag operator relies on, because
within Division Z the week sequence restarts partway through the frame.

What I did: dropped the exact duplicate rows, keeping the first occurrence, then
re-verified that (Division, Calendar_Week) is unique and every division has 113 weeks.

What I did not do: I did not deduplicate on the key alone. Dropping by key would silently
discard genuinely conflicting rows if any existed. I check first that the duplicates are
exact, and if a key collision has differing measure values the function refuses to guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import DotDict, load_config

logger = logging.getLogger(__name__)


@dataclass
class CleaningLog:
    """A record of everything the cleaning step changed, so the README can quote it."""

    rows_in: int = 0
    rows_out: int = 0
    exact_duplicates_removed: int = 0
    duplicate_entities: dict[str, int] = field(default_factory=dict)
    conflicting_key_rows: int = 0
    dropped_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed": self.rows_in - self.rows_out,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "duplicate_entities": self.duplicate_entities,
            "conflicting_key_rows": self.conflicting_key_rows,
            "dropped_columns": self.dropped_columns,
            "notes": self.notes,
        }


def diagnose_duplicates(
    df: pd.DataFrame, entity_col: str, date_col: str
) -> dict[str, Any]:
    """Describe the duplication before touching anything.

    Separating diagnosis from repair is the part that turns a one line dedupe into a data
    quality finding I can actually talk about.
    """
    key = [entity_col, date_col]
    key_dupes = df.duplicated(subset=key, keep=False)
    exact_dupes = df.duplicated(keep=False)

    # A key collision where the measure values differ is a different, worse problem than
    # an exact repeat. It means two sources disagree, and no automatic rule is safe.
    conflicting = int((key_dupes & ~exact_dupes).sum())

    per_entity = (
        df.loc[key_dupes, entity_col].value_counts().to_dict() if key_dupes.any() else {}
    )
    row_counts = df.groupby(entity_col).size().to_dict()

    return {
        "n_rows": int(len(df)),
        "n_key_duplicate_rows": int(key_dupes.sum()),
        "n_exact_duplicate_rows": int(exact_dupes.sum()),
        "n_conflicting_key_rows": conflicting,
        "duplicate_rows_by_entity": {str(k): int(v) for k, v in per_entity.items()},
        "row_count_by_entity": {str(k): int(v) for k, v in row_counts.items()},
    }


def fix_duplicate_rows(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    log: CleaningLog | None = None,
    allow_conflicts: bool = False,
) -> tuple[pd.DataFrame, CleaningLog]:
    """Remove exact duplicate rows and confirm the key is unique afterwards."""
    log = log or CleaningLog()
    log.rows_in = int(len(df))

    diag = diagnose_duplicates(df, entity_col, date_col)
    log.duplicate_entities = diag["duplicate_rows_by_entity"]
    log.conflicting_key_rows = diag["n_conflicting_key_rows"]

    if diag["n_conflicting_key_rows"] > 0 and not allow_conflicts:
        raise ValueError(
            f"{diag['n_conflicting_key_rows']} rows share ({entity_col}, {date_col}) but "
            "hold different measure values. That is a source conflict, not a duplicate. "
            "I will not pick a winner automatically. Inspect the rows and decide, or pass "
            "allow_conflicts=True to keep the first occurrence."
        )

    before = len(df)
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    log.exact_duplicates_removed = int(before - len(out))

    if allow_conflicts:
        before_key = len(out)
        out = out.drop_duplicates(subset=[entity_col, date_col], keep="first").reset_index(
            drop=True
        )
        removed_conflicting = int(before_key - len(out))
        if removed_conflicting:
            log.notes.append(
                f"Kept the first row for {removed_conflicting} conflicting key collisions "
                "because allow_conflicts was set."
            )

    log.rows_out = int(len(out))
    if log.exact_duplicates_removed:
        entities = ", ".join(sorted(log.duplicate_entities)) or "none identified"
        log.notes.append(
            f"Removed {log.exact_duplicates_removed} exact duplicate rows. "
            f"Affected entities: {entities}."
        )
        logger.warning(
            "Removed %d exact duplicate rows affecting %s",
            log.exact_duplicates_removed,
            entities,
        )

    remaining = out.duplicated(subset=[entity_col, date_col]).sum()
    if remaining:
        raise ValueError(
            f"{remaining} key duplicates survived cleaning. The panel is still invalid."
        )
    return out, log


def drop_derived_columns(
    df: pd.DataFrame,
    aggregate_col: str,
    component_cols: list[str],
    log: CleaningLog | None = None,
    rel_tol: float = 1e-6,
) -> tuple[pd.DataFrame, CleaningLog]:
    """Drop an aggregate column when it is exactly the sum of columns already present.

    I only drop when the identity actually holds in this data. If Overall_Views turns out
    not to be the exact sum, it carries independent information and I keep it.
    """
    log = log or CleaningLog()
    if aggregate_col not in df.columns:
        return df, log
    present = [c for c in component_cols if c in df.columns]
    if not present:
        return df, log

    total = df[present].sum(axis=1)
    denom = df[aggregate_col].abs().clip(lower=1.0)
    max_rel_diff = float(((df[aggregate_col] - total).abs() / denom).max())

    if max_rel_diff <= rel_tol:
        out = df.drop(columns=[aggregate_col])
        log.dropped_columns.append(aggregate_col)
        log.notes.append(
            f"Dropped {aggregate_col} because it equals {' + '.join(present)} exactly. "
            "Keeping it would put a perfect linear combination of existing regressors "
            "into the design matrix."
        )
        logger.info("Dropped derived column %s", aggregate_col)
        return out, log

    log.notes.append(
        f"Kept {aggregate_col}. It is not the exact sum of {present}, maximum relative "
        f"difference {max_rel_diff:.4g}, so it carries information those columns do not."
    )
    return df, log


def add_panel_index(
    df: pd.DataFrame, entity_col: str, date_col: str
) -> pd.DataFrame:
    """Add integer indices the model uses for the hierarchical dimensions."""
    out = df.sort_values([entity_col, date_col]).reset_index(drop=True)
    entities = sorted(out[entity_col].unique())
    weeks = sorted(out[date_col].unique())
    out["entity_idx"] = out[entity_col].map({e: i for i, e in enumerate(entities)})
    out["time_idx"] = out[date_col].map({w: i for i, w in enumerate(weeks)})
    out["week_of_year"] = pd.to_datetime(out[date_col]).dt.isocalendar().week.astype(int)
    out["year"] = pd.to_datetime(out[date_col]).dt.year.astype(int)
    return out


def clean_primary(
    df: pd.DataFrame, cfg: DotDict | None = None, allow_conflicts: bool = False
) -> tuple[pd.DataFrame, CleaningLog]:
    """Full cleaning pass over the primary panel."""
    cfg = cfg or load_config()
    primary = cfg["data"]["primary"]
    entity, date_col = primary["entity_col"], primary["date_col"]

    out, log = fix_duplicate_rows(df, entity, date_col, allow_conflicts=allow_conflicts)

    for agg in cfg["channels"]["suspected_derived"]:
        out, log = drop_derived_columns(
            out, agg, list(cfg["channels"]["organic_controls"]) + ["Paid_Views"], log
        )

    out = add_panel_index(out, entity, date_col)

    n_entities = out[entity].nunique()
    n_weeks = out[date_col].nunique()
    log.notes.append(
        f"Final panel: {n_entities} divisions by {n_weeks} weeks, {len(out)} rows."
    )
    logger.info("Cleaned panel: %d divisions, %d weeks, %d rows", n_entities, n_weeks, len(out))
    return out, log
