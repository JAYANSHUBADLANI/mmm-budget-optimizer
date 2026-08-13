"""Tests for the data quality and cleaning layer.

The first test is the one that matters most. It is a regression test for the actual bug I
found in the primary dataset, so if a future refactor makes the duplicate handling silent,
this fails.
"""

from __future__ import annotations

import pytest

from mmm import cleaning, validation


def test_duplicate_division_is_detected(panel_with_duplicate_division):
    """The primary key check must fail on a division that appears twice."""
    report = validation.DataQualityReport()
    result = validation.check_primary_key(
        panel_with_duplicate_division, ["Division", "Calendar_Week"], report
    )
    assert result.passed is False
    assert result.severity == "error"
    assert "D" in result.detail["offenders"]
    assert result.detail["n_violating_rows"] == 60  # 30 original plus 30 duplicated


def test_balanced_panel_check_catches_the_extra_rows(panel_with_duplicate_division):
    report = validation.DataQualityReport()
    result = validation.check_balanced_panel(
        panel_with_duplicate_division, "Division", "Calendar_Week", report
    )
    # Distinct weeks are unchanged by exact duplication, so this specific check passes.
    # That is the point of running several checks: duplication hides from some of them.
    assert result.passed is True


def test_cleaning_removes_exact_duplicates_and_records_what_it_did(
    panel_with_duplicate_division,
):
    cleaned, log = cleaning.fix_duplicate_rows(
        panel_with_duplicate_division, "Division", "Calendar_Week"
    )
    assert log.exact_duplicates_removed == 30
    assert log.rows_in - log.rows_out == 30
    assert "D" in log.duplicate_entities
    assert cleaned.duplicated(subset=["Division", "Calendar_Week"]).sum() == 0
    assert any("Removed 30 exact duplicate rows" in note for note in log.notes)


def test_cleaning_refuses_to_guess_on_conflicting_rows(panel_with_conflicting_rows):
    """A key collision with different values is a source conflict, not a duplicate.

    Dropping it silently would be the dangerous behaviour, so the function raises.
    """
    with pytest.raises(ValueError, match="source conflict"):
        cleaning.fix_duplicate_rows(
            panel_with_conflicting_rows, "Division", "Calendar_Week"
        )


def test_conflicting_rows_can_be_kept_explicitly(panel_with_conflicting_rows):
    cleaned, log = cleaning.fix_duplicate_rows(
        panel_with_conflicting_rows, "Division", "Calendar_Week", allow_conflicts=True
    )
    assert cleaned.duplicated(subset=["Division", "Calendar_Week"]).sum() == 0
    assert log.conflicting_key_rows > 0


def test_derived_column_is_detected_and_dropped(tiny_panel):
    """Overall_Views is exactly Paid_Views plus Organic_Views in the fixture."""
    out, log = cleaning.drop_derived_columns(
        tiny_panel, "Overall_Views", ["Organic_Views", "Paid_Views"]
    )
    assert "Overall_Views" not in out.columns
    assert "Overall_Views" in log.dropped_columns


def test_non_derived_column_is_kept(tiny_panel):
    frame = tiny_panel.copy()
    frame["Overall_Views"] = frame["Overall_Views"] * 1.3  # no longer the exact sum
    out, log = cleaning.drop_derived_columns(
        frame, "Overall_Views", ["Organic_Views", "Paid_Views"]
    )
    assert "Overall_Views" in out.columns
    assert log.dropped_columns == []


def test_date_continuity_catches_a_missing_week(tiny_panel):
    broken = tiny_panel.drop(
        tiny_panel[
            (tiny_panel["Division"] == "A")
            & (tiny_panel["Calendar_Week"] == tiny_panel["Calendar_Week"].unique()[5])
        ].index
    )
    report = validation.DataQualityReport()
    result = validation.check_date_continuity(broken, "Division", "Calendar_Week", report)
    assert result.passed is False
    assert "A" in result.detail["offenders"]


def test_zero_variance_channel_is_flagged(tiny_panel):
    flat = tiny_panel.copy()
    flat["Google_Impressions"] = 1000.0
    report = validation.DataQualityReport()
    result = validation.check_zero_variance(flat, ["Google_Impressions"], report)
    assert result.passed is False
    assert "Google_Impressions" in result.detail["zero_variance_columns"]


def test_panel_index_is_added_and_ordered(tiny_panel):
    out = cleaning.add_panel_index(tiny_panel, "Division", "Calendar_Week")
    assert {"entity_idx", "time_idx", "week_of_year", "year"} <= set(out.columns)
    assert out["entity_idx"].nunique() == 4
    assert out["time_idx"].max() == 29
    first = out[out["Division"] == "A"].sort_values("Calendar_Week")
    assert list(first["time_idx"]) == sorted(first["time_idx"])


def test_report_raises_only_on_error_severity():
    report = validation.DataQualityReport()
    report.add(
        validation.CheckResult("warn_only", passed=False, severity="warning", message="x")
    )
    report.raise_if_failed()  # must not raise
    report.add(
        validation.CheckResult("blocking", passed=False, severity="error", message="y")
    )
    with pytest.raises(ValueError, match="blocking"):
        report.raise_if_failed()


def test_full_clean_pipeline_produces_a_balanced_panel(panel_with_duplicate_division, cfg):
    cleaned, log = cleaning.clean_primary(panel_with_duplicate_division, cfg)
    counts = cleaned.groupby("Division").size()
    assert counts.nunique() == 1
    assert len(cleaned) == 120
    assert isinstance(log.to_dict(), dict)
