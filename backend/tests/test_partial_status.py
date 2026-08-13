"""Machine-readable partial-result signal on the profitability response.

Review issue #4: previously incompleteness was only signalled via a
free-text `warnings` array — a frontend that forgot to render it would
show a plausibly-precise profit number that was actually missing fee
data. Now the response also carries a top-level `complete` bool and
`partial_sections` list so downstream code can refuse to render an
incomplete total. These tests lock in the pure helper."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from agent import build_incompleteness_report


def test_all_done_reports_complete():
    report = build_incompleteness_report(
        {"finances": True, "storage": True, "ads": True}
    )
    assert report == {"complete": True, "partial_sections": []}


def test_empty_task_map_is_trivially_complete():
    """An empty request (no critical loaders ran) is a degenerate case —
    should not falsely flag as incomplete."""
    assert build_incompleteness_report({}) == {
        "complete": True,
        "partial_sections": [],
    }


def test_single_missing_section_flags_incomplete():
    report = build_incompleteness_report(
        {"finances": True, "storage": False, "ads": True}
    )
    assert report["complete"] is False
    assert report["partial_sections"] == ["storage"]


def test_multiple_missing_sections_sorted_alphabetically():
    """Sorted output is a hidden contract — tests would break if the
    helper started returning insertion order, which frontends might sort
    on their side and get inconsistent groupings."""
    report = build_incompleteness_report(
        {
            "finances": False,
            "storage": True,
            "reimbursements": False,
            "product fees": True,
            "aged inventory": False,
        }
    )
    assert report["complete"] is False
    assert report["partial_sections"] == [
        "aged inventory", "finances", "reimbursements",
    ]


def test_all_missing_reports_all_sections():
    report = build_incompleteness_report(
        {"finances": False, "storage": False, "ads": False}
    )
    assert report["complete"] is False
    assert set(report["partial_sections"]) == {"finances", "storage", "ads"}
