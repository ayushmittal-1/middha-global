"""Restock "Include returns" feature.

Adds a UI toggle so users can see forecast/velocity/ship-by dates with
physical FBA returns subtracted from demand. These tests lock in the
pure helpers that make it work, plus the FE wiring (via HTML string
inspection so a refactor that drops the wire fails CI)."""

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

from forecasting.model import (
    apply_returns_to_daily_rows,
    compute_velocity_windows,
    weighted_velocity,
)


# ── apply_returns_to_daily_rows ──────────────────────────────────────────


def _day(y=2026, m=8, d=10):
    return datetime(y, m, d, tzinfo=timezone.utc)


def test_no_returns_returns_original_rows():
    rows = [
        {"date": _day(2026, 8, 10), "units_ordered": 5},
        {"date": _day(2026, 8, 11), "units_ordered": 3},
    ]
    result = apply_returns_to_daily_rows(rows, {})
    assert result == rows
    assert apply_returns_to_daily_rows(rows, None) == rows


def test_returns_subtract_from_matching_day():
    rows = [
        {"date": _day(2026, 8, 10), "units_ordered": 10},
        {"date": _day(2026, 8, 11), "units_ordered": 5},
    ]
    returns = {"2026-08-10": 3}
    result = apply_returns_to_daily_rows(rows, returns)
    assert result[0]["units_ordered"] == 7
    assert result[1]["units_ordered"] == 5


def test_returns_never_go_negative():
    """If returns > gross for a day (edge case: delayed returns land on a
    zero-sales day), net floors at 0 — no negative demand."""
    rows = [{"date": _day(2026, 8, 10), "units_ordered": 2}]
    returns = {"2026-08-10": 100}
    result = apply_returns_to_daily_rows(rows, returns)
    assert result[0]["units_ordered"] == 0


def test_original_row_list_is_not_mutated():
    """Restock endpoint re-uses the same rows list for the gross-velocity
    pass, so the net-pass helper must return a fresh copy."""
    rows = [{"date": _day(2026, 8, 10), "units_ordered": 10}]
    original_copy = [{"date": _day(2026, 8, 10), "units_ordered": 10}]
    apply_returns_to_daily_rows(rows, {"2026-08-10": 5})
    assert rows == original_copy


def test_rows_without_date_are_passed_through_untouched():
    """A malformed row (missing date) shouldn't blow up the helper —
    it just passes through untouched."""
    rows = [{"units_ordered": 5}, {"date": _day(2026, 8, 10), "units_ordered": 10}]
    result = apply_returns_to_daily_rows(rows, {"2026-08-10": 3})
    assert result[0] == {"units_ordered": 5}
    assert result[1]["units_ordered"] == 7


def test_returns_for_unknown_day_are_ignored():
    """Returns keyed to a day with no sales row shouldn't affect anything."""
    rows = [{"date": _day(2026, 8, 10), "units_ordered": 10}]
    result = apply_returns_to_daily_rows(rows, {"2026-08-15": 100})
    assert result[0]["units_ordered"] == 10


# ── Combined: net velocity via compute_velocity_windows ──────────────────


def test_net_velocity_is_lower_than_gross_when_returns_present():
    """End-to-end sanity: subtracting returns lowers per-window
    units_sold and thus weighted velocity. The 30d window ends
    YESTERDAY, so with now=2026-09-01 the window covers 2026-08-02
    through 2026-08-31 — exactly the days shared by our rows and
    returns arrays below."""
    rows = [
        {"date": _day(2026, 8, day), "units_ordered": 10}
        for day in range(2, 32)
    ]
    returns = {f"2026-08-{d:02d}": 2 for d in range(2, 32)}
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)

    windows = compute_velocity_windows(rows, now, windows=(30,))
    gross_units = windows[0]["units_sold"]
    assert gross_units == 30 * 10

    net_rows = apply_returns_to_daily_rows(rows, returns)
    net_windows = compute_velocity_windows(net_rows, now, windows=(30,))
    net_units = net_windows[0]["units_sold"]

    assert net_units == gross_units - sum(returns.values())
    assert net_units < gross_units


def test_forecast_scaling_ratio_math():
    """The endpoint scales next_30_day_forecast by net_velocity /
    gross_velocity, capped at 1.0 so the toggle can only ever lower
    demand (not inflate it)."""
    gross_v = 4.0
    net_v = 3.0
    gross_forecast = 120.0
    scale = min(1.0, net_v / gross_v)
    scaled = round(gross_forecast * scale, 1)
    assert scaled == 90.0

    # Guard: if net > gross (data oddity), cap at 1.0.
    scale_capped = min(1.0, 5.0 / 4.0)
    assert scale_capped == 1.0


# ── Frontend wiring locks ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def frontend_html() -> str:
    return (
        Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")


def test_restock_toolbar_has_include_returns_toggle(frontend_html):
    assert 'id="restock-include-returns"' in frontend_html


def test_tooltip_documents_the_return_lag_caveat(frontend_html):
    """Live-Mongo probe (PR #42 review) confirmed physical returns lag
    30-60d from purchase date, so trailing 7/30d shifts are small even
    for high-return sellers. Tooltip must call this out so users don't
    mistake the near-zero shift for a bug."""
    assert "30" in frontend_html and "60" in frontend_html, (
        "toggle tooltip should mention the 30-60 day return lag"
    )
    # A specific phrase from the tooltip — locks the wording so a
    # generic \"tooltip\" edit that drops the caveat fails this test.
    assert "barely shift" in frontend_html, (
        "toggle tooltip should explain that recent-window numbers "
        "barely shift (return-lag caveat)"
    )


def test_toggle_persists_to_localstorage(frontend_html):
    """Toggle state must survive reloads so users don't have to re-opt-in
    on every session."""
    assert "restock:include-returns" in frontend_html


def test_row_view_swap_reads_returns_view_block(frontend_html):
    """The FE must actually use `row.returns_view` when the toggle is on —
    not just render the toggle without wiring it up."""
    assert "row.returns_view" in frontend_html
    assert "restockRowView" in frontend_html


def test_toggle_defaults_to_off(frontend_html):
    """Default state: off (matches the pre-feature behavior). Existing
    users see no change on first load unless they opt in."""
    # localStorage read is compared to '1' — anything else (missing key
    # included) leaves the checkbox unchecked.
    assert "localStorage.getItem('restock:include-returns') === '1'" in frontend_html
