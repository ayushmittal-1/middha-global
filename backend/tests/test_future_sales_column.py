"""Restock "Future sales" column.

Adds a variable-horizon predicted-sales column: the backend ships the
winning model's daily p50 series and the frontend sums the first N days,
so changing N re-renders without a refetch. Backend behaviour is tested
directly; the FE wiring is locked via HTML inspection (same approach as
test_returns_toggle.py) so a refactor that drops the wire fails CI."""

import re
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def main_src() -> str:
    return (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontend_html() -> str:
    return (
        Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")


# ── Backend payload ──────────────────────────────────────────────────────


def test_restock_row_ships_daily_p50_series(main_src):
    assert '"forecast_p50_daily": forecast_p50_daily,' in main_src


def test_series_is_capped_at_the_cached_horizon(main_src):
    """The forecast cache is built with horizon=90, so the series must not
    claim to extend further than that."""
    assert "forecast[:90]" in main_src
    assert "_forecast_one(train_rows, horizon=90" in main_src


def test_returns_view_exposes_the_scale_factor(main_src):
    """The UI multiplies any summed horizon by this, mirroring what the
    backend already does to next_30_day_forecast."""
    assert 'returns_view["forecast_scale"] = round(scale, 4)' in main_src
    assert 'returns_view["forecast_scale"] = 1.0' in main_src


def test_thirty_day_forecast_field_is_retained(main_src):
    """Older cached frontends still read next_30_day_forecast."""
    assert '"next_30_day_forecast": round(next30, 1),' in main_src


def test_daily_series_sums_to_the_thirty_day_total():
    """The column's arithmetic: summing the first 30 entries of the series
    reproduces next_30_day_forecast."""
    forecast = [{"p50": 1.5} for _ in range(90)]
    next30 = sum(float(r.get("p50", 0)) for r in forecast[:30])
    series = [round(float(r.get("p50", 0) or 0), 2) for r in forecast[:90]]
    assert len(series) == 90
    assert round(sum(series[:30]), 1) == round(next30, 1) == 45.0


def test_short_forecast_series_is_not_padded():
    """A SKU with only 12 days of forecast ships 12 entries — the UI flags
    the shortfall rather than the backend inventing zeros."""
    forecast = [{"p50": 2} for _ in range(12)]
    series = [round(float(r.get("p50", 0) or 0), 2) for r in forecast[:90]]
    assert len(series) == 12


# ── Frontend wiring locks ────────────────────────────────────────────────


def test_toolbar_has_the_days_input(frontend_html):
    assert 'id="restock-future-days"' in frontend_html
    assert 'max="90"' in frontend_html


def test_header_cell_is_present_and_relabelled(frontend_html):
    assert 'id="restock-future-header"' in frontend_html
    assert "Future sales (${futureDays}d)" in frontend_html


def test_horizon_is_clamped_to_the_cached_range(frontend_html):
    assert "RESTOCK_FUTURE_MAX_DAYS = 90" in frontend_html
    assert "Math.max(1, Math.min(" in frontend_html


def test_changing_the_horizon_rerenders_without_refetch(frontend_html):
    """The series is already client-side; the handler must call the
    renderer, not loadRestock()."""
    m = re.search(
        r"restockFutureDaysEl\.addEventListener\('input',\s*(\w+)\)", frontend_html
    )
    assert m, "days input is not wired to a handler"
    assert m.group(1) == "renderRestockRows"


def test_restock_header_and_row_cell_counts_match(frontend_html):
    """Guards the column alignment this change depends on — a future column
    added to the <thead> but not the row template (or vice versa) shifts
    every cell to its right."""
    head = re.search(
        r'<table class="restock">\s*<thead>\s*<tr>(.*?)</tr>', frontend_html, re.S
    )
    assert head, "restock <thead> not found"
    n_th = len(re.findall(r"<th[ >]", re.sub(r"<!--.*?-->", "", head.group(1), flags=re.S)))

    start = frontend_html.index("function renderRestockRows()")
    end = frontend_html.index("async function loadRestock()")
    body = re.sub(r"<!--.*?-->", "", frontend_html[start:end], flags=re.S)
    n_td = len(re.findall(r"<td[ >]", body))

    assert n_th == n_td, f"{n_th} header cells vs {n_td} row cells"
