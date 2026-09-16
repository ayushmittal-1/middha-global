"""Q4 accuracy: separating "not scorable" from "scored badly".

Two situations produced misleading Q4 cells. A SKU whose first sale falls
inside the Oct-Dec holdout has nothing to train on, so every model
correctly predicts 0 and the volume-error formula renders that as
"Naive 0%" — which reads as a model failure. A SKU with zero actual Q4
units divides by zero, so every candidate scores None and the cell went
blank with no explanation.

These tests drive the real `_q4_backtest_one_sku` with synthetic sales so
the classification is exercised end to end, and lock the guarantee that
SKUs which DO have history and demand keep their existing score."""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import q4_backtest
from q4_backtest import _q4_backtest_one_sku, _most_recent_completed_q4

YEAR = 2025
CUTOFF = datetime(YEAR, 10, 1, tzinfo=timezone.utc)
Q4_START = CUTOFF
Q4_END = datetime(YEAR, 12, 31, tzinfo=timezone.utc)


def _rows(start: datetime, n_days: int, units: int, sku="TEST-SKU"):
    return [
        {
            "date": start + timedelta(days=i),
            "sku": sku,
            "units_ordered": units,
            "ad_spend": 0.0,
            "stockout_corrected": False,
        }
        for i in range(n_days)
    ]


@pytest.fixture
def patched(monkeypatch):
    """Feed `_q4_backtest_one_sku` a fixed sales series, no DB."""
    def _install(rows):
        async def fake_sales(user_id, sku=None, since=None):
            return list(rows)

        async def fake_settings(user_id, sku):
            return {}

        monkeypatch.setattr(q4_backtest, "get_sales_daily_for_user", fake_sales)
        monkeypatch.setattr(q4_backtest, "get_product_settings_for_user", fake_settings)
    return _install


def _run(rows, patched):
    patched(rows)
    return asyncio.run(
        _q4_backtest_one_sku(None, "TEST-SKU", CUTOFF, Q4_START, Q4_END, YEAR)
    )


# ── The two unscorable states ────────────────────────────────────────────


def test_first_sale_inside_the_holdout_is_not_scorable(patched):
    """The LOCTITE case: nothing before Oct 1, real sales during Q4."""
    out = _run(_rows(Q4_START, 60, 2), patched)
    assert out["n_pre_cutoff_days"] == 0
    assert out["actual_q4_units"] > 0
    assert out["not_scorable"] == "no_pre_q4_history"


def test_no_winner_is_crowned_off_an_empty_training_series(patched):
    """Regression: this used to report "Naive 0%", implying the model was
    evaluated and failed."""
    out = _run(_rows(Q4_START, 60, 2), patched)
    assert out["winner"] is None
    # The per-model numbers are still recorded — only the verdict is withheld.
    assert out["models"], "per-model results should still be persisted"


def test_zero_q4_demand_is_not_scorable(patched):
    """The Kiwi Sponge / Multi Knobs case: plenty of history, no Q4 sales."""
    out = _run(_rows(CUTOFF - timedelta(days=200), 200, 3), patched)
    assert out["n_pre_cutoff_days"] == 200
    assert out["actual_q4_units"] == 0
    assert out["not_scorable"] == "no_q4_demand"
    assert out["winner"] is None


# ── Everything else must be untouched ────────────────────────────────────


def test_history_plus_demand_still_scores_normally(patched):
    rows = _rows(CUTOFF - timedelta(days=200), 200, 3) + _rows(Q4_START, 92, 3)
    out = _run(rows, patched)
    assert out["not_scorable"] is None
    assert out["winner"] is not None
    assert out["winner"]["accuracy_pct"] is not None


def test_a_genuinely_wrong_model_keeps_its_zero_percent(patched):
    """A SKU with history whose demand collapses in Q4 scores a real 0%.
    That must survive — it is a verdict, not a missing measurement."""
    rows = _rows(CUTOFF - timedelta(days=200), 200, 50) + _rows(Q4_START, 92, 1)
    out = _run(rows, patched)
    assert out["not_scorable"] is None
    assert out["winner"] is not None
    accs = [m["accuracy_pct"] for m in out["models"].values()
            if m.get("accuracy_pct") is not None]
    assert accs, "models should still be scored"
    assert min(accs) == 0.0, "over-prediction by >100% still floors at 0%"


def test_no_sales_at_all_reports_the_history_reason(patched):
    out = _run([], patched)
    assert out["skipped"] == "no sales history"
    assert out["not_scorable"] == "no_pre_q4_history"


def test_target_year_is_the_last_completed_q4():
    _, _, _, year = _most_recent_completed_q4(datetime(2026, 9, 7, tzinfo=timezone.utc))
    assert year == 2025


# ── Wiring: endpoint + frontend ──────────────────────────────────────────


@pytest.fixture(scope="module")
def main_src() -> str:
    return (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontend_html() -> str:
    return (
        Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")


def test_restock_row_carries_the_reason(main_src):
    assert '"q4_not_scorable": q4_not_scorable' in main_src


def test_legacy_docs_get_the_reason_derived_on_read(main_src):
    """Docs written before the field existed must still explain their blank
    without waiting for a fleet re-run."""
    assert 'q4_not_scorable = _q4.get("not_scorable")' in main_src
    assert 'elif not _q4_winner and q4_actual_units == 0:' in main_src


def test_legacy_all_zero_predictions_are_flagged(main_src):
    """Docs predating the field are rescued by the all-zero-prediction
    signature, so the LOCTITE SKUs read correctly before any re-run."""
    assert "_q4_predictions_all_zero" in main_src


def test_all_zero_helper_selects_only_signal_free_docs():
    from main import _q4_predictions_all_zero

    no_signal = {
        "models": {"naive": {"predicted_q4": 0.0}},
        "per_config": [{"predicted_q4": 0.0}, {"predicted_q4": 0.0}],
    }
    assert _q4_predictions_all_zero(no_signal) is True

    # One non-zero prediction anywhere means a model DID have something to
    # say — that 0% is a verdict and must be left alone.
    has_signal = {
        "models": {"naive": {"predicted_q4": 0.0}},
        "per_config": [{"predicted_q4": 0.0}, {"predicted_q4": 59.8}],
    }
    assert _q4_predictions_all_zero(has_signal) is False

    wrong_by_a_lot = {
        "models": {"naive": {"predicted_q4": 2619.6}},
        "per_config": [{"predicted_q4": 400.0}],
    }
    assert _q4_predictions_all_zero(wrong_by_a_lot) is False

    assert _q4_predictions_all_zero({}) is False


def test_cell_renders_a_labelled_na(frontend_html):
    assert "row.q4_not_scorable" in frontend_html
    assert "no_pre_q4_history" in frontend_html
    assert re.search(r'title="\$\{escapeHtml\(why\)\}">n/a<', frontend_html)


def test_scored_cells_are_untouched(frontend_html):
    """The reason branch must sit ahead of the existing render path without
    replacing it."""
    assert "const acc = row.q4_accuracy_pct;" in frontend_html
    assert "if (acc == null) return '<span style=\"color:#888;\">—</span>';" in frontend_html
