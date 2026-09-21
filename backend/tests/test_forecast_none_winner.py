"""A SKU whose backtest scores nothing must still produce a forecast.

`refit_choice` is `bt_winner`, which `_pick_winner` returns as None when no
candidate scored. None matched no branch in the refit chain, and the `in`
test was False, so evaluation reached `.startswith()` on None and raised
AttributeError. refresh_forecasts_for_user catches that with a bare
`except Exception: continue`, so the SKU was skipped in silence and its
previous forecastCache row survived — which is how some SKUs ended up
showing forecasts two months out of date while the nightly reported success.

Fixing the guard took one seller from 8 refreshed SKUs to 50.
"""

from datetime import datetime, timedelta, timezone

import pytest

from forecasting.model import _multimodel_forecast, _pick_winner

TODAY = datetime(2026, 9, 21, tzinfo=timezone.utc)


def test_pick_winner_returns_none_when_nothing_scored():
    """The precondition that makes refit_choice None."""
    assert _pick_winner({}) is None
    assert _pick_winner({"prophet": {"backtest_metrics": {"accuracy_pct": None}}}) is None


def _sparse_rows(n_days: int = 8) -> list[dict]:
    """Too little signal for any candidate to score — the shape that used to
    crash. One unit, long ago, then nothing."""
    return [
        {
            "sku": "SPARSE-SKU",
            "date": TODAY - timedelta(days=400 - i),
            "units_ordered": 1 if i == 0 else 0,
            "stockout_corrected": False,
        }
        for i in range(n_days)
    ]


def test_none_winner_falls_through_to_naive(monkeypatch):
    """Force the exact precondition rather than hoping sparse data produces
    it: with _pick_winner returning None, refit_choice is None and the refit
    chain must not raise."""
    import forecasting.model as model
    monkeypatch.setattr(model, "_pick_winner", lambda *_a, **_k: None)

    result = model._multimodel_forecast(
        "SPARSE-SKU",
        _sparse_rows(),
        {}, None, None, 90, TODAY, TODAY - timedelta(days=30),
    )

    assert result is not None
    assert result.get("method"), "a method must always be chosen"
    assert result.get("forecast"), "a forecast must always be produced"


def test_sparse_sku_does_not_raise_and_still_forecasts():
    result = _multimodel_forecast(
        "SPARSE-SKU",
        _sparse_rows(),
        {},          # all_train_series_by_sku
        None, None,  # lgbm_state, lgbm_module
        90,          # horizon
        TODAY,
        TODAY - timedelta(days=30),   # cutoff
    )

    assert result is not None
    assert result.get("method"), "a method must always be chosen"
    assert result.get("forecast"), "a forecast must always be produced"
