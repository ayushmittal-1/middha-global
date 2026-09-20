"""Ship-by dates are dispatch deadlines, not "start making it" dates.

`manufacturing_time_days` used to be folded into both lead times, which
pushed every ship-by weeks early. Because DEFAULT_PRODUCT_SETTINGS carries
35 and `_merge_settings` overlays defaults on read, compute_reorder saw 35
for *every* SKU — so ocean lead was 35 + 45 = 80 days catalogue-wide and any
SKU with under 80 days of cover silently clamped to "ship today".

Nothing covered this maths, so the regression shipped unnoticed. These tests
pin the contract: ship-by = stockout − transit, manufacturing excluded.
"""

from datetime import datetime, timedelta, timezone

from forecasting.reorder import compute_reorder

TODAY = datetime(2026, 9, 19, tzinfo=timezone.utc)
SETTINGS = {"moq": 1, "target_cover_days": 90, "air_transit_days": 10, "ocean_transit_days": 45}


def _forecast(daily: float, days: int = 120) -> list[dict]:
    return [
        {"date": (TODAY + timedelta(days=i)).date().isoformat(), "p50": daily}
        for i in range(days)
    ]


def _reorder(product_settings: dict, on_hand: int = 100, daily: float = 1.0) -> dict:
    return compute_reorder(
        forecast=_forecast(daily),
        inv_snapshot={"fulfillable": on_hand},
        drivers={},
        settings=SETTINGS,
        shipments=[],
        today=TODAY,
        product_settings=product_settings,
    )


def _days_before_stockout(res: dict, key: str) -> int:
    stockout = datetime.fromisoformat(res["stockout_date"]).date()
    shipby = datetime.fromisoformat(res[key]).date()
    return (stockout - shipby).days


def test_manufacturing_time_does_not_move_ship_by_dates():
    """The whole point: a 35-day manufacturing time must not shift either
    deadline, because manufacturing happens before dispatch."""
    without = _reorder({"manufacturing_time_days": 0})
    with_mfg = _reorder({"manufacturing_time_days": 35})

    assert with_mfg["reorder_by_date_air"] == without["reorder_by_date_air"]
    assert with_mfg["reorder_by_date_ocean"] == without["reorder_by_date_ocean"]


def test_ship_by_equals_stockout_minus_transit():
    """Air uses the FBA transit leg, ocean uses the ocean leg — nothing else,
    when prep and buffer are zero."""
    res = _reorder({"manufacturing_time_days": 35})

    assert _days_before_stockout(res, "reorder_by_date_air") == 10
    assert _days_before_stockout(res, "reorder_by_date_ocean") == 45


def test_prep_and_buffer_still_count():
    """Removing manufacturing must not remove the components that genuinely
    sit between "dispatch" and "sellable", or this would over-correct."""
    res = _reorder({
        "manufacturing_time_days": 35,
        "use_prep_center": True,
        "shipping_to_prep_days": 4,
        "fba_buffer_days": 3,
    })

    assert _days_before_stockout(res, "reorder_by_date_air") == 10 + 4 + 3
    assert _days_before_stockout(res, "reorder_by_date_ocean") == 45 + 4 + 3


def test_ship_by_clamps_to_today_rather_than_showing_a_past_date():
    """Cover shorter than the transit leg means the deadline has passed. The
    value clamps to today so the seller never sees a date in the past."""
    res = _reorder({"manufacturing_time_days": 0}, on_hand=5, daily=1.0)

    assert res["reorder_by_date_ocean"] == TODAY.date().isoformat()
