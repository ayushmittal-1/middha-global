"""
Reorder math — pure functions, no I/O.

Takes a forecast (output of `model.py`), the current inventory snapshot,
and the user's forecast settings; returns the reorder fields the dashboard
and the agent surface.

Formulas:
- safety_stock = z(service_level) * std(daily_demand) * sqrt(lead_time)
- reorder_point = avg_daily_demand_over_lead_time * lead_time + safety_stock
- days_of_cover = (on_hand + inbound) / avg_daily_demand
- reorder_by_date = today + max(0, days_of_cover - lead_time) days
- recommended_po_qty = max(0, ceil(target_cover_days * avg_daily -
                                   on_hand - inbound - safety_stock))
  rounded up to the next MOQ multiple.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


# Common service levels → z-scores. We snap to the nearest bucket rather
# than carrying scipy as a dependency for this single lookup.
_Z_TABLE: list[tuple[float, float]] = [
    (0.80, 0.84), (0.85, 1.04), (0.90, 1.28),
    (0.95, 1.65), (0.975, 1.96), (0.99, 2.33),
]


def _z(service_level: float) -> float:
    return min(_Z_TABLE, key=lambda kv: abs(kv[0] - service_level))[1]


def _round_up_to_moq(qty: float, moq: int) -> int:
    if moq <= 1:
        return max(0, math.ceil(qty))
    if qty <= 0:
        return 0
    return moq * math.ceil(qty / moq)


def compute_reorder(
    forecast: list[dict],
    inv_snapshot: dict | None,
    drivers: dict,
    settings: dict,
    today: datetime | None = None,
) -> dict:
    """All four reorder fields plus the supporting numbers the UI shows.

    `forecast` is the list-of-dicts emitted by `model.py` (each row has
    `p50`). `inv_snapshot` is one row from `inventorySnapshot` or None
    (treated as zero). `drivers.recent_std` is used for the safety-stock
    sigma — that's the std of observed daily demand, which is what the
    standard formula wants (not the forecast spread).
    """
    if today is None:
        today = datetime.now(timezone.utc)

    lead_time = int(settings.get("lead_time_days", 30))
    moq = int(settings.get("moq", 1))
    target_cover = int(settings.get("target_cover_days", 90))
    service_level = float(settings.get("service_level", 0.95))

    on_hand = int((inv_snapshot or {}).get("fulfillable", 0))
    inbound = int((inv_snapshot or {}).get("inbound_working", 0)) + \
              int((inv_snapshot or {}).get("inbound_shipped", 0))

    horizon_p50 = [float(r.get("p50", 0)) for r in forecast]
    if not horizon_p50:
        return {
            "on_hand": on_hand, "inbound": inbound,
            "avg_daily_demand": 0.0, "safety_stock": 0,
            "reorder_point": 0, "days_of_cover": None,
            "reorder_by_date": None, "recommended_po_qty": 0,
        }

    lead = horizon_p50[:lead_time] or horizon_p50
    avg_daily_lead = sum(lead) / len(lead)
    avg_daily_overall = sum(horizon_p50) / len(horizon_p50)

    sigma = float(drivers.get("recent_std", 0.0))
    safety_stock = _z(service_level) * sigma * math.sqrt(max(lead_time, 1))

    reorder_point = avg_daily_lead * lead_time + safety_stock

    available = on_hand + inbound
    days_of_cover = (available / avg_daily_overall) if avg_daily_overall > 0 else None

    reorder_by_date = None
    if days_of_cover is not None:
        slack_days = max(0.0, days_of_cover - lead_time)
        reorder_by_date = (today + timedelta(days=slack_days)).date().isoformat()

    target_units = target_cover * avg_daily_overall + safety_stock
    raw_po = target_units - available
    recommended_po_qty = _round_up_to_moq(raw_po, moq)

    return {
        "on_hand": on_hand,
        "inbound": inbound,
        "avg_daily_demand": round(avg_daily_overall, 2),
        "avg_daily_demand_lead": round(avg_daily_lead, 2),
        "safety_stock": int(math.ceil(safety_stock)),
        "reorder_point": int(math.ceil(reorder_point)),
        "days_of_cover": round(days_of_cover, 1) if days_of_cover is not None else None,
        "reorder_by_date": reorder_by_date,
        "recommended_po_qty": recommended_po_qty,
        "service_level": service_level,
        "lead_time_days": lead_time,
        "moq": moq,
        "target_cover_days": target_cover,
    }
