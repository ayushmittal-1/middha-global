"""
Reorder math — pure functions, no I/O.

The Restock tab used to frame reorder dates as "when the seller places a PO
with their supplier", using a coarse per-org lead time. That was wrong for
FBA sellers: they don't manage supplier POs through this app, they manage
what's already in transit to Amazon.

The new model:

  1. Read the seller's current on-hand at FBA.
  2. Walk every in-flight inbound shipment (from Aurora's `shipments`
     collection) — each has an ETA and outstanding units per SKU.
  3. Simulate the stock timeline: start at on_hand today, deplete at the
     forecasted daily demand, add each shipment's outstanding units on its
     ETA. Find the first day the balance hits zero — the `stockout_date`.
  4. Ship-by-air date  = stockout_date − AIR_TRANSIT_DAYS
     Ship-by-ocean date = stockout_date − OCEAN_TRANSIT_DAYS

Ship-by dates are dispatch deadlines — the seller factors in their own
prep/packing time on top. AIR/OCEAN transit constants are the time from
dispatch → arrival at the Amazon FC.

Safety stock is intentionally zero — the seller asked to be alerted at
pure depletion, and the timeline sim already accounts for real inbound.
`recommended_po_qty` targets `target_cover_days` of forward demand net of
what's already inbound.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


# ── Transit time constants ───────────────────────────────────────────────
# Time from the seller's origin to Amazon FC once a shipment physically
# dispatches. Rough China → US defaults; the compute_reorder settings
# dict (populated from the org-wide DEFAULT_FORECAST_SETTINGS + any
# saved override) supersedes these when it carries `air_transit_days` /
# `ocean_transit_days`. Kept as constants for tests and callers that
# pass an empty settings dict.
AIR_TRANSIT_DAYS = 10
OCEAN_TRANSIT_DAYS = 45


# Common service levels → z-scores. Kept for backwards compat with the
# response shape — safety stock is no longer applied but the coefficient
# is still surfaced in case a future feature wants it.
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


# ── Shipment mode inference ─────────────────────────────────────────────

# Case-insensitive substring keywords. Matched against Aurora's stored
# `carrierName` (which is populated for shipped shipments and often for
# WORKING ones too). `ocean` wins ties over `air` because carriers like
# "OceanX Air" don't exist — ocean keywords are more specific.
_AIR_KEYWORDS = (
    "air", "express", "dhl", "fedex", "sf express", "aramex", "tnt",
    "ups worldwide", "ups saver",
)
_OCEAN_KEYWORDS = (
    "ocean", "sea", "fcl", "lcl", "maersk", "msc", "cosco", "hapag",
    "yang ming", "cma cgm", "one line", "evergreen",
)


def infer_shipment_mode(carrier_name: str) -> str:
    """Return 'air' | 'ocean' | 'ground'. Ground is the fallback for
    empty carriers and Amazon-partnered/small-parcel domestic runs."""
    hay = (carrier_name or "").lower().strip()
    if not hay:
        return "ground"
    if any(k in hay for k in _OCEAN_KEYWORDS):
        return "ocean"
    if any(k in hay for k in _AIR_KEYWORDS):
        return "air"
    return "ground"


# ── Stockout timeline simulation ────────────────────────────────────────

def _simulate_stockout_date(
    on_hand: int,
    daily_demand: float,
    shipments: list[dict],
    today: datetime,
    horizon_days: int = 365,
) -> datetime | None:
    """Walk the stock balance from `today` forward. Deplete at `daily_demand`
    per day and top up on each shipment's ETA. Return the first date the
    balance would hit zero, or None if it doesn't within horizon_days.

    `shipments` is a list of {eta: datetime, qty_outstanding: int}, already
    sorted by ETA.
    """
    if daily_demand <= 0:
        # No demand → never stocks out (this is what "no demand" SKUs hit).
        return None

    balance = float(on_hand)
    cursor_day = today

    for shp in shipments:
        eta = shp["eta"]
        if eta <= cursor_day:
            # Late/current arrivals — treat as landing at the cursor.
            balance += shp["qty_outstanding"]
            continue
        days_between = (eta - cursor_day).total_seconds() / 86400.0
        depletion = days_between * daily_demand
        if balance - depletion <= 0:
            # Stockout falls in this interval.
            days_until = balance / daily_demand
            return cursor_day + timedelta(days=math.ceil(days_until))
        balance -= depletion
        balance += shp["qty_outstanding"]
        cursor_day = eta

    # No more shipments — extrapolate.
    days_until = balance / daily_demand
    if days_until > horizon_days:
        return None
    return cursor_day + timedelta(days=math.ceil(days_until))


# ── Main entry point ────────────────────────────────────────────────────

def compute_reorder(
    forecast: list[dict],
    inv_snapshot: dict | None,
    drivers: dict,
    settings: dict,
    shipments: list[dict] | None = None,
    today: datetime | None = None,
    product_settings: dict | None = None,
) -> dict:
    """All reorder fields the dashboard / drawer / agent surface.

    Parameters
    ----------
    forecast : list of dicts with `p50` — output of model.py.
    inv_snapshot : one row from products.inventory (via
        latest_inventory_for_user) or None.
    drivers : forecast drivers dict — used for `recent_avg` PO ceiling.
    settings : forecast settings (moq, target_cover_days).
    shipments : per-SKU active inbound shipments, each
        {shipment_id, eta, qty_outstanding, mode, carrier_name, status}.
        Empty list = no shipments in-flight for this SKU.
    product_settings : per-SKU overrides from the Actions modal — controls
        lead-time components (manufacturing, prep, FBA transit, buffer) and
        can override the org-wide `target_cover_days`.
    """
    if today is None:
        today = datetime.now(timezone.utc)
    # Normalise `today` to midnight UTC so the returned dates align with
    # the frontend's day-granularity display.
    today = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)

    ps = product_settings or {}

    # ── Per-SKU lead-time build-up ───────────────────────────────────────
    # Air transit gets the seller's real supply chain: manufacturing time
    # → shipping to prep center (only if they use one) → shipping to FBA
    # → FBA buffer. Falls back to our regional default when nothing set.
    air_default = int(settings.get("air_transit_days") or AIR_TRANSIT_DAYS)
    ocean_default = int(settings.get("ocean_transit_days") or OCEAN_TRANSIT_DAYS)
    fba_transit = ps.get("shipping_to_fba_days")
    fba_transit = int(fba_transit) if fba_transit is not None else air_default
    mfg_time = int(ps.get("manufacturing_time_days") or 0)
    to_prep = int(ps.get("shipping_to_prep_days") or 0) if ps.get("use_prep_center") else 0
    buffer = int(ps.get("fba_buffer_days") or 0)
    air_lead_days = mfg_time + to_prep + fba_transit + buffer
    ocean_lead_days = mfg_time + to_prep + ocean_default + buffer

    moq = int(settings.get("moq", 1))
    target_cover = int(
        ps.get("target_stock_days") or settings.get("target_cover_days", 90)
    )
    service_level = float(settings.get("service_level", 0.95))

    shipments = shipments or []

    inv = inv_snapshot or {}
    on_hand = int(inv.get("fulfillable", 0))
    reserved = int(inv.get("reserved", 0))
    # Amazon-side "sent to FBA" — units checked in / in-transit at Amazon,
    # distinct from our own shipments-collection `inbound` sum.
    sent_to_fba = int(inv.get("inbound_shipped", 0))
    inbound_working = int(inv.get("inbound_working", 0))
    unfulfillable = int(inv.get("unfulfillable", 0))
    inbound_outstanding = sum(int(s.get("qty_outstanding", 0)) for s in shipments)

    horizon_p50 = [float(r.get("p50", 0)) for r in forecast]
    empty_response = {
        "on_hand": on_hand,
        "reserved": reserved,
        "sent_to_fba": sent_to_fba,
        "inbound_working": inbound_working,
        "unfulfillable": unfulfillable,
        "inbound": inbound_outstanding,
        "avg_daily_demand": 0.0,
        "safety_stock": 0,
        "reorder_point": 0,
        "days_of_cover": None,
        "stockout_date": None,
        "reorder_by_date": None,
        "reorder_by_date_air": None,
        "reorder_by_date_ocean": None,
        "reorder_by_date_sea": None,  # legacy alias
        "air_transit_days": air_lead_days,
        "ocean_transit_days": ocean_lead_days,
        "recommended_po_qty": 0,
        "service_level": service_level,
        "moq": moq,
        "target_cover_days": target_cover,
        "inbound_shipments_count": len(shipments),
        "next_shipment_eta": (
            shipments[0]["eta"].date().isoformat() if shipments else None
        ),
        "next_shipment_qty": (
            int(shipments[0]["qty_outstanding"]) if shipments else None
        ),
    }
    if not horizon_p50:
        return empty_response

    avg_daily_overall = sum(horizon_p50) / len(horizon_p50)

    days_of_cover: float | None = None
    if avg_daily_overall > 0:
        days_of_cover = (on_hand + inbound_outstanding) / avg_daily_overall

    stockout_dt = _simulate_stockout_date(
        on_hand=on_hand,
        daily_demand=avg_daily_overall,
        shipments=shipments,
        today=today,
    )
    stockout_date = stockout_dt.date().isoformat() if stockout_dt else None

    def _ship_by(transit_days: int) -> str | None:
        if stockout_dt is None:
            return None
        latest = stockout_dt - timedelta(days=transit_days)
        # Clamp negative values ("you're already too late") to today so
        # the seller sees "ship NOW" rather than a past date.
        if latest < today:
            latest = today
        return latest.date().isoformat()

    reorder_by_date_air = _ship_by(air_lead_days)
    reorder_by_date_ocean = _ship_by(ocean_lead_days)

    # Recommend enough to hit target_cover days after the ocean shipment
    # arrives, net of what's already inbound. Rough model — good enough
    # to drive the "how much to ship" column.
    target_units = target_cover * avg_daily_overall
    raw_po = target_units - (on_hand + inbound_outstanding)

    # Sanity ceiling — same guard as before. Prophet on sparse SKUs can
    # balloon the projection; cap against recent 28-day rate × 180 days.
    recent_avg = float(drivers.get("recent_avg") or 0)
    if recent_avg > 0:
        ceiling = max(recent_avg * 180.0, 50.0)
    else:
        ceiling = 30.0
    raw_po = min(raw_po, ceiling)
    recommended_po_qty = _round_up_to_moq(raw_po, moq)

    return {
        "on_hand": on_hand,
        "reserved": reserved,
        "sent_to_fba": sent_to_fba,
        "inbound_working": inbound_working,
        "unfulfillable": unfulfillable,
        "inbound": inbound_outstanding,
        "avg_daily_demand": round(avg_daily_overall, 2),
        # Safety stock kept in the response for backwards compat, but the
        # sim uses pure depletion so we report zero here.
        "safety_stock": 0,
        "reorder_point": int(math.ceil(avg_daily_overall * air_lead_days)),
        "days_of_cover": round(days_of_cover, 1) if days_of_cover is not None else None,
        "stockout_date": stockout_date,
        # Legacy field — some agent code still reads it. Point at the
        # ocean date since that's the earlier / more conservative one.
        "reorder_by_date": reorder_by_date_ocean,
        "reorder_by_date_air": reorder_by_date_air,
        "reorder_by_date_ocean": reorder_by_date_ocean,
        "reorder_by_date_sea": reorder_by_date_ocean,  # legacy alias
        "air_transit_days": air_lead_days,
        "ocean_transit_days": ocean_lead_days,
        "recommended_po_qty": recommended_po_qty,
        "service_level": service_level,
        "moq": moq,
        "target_cover_days": target_cover,
        "inbound_shipments_count": len(shipments),
        "next_shipment_eta": (
            shipments[0]["eta"].date().isoformat() if shipments else None
        ),
        "next_shipment_qty": (
            int(shipments[0]["qty_outstanding"]) if shipments else None
        ),
    }
