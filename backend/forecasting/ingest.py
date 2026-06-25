"""
Forecasting data spine — pulls Amazon SP-API data into `salesDaily` and
`inventorySnapshot` collections.

Two entry points:

- `backfill_user(user, days_back=540)` — initial 18-month pull. Run once
  per user (or whenever they reconnect SP-API).
- `ingest_user_incremental(user)` — yesterday's orders + today's inventory
  snapshot. The nightly APScheduler job calls this for every user with a
  saved SP-API refresh token.

The orders report (`GET_FLAT_FILE_ALL_ORDERS_DATA_BY_ORDER_DATE_GENERAL`)
gives us per-line-item rows with SKU + ASIN + quantity + item-price +
purchase-date. We aggregate to (sku, date) ourselves.

Sessions / page-views / buy-box% and ads-by-day are left as zero-filled
fields for now — they require separate report pulls and will be plugged
in in a follow-up. Schema is forward-compatible.

Stockout correction: when today's `fulfillable == 0`, today's salesDaily
row is flagged `stockout_corrected=true` so the forecaster excludes it
from training. Historical stockout detection during backfill is best-
effort (we don't have historical inventory snapshots until we start
recording them).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Iterable

from bson import ObjectId

import amazon_sp
from auth import current_user, _db
from database import (
    upsert_inventory_snapshot,
    upsert_sales_daily,
)
from pymongo import UpdateOne

log = logging.getLogger("forecasting.ingest")

ORDERS_REPORT = "GET_FLAT_FILE_ALL_ORDERS_DATA_BY_ORDER_DATE_GENERAL"
SALES_TRAFFIC_REPORT = "GET_SALES_AND_TRAFFIC_REPORT"


# ── Helpers ────────────────────────────────────────────────────────────────


def _day_floor(dt: datetime) -> datetime:
    """Drop a datetime to midnight UTC of its date."""
    return datetime.combine(dt.date(), time.min, tzinfo=timezone.utc)


def _parse_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _parse_int(s: str) -> int:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def _parse_purchase_date(s: str) -> datetime | None:
    """The orders flat-file uses ISO 8601 with timezone (e.g.
    '2026-06-21T17:30:54+00:00'). Return midnight UTC of that day."""
    if not s:
        return None
    try:
        # Python 3.11+ handles the offset directly.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _day_floor(dt.astimezone(timezone.utc))


def _aggregate_orders_tsv(text: str) -> list[dict]:
    """Parse the orders flat-file TSV and aggregate to per-(sku, date) rows.

    Cancelled rows are skipped — they shouldn't count as demand. We sum
    quantity and item-price; promos are not subtracted (matches how the
    profitability tool computes gross revenue).
    """
    by_key: dict[tuple[str, datetime], dict] = {}
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    for row in reader:
        sku = (row.get("sku") or row.get("seller-sku") or "").strip()
        if not sku:
            continue
        status = (row.get("item-status") or row.get("order-status") or "").strip().lower()
        if status == "cancelled":
            continue
        date = _parse_purchase_date(row.get("purchase-date") or row.get("last-updated-date") or "")
        if not date:
            continue
        qty = _parse_int(row.get("quantity") or row.get("quantity-shipped") or "0")
        if qty <= 0:
            continue
        price = _parse_float(row.get("item-price") or "0")
        asin = (row.get("asin") or "").strip()

        key = (sku, date)
        agg = by_key.setdefault(key, {
            "sku": sku, "date": date, "asin": asin,
            "units_ordered": 0, "ordered_revenue": 0.0,
            "sessions": 0, "page_views": 0, "buy_box_pct": 0.0,
            "ad_spend": 0.0, "ad_impressions": 0, "ad_clicks": 0,
            "stockout_corrected": False,
        })
        agg["units_ordered"] += qty
        agg["ordered_revenue"] += price
        if asin and not agg.get("asin"):
            agg["asin"] = asin
    return list(by_key.values())


# ── Inventory snapshot ─────────────────────────────────────────────────────


async def _snapshot_inventory_today(user_id: ObjectId) -> int:
    """Write today's FBA inventory snapshot for every SKU the user has.
    Also returns a map of SKU → fulfillable that the caller can use for
    stockout flagging."""
    try:
        data = await amazon_sp.get_inventory_summaries(skus=None)
    except Exception as e:
        log.warning("inventory snapshot failed: %s", e)
        return 0
    today = _day_floor(datetime.now(timezone.utc))
    rows: list[dict] = []
    for s in data.get("payload", {}).get("inventorySummaries", []):
        sku = (s.get("sellerSku") or "").strip()
        if not sku:
            continue
        inv = s.get("inventoryDetails") or {}
        rows.append({
            "sku": sku,
            "date": today,
            "fulfillable": int(inv.get("fulfillableQuantity") or 0),
            "inbound_working": int((inv.get("inboundWorkingQuantity") or 0)),
            "inbound_shipped": int((inv.get("inboundShippedQuantity") or 0)),
            "reserved": int((inv.get("reservedQuantity") or {}).get("totalReservedQuantity") or 0),
            "unfulfillable": int((inv.get("unfulfillableQuantity") or {}).get("totalUnfulfillableQuantity") or 0),
        })
    return await upsert_inventory_snapshot(user_id, rows)


async def _latest_fulfillable_map(user_id: ObjectId) -> dict[str, int]:
    """Pull today's fulfillable counts straight from the snapshot we just
    wrote, so the sales-side ingest can stockout-flag today's row."""
    today = _day_floor(datetime.now(timezone.utc))
    cursor = _db().inventorySnapshot.find(
        {"userId": user_id, "date": today},
        {"sku": 1, "fulfillable": 1, "_id": 0},
    )
    out: dict[str, int] = {}
    async for r in cursor:
        out[r["sku"]] = int(r.get("fulfillable") or 0)
    return out


# ── Orders ingest ──────────────────────────────────────────────────────────


async def _pull_orders_range(
    user_id: ObjectId,
    start: datetime,
    end: datetime,
    stockout_map: dict[str, int] | None = None,
) -> int:
    """Request the orders report for [start, end), parse it, aggregate to
    (sku, date), and upsert. Returns the number of rows written."""
    log.info("orders ingest: %s → %s", start.date(), end.date())
    resp = await amazon_sp.create_report(
        ORDERS_REPORT,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )
    report_id = resp.get("reportId")
    if not report_id:
        log.error("create_report returned no reportId: %s", resp)
        return 0
    text = await amazon_sp.download_report_raw(report_id)
    rows = _aggregate_orders_tsv(text)

    today = _day_floor(datetime.now(timezone.utc))
    if stockout_map:
        for r in rows:
            # We only have *today's* fulfillable count, so we can only
            # stockout-correct rows from today. Historical stockouts will
            # be detected forward-going as we accumulate snapshots.
            if r["date"] == today and stockout_map.get(r["sku"], 1) == 0:
                r["stockout_corrected"] = True

    return await upsert_sales_daily(user_id, rows)


# ── Public entry points ────────────────────────────────────────────────────


async def backfill_user(user: dict, days_back: int = 540) -> dict:
    """Initial pull for a freshly-connected user. ~18 months by default.

    The orders report has a max window of 30 days per request, so we
    chunk into 30-day windows.
    """
    user_id = ObjectId(str(user["_id"]))
    current_user.set(user)
    end = _day_floor(datetime.now(timezone.utc)) + timedelta(days=1)
    earliest = end - timedelta(days=days_back)

    inv_written = await _snapshot_inventory_today(user_id)
    stockout_map = await _latest_fulfillable_map(user_id)

    sales_written = 0
    cursor = earliest
    while cursor < end:
        window_end = min(cursor + timedelta(days=30), end)
        try:
            sales_written += await _pull_orders_range(
                user_id, cursor, window_end, stockout_map
            )
        except Exception as e:
            log.error("backfill window %s–%s failed: %s", cursor.date(), window_end.date(), e)
        cursor = window_end

    try:
        from forecasting.ads_ingest import pull_ads_backfill
        ads_written = await pull_ads_backfill(user, days_back=days_back)
    except Exception as e:
        log.warning("ads backfill failed for user=%s: %s", user_id, e)
        ads_written = 0

    try:
        stockout_flagged = await mark_stockouts_for_user(user_id)
    except Exception as e:
        log.warning("stockout heuristic failed during backfill for user=%s: %s", user_id, e)
        stockout_flagged = 0

    log.info(
        "backfill done: user=%s sales=%d inv=%d ads=%d stockouts=%d",
        user_id, sales_written, inv_written, ads_written, stockout_flagged,
    )
    return {
        "sales_rows": sales_written,
        "inventory_rows": inv_written,
        "ads_rows": ads_written,
        "stockouts_flagged": stockout_flagged,
    }


async def ingest_user_incremental(user: dict) -> dict:
    """Nightly job — pulls yesterday's orders, today's inventory, the
    last 7 days of Sales & Traffic, and runs the stockout heuristic."""
    user_id = ObjectId(str(user["_id"]))
    current_user.set(user)

    inv_written = await _snapshot_inventory_today(user_id)
    stockout_map = await _latest_fulfillable_map(user_id)

    today = _day_floor(datetime.now(timezone.utc))
    # Pull a 3-day window to absorb late-settling orders. Upserts dedupe.
    start = today - timedelta(days=3)
    end = today + timedelta(days=1)
    try:
        sales_written = await _pull_orders_range(user_id, start, end, stockout_map)
    except Exception as e:
        log.error("incremental ingest failed for user=%s: %s", user_id, e)
        sales_written = 0

    try:
        traffic_written = await pull_traffic_window(user_id, days_back=7)
    except Exception as e:
        log.error("traffic ingest failed for user=%s: %s", user_id, e)
        traffic_written = 0

    try:
        from forecasting.ads_ingest import pull_ads_window
        ads_written = await pull_ads_window(user, days_back=7)
    except Exception as e:
        log.error("ads ingest failed for user=%s: %s", user_id, e)
        ads_written = 0

    try:
        stockout_flagged = await mark_stockouts_for_user(user_id)
    except Exception as e:
        log.warning("stockout heuristic failed for user=%s: %s", user_id, e)
        stockout_flagged = 0

    return {
        "sales_rows": sales_written,
        "inventory_rows": inv_written,
        "traffic_rows": traffic_written,
        "ads_rows": ads_written,
        "stockouts_flagged": stockout_flagged,
    }


# ── Sales & Traffic ingest (sessions / page-views / buy-box) ───────────────
#
# The Sales & Traffic report's response object has two top-level arrays —
# `salesAndTrafficByDate` (no SKU breakdown) and `salesAndTrafficByAsin`
# (no date breakdown). To get per-(sku, date) cells we have to request
# the report once per day. That's fine for incremental (~7 calls/night)
# but too expensive for backfill (~540 calls). So backfill is skipped;
# traffic only accumulates going forward.


async def _pull_traffic_for_day(user_id: ObjectId, day: datetime) -> int:
    day_str = day.date().isoformat()
    resp = await amazon_sp.create_report(
        SALES_TRAFFIC_REPORT,
        start_date=day_str,
        end_date=day_str,
        report_options={"dateGranularity": "DAY", "asinGranularity": "SKU"},
        single_marketplace=True,
    )
    report_id = resp.get("reportId")
    if not report_id:
        log.warning("traffic report create returned no reportId for %s: %s", day_str, resp)
        return 0
    try:
        text = await amazon_sp.download_report_raw(report_id)
    except Exception as e:
        log.warning("traffic download failed for %s: %s", day_str, e)
        return 0
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        log.warning("traffic report for %s is not JSON", day_str)
        return 0

    rows: list[dict] = []
    for entry in payload.get("salesAndTrafficByAsin", []):
        sku = (entry.get("sku") or "").strip()
        if not sku:
            continue
        traffic = entry.get("trafficByAsin") or {}
        sessions = int(traffic.get("sessions") or 0)
        page_views = int(traffic.get("pageViews") or 0)
        # Amazon returns 0–100; persist as the same scale we read it in.
        buy_box = float(traffic.get("buyBoxPercentage") or 0)
        rows.append({
            "sku": sku,
            "date": day,
            "sessions": sessions,
            "page_views": page_views,
            "buy_box_pct": buy_box,
        })
    return await upsert_sales_daily(user_id, rows)


async def pull_traffic_window(
    user_id: ObjectId, days_back: int = 7
) -> int:
    """Pull the last `days_back` days of Sales & Traffic, one report per
    day. Skips today (incomplete). Returns total rows touched."""
    today = _day_floor(datetime.now(timezone.utc))
    written = 0
    for offset in range(1, days_back + 1):
        day = today - timedelta(days=offset)
        try:
            written += await _pull_traffic_for_day(user_id, day)
        except Exception as e:
            log.warning("traffic ingest failed for %s: %s", day.date(), e)
    return written


# ── Stockout heuristic (post-processing) ───────────────────────────────────


async def mark_stockouts_for_user(
    user_id: ObjectId,
    min_run: int = 3,
    velocity_threshold: float = 0.5,
    window_days: int = 28,
) -> int:
    """Flag historical zero-sales runs as stockout-corrected so the
    forecaster excludes them.

    Heuristic: for each SKU, walk the dense daily series. A run of
    `min_run` or more consecutive zero-sales days is suspicious if the
    SKU's rolling `window_days` mean *outside* the run is above
    `velocity_threshold`. Such runs almost certainly reflect inventory
    gaps, not collapsing demand — and Amazon doesn't expose historical
    inventory snapshots, so this is the only path.

    Returns the number of rows newly flagged.
    """
    coll = _db().salesDaily
    # All rows for the user, sorted by sku then date so we can stream
    # one SKU at a time without holding the whole history in memory.
    cursor = coll.find(
        {"userId": user_id},
        {"sku": 1, "date": 1, "units_ordered": 1, "stockout_corrected": 1, "_id": 0},
    ).sort([("sku", 1), ("date", 1)])

    by_sku: dict[str, list[dict]] = {}
    async for r in cursor:
        by_sku.setdefault(r["sku"], []).append(r)

    ops: list[UpdateOne] = []
    for sku, rows in by_sku.items():
        units = [float(r.get("units_ordered") or 0) for r in rows]
        already = [bool(r.get("stockout_corrected")) for r in rows]
        if not units:
            continue

        # Find runs of zeros.
        i = 0
        while i < len(units):
            if units[i] != 0:
                i += 1
                continue
            j = i
            while j < len(units) and units[j] == 0:
                j += 1
            run_len = j - i
            if run_len >= min_run:
                # Compute velocity OUTSIDE the run, looking back window_days
                # days before the run start.
                lookback_start = max(0, i - window_days)
                outside = units[lookback_start:i]
                if outside:
                    mean_outside = sum(outside) / len(outside)
                    if mean_outside >= velocity_threshold:
                        for k in range(i, j):
                            if not already[k]:
                                ops.append(UpdateOne(
                                    {"userId": user_id, "sku": sku, "date": rows[k]["date"]},
                                    {"$set": {"stockout_corrected": True}},
                                ))
            i = j

    if ops:
        await coll.bulk_write(ops, ordered=False)
    log.info("stockout heuristic: user=%s flagged=%d", user_id, len(ops))
    return len(ops)


# ── Multi-user scheduler entry ─────────────────────────────────────────────


async def _users_with_sp_token() -> Iterable[dict]:
    """Every user with a saved SP-API refresh token. Loaded eagerly because
    the scheduler runs through them sequentially anyway."""
    cursor = _db().users.find(
        {"amazonRefreshToken": {"$exists": True, "$ne": None, "$ne": ""}},
        {"password": 0},
    )
    return await cursor.to_list(length=None)


async def run_nightly_ingest() -> dict:
    """APScheduler entry point. Iterates every connected user, refreshes
    forecasts immediately after ingest so the dashboard is up-to-date by
    morning."""
    # Local import — avoids importing prophet at FastAPI startup time.
    from forecasting.model import refresh_forecasts_for_user

    users = await _users_with_sp_token()
    results: dict[str, dict] = {}
    for u in users:
        uid = str(u["_id"])
        try:
            ingest_res = await ingest_user_incremental(u)
            refresh_res = await refresh_forecasts_for_user(ObjectId(uid))
            results[uid] = {"ingest": ingest_res, "refresh": refresh_res}
        except Exception as e:
            log.exception("nightly run failed for user %s", uid)
            results[uid] = {"error": str(e)}
    log.info("nightly run done for %d users", len(results))
    return {"users": len(results), "details": results}
