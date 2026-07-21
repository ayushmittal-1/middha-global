"""
Chat history + COGS persistence — Mongo-backed, per-user-scoped.

One `conversations` document per chat with its messages embedded as an array.
Shape:
    {
      _id: ObjectId,
      convId: str,              # client-supplied UUID, matches the frontend's session_id
      userId: ObjectId,         # owning user
      title: str,
      messages: [
        { role: "user"|"assistant"|"tool_call"|"tool", content: str, createdAt: Date }
      ],
      createdAt: Date,
      updatedAt: Date,
    }

COGS lives in its own `userCogs` collection (one doc per user-SKU pair).

Caveat: Mongo caps documents at 16 MB. With the current policy of persisting
full tool results, a very long conversation that includes large inventory /
report dumps can approach that limit. We refuse the write past 15 MB so the
caller sees a clear error instead of a silent corruption.
"""

import json
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import UpdateOne
from pymongo.errors import DocumentTooLarge

from auth import _db, require_user

MAX_HISTORY = 40  # max messages loaded per session to avoid token overflow
MAX_DOC_BYTES = 15 * 1024 * 1024  # leave headroom under Mongo's 16 MB cap


def _conversations():
    return _db().conversations


def _cogs():
    return _db().userCogs


def _sales_daily():
    return _db().salesDaily


def _inventory_snapshot():
    return _db().inventorySnapshot


def _forecast_cache():
    return _db().forecastCache


def _forecast_settings():
    return _db().forecastSettings


def _storage_cache():
    return _db().storageFeeCache


def _placement_fee_cache():
    return _db().inboundPlacementFeeCache


def _aged_inventory_cache():
    return _db().agedInventoryFeeCache


def _product_settings():
    return _db().productSettings


def _purchase_orders():
    return _db().purchaseOrders


def _user_oid() -> ObjectId:
    user = require_user()
    return ObjectId(str(user["_id"]))


async def init_db():
    """Create indexes if they don't exist. Safe to run repeatedly."""
    await _conversations().create_index([("userId", 1), ("updatedAt", -1)])
    await _conversations().create_index(
        [("convId", 1), ("userId", 1)], unique=True
    )
    await _cogs().create_index([("userId", 1), ("sku", 1)], unique=True)
    await _sales_daily().create_index(
        [("userId", 1), ("sku", 1), ("date", 1)], unique=True
    )
    await _sales_daily().create_index([("userId", 1), ("date", -1)])
    await _inventory_snapshot().create_index(
        [("userId", 1), ("sku", 1), ("date", 1)], unique=True
    )
    await _forecast_cache().create_index(
        [("userId", 1), ("sku", 1)], unique=True
    )
    await _forecast_settings().create_index([("userId", 1)], unique=True)
    await _storage_cache().create_index([("userId", 1)], unique=True)
    await _placement_fee_cache().create_index(
        [("userId", 1)], unique=True
    )
    await _aged_inventory_cache().create_index(
        [("userId", 1)], unique=True
    )
    await _product_settings().create_index(
        [("userId", 1), ("sku", 1)], unique=True
    )
    await _purchase_orders().create_index(
        [("userId", 1), ("poId", 1)], unique=True
    )
    await _purchase_orders().create_index([("userId", 1), ("sku", 1), ("status", 1)])


# ── Conversations ────────────────────────────────────────────────────────

async def create_session(session_id: str, title: str = "New Chat"):
    """Insert a conversation doc if one with this (convId, userId) doesn't
    already exist. The function name keeps the old signature so callers
    don't need to change."""
    user_id = _user_oid()
    now = datetime.now(timezone.utc)
    await _conversations().update_one(
        {"convId": session_id, "userId": user_id},
        {
            "$setOnInsert": {
                "convId": session_id,
                "userId": user_id,
                "title": title,
                "messages": [],
                "createdAt": now,
                "updatedAt": now,
            }
        },
        upsert=True,
    )


async def list_sessions() -> list[dict]:
    """Return the current user's conversations, newest-updated first."""
    user_id = _user_oid()
    cursor = (
        _conversations()
        .find(
            {"userId": user_id},
            {"_id": 0, "convId": 1, "title": 1, "createdAt": 1},
        )
        .sort("updatedAt", -1)
    )
    rows = await cursor.to_list(length=500)
    return [
        {
            "id": r["convId"],
            "title": r.get("title", "New Chat"),
            "created_at": r["createdAt"].isoformat() if r.get("createdAt") else None,
        }
        for r in rows
    ]


async def get_messages(session_id: str) -> list[dict]:
    """Return the last MAX_HISTORY messages for the user's conversation,
    oldest first. Uses $slice so we don't drag the whole array off the
    wire when only the tail matters."""
    user_id = _user_oid()
    doc = await _conversations().find_one(
        {"convId": session_id, "userId": user_id},
        {"messages": {"$slice": -MAX_HISTORY}},
    )
    if not doc:
        return []

    results: list[dict] = []
    for m in doc.get("messages", []):
        role = m.get("role")
        raw = m.get("content")
        if role in ("tool_call", "tool"):
            results.append(json.loads(raw))
        else:
            results.append({"role": role, "content": raw})
    return results


async def save_message(session_id: str, role: str, content: str):
    """Append a message to the conversation. Auto-creates the conversation if
    it doesn't exist yet (matches the old behavior; `create_session` is
    idempotent anyway)."""
    user_id = _user_oid()
    now = datetime.now(timezone.utc)
    msg = {"role": role, "content": content, "createdAt": now}
    try:
        await _conversations().update_one(
            {"convId": session_id, "userId": user_id},
            {
                "$push": {"messages": msg},
                "$set": {"updatedAt": now},
                "$setOnInsert": {
                    "convId": session_id,
                    "userId": user_id,
                    "title": "New Chat",
                    "createdAt": now,
                },
            },
            upsert=True,
        )
    except DocumentTooLarge:
        # Conversation hit Mongo's 16 MB cap — almost always because tool
        # results were persisted verbatim. Surface the cause; the caller
        # can decide whether to truncate or start a fresh conversation.
        raise RuntimeError(
            f"Conversation {session_id} exceeds {MAX_DOC_BYTES // (1024*1024)} MB. "
            "Tool results are being stored in full — consider trimming or starting "
            "a new chat."
        )


async def update_session_title(session_id: str, title: str):
    user_id = _user_oid()
    await _conversations().update_one(
        {"convId": session_id, "userId": user_id},
        {"$set": {"title": title, "updatedAt": datetime.now(timezone.utc)}},
    )


async def delete_session(session_id: str):
    user_id = _user_oid()
    await _conversations().delete_one(
        {"convId": session_id, "userId": user_id}
    )


# ── COGS (Cost of Goods Sold) ─────────────────────────────────────────────

async def upsert_cogs(rows: list[dict]) -> int:
    """Insert or update COGS rows for the current user. Each row needs sku and
    unit_cost; inbound_shipping_per_unit is optional. Returns the count actually
    written.
    """
    user_id = _user_oid()
    written = 0
    now = datetime.now(timezone.utc)
    for r in rows:
        sku = (r.get("sku") or "").strip()
        if not sku:
            continue
        try:
            unit_cost = float(r.get("unit_cost") or 0)
        except (TypeError, ValueError):
            continue
        if unit_cost <= 0:
            continue
        try:
            shipping = float(r.get("inbound_shipping_per_unit") or 0)
        except (TypeError, ValueError):
            shipping = 0.0
        await _cogs().update_one(
            {"userId": user_id, "sku": sku},
            {
                "$set": {
                    "unitCost": unit_cost,
                    "inboundShippingPerUnit": shipping,
                    "updatedAt": now,
                },
                "$setOnInsert": {"userId": user_id, "sku": sku},
            },
            upsert=True,
        )
        written += 1
    return written


async def delete_cogs(sku: str) -> int:
    """Delete a single COGS row for the current user. Returns number of
    docs removed (0 if the SKU wasn't in the collection)."""
    user_id = _user_oid()
    result = await _cogs().delete_one({"userId": user_id, "sku": (sku or "").strip()})
    return result.deleted_count


async def get_cogs(skus: list[str] | None = None) -> list[dict]:
    user_id = _user_oid()
    query: dict = {"userId": user_id}
    if skus:
        query["sku"] = {"$in": skus}
    cursor = _cogs().find(query, {"_id": 0, "userId": 0}).sort("sku", 1)
    rows = await cursor.to_list(length=5000)
    return [
        {
            "sku": r["sku"],
            "unit_cost": r.get("unitCost"),
            "inbound_shipping_per_unit": r.get("inboundShippingPerUnit", 0),
            "updated_at": r["updatedAt"].isoformat() if r.get("updatedAt") else None,
        }
        for r in rows
    ]


# ── Storage fee cache (24h TTL for FBA storage report) ────────────────────


async def get_storage_cache(max_age_hours: int = 24) -> dict | None:
    """Return cached per-SKU monthly storage fee map or None if stale/missing."""
    user_id = _user_oid()
    doc = await _storage_cache().find_one({"userId": user_id})
    if not doc:
        return None
    updated = doc.get("updatedAt")
    if not updated:
        return None
    # Mongo returns naive UTC datetimes — make tz-aware so subtraction works.
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None
    return {
        "per_sku_monthly": doc.get("perSkuMonthly", {}),
        "months_covered": doc.get("monthsCovered", []),
        "updated_at": updated.isoformat(),
    }


async def put_storage_cache(per_sku_monthly: dict, months_covered: list[str]) -> None:
    user_id = _user_oid()
    await _storage_cache().update_one(
        {"userId": user_id},
        {
            "$set": {
                "perSkuMonthly": per_sku_monthly,
                "monthsCovered": months_covered,
                "updatedAt": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"userId": user_id},
        },
        upsert=True,
    )


# ── Inbound placement fee report cache (24h TTL, like storage fees) ─────


async def get_placement_fee_cache(max_age_hours: int = 24) -> dict | None:
    user_id = _user_oid()
    doc = await _placement_fee_cache().find_one({"userId": user_id})
    if not doc:
        return None
    updated = doc.get("updatedAt")
    if not updated:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None
    per_sku = doc.get("perSku", {})
    months = doc.get("monthsCovered", [])
    if doc.get("accessDenied"):
        # per_sku may still hold rates derived from Finances shipment-level
        # placement fees joined with Aurora shipments (see agent.py).
        return {
            "per_sku": per_sku,
            "months_covered": months,
            "updated_at": updated.isoformat(),
            "access_denied": True,
        }
    # Legacy empty cache from a failed 403 — treat as miss so we retry once.
    if not per_sku and not months:
        return None
    return {
        "per_sku": per_sku,
        "months_covered": months,
        "updated_at": updated.isoformat(),
        "access_denied": False,
    }


async def put_placement_fee_cache(
    per_sku: dict,
    months_covered: list[str],
    *,
    access_denied: bool = False,
) -> None:
    user_id = _user_oid()
    await _placement_fee_cache().update_one(
        {"userId": user_id},
        {
            "$set": {
                "perSku": per_sku,
                "monthsCovered": months_covered,
                "accessDenied": access_denied,
                "updatedAt": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"userId": user_id},
        },
        upsert=True,
    )


# ── Aged inventory fee cache (24h TTL) ──────────────────────────────────


async def get_aged_inventory_cache(max_age_hours: int = 24) -> dict | None:
    user_id = _user_oid()
    doc = await _aged_inventory_cache().find_one({"userId": user_id})
    if not doc:
        return None
    updated = doc.get("updatedAt")
    if not updated:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None
    return {
        "per_sku": doc.get("perSku", {}),
        "updated_at": updated.isoformat(),
    }


async def put_aged_inventory_cache(per_sku: dict) -> None:
    user_id = _user_oid()
    await _aged_inventory_cache().update_one(
        {"userId": user_id},
        {
            "$set": {
                "perSku": per_sku,
                "updatedAt": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"userId": user_id},
        },
        upsert=True,
    )


# ── Forecasting: salesDaily / inventorySnapshot / forecastCache ───────────
#
# These helpers take an explicit `user_id` because they are also called from
# the APScheduler nightly job, which has no request context (no ContextVar).
# The agent / UI surface uses the request-scoped wrappers further below.

async def upsert_sales_daily(user_id: ObjectId, rows: list[dict]) -> int:
    """Bulk-upsert daily sales rows for one user. Each row must include
    `sku` and `date` (datetime, UTC midnight). All other fields are
    persisted as-is. Returns the count of operations attempted."""
    if not rows:
        return 0
    ops: list[UpdateOne] = []
    for r in rows:
        sku = (r.get("sku") or "").strip()
        date = r.get("date")
        if not sku or not isinstance(date, datetime):
            continue
        payload = {k: v for k, v in r.items() if k not in ("sku", "date")}
        ops.append(UpdateOne(
            {"userId": user_id, "sku": sku, "date": date},
            {"$set": payload,
             "$setOnInsert": {"userId": user_id, "sku": sku, "date": date}},
            upsert=True,
        ))
    if not ops:
        return 0
    await _sales_daily().bulk_write(ops, ordered=False)
    return len(ops)


async def get_sales_daily_for_user(
    user_id: ObjectId,
    sku: str | None = None,
    since: datetime | None = None,
) -> list[dict]:
    """Return daily units-sold rows for the user, keyed by (sku, date).

    Aurora is the source of truth: when AURORA_DATA_SOURCE=db we aggregate
    the shared `orders` collection on every read, so the modal / restock
    numbers never drift from Aurora. The stockout heuristic runs in-memory
    on the aggregated series so the row shape stays identical to the
    former `salesDaily` cache — callers don't change.

    Falls back to the `salesDaily` cache in SP-API mode.
    """
    from aurora_data import aurora_db_enabled, aggregate_sales_daily_lean

    if aurora_db_enabled():
        end = datetime.now(timezone.utc) + timedelta(days=1)
        start = since or (end - timedelta(days=540))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        rows = await aggregate_sales_daily_lean(user_id, start, end, sku=sku)
        _flag_stockout_runs(rows)
        return rows

    query: dict = {"userId": user_id}
    if sku:
        query["sku"] = sku
    if since:
        query["date"] = {"$gte": since}
    cursor = _sales_daily().find(query, {"_id": 0, "userId": 0}).sort("date", 1)
    return await cursor.to_list(length=None)


def _flag_stockout_runs(
    rows: list[dict],
    *,
    min_run: int = 3,
    velocity_threshold: float = 0.5,
    window_days: int = 28,
) -> None:
    """Same heuristic as `mark_stockouts_for_user`, applied in-memory.

    Runs of ≥`min_run` consecutive zero-sales days flanked by a rolling
    mean above `velocity_threshold` almost certainly reflect inventory
    gaps rather than dead demand — the forecaster excludes them.
    """
    if not rows:
        return
    by_sku: dict[str, list[dict]] = {}
    for r in rows:
        by_sku.setdefault(r.get("sku") or "", []).append(r)

    for sku_rows in by_sku.values():
        sku_rows.sort(key=lambda r: r.get("date"))
        # Densify: fill missing days with synthetic zero rows so runs are
        # detectable. The synthetic rows aren't returned unless a stockout
        # gets flagged on them — see below.
        if not sku_rows:
            continue
        d0 = sku_rows[0].get("date")
        d1 = sku_rows[-1].get("date")
        if not isinstance(d0, datetime) or not isinstance(d1, datetime):
            continue
        by_date = {r["date"]: r for r in sku_rows}
        span_days = (d1 - d0).days + 1
        dense: list[dict] = []
        for i in range(span_days):
            day = d0 + timedelta(days=i)
            dense.append(by_date.get(day) or {
                "sku": sku_rows[0].get("sku"),
                "date": day,
                "asin": sku_rows[0].get("asin"),
                "units_ordered": 0,
                "ordered_revenue": 0.0,
                "stockout_corrected": False,
                "_synthetic": True,
            })
        units = [float(r.get("units_ordered") or 0) for r in dense]
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
                lookback_start = max(0, i - window_days)
                outside = units[lookback_start:i]
                if outside:
                    mean_outside = sum(outside) / len(outside)
                    if mean_outside >= velocity_threshold:
                        for k in range(i, j):
                            dense[k]["stockout_corrected"] = True
                            # Promote synthetic zero-rows into the output
                            # only when they carry a stockout flag; the
                            # forecaster needs to see them.
                            if dense[k].get("_synthetic"):
                                dense[k].pop("_synthetic", None)
                                rows.append(dense[k])
            i = j


async def upsert_inventory_snapshot(user_id: ObjectId, rows: list[dict]) -> int:
    """Bulk-upsert daily inventory snapshots."""
    if not rows:
        return 0
    ops: list[UpdateOne] = []
    for r in rows:
        sku = (r.get("sku") or "").strip()
        date = r.get("date")
        if not sku or not isinstance(date, datetime):
            continue
        payload = {k: v for k, v in r.items() if k not in ("sku", "date")}
        ops.append(UpdateOne(
            {"userId": user_id, "sku": sku, "date": date},
            {"$set": payload,
             "$setOnInsert": {"userId": user_id, "sku": sku, "date": date}},
            upsert=True,
        ))
    if not ops:
        return 0
    await _inventory_snapshot().bulk_write(ops, ordered=False)
    return len(ops)


async def latest_inventory_for_user(user_id: ObjectId) -> dict[str, dict]:
    """Latest on-hand + inbound per SKU, keyed by sku.

    Sources from Aurora's `products.inventory` subdocument — Aurora's sync
    is the source of truth. Our own `inventorySnapshot` ingest currently
    writes zeros (fixed elsewhere), so we bypass it here.

    Shape matches the historic inventorySnapshot rows so `compute_reorder`
    doesn't need to change:
      {sku, date, fulfillable, inbound_shipped, inbound_working,
       reserved, unfulfillable}.
    """
    cursor = _db().products.find(
        {"sellerId": user_id},
        {"sku": 1, "asin": 1, "fnSku": 1, "inventory": 1, "lastSynced": 1,
         "status": 1, "listingStatus": 1, "_id": 0},
    )
    out: dict[str, dict] = {}
    async for p in cursor:
        sku = (p.get("sku") or "").strip()
        if not sku:
            continue
        inv = p.get("inventory") or {}
        # Buyability signal for the Restock UI. Aurora surfaces two
        # independent Amazon fields:
        #   status         — 'Active' | 'Inactive' (listing-level state)
        #   listingStatus  — comma-separated flags like 'DISCOVERABLE, BUYABLE'
        # Both need to be positive for the SKU to actually be sellable;
        # e.g. a suppressed listing is often DISCOVERABLE but not BUYABLE.
        status = (p.get("status") or "").strip()
        listing_status = (p.get("listingStatus") or "").strip()
        is_buyable = (
            status.lower() == "active"
            and "buyable" in listing_status.lower()
        )
        out[sku] = {
            "sku": sku,
            "asin": (p.get("asin") or "").strip() or None,
            "fnsku": (p.get("fnSku") or "").strip() or None,
            "date": p.get("lastSynced"),
            "status": status or None,
            "listing_status": listing_status or None,
            "is_buyable": is_buyable,
            "fulfillable": int(inv.get("fulfillableQuantity") or 0),
            "inbound_shipped": int(inv.get("inboundShippedQuantity") or 0),
            "inbound_working": int(inv.get("inboundWorkingQuantity") or 0),
            "reserved": int(inv.get("reservedQuantity") or 0),
            "unfulfillable": int(inv.get("unfulfillableQuantity") or 0),
            # Seller Central's Manage Inventory page displays "On-hand (FBA)"
            # equal to "Available" (fulfillableQuantity) — Reserved,
            # Unfulfillable, and Inbound are shown as separate, non-additive
            # columns, NOT summed into on-hand. Verified directly against a
            # live Seller Central export (2026-07-21): SKU ASG-KIWI Liquid
            # Polish Black -TA8 showed On-hand 687 == Available 687 with
            # Reserved 4 / Unfulfillable 0 tracked separately — confirming
            # fulfillableQuantity is the correct on-hand value, not
            # fulfillable + reserved + unfulfillable.
            "on_hand": int(inv.get("fulfillableQuantity") or 0),
        }
    return out


async def active_inbound_shipments_for_user(
    user_id: ObjectId,
) -> dict[str, list[dict]]:
    """Return per-SKU list of in-flight FBA shipments arriving at Amazon.

    Source is the shared `shipments` collection written by auroraBackend's
    ShipmentSyncManager. We surface only outstanding units (expected − received)
    so the reorder simulation doesn't double-count what Amazon has already
    checked in.

    Shape per shipment entry:
      {shipment_id, name, status, eta (datetime, UTC midnight),
       qty_outstanding, carrier_name, mode}

    `mode` is inferred from carrierName ('air' | 'ocean' | 'ground').
    """
    # In-flight = not yet received by Amazon and not cancelled. Delivered
    # shipments are excluded because Aurora sets `unitsLocated` shortly
    # after and the on-hand snapshot picks them up — counting them here
    # too would double-book.
    active_statuses = [
        "WORKING", "READY_TO_SHIP", "CHECKED_IN",
        "SHIPPED", "IN_TRANSIT", "RECEIVING",
    ]
    cursor = _db().shipments.find(
        {
            "sellerId": user_id,
            "shipmentType": "fba_fc",
            "status": {"$in": active_statuses},
        },
        {
            "_id": 0,
            "shipmentId": 1, "referenceId": 1, "status": 1, "displayStatus": 1,
            "estimatedDeliveryDate": 1, "shipDate": 1, "lastUpdatedDate": 1,
            "carrierName": 1, "lineItems": 1,
        },
    )
    from forecasting.reorder import infer_shipment_mode

    now = datetime.now(timezone.utc)
    by_sku: dict[str, list[dict]] = {}
    async for shp in cursor:
        eta = (
            shp.get("estimatedDeliveryDate")
            or shp.get("shipDate")
            or shp.get("lastUpdatedDate")
        )
        if eta and eta.tzinfo is None:
            eta = eta.replace(tzinfo=timezone.utc)
        # Undated shipments (rare, mostly WORKING plans) don't help the
        # sim — skip them rather than injecting arbitrary dates.
        if not eta:
            continue
        # Clamp to today: a delivery date in the past on a still-active
        # shipment just means it's late, treat it as landing now.
        if eta < now:
            eta = now
        carrier = shp.get("carrierName") or ""
        mode = infer_shipment_mode(carrier)
        for li in shp.get("lineItems") or []:
            sku = (li.get("sku") or "").strip()
            if not sku:
                continue
            expected = int(li.get("unitsExpected") or 0)
            received = int(li.get("unitsReceived") or 0)
            outstanding = max(0, expected - received)
            if outstanding <= 0:
                continue
            by_sku.setdefault(sku, []).append({
                "shipment_id": shp.get("shipmentId"),
                "name": shp.get("referenceId"),
                "status": shp.get("status"),
                "display_status": shp.get("displayStatus"),
                "eta": eta,
                "qty_outstanding": outstanding,
                "carrier_name": carrier or None,
                "mode": mode,
            })
    # Sort each SKU's shipments by ETA — the reorder sim walks them in order.
    for sku in by_sku:
        by_sku[sku].sort(key=lambda s: s["eta"])
    return by_sku


async def upsert_forecast_cache(user_id: ObjectId, sku: str, payload: dict) -> None:
    # Keep userId/sku out of $set — Mongo rejects an update that touches
    # the same path in both $set and $setOnInsert.
    set_payload = {k: v for k, v in payload.items() if k not in ("userId", "sku")}
    set_payload["generated_at"] = datetime.now(timezone.utc)
    await _forecast_cache().update_one(
        {"userId": user_id, "sku": sku},
        {"$set": set_payload, "$setOnInsert": {"userId": user_id, "sku": sku}},
        upsert=True,
    )


# Request-scoped wrappers — used by FastAPI endpoints and agent tools that
# run inside an authenticated request context.

async def get_sales_daily(
    sku: str | None = None, since: datetime | None = None
) -> list[dict]:
    return await get_sales_daily_for_user(_user_oid(), sku=sku, since=since)


async def get_forecast_cache(skus: list[str] | None = None) -> list[dict]:
    query: dict = {"userId": _user_oid()}
    if skus:
        query["sku"] = {"$in": skus}
    cursor = _forecast_cache().find(query, {"_id": 0, "userId": 0}).sort("sku", 1)
    return await cursor.to_list(length=None)


async def latest_inventory() -> dict[str, dict]:
    return await latest_inventory_for_user(_user_oid())


# ── Forecast settings ─────────────────────────────────────────────────────

DEFAULT_FORECAST_SETTINGS = {
    "lead_time_days": 30,
    "moq": 1,
    "target_cover_days": 90,
    "service_level": 0.95,
    # Transit time from the seller's origin to the Amazon FC once a
    # shipment physically dispatches. Historic defaults matched a rough
    # China → US benchmark; sellers can now override org-wide from the
    # Restock settings panel.
    "air_transit_days": 10,
    "ocean_transit_days": 45,
}


async def get_forecast_settings_for_user(user_id: ObjectId) -> dict:
    doc = await _forecast_settings().find_one({"userId": user_id}) or {}
    return {**DEFAULT_FORECAST_SETTINGS, **{
        k: doc[k] for k in DEFAULT_FORECAST_SETTINGS if k in doc
    }}


async def get_forecast_settings() -> dict:
    return await get_forecast_settings_for_user(_user_oid())


async def update_forecast_settings(patch: dict) -> dict:
    user_id = _user_oid()
    allowed = {k: v for k, v in patch.items() if k in DEFAULT_FORECAST_SETTINGS}
    if not allowed:
        return await get_forecast_settings_for_user(user_id)
    allowed["updatedAt"] = datetime.now(timezone.utc)
    await _forecast_settings().update_one(
        {"userId": user_id},
        {"$set": allowed, "$setOnInsert": {"userId": user_id}},
        upsert=True,
    )
    return await get_forecast_settings_for_user(user_id)


# ── Per-SKU product settings (Actions modal) ─────────────────────────────

DEFAULT_PRODUCT_SETTINGS: dict = {
    # Manufacturing & logistics tab
    "manufacturing_time_days": 35,
    "use_prep_center": False,
    "shipping_to_prep_days": 0,
    "shipping_to_fba_days": None,     # None → falls back to global AIR_TRANSIT_DAYS
    "fba_buffer_days": 0,
    "target_stock_days": None,        # None → falls back to global target_cover_days
    # Forecast tab — SellerBoard-style defaults so recent demand dominates.
    # Users can override per-SKU; leaving all weights at 0 falls back to Prophet.
    "velocity_weights": {"d3": 0.5, "d7": 0.3, "d30": 0.2, "d60": 0.0, "d180": 0.0},
    # Shipping to FBA tab (packing template — pure storage)
    "packing": None,
    # Purchase order tab (supplier — pure storage)
    "supplier": None,
    # Free-text comment (surfaces in restock table)
    "comment": "",
}


def _merge_settings(doc: dict | None) -> dict:
    """Overlay stored fields on the defaults so the caller always gets the
    full shape (missing keys → defaults)."""
    doc = doc or {}
    merged = {**DEFAULT_PRODUCT_SETTINGS}
    for k in DEFAULT_PRODUCT_SETTINGS:
        if k in doc and doc[k] is not None:
            merged[k] = doc[k]
    return merged


async def get_product_settings_for_user(user_id: ObjectId, sku: str) -> dict:
    doc = await _product_settings().find_one(
        {"userId": user_id, "sku": (sku or "").strip()},
        {"_id": 0, "userId": 0},
    )
    merged = _merge_settings(doc)
    # Weights are platform-locked for now — always return the default so
    # every SKU (including ones with old per-SKU overrides in Mongo)
    # renders the same numbers in the Forecast tab and drives the same
    # weighted velocity downstream.
    merged["velocity_weights"] = dict(DEFAULT_PRODUCT_SETTINGS["velocity_weights"])
    return merged


async def get_product_settings(sku: str) -> dict:
    return await get_product_settings_for_user(_user_oid(), sku)


async def all_product_settings_for_user(user_id: ObjectId) -> dict[str, dict]:
    """Bulk-load settings for every SKU that has a saved row. Returned map
    is keyed by sku and holds only the stored (non-default) subset — callers
    that need defaults should merge on top."""
    out: dict[str, dict] = {}
    cursor = _product_settings().find(
        {"userId": user_id}, {"_id": 0, "userId": 0},
    )
    async for r in cursor:
        sku = (r.get("sku") or "").strip()
        if not sku:
            continue
        merged = _merge_settings(r)
        # Same platform-locked weights override as the per-SKU getter.
        merged["velocity_weights"] = dict(DEFAULT_PRODUCT_SETTINGS["velocity_weights"])
        out[sku] = merged
    return out


_SETTINGS_ALLOWED = set(DEFAULT_PRODUCT_SETTINGS.keys())


async def upsert_product_settings(sku: str, patch: dict) -> dict:
    user_id = _user_oid()
    sku = (sku or "").strip()
    if not sku:
        raise ValueError("sku required")
    allowed = {k: v for k, v in patch.items() if k in _SETTINGS_ALLOWED}
    allowed["updatedAt"] = datetime.now(timezone.utc)
    await _product_settings().update_one(
        {"userId": user_id, "sku": sku},
        {"$set": allowed, "$setOnInsert": {"userId": user_id, "sku": sku}},
        upsert=True,
    )
    return await get_product_settings_for_user(user_id, sku)


# ── Purchase orders (drives the "Ordered" column) ────────────────────────

async def list_purchase_orders(status: str | None = None) -> list[dict]:
    query: dict = {"userId": _user_oid()}
    if status:
        query["status"] = status
    cursor = _purchase_orders().find(query, {"_id": 0, "userId": 0}).sort("createdAt", -1)
    rows = await cursor.to_list(length=None)
    for r in rows:
        for k in ("orderDate", "expectedDate", "createdAt", "updatedAt"):
            v = r.get(k)
            if isinstance(v, datetime):
                r[k] = v.isoformat()
    return rows


async def open_ordered_qty_by_sku(user_id: ObjectId) -> dict[str, int]:
    """Sum of outstanding (qty_ordered − qty_received) across all open POs,
    keyed by SKU. Drives the "Ordered" column in the restock table."""
    pipeline = [
        {"$match": {"userId": user_id, "status": "open"}},
        {"$group": {
            "_id": "$sku",
            "outstanding": {"$sum": {"$subtract": [
                {"$ifNull": ["$qtyOrdered", 0]},
                {"$ifNull": ["$qtyReceived", 0]},
            ]}},
        }},
    ]
    out: dict[str, int] = {}
    async for r in _purchase_orders().aggregate(pipeline):
        sku = (r.get("_id") or "").strip()
        if sku:
            out[sku] = int(max(0, r.get("outstanding", 0)))
    return out


async def upsert_purchase_order(patch: dict) -> dict:
    """Create or update a PO. If `poId` is present the record is updated in
    place; otherwise a new poId is minted."""
    import uuid as _uuid
    user_id = _user_oid()
    po_id = (patch.get("poId") or _uuid.uuid4().hex).strip()
    now = datetime.now(timezone.utc)

    set_payload: dict = {"updatedAt": now}
    for src, dst in [
        ("sku", "sku"),
        ("qty_ordered", "qtyOrdered"),
        ("qty_received", "qtyReceived"),
        ("status", "status"),          # 'open' | 'received' | 'cancelled'
        ("supplier", "supplier"),
        ("notes", "notes"),
    ]:
        if src in patch:
            set_payload[dst] = patch[src]
    # ISO strings from the FE → datetime for Mongo
    for src, dst in [("order_date", "orderDate"), ("expected_date", "expectedDate")]:
        if src in patch and patch[src]:
            try:
                set_payload[dst] = datetime.fromisoformat(patch[src].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

    set_payload.setdefault("status", "open")

    await _purchase_orders().update_one(
        {"userId": user_id, "poId": po_id},
        {
            "$set": set_payload,
            "$setOnInsert": {"userId": user_id, "poId": po_id, "createdAt": now},
        },
        upsert=True,
    )
    doc = await _purchase_orders().find_one(
        {"userId": user_id, "poId": po_id}, {"_id": 0, "userId": 0},
    )
    for k in ("orderDate", "expectedDate", "createdAt", "updatedAt"):
        v = (doc or {}).get(k)
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc or {}


async def delete_purchase_order(po_id: str) -> int:
    r = await _purchase_orders().delete_one(
        {"userId": _user_oid(), "poId": (po_id or "").strip()},
    )
    return r.deleted_count
