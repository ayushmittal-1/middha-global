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
import re
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


def _aged_surcharge_charges_cache():
    """Actual AIS charged amounts (LONGTERM_STORAGE_FEE_CHARGES report)."""
    return _db().agedSurchargeChargesCache


def _removal_fees_cache():
    """Actual removal fees (REMOVAL_ORDER_DETAIL report) for a date window."""
    return _db().removalFeesCache


def _reimbursements_cache():
    """FBA Reimbursements report (GET_FBA_REIMBURSEMENTS_DATA) for a date window."""
    return _db().reimbursementsCache


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
    await _aged_surcharge_charges_cache().create_index(
        [("userId", 1)], unique=True
    )
    await _removal_fees_cache().create_index(
        [("userId", 1)], unique=True
    )
    await _reimbursements_cache().create_index(
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

def _build_cogs_bulk_ops(
    rows: list[dict], user_id: ObjectId, now: datetime,
) -> list[UpdateOne]:
    """Validate + shape the incoming COGS rows into a list of UpdateOne
    ops. Pure function so the parsing/skip rules are unit-testable
    without any Mongo mocking (audit H3)."""
    ops: list[UpdateOne] = []
    for r in rows or []:
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
        ops.append(
            UpdateOne(
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
            ),
        )
    return ops


async def upsert_cogs(rows: list[dict]) -> int:
    """Bulk insert/update COGS rows for the current user (audit H3).

    Previously issued one `update_one` per row, sequentially awaited —
    an O(N) round-trip pattern that made large CSV uploads slow and
    piled pointless load on Mongo. Now shapes every valid row into a
    pymongo.UpdateOne and dispatches a single `bulk_write(ordered=False)`
    per 1000-row chunk. Returns the count of rows submitted (which,
    combined with `ordered=False`, is the count actually written unless
    Mongo rejected a specific op)."""
    user_id = _user_oid()
    now = datetime.now(timezone.utc)
    ops = _build_cogs_bulk_ops(rows, user_id, now)
    if not ops:
        return 0
    coll = _cogs()
    written = 0
    # Chunk to bound single-command size on very large uploads.
    CHUNK = 1000
    for i in range(0, len(ops), CHUNK):
        batch = ops[i:i + CHUNK]
        await coll.bulk_write(batch, ordered=False)
        written += len(batch)
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

# When Amazon returns only a neighboring month_of_charge (requested July,
# report has June), do not permanently mark July covered. Use a short
# empty-check so every account retries after Amazon publishes, without
# re-creating a storage report on every profitability page load.
_STORAGE_EMPTY_CHECK_HOURS = 6.0


def _parse_cache_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fresh_empty_checked_months(
    empty_checked_at: dict | None,
    *,
    now: datetime | None = None,
    max_age_hours: float = _STORAGE_EMPTY_CHECK_HOURS,
) -> set[str]:
    """Months recently confirmed empty / unpublished (short negative cache)."""
    now = now or datetime.now(timezone.utc)
    fresh: set[str] = set()
    for month, raw_ts in (empty_checked_at or {}).items():
        key = str(month).strip()
        if not re.match(r"^\d{4}-\d{2}$", key):
            continue
        ts = _parse_cache_dt(raw_ts)
        if ts is None:
            continue
        age_h = (now - ts).total_seconds() / 3600.0
        if 0 <= age_h <= max_age_hours:
            fresh.add(key)
    return fresh


async def get_storage_cache(max_age_hours: int = 24) -> dict | None:
    """Return cached per-SKU monthly storage fee map or None if stale/missing.

    ``months_covered`` is effective coverage for the current user only:
    months with real fee rows, plus months empty-checked within 6h.
    Phantom monthsCovered entries from older builds are ignored (and healed).
    """
    from amazon_sp import storage_months_with_fees

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

    per_sku = doc.get("perSkuMonthly", {}) or {}
    fee_months = storage_months_with_fees(per_sku)
    claimed = set(_normalize_month_keys(doc.get("monthsCovered", [])))
    empty_checked = _fresh_empty_checked_months(doc.get("emptyCheckedAt") or {})
    # Never treat a phantom monthsCovered entry as covered.
    effective = sorted(set(fee_months) | (empty_checked - set(fee_months)))

    # Self-heal poisoned docs for this seller (any account / any month).
    # Older builds wrote missing months into monthsCovered / emptyCheckedAt
    # after Amazon returned only a neighboring month_of_charge.
    if claimed - set(fee_months):
        try:
            await _storage_cache().update_one(
                {"userId": user_id},
                {
                    "$set": {
                        "monthsCovered": fee_months,
                        "emptyCheckedAt": {},
                    },
                },
            )
        except Exception:
            pass
        # Do not honor stale empty-checks from the poisoned doc.
        empty_checked = set()
        effective = list(fee_months)

    return {
        "per_sku_monthly": per_sku,
        "months_covered": effective,
        "months_with_fees": fee_months,
        "updated_at": updated.isoformat(),
    }


async def put_storage_cache(
    per_sku_monthly: dict,
    months_covered: list[str],
    empty_checked_at: dict | None = None,
) -> None:
    user_id = _user_oid()
    payload: dict = {
        "perSkuMonthly": per_sku_monthly,
        "monthsCovered": months_covered,
        "updatedAt": datetime.now(timezone.utc),
    }
    if empty_checked_at is not None:
        payload["emptyCheckedAt"] = empty_checked_at
    await _storage_cache().update_one(
        {"userId": user_id},
        {
            "$set": payload,
            "$setOnInsert": {"userId": user_id},
        },
        upsert=True,
    )


def _normalize_month_keys(months) -> list[str]:
    """Coerce monthsCovered from Mongo to sorted YYYY-MM strings only."""
    out: list[str] = []
    for m in months or []:
        s = str(m).strip()
        if re.match(r"^\d{4}-\d{2}$", s):
            out.append(s)
    return sorted(set(out))


async def merge_storage_cache(
    new_per_asin_month: dict,
    new_months: list[str],
    empty_months: list[str] | None = None,
) -> dict:
    """Merge freshly fetched months into the seller's storage cache.

    Works for every authenticated seller (current + future accounts):
    - ``monthsCovered`` is always derived from months that have fee rows.
    - ``empty_months`` get a short empty-check stamp so unpublished months
      retry after ``_STORAGE_EMPTY_CHECK_HOURS`` instead of being poisoned
      for the full 24h cache TTL.
    """
    from amazon_sp import (
        merge_storage_by_asin_month,
        normalize_storage_fee_map,
        storage_months_with_fees,
    )

    user_id = _user_oid()
    doc = await _storage_cache().find_one({"userId": user_id})
    old_map = (doc or {}).get("perSkuMonthly") or {}
    old_empty = dict((doc or {}).get("emptyCheckedAt") or {})

    merged_map = normalize_storage_fee_map(
        merge_storage_by_asin_month(old_map, new_per_asin_month)
    )
    fee_months = storage_months_with_fees(merged_map)

    now = datetime.now(timezone.utc)
    # Drop empty-checks for months that now have fees.
    cleaned_empty: dict[str, datetime] = {}
    for month, raw_ts in old_empty.items():
        key = str(month).strip()
        if key in fee_months:
            continue
        ts = _parse_cache_dt(raw_ts)
        if ts is None:
            continue
        cleaned_empty[key] = ts

    for month in _normalize_month_keys(empty_months):
        if month in fee_months:
            continue
        cleaned_empty[month] = now

    # new_months arg kept for call-site compat; coverage is fee-derived.
    _ = new_months

    await put_storage_cache(merged_map, fee_months, empty_checked_at=cleaned_empty)
    empty_fresh = _fresh_empty_checked_months(cleaned_empty, now=now)
    effective = sorted(set(fee_months) | empty_fresh)
    return {
        "per_sku_monthly": merged_map,
        "months_covered": effective,
        "months_with_fees": fee_months,
        "updated_at": now.isoformat(),
    }


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


async def get_aged_surcharge_charges_cache(
    start_iso: str,
    end_iso: str,
    max_age_hours: int = 24,
) -> dict | None:
    """Cached actual AIS charged amounts for a profitability window.

    Invalidates when the window changes or the doc is older than TTL —
    different date ranges must not reuse each other's charge totals.
    """
    user_id = _user_oid()
    doc = await _aged_surcharge_charges_cache().find_one({"userId": user_id})
    if not doc:
        return None
    # v3 = marketplace-TZ months + reuse validated by snapshot Event Month.
    if int(doc.get("schemaVersion") or 1) < 3:
        return None
    if doc.get("startIso") != start_iso or doc.get("endIso") != end_iso:
        return None
    updated = doc.get("updatedAt")
    if not updated:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None
    # Poisoned 403 / empty denial — retry within 1h so a transient denial
    # does not blank Storage/Aged/Removal for a full day.
    if doc.get("accessDenied") and age_hours > 1:
        return None
    return {
        "per_sku": doc.get("perSku", {}),
        "updated_at": updated.isoformat(),
        "access_denied": bool(doc.get("accessDenied")),
    }


async def put_aged_surcharge_charges_cache(
    per_sku: dict,
    start_iso: str,
    end_iso: str,
    access_denied: bool = False,
) -> None:
    user_id = _user_oid()
    await _aged_surcharge_charges_cache().update_one(
        {"userId": user_id},
        {
            "$set": {
                "perSku": per_sku,
                "startIso": start_iso,
                "endIso": end_iso,
                "accessDenied": access_denied,
                "schemaVersion": 3,
                "updatedAt": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"userId": user_id},
        },
        upsert=True,
    )


async def get_removal_fees_cache(
    start_iso: str,
    end_iso: str,
    max_age_hours: int = 24,
) -> dict | None:
    """Cached Removal Order Detail fees for one profitability window.

    Different start/end must not reuse each other — Amazon is only called for
    the selected filter, and the cache is keyed to that exact window.
    """
    user_id = _user_oid()
    doc = await _removal_fees_cache().find_one({"userId": user_id})
    if not doc:
        return None
    if int(doc.get("schemaVersion") or 0) < 5:
        return None
    if doc.get("startIso") != start_iso or doc.get("endIso") != end_iso:
        return None
    updated = doc.get("updatedAt")
    if not updated:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None
    # Transient 403 should not blank Removal for a full day.
    if doc.get("accessDenied") and age_hours > 1:
        return None
    return {
        "per_sku": doc.get("perSku", {}),
        "updated_at": updated.isoformat(),
        "access_denied": bool(doc.get("accessDenied")),
        "report_total": float(doc.get("reportTotal") or 0),
    }


async def put_removal_fees_cache(
    per_sku: dict,
    start_iso: str,
    end_iso: str,
    access_denied: bool = False,
) -> None:
    user_id = _user_oid()
    report_total = 0.0
    for v in (per_sku or {}).values():
        if isinstance(v, dict):
            report_total += float(v.get("removal_fee") or 0)
        else:
            report_total += float(v or 0)
    await _removal_fees_cache().update_one(
        {"userId": user_id},
        {
            "$set": {
                "perSku": per_sku,
                "startIso": start_iso,
                "endIso": end_iso,
                "accessDenied": access_denied,
                "reportTotal": round(report_total, 2),
                "schemaVersion": 5,
                "updatedAt": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"userId": user_id},
        },
        upsert=True,
    )


async def get_reimbursements_cache(
    start_iso: str,
    end_iso: str,
    max_age_hours: int = 24,
) -> dict | None:
    """Cached FBA Reimbursements for one profitability window."""
    user_id = _user_oid()
    doc = await _reimbursements_cache().find_one({"userId": user_id})
    if not doc:
        return None
    if int(doc.get("schemaVersion") or 0) < 2:
        return None
    if doc.get("startIso") != start_iso or doc.get("endIso") != end_iso:
        return None
    updated = doc.get("updatedAt")
    if not updated:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None
    if doc.get("accessDenied") and age_hours > 1:
        return None
    return {
        "per_sku": doc.get("perSku", {}),
        "updated_at": updated.isoformat(),
        "access_denied": bool(doc.get("accessDenied")),
        "report_total": float(doc.get("reportTotal") or 0),
    }


async def put_reimbursements_cache(
    per_sku: dict,
    start_iso: str,
    end_iso: str,
    report_total: float | None = None,
    access_denied: bool = False,
) -> None:
    user_id = _user_oid()
    if report_total is None:
        report_total = 0.0
        for v in (per_sku or {}).values():
            if isinstance(v, dict):
                report_total += float(v.get("reimbursement") or 0)
            else:
                report_total += float(v or 0)
    await _reimbursements_cache().update_one(
        {"userId": user_id},
        {
            "$set": {
                "perSku": per_sku or {},
                "startIso": start_iso,
                "endIso": end_iso,
                "accessDenied": access_denied,
                "reportTotal": round(float(report_total or 0), 2),
                "schemaVersion": 2,
                "updatedAt": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"userId": user_id},
        },
        upsert=True,
    )


async def clear_profitability_fee_caches(user_id: ObjectId | None = None) -> dict:
    """Drop storage / aged / removal / placement / reimbursement fee caches.

    Used after fee-pipeline fixes so a poisoned empty/403 cache cannot stick
    for the normal 24h TTL. Pass ``user_id`` from scripts; defaults to the
    authenticated request user.
    """
    oid = ObjectId(str(user_id)) if user_id is not None else _user_oid()
    deleted = {}
    for name, coll in (
        ("storage", _storage_cache()),
        ("aged_planning", _aged_inventory_cache()),
        ("aged_charges", _aged_surcharge_charges_cache()),
        ("removal", _removal_fees_cache()),
        ("placement", _placement_fee_cache()),
        ("reimbursements", _reimbursements_cache()),
    ):
        result = await coll.delete_many({"userId": oid})
        deleted[name] = int(result.deleted_count or 0)
    return deleted


# ── Forecasting: salesDaily / inventorySnapshot / forecastCache ───────────
## These helpers take an explicit `user_id` because they are also called from
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
            # Amazon's reservedQuantity is a sum of 3 sub-buckets:
            #   pendingCustomerOrderQuantity → allocated to open orders
            #     not yet shipped
            #   fcProcessingQuantity → at an FC undergoing processing
            #     (moves between bins, damage inspection, etc.)
            #   pendingTransshipmentQuantity → in transit between FCs
            #     (surfaced separately as fc_transfer for the on-hand math)
            # Aurora Node's inventory sync flattens the first two into
            # their own fields on `products.inventory` so we can show
            # the customer-order vs FC-processing bifurcation the seller
            # asks about ("why is my inventory reserved?"). See
            # auroraBackend/src/utils/fbaInventoryFields.js.
            "reserved_customer_order": int(inv.get("reservedPendingCustomerOrder") or 0),
            "reserved_fc_processing": int(inv.get("reservedFcProcessing") or 0),
            "unfulfillable": int(inv.get("unfulfillableQuantity") or 0),
            # Seller Central "On-hand (FBA)" = Available + FC Transfer
            # (pendingTransshipment). Reserved (customer orders + FC
            # processing), Unfulfillable, and Inbound are separate columns
            # and are NOT part of on-hand. Verified 2026-07-21 against live
            # Manage Inventory: Liquid Polish Brown 555+4=559; Shoe Shine
            # Sponge Black 88+1=89; Sponge Combo 101+6=107.
            "fc_transfer": int(inv.get("reservedPendingTransshipment") or 0),
            "on_hand": (
                int(inv.get("fulfillableQuantity") or 0)
                + int(inv.get("reservedPendingTransshipment") or 0)
            ),
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
