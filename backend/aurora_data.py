"""
Aurora MongoDB integration — DB-first reads with SP-API fallback.

Set AURORA_DATA_SOURCE=db (default) so the AI backend reads Aurora's
synced collections (orders, products, ads, users). Missing required
fields trigger live Amazon API calls via data_resolver.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from auth import _db
from amazon_sp import MARKETPLACE_NAMES, resolve_marketplace
from amazon_sp import parse_fee_detail_lines, split_bundled_fulfillment_total

# Match auroraBackend dashboardMetrics — cancelled/unfulfillable orders are not sales.
# `Pending` is also excluded: those orders aren't confirmed yet and Amazon can flip
# them to Canceled before the buyer is charged, so counting them inflates velocity.
CANCELLED_ORDER_STATUSES = frozenset({
    "Canceled", "Cancelled", "Unfulfillable", "Pending", "InvoiceUnconfirmed",
})

# Re-export order helpers from aurora_orders for a single import surface.
from aurora_orders import (  # noqa: F401
    fetch_orders_with_items,
    get_order_items,
    list_orders,
)


def aurora_db_enabled() -> bool:
    source = os.getenv("AURORA_DATA_SOURCE") or os.getenv("AURORA_ORDERS_SOURCE", "db")
    return source.strip().lower() == "db"


def _money_amount(block: Optional[dict]) -> float:
    if not block:
        return 0.0
    try:
        return float(block.get("amount") or 0)
    except (TypeError, ValueError):
        return 0.0


def is_excluded_order_status(status: Optional[str]) -> bool:
    """True for Canceled / Cancelled / Unfulfillable / Pending — anything
    that isn't a confirmed, revenue-recognisable sale."""
    if not status:
        return False
    if status in CANCELLED_ORDER_STATUSES:
        return True
    return status.strip().lower() in {
        "canceled", "cancelled", "unfulfillable", "pending", "invoiceunconfirmed",
    }


def line_item_sales_amount(item: dict) -> float:
    """Net line revenue: itemSubtotal − promotionDiscount (Aurora order sync shape)."""
    subtotal = _money_amount(item.get("itemSubtotal"))
    promo = _money_amount(item.get("promotionDiscount"))
    unit_price = _money_amount(item.get("itemPrice"))
    if subtotal > 0:
        return max(0.0, subtotal - promo)
    if unit_price > 0:
        # itemPrice in Aurora DB is the line total, not per-unit (orderSyncService).
        return max(0.0, unit_price - promo)
    return 0.0


def aggregate_sku_metrics_from_orders(
    order_docs: list[dict],
) -> tuple[dict[str, dict], list[dict], int]:
    """Build per-SKU units/revenue/fees from Aurora order line items.

    Returns (sku_data, na_price_rows, orders_count).
    referralFee / fulfillmentFee on each line item are line totals (see
    auroraBackend orderSyncService).
    """
    sku_data: dict[str, dict] = defaultdict(
        lambda: {
            "units": 0,
            "revenue": 0.0,
            "asin": None,
            "referral_total": 0.0,
            "fba_total": 0.0,
        }
    )
    na_price_rows: list[dict] = []
    eligible_orders = 0

    for doc in order_docs:
        if is_excluded_order_status(doc.get("orderStatus")):
            continue
        eligible_orders += 1
        oid = doc.get("amazonOrderId") or ""
        for it in doc.get("orderItems") or []:
            sku = it.get("sellerSku")
            if not sku:
                continue
            qty = int(it.get("quantityOrdered") or 0)
            amount = line_item_sales_amount(it)
            if amount <= 0 and qty > 0:
                na_price_rows.append({"order_id": oid, "sku": sku, "qty": qty})
                continue
            sku_data[sku]["units"] += qty
            sku_data[sku]["revenue"] += amount
            sku_data[sku]["referral_total"] += _money_amount(it.get("referralFee"))
            sku_data[sku]["fba_total"] += _money_amount(it.get("fulfillmentFee"))
            if not sku_data[sku]["asin"] and it.get("asin"):
                sku_data[sku]["asin"] = it["asin"]

    return sku_data, na_price_rows, eligible_orders


async def product_fee_estimates_by_sku(user: dict, skus: list[str]) -> dict[str, dict]:
    """Per-unit fees from Aurora `products.fees` (same source as Products page).

    Used as the primary fee basis for Profitability — not a last-resort fallback.
    Referral is stored as a per-unit amount at the listing price; callers should
    scale it by revenue when the sale price differs.
    """
    if not skus:
        return {}
    seller_id = ObjectId(str(user["_id"]))
    cursor = _db().products.find(
        {"sellerId": seller_id, "sku": {"$in": skus}},
        {"sku": 1, "asin": 1, "fees": 1, "price": 1},
    )
    out: dict[str, dict] = {}
    async for doc in cursor:
        sku = doc.get("sku")
        if not sku:
            continue
        fees = doc.get("fees") or {}
        price = _money_amount((doc.get("price") or {}))
        breakdown = fees.get("breakdown") or []
        if breakdown:
            parsed = parse_fee_detail_lines(breakdown)
            referral = parsed["referral"]
            fba = parsed["fba"]
            fuel = parsed["fuel_surcharge"]
            # If breakdown only had a bundled FBAFees parent, parser already split.
            # If fbaFee field is the full fulfillment total and breakdown missed fuel,
            # prefer splitting the stored fbaFee when it is clearly larger.
            stored_fba = _money_amount(fees.get("fbaFee"))
            if stored_fba > 0 and fuel <= 0 and abs(stored_fba - (fba + fuel)) > 0.02:
                fba, fuel = split_bundled_fulfillment_total(stored_fba)
        else:
            fba_total = _money_amount(fees.get("fbaFee"))
            referral = _money_amount(fees.get("referralFee"))
            total = _money_amount(fees.get("totalFees"))
            if referral <= 0 and total > 0 and fba_total > 0:
                referral = max(total - fba_total, 0.0)
            elif referral <= 0 and price > 0:
                referral = price * 0.15
            fba, fuel = split_bundled_fulfillment_total(fba_total)
        if referral <= 0 and fba <= 0 and fuel <= 0:
            continue
        out[sku] = {
            "referral_per_unit": referral,
            "fba_per_unit": fba,
            "fuel_per_unit": fuel,
            "listing_price": price,
            "asin": doc.get("asin"),
        }
    return out


async def fba_aged_inventory_by_sku(user: dict) -> Optional[dict[str, dict]]:
    """Read Aurora's `fbaagedinventoryfees` snapshot for this seller.

    Aurora's fbaAgedInventorySyncService runs from inventorySyncService and
    persists per-SKU projections from GET_FBA_INVENTORY_PLANNING_DATA:
      { monthlyFee, totalAgedUnits, asin, historicalDaysOfSupply }.

    Returns None when the collection has no doc for this seller (Aurora
    hasn't synced yet). Returns an empty dict when the doc exists but is
    empty (Aurora synced, no aged inventory) — caller can distinguish
    "not synced" from "no charges" by the None vs {} return.
    """
    seller_id = ObjectId(str(user["_id"]))
    doc = await _db().fbaagedinventoryfees.find_one({"sellerId": seller_id})
    if not doc:
        return None
    per_sku_raw = doc.get("perSku") or {}
    # Normalize the JS camelCase to Python snake_case so consumers don't
    # need to know which side wrote the doc.
    out: dict[str, dict] = {}
    for sku, v in per_sku_raw.items():
        if not isinstance(v, dict):
            continue
        out[sku] = {
            "monthly_fee": float(v.get("monthlyFee") or 0.0),
            "total_aged_units": int(v.get("totalAgedUnits") or 0),
            "asin": v.get("asin"),
            "historical_days_of_supply": (
                float(v["historicalDaysOfSupply"])
                if v.get("historicalDaysOfSupply") is not None else None
            ),
        }
    return out


async def fba_inbound_placement_by_sku(user: dict) -> Optional[dict[str, dict]]:
    """Read Aurora's `fbainboundplacementfees` snapshot for this seller.

    Aurora's fbaInboundPlacementSyncService persists per-SKU inbound
    placement fees:
      { totalUnits, totalFee, avgFeePerUnit, asin }
    plus a top-level `source: 'report' | 'finances_join'` marker.

    Returns the shape used by agent._build_placement_rates:
      {sku: {fee_total, units_received, fee_bearing_units, asin}}
    so the caller's rate-building code is unchanged.

    Returns None when the collection has no doc for this seller.
    """
    seller_id = ObjectId(str(user["_id"]))
    doc = await _db().fbainboundplacementfees.find_one({"sellerId": seller_id})
    if not doc:
        return None
    per_sku_raw = doc.get("perSku") or {}
    out: dict[str, dict] = {}
    for sku, v in per_sku_raw.items():
        if not isinstance(v, dict):
            continue
        units = int(v.get("totalUnits") or 0)
        fee = float(v.get("totalFee") or 0.0)
        if units <= 0 or fee <= 0:
            continue
        out[sku] = {
            "fee_total": fee,
            "units_received": units,
            "fee_bearing_units": units,
            "asin": v.get("asin"),
        }
    return out


async def placement_rates_from_shipments(
    user: dict,
    fees_by_shipment: dict[str, float],
) -> dict[str, dict]:
    """Rebuild per-SKU inbound placement fees from shipment-level Finances
    lump sums (FBAInboundConvenienceFee, keyed by FBA shipment id) joined
    with Aurora `shipments.lineItems` units received.

    Returns the same shape as amazon_sp.fetch_inbound_placement_fees_per_sku:
    {sku: {fee_total, units_received, fee_bearing_units, asin}} so the
    caller's rate-building and cache format are unchanged.
    """
    if not fees_by_shipment:
        return {}
    seller_id = ObjectId(str(user["_id"]))
    ship_ids = [sid for sid in fees_by_shipment if sid and sid != "_unknown"]
    if not ship_ids:
        return {}
    cursor = _db().shipments.find(
        {"sellerId": seller_id, "shipmentId": {"$in": ship_ids}},
        {"shipmentId": 1, "lineItems": 1},
    )
    per_sku: dict[str, dict] = {}
    async for doc in cursor:
        fee = float(fees_by_shipment.get(doc.get("shipmentId")) or 0)
        items = doc.get("lineItems") or []
        total_units = sum(
            max(int(it.get("unitsReceived") or 0), 0) for it in items
        )
        if fee <= 0 or total_units <= 0:
            continue
        # Allocate the shipment's lump sum across SKUs by units received —
        # the per-unit rate within one shipment is near-uniform (size tier).
        for it in items:
            units = max(int(it.get("unitsReceived") or 0), 0)
            sku = (it.get("sku") or "").strip()
            if units <= 0 or not sku:
                continue
            bucket = per_sku.setdefault(
                sku,
                {"fee_total": 0.0, "units_received": 0,
                 "fee_bearing_units": 0, "asin": None},
            )
            bucket["fee_total"] += fee * units / total_units
            bucket["units_received"] += units
            bucket["fee_bearing_units"] += units
    if not per_sku:
        return {}
    prod_cursor = _db().products.find(
        {"sellerId": seller_id, "sku": {"$in": list(per_sku.keys())}},
        {"sku": 1, "asin": 1},
    )
    async for doc in prod_cursor:
        bucket = per_sku.get(doc.get("sku"))
        if bucket is not None and doc.get("asin"):
            bucket["asin"] = str(doc["asin"]).upper()
    return {
        sku: {**b, "fee_total": round(b["fee_total"], 2)}
        for sku, b in per_sku.items()
    }


def list_user_marketplaces(user: dict) -> list[dict]:
    ids = user.get("amazonMarketplaceIds") or []
    if not ids:
        ids = ["ATVPDKIKX0DER"]
    primary = ids[0]
    if "ATVPDKIKX0DER" in ids:
        primary = "ATVPDKIKX0DER"
    return [
        {
            "id": mid,
            "name": MARKETPLACE_NAMES.get(mid, "Unknown"),
            "is_primary": mid == primary,
        }
        for mid in [str(x) for x in ids]
    ]


async def fetch_inventory_summaries(
    user: dict,
    skus: Optional[list[str]] = None,
) -> list[dict]:
    """FBA inventory from Aurora `products` collection."""
    seller_id = ObjectId(str(user["_id"]))
    query: dict[str, Any] = {"sellerId": seller_id}
    if skus:
        query["sku"] = {"$in": skus}
    cursor = _db().products.find(
        query,
        {"sku": 1, "asin": 1, "inventory": 1, "fulfillmentType": 1},
    )
    rows: list[dict] = []
    async for doc in cursor:
        inv = doc.get("inventory") or {}
        rows.append({
            "sellerSku": doc.get("sku"),
            "asin": doc.get("asin"),
            "inventoryDetails": {
                "fulfillableQuantity": int(inv.get("fulfillableQuantity") or 0),
                "inboundWorkingQuantity": int(inv.get("inboundWorkingQuantity") or 0),
                "inboundShippedQuantity": int(inv.get("inboundShippedQuantity") or 0),
                "reservedQuantity": {
                    "totalReservedQuantity": int(inv.get("reservedQuantity") or 0),
                },
                "unfulfillableQuantity": {
                    "totalUnfulfillableQuantity": int(inv.get("unfulfillableQuantity") or 0),
                },
            },
            "fulfillmentType": doc.get("fulfillmentType"),
        })
    return rows


async def fetch_campaigns(user: dict) -> list[dict]:
    """Campaign metrics from Aurora `ads` collection (+ optional extras)."""
    seller_id = ObjectId(str(user["_id"]))
    cursor = _db().ads.find({"sellerId": seller_id})
    campaigns = await cursor.to_list(length=None)
    for doc in campaigns:
        doc.pop("_id", None)
        for key in ("startDate", "endDate", "metricsStartDate", "metricsEndDate", "lastSynced"):
            val = doc.get(key)
            if isinstance(val, datetime):
                doc[key] = val.isoformat()
    try:
        extras = await _db().middhaAdCampaigns.find(
            {"sellerId": seller_id}, {"_id": 0, "sellerId": 0},
        ).to_list(length=None)
        seen = {c.get("campaignId") for c in campaigns}
        for extra in extras:
            if extra.get("campaignId") not in seen:
                campaigns.append(extra)
    except Exception:
        pass
    return campaigns


async def aggregate_sales_daily_from_orders(
    user_id: ObjectId,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Aggregate Aurora orders into (sku, date) rows for forecasting ingest."""
    user = await _db().users.find_one({"_id": user_id})
    if not user:
        return []
    created_after = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    created_before = end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    docs = await fetch_orders_with_items(
        user, created_after=created_after, created_before=created_before,
    )
    by_key: dict[tuple[str, datetime], dict] = {}
    for doc in docs:
        if is_excluded_order_status(doc.get("orderStatus")):
            continue
        pd = doc.get("purchaseDate")
        if not isinstance(pd, datetime):
            continue
        day = datetime.combine(pd.date(), datetime.min.time(), tzinfo=timezone.utc)
        if day < start or day >= end:
            continue
        for it in doc.get("orderItems") or []:
            sku = (it.get("sellerSku") or "").strip()
            if not sku:
                continue
            qty = int(it.get("quantityOrdered") or 0)
            if qty <= 0:
                continue
            revenue = line_item_sales_amount(it)
            key = (sku, day)
            agg = by_key.setdefault(key, {
                "sku": sku,
                "date": day,
                "asin": it.get("asin") or "",
                "units_ordered": 0,
                "ordered_revenue": 0.0,
                "sessions": 0,
                "page_views": 0,
                "buy_box_pct": 0.0,
                "ad_spend": 0.0,
                "ad_impressions": 0,
                "ad_clicks": 0,
                "stockout_corrected": False,
            })
            agg["units_ordered"] += qty
            agg["ordered_revenue"] += revenue
            if it.get("asin") and not agg.get("asin"):
                agg["asin"] = it["asin"]
    return list(by_key.values())


async def aggregate_sales_daily_lean(
    user_id: ObjectId,
    start: datetime,
    end: datetime,
    sku: Optional[str] = None,
) -> list[dict]:
    """Server-side (sku, date) aggregation from Aurora `orders`.

    Uses a Mongo aggregation pipeline so the AI backend only receives the
    already-summed rows — the naive per-doc fetch in
    `aggregate_sales_daily_from_orders` was OOM-ing Render's 512 MB tier
    on multi-hundred-day windows.
    """
    # For velocity/forecasting we count every order that was placed —
    # including Pending and Cancelled — because they still represent
    # customer demand at the moment of purchase. Profitability paths
    # (aggregate_sku_metrics_from_orders) keep the CANCELLED_ORDER_STATUSES
    # filter because those don't recognise revenue.
    pipeline: list[dict] = [
        {
            "$match": {
                "sellerId": user_id,
                "purchaseDate": {"$gte": start, "$lt": end},
            },
        },
        {"$unwind": "$orderItems"},
    ]
    if sku:
        pipeline.append({"$match": {"orderItems.sellerSku": sku}})
    pipeline += [
        {
            "$project": {
                "_id": 0,
                "sku": "$orderItems.sellerSku",
                "asin": "$orderItems.asin",
                "qty": {"$ifNull": ["$orderItems.quantityOrdered", 0]},
                "subtotal": {"$ifNull": ["$orderItems.itemSubtotal.amount", 0]},
                "item_price": {"$ifNull": ["$orderItems.itemPrice.amount", 0]},
                "promo": {"$ifNull": ["$orderItems.promotionDiscount.amount", 0]},
                "day": {
                    "$dateTrunc": {"date": "$purchaseDate", "unit": "day", "timezone": "UTC"},
                },
            },
        },
        {"$match": {"sku": {"$ne": None, "$nin": ["", None]}, "qty": {"$gt": 0}}},
        {
            "$group": {
                "_id": {"sku": "$sku", "date": "$day"},
                "asin": {"$first": "$asin"},
                "units_ordered": {"$sum": "$qty"},
                "ordered_revenue": {
                    "$sum": {
                        "$max": [
                            0,
                            {"$subtract": [
                                {"$cond": [{"$gt": ["$subtotal", 0]}, "$subtotal", "$item_price"]},
                                "$promo",
                            ]},
                        ],
                    },
                },
            },
        },
        {
            "$project": {
                "_id": 0,
                "sku": "$_id.sku",
                "date": "$_id.date",
                "asin": 1,
                "units_ordered": 1,
                "ordered_revenue": 1,
                "sessions": {"$literal": 0},
                "page_views": {"$literal": 0},
                "buy_box_pct": {"$literal": 0.0},
                "ad_spend": {"$literal": 0.0},
                "ad_impressions": {"$literal": 0},
                "ad_clicks": {"$literal": 0},
                "stockout_corrected": {"$literal": False},
            },
        },
        {"$sort": {"sku": 1, "date": 1}},
    ]
    cursor = _db().orders.aggregate(pipeline, allowDiskUse=True)
    return await cursor.to_list(length=None)


async def inventory_snapshot_rows_from_products(user_id: ObjectId) -> list[dict]:
    """Today's inventory snapshot rows from Aurora products."""
    today = datetime.combine(
        datetime.now(timezone.utc).date(), datetime.min.time(), tzinfo=timezone.utc,
    )
    cursor = _db().products.find(
        {"sellerId": user_id},
        {"sku": 1, "inventory": 1},
    )
    rows: list[dict] = []
    async for doc in cursor:
        sku = (doc.get("sku") or "").strip()
        if not sku:
            continue
        inv = doc.get("inventory") or {}
        rows.append({
            "sku": sku,
            "date": today,
            "fulfillable": int(inv.get("fulfillableQuantity") or 0),
            "inbound_working": int(inv.get("inboundWorkingQuantity") or 0),
            "inbound_shipped": int(inv.get("inboundShippedQuantity") or 0),
            "reserved": int(inv.get("reservedQuantity") or 0),
            "unfulfillable": int(inv.get("unfulfillableQuantity") or 0),
        })
    return rows
