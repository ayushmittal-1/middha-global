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

# Match auroraBackend dashboardMetrics — cancelled/unfulfillable orders are not sales.
CANCELLED_ORDER_STATUSES = frozenset({"Canceled", "Cancelled", "Unfulfillable"})

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
    """True for Canceled / Cancelled / Unfulfillable (Aurora dashboard parity)."""
    if not status:
        return False
    if status in CANCELLED_ORDER_STATUSES:
        return True
    return status.strip().lower() in {"canceled", "cancelled", "unfulfillable"}


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
    """Fallback unit fees from Aurora `products.fees` when order lines lack them."""
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
        fba = _money_amount(fees.get("fbaFee"))
        total = _money_amount(fees.get("totalFees"))
        price = _money_amount((doc.get("price") or {}))
        referral = max(total - fba, price * 0.15) if total and fba else price * 0.15
        out[sku] = {
            "referral_per_unit": referral if price else 0.0,
            "fba_per_unit": fba,
            "asin": doc.get("asin"),
        }
    return out


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
