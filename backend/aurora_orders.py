"""
Read Aurora-synced orders from MongoDB for the AI Assistant Orders tab.

Maps Aurora `orders` documents into the SP-API JSON shape the aiModel
frontend already expects — no Amazon credentials or live SP-API calls.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from auth import _db
from amazon_sp import resolve_marketplace


def _parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _money_block(block: Optional[dict]) -> Optional[dict]:
    if not block or block.get("amount") is None:
        return None
    currency = block.get("currencyCode") or "USD"
    amount = block["amount"]
    return {"Amount": str(amount), "CurrencyCode": currency}


def _iso_z(dt: Any) -> Optional[str]:
    if not dt:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(dt)


def _order_status_query(status_param: str) -> dict:
    """Match Aurora orderStatusFilters.js for common filter keys."""
    key = status_param.strip().lower()
    mapping = {
        "pending": {"orderStatus": {"$in": ["Pending", "InvoiceUnconfirmed"]}},
        "unshipped": {"orderStatus": {"$in": ["Unshipped", "PartiallyShipped"]}},
        "shipped": {"orderStatus": {"$in": ["Shipped", "PartiallyShipped"]}},
        "canceled": {"orderStatus": {"$in": ["Canceled", "Cancelled", "Unfulfillable"]}},
        "cancelled": {"orderStatus": {"$in": ["Canceled", "Cancelled", "Unfulfillable"]}},
    }
    if key in mapping:
        return mapping[key]
    if "," in status_param:
        values = [s.strip() for s in status_param.split(",") if s.strip()]
        return {"orderStatus": {"$in": values}}
    return {"orderStatus": status_param.strip()}


def _order_doc_to_sp_api(doc: dict) -> dict:
    order_total = doc.get("orderTotal") or {}
    out: dict[str, Any] = {
        "AmazonOrderId": doc.get("amazonOrderId"),
        "PurchaseDate": _iso_z(doc.get("purchaseDate")),
        "OrderStatus": doc.get("orderStatus"),
        "MarketplaceId": doc.get("marketplaceId"),
        "NumberOfItemsShipped": doc.get("numberOfItemsShipped"),
        "NumberOfItemsUnshipped": doc.get("numberOfItemsUnshipped"),
        "OrderTotal": _money_block(order_total),
        "LatestShipDate": _iso_z(doc.get("promiseResponseDueDate")),
    }
    if doc.get("buyerEmail"):
        out["BuyerInfo"] = {"BuyerEmail": doc.get("buyerEmail")}
    return out


def _item_doc_to_sp_api(item: dict) -> dict:
    return {
        "SellerSKU": item.get("sellerSku"),
        "ASIN": item.get("asin"),
        "Title": item.get("title"),
        "QuantityOrdered": item.get("quantityOrdered"),
        "QuantityShipped": item.get("quantityShipped"),
        "ItemPrice": _money_block(item.get("itemPrice")),
        "PromotionDiscount": _money_block(item.get("promotionDiscount")),
    }


async def list_orders(
    user: dict,
    *,
    created_after: str,
    created_before: Optional[str] = None,
    statuses: Optional[list[str]] = None,
    marketplace: Optional[str] = None,
    buyer_email: Optional[str] = None,
) -> dict:
    seller_id = ObjectId(str(user["_id"]))
    query: dict[str, Any] = {"sellerId": seller_id}

    after_dt = _parse_iso_dt(created_after)
    before_dt = _parse_iso_dt(created_before)
    if after_dt or before_dt:
        purchase_filter: dict[str, Any] = {}
        if after_dt:
            purchase_filter["$gte"] = after_dt
        if before_dt:
            purchase_filter["$lte"] = before_dt
        query["purchaseDate"] = purchase_filter

    if statuses:
        if len(statuses) == 1:
            query.update(_order_status_query(statuses[0]))
        else:
            query["orderStatus"] = {"$in": statuses}

    if marketplace:
        marketplace_ids = resolve_marketplace(user, marketplace, multiple=True)
        query["marketplaceId"] = {"$in": marketplace_ids}

    if buyer_email:
        query["buyerEmail"] = {"$regex": re.escape(buyer_email.strip()), "$options": "i"}

    coll = _db().orders
    projection = {
        "amazonOrderId": 1,
        "purchaseDate": 1,
        "orderStatus": 1,
        "marketplaceId": 1,
        "numberOfItemsShipped": 1,
        "numberOfItemsUnshipped": 1,
        "orderTotal": 1,
        "buyerEmail": 1,
        "promiseResponseDueDate": 1,
    }
    docs = await coll.find(query, projection).sort("purchaseDate", -1).to_list(length=None)
    orders = [_order_doc_to_sp_api(doc) for doc in docs]

    return {
        "count": len(orders),
        "created_after": created_after,
        "created_before": created_before,
        "orders": orders,
        "source": "aurora_db",
    }


def _build_window_query(
    user: dict,
    *,
    created_after: str,
    created_before: Optional[str] = None,
) -> dict[str, Any]:
    seller_id = ObjectId(str(user["_id"]))
    query: dict[str, Any] = {"sellerId": seller_id}
    after_dt = _parse_iso_dt(created_after)
    before_dt = _parse_iso_dt(created_before)
    if after_dt or before_dt:
        purchase_filter: dict[str, Any] = {}
        if after_dt:
            purchase_filter["$gte"] = after_dt
        if before_dt:
            purchase_filter["$lte"] = before_dt
        query["purchaseDate"] = purchase_filter
    return query


async def fetch_orders_with_items(
    user: dict,
    *,
    created_after: str,
    created_before: Optional[str] = None,
) -> list[dict]:
    """Full order docs (with embedded line items) for profitability."""
    query = _build_window_query(
        user, created_after=created_after, created_before=created_before,
    )
    return await _db().orders.find(query).sort("purchaseDate", -1).to_list(length=None)


async def get_order_items(user: dict, order_id: str) -> dict:
    seller_id = ObjectId(str(user["_id"]))
    doc = await _db().orders.find_one(
        {"sellerId": seller_id, "amazonOrderId": order_id},
        {"orderItems": 1},
    )
    if not doc:
        return {"items": [], "source": "aurora_db"}
    items = [_item_doc_to_sp_api(item) for item in (doc.get("orderItems") or [])]
    return {"items": items, "source": "aurora_db"}
