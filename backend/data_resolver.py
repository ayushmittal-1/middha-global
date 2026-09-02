"""
DB-first data access with Amazon SP-API fallback.

Read Aurora Mongo when data exists. Call Amazon only when required fields
are missing from the DB (empty window, missing line items, missing fees, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import amazon_sp
import aurora_data
from aurora_data import aurora_db_enabled
from currency_fx import looks_like_unconverted_foreign_face

log = logging.getLogger(__name__)


def _sp_orders_result(
    data: dict,
    created_after: str,
    created_before: Optional[str],
) -> dict:
    orders = (data.get("payload") or {}).get("Orders") or []
    out: dict[str, Any] = {
        "count": len(orders),
        "created_after": created_after,
        "created_before": created_before,
        "orders": orders,
        "source": "sp_api",
    }
    if data.get("_partial"):
        out["partial_warning"] = data["_partial"]
    return out


async def list_orders_resolved(
    user: dict,
    *,
    created_after: str,
    created_before: Optional[str] = None,
    statuses: Optional[list[str]] = None,
    marketplace: Optional[str] = None,
    buyer_email: Optional[str] = None,
    paginate: bool = True,
) -> dict:
    """Orders list — DB when available; SP-API when the DB window is empty."""
    if not aurora_db_enabled():
        data = await amazon_sp.get_orders(
            created_after=created_after,
            created_before=created_before,
            statuses=statuses,
            marketplace=marketplace,
            paginate=paginate,
        )
        result = _sp_orders_result(data, created_after, created_before)
    else:
        result = await aurora_data.list_orders(
            user,
            created_after=created_after,
            created_before=created_before,
            statuses=statuses,
            marketplace=marketplace,
            buyer_email=buyer_email,
        )
        if result.get("count", 0) > 0:
            return result
        log.info("orders: DB empty for window — falling back to SP-API")
        try:
            data = await amazon_sp.get_orders(
                created_after=created_after,
                created_before=created_before,
                statuses=statuses,
                marketplace=marketplace,
                paginate=paginate,
            )
            result = _sp_orders_result(data, created_after, created_before)
            result["source"] = "sp_api_fallback"
        except Exception as e:
            log.warning("orders SP-API fallback failed: %s", e)
            result["fallback_error"] = str(e)
            return result

    if buyer_email and result.get("source", "").startswith("sp_api"):
        needle = buyer_email.lower()
        result["orders"] = [
            o for o in result["orders"]
            if needle in (o.get("BuyerInfo", {}).get("BuyerEmail") or "").lower()
        ]
        result["count"] = len(result["orders"])
    return result


async def get_order_items_resolved(user: dict, order_id: str) -> dict:
    """Line items — DB when present; SP-API when order has no embedded items."""
    if not aurora_db_enabled():
        data = await amazon_sp.get_order_items(order_id)
        return {
            "items": (data.get("payload") or {}).get("OrderItems") or [],
            "source": "sp_api",
        }

    db_result = await aurora_data.get_order_items(user, order_id)
    if db_result.get("items"):
        return db_result

    log.info("order items: DB empty for %s — falling back to SP-API", order_id)
    try:
        data = await amazon_sp.get_order_items(order_id)
        return {
            "items": (data.get("payload") or {}).get("OrderItems") or [],
            "source": "sp_api_fallback",
        }
    except Exception as e:
        log.warning("order items SP-API fallback failed: %s", e)
        db_result["fallback_error"] = str(e)
        return db_result


async def fetch_inventory_resolved(
    user: dict,
    skus: Optional[list[str]] = None,
    marketplace: Optional[str] = None,
) -> tuple[list[dict], str]:
    """FBA inventory — DB when present; SP-API for missing SKUs or empty DB."""
    if not aurora_db_enabled():
        data = await amazon_sp.get_inventory_summaries(skus=skus, marketplace=marketplace)
        return data.get("payload", {}).get("inventorySummaries", []), "sp_api"

    db_rows = await aurora_data.fetch_inventory_summaries(user, skus=skus)
    if db_rows:
        if not skus:
            return db_rows, "aurora_db"
        found = {r.get("sellerSku") for r in db_rows}
        missing = [s for s in skus if s not in found]
        if not missing:
            return db_rows, "aurora_db"
        log.info("inventory: DB missing %d SKU(s) — merging SP-API", len(missing))
        try:
            data = await amazon_sp.get_inventory_summaries(
                skus=missing, marketplace=marketplace,
            )
            extra = data.get("payload", {}).get("inventorySummaries", [])
            return db_rows + extra, "aurora_db+sp_api_fallback"
        except Exception as e:
            log.warning("inventory partial SP-API fallback failed: %s", e)
            return db_rows, "aurora_db"

    log.info("inventory: DB empty — falling back to SP-API")
    try:
        data = await amazon_sp.get_inventory_summaries(skus=skus, marketplace=marketplace)
        return data.get("payload", {}).get("inventorySummaries", []), "sp_api_fallback"
    except Exception as e:
        log.warning("inventory SP-API fallback failed: %s", e)
        return [], "aurora_db"


def list_marketplaces_resolved(user: dict) -> list[dict]:
    """Marketplaces from user profile; SP-API when profile has none."""
    rows = aurora_data.list_user_marketplaces(user)
    if user.get("amazonMarketplaceIds"):
        return rows
    try:
        return amazon_sp.list_marketplaces()
    except Exception:
        return rows


def orders_missing_line_items(order_docs: list[dict]) -> list[str]:
    return [
        oid
        for doc in order_docs
        if (oid := doc.get("amazonOrderId")) and not (doc.get("orderItems") or [])
    ]


def _sp_item_to_db_shape(item: dict) -> dict:
    price = item.get("ItemPrice") or {}
    try:
        amount = float(price.get("Amount") or 0)
    except (TypeError, ValueError):
        amount = 0.0
    return {
        "sellerSku": item.get("SellerSKU"),
        "asin": item.get("ASIN"),
        "quantityOrdered": item.get("QuantityOrdered"),
        "itemPrice": {
            "amount": amount,
            "currencyCode": price.get("CurrencyCode") or "USD",
        },
        "referralFee": None,
        "fulfillmentFee": None,
    }


async def supplement_order_items_from_sp_api(
    order_docs: list[dict],
) -> tuple[list[dict], list[str]]:
    """Fill orders that lack embedded line items via SP-API GetOrderItems."""
    missing_ids = orders_missing_line_items(order_docs)
    if not missing_ids:
        return order_docs, []

    errors: list[str] = []
    id_to_items: dict[str, list[dict]] = {}

    for oid in missing_ids:
        try:
            resp = await amazon_sp.get_order_items(oid)
            id_to_items[oid] = [
                _sp_item_to_db_shape(it)
                for it in (resp.get("payload") or {}).get("OrderItems") or []
            ]
            await asyncio.sleep(2.1)
        except Exception as e:
            errors.append(f"{oid}: {e}")

    out: list[dict] = []
    for doc in order_docs:
        patched = dict(doc)
        oid = patched.get("amazonOrderId")
        if oid in id_to_items:
            patched["orderItems"] = id_to_items[oid]
        out.append(patched)
    return out, errors


def skus_needing_fees_api(
    sku_data: dict[str, dict],
    product_fee_fallback: dict[str, dict],
    *,
    price_divergence_pct: float = 0.01,
) -> list[str]:
    """SKUs that need a live Product Fees API estimate at sale price.

    Always fetch when ``products.fees`` is missing. Also fetch when the
    window avg selling price diverges from the listing price used for the
    stored estimate (referral scales with price; stale listing fees drift
    from Amazon Fee Preview / Revenue Calculator).
    """
    need: list[str] = []
    pf_lower = {str(k).lower(): v for k, v in (product_fee_fallback or {}).items()}
    for sku, d in sku_data.items():
        units = int(d.get("units") or 0)
        if units <= 0:
            continue
        if not d.get("asin") or float(d.get("revenue") or 0) <= 0:
            continue
        pf = product_fee_fallback.get(sku) or pf_lower.get(str(sku).lower()) or {}
        has_pf = (
            float(pf.get("referral_per_unit") or 0) > 0
            or float(pf.get("fba_per_unit") or 0) > 0
            or float(pf.get("fulfillment_per_unit") or 0) > 0
        )
        if not has_pf:
            need.append(sku)
            continue
        listing = float(pf.get("listing_price") or 0)
        avg_sale = float(d.get("revenue") or 0) / units
        if listing > 0 and avg_sale > 0:
            if looks_like_unconverted_foreign_face(avg_sale, listing):
                continue
            drift = abs(avg_sale - listing) / listing
            if drift >= price_divergence_pct:
                need.append(sku)
                continue
        # Always refresh when we only have referral heuristic / no real FBA
        # for an FBA SKU (or vice versa) — products.fees may be incomplete.
        is_fba = pf.get("is_fba")
        if is_fba is True and float(pf.get("fulfillment_per_unit") or 0) <= 0:
            need.append(sku)
    return need


def fee_batch_items_for_skus(
    skus: list[str],
    sku_data: dict[str, dict],
    product_fee_fallback: dict[str, dict] | None = None,
) -> list[tuple[str, float, bool]]:
    """Build (asin, avg_price, is_fba) tuples for ``get_fees_estimates_batch``."""
    pf_lower = {
        str(k).lower(): v for k, v in (product_fee_fallback or {}).items()
    }
    items: list[tuple[str, float, bool]] = []
    for sku in skus:
        d = sku_data.get(sku) or {}
        asin = d.get("asin")
        units = int(d.get("units") or 0)
        revenue = float(d.get("revenue") or 0)
        if not asin or units <= 0 or revenue <= 0:
            continue
        avg_price = revenue / units
        pf = (product_fee_fallback or {}).get(sku) or pf_lower.get(str(sku).lower()) or {}
        listing = float(pf.get("listing_price") or 0)
        if looks_like_unconverted_foreign_face(avg_price, listing):
            continue
        if "is_fba" in pf:
            item_is_fba = bool(pf.get("is_fba"))
        else:
            # Default FBA when unknown — most Profitability rows are FBA;
            # FBM products with synced fees expose is_fba=False via pf.
            item_is_fba = True
        items.append((asin, avg_price, item_is_fba))
    return items
