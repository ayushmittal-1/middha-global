"""
Aurora MongoDB integration — DB-first reads with SP-API fallback.

Set AURORA_DATA_SOURCE=db (default) so the AI backend reads Aurora's
synced collections (orders, products, ads, users). Missing required
fields trigger live Amazon API calls via data_resolver.
"""

from __future__ import annotations

import html
import os
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from auth import _db
from amazon_sp import MARKETPLACE_NAMES, resolve_marketplace
from amazon_sp import parse_fee_detail_lines, split_bundled_fulfillment_total
from currency_fx import (
    UsdFx,
    identity_fx,
    infer_line_currency,
    infer_marketplace_id,
    infer_order_date,
    is_us_marketplace,
    referral_fee_currency,
    fba_fee_currency,
    FALLBACK_USD_PER_UNIT,
    US_MARKETPLACE_ID,
)

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


from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

_PENNY = Decimal("0.01")


# Row keys whose account-level total is a naive per-SKU sum (no blending
# with account-level report totals or unallocated residuals). Kept here so
# both the reconciliation stage and its unit tests reference the same list.
ROW_SUM_MONEY_KEYS = (
    "revenue", "referral_fee", "fba_fee", "fuel_surcharge",
    "amazon_fees", "ad_cost", "product_cost", "inbound_shipping",
    "cogs_total", "return_processing_fee", "low_inventory_fee",
)


def reconcile_row_derived_totals(rows: list[dict]) -> dict[str, float]:
    """Extracted stage of compute_profitability_data (review issue #8, #7).

    Returns Decimal-accurate totals for the money keys that are pure
    per-SKU sums. Blended/overwritten totals (placement, storage w/
    unallocated, aged w/ unallocated, removal, reimbursement) are the
    caller's responsibility — they aren't row sums."""
    return {
        key: sum_money(row.get(key, 0) for row in (rows or []))
        for key in ROW_SUM_MONEY_KEYS
    }


def reconcile_net_total(rows: list[dict]) -> float:
    """Penny-accurate `net` total (verification-review followup on issue #7).

    The first pass only reconciled the row-sum COMPONENT keys with Decimal,
    leaving `totals["net"]` on the float-accumulator path — so components
    and net could disagree by a cent even though the row-level
    `row["net"] = revenue - fees - ...` was rounded to 2dp. Summing
    `row["net"]` with the same Decimal accumulator closes that gap. The
    caller still applies blended-total adjustments to `net` after
    reconciliation (unallocated storage/aged, blended placement, blended
    removal, reimbursements) — those adjustments themselves land at 2dp,
    so the Decimal-accurate base plus penny-quantized adjustments keeps
    the whole path penny-consistent."""
    return sum_money(row.get("net", 0) for row in (rows or []))


def sum_money(values: Iterable) -> float:
    """Penny-accurate sum of money values (review issue #7).

    Each addend is converted via `Decimal(str(v))` and quantized to 2dp
    with ROUND_HALF_UP before accumulation, so binary-float drift can't
    build up over thousands of rows and produce a total that's off by a
    cent from Seller Central's own figure. Returns float for JSON
    compatibility. Ignores None and non-numeric values."""
    total = Decimal("0")
    for v in values:
        if v is None:
            continue
        try:
            d = v if isinstance(v, Decimal) else Decimal(str(v))
        except Exception:
            continue
        total += d.quantize(_PENNY, rounding=ROUND_HALF_UP)
    return float(total.quantize(_PENNY, rounding=ROUND_HALF_UP))


def filter_orders_by_marketplace(
    order_docs: list[dict], marketplace_id: Optional[str],
) -> list[dict]:
    """Keep only orders belonging to the given marketplace.

    When marketplace_id is None or empty, the list is returned unchanged.
    Handles both Aurora-DB shape (`marketplaceId`, lower-camel) and SP-API
    shape (`MarketplaceId`, PascalCase) so the same helper works on both
    profitability code paths. This is the fix for review issue #1 —
    scoping a profitability request to a single marketplace guarantees a
    single-currency response instead of a silently-mixed blended total."""
    if not marketplace_id:
        return list(order_docs or [])
    target = str(marketplace_id).strip()
    if not target:
        return list(order_docs or [])
    kept: list[dict] = []
    for doc in order_docs or []:
        mid = infer_marketplace_id(doc)
        if mid == target:
            kept.append(doc)
    return kept


def detect_order_currencies(order_docs: list[dict]) -> set[str]:
    """Return the distinct non-empty currency codes present across order-line
    money fields. Used by the profitability path to detect when a request
    would silently mix multiple currencies into one total (review issue #1).

    Cancelled/unfulfillable orders are excluded to match the revenue path.
    Only 3-letter ISO-like codes are considered; blank/None values are ignored
    so a partial fixture doesn't produce a false-positive '' currency."""
    codes: set[str] = set()
    for doc in order_docs or []:
        if is_excluded_order_status(doc.get("orderStatus") or doc.get("OrderStatus")):
            continue
        items = doc.get("orderItems") or doc.get("OrderItems") or []
        if items:
            for it in items:
                code = infer_line_currency(doc, it, default="")
                if code:
                    codes.add(code)
        else:
            code = infer_line_currency(doc, None, default="")
            if code:
                codes.add(code)
    return codes


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


def _money_amount(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        try:
            return float(value.get("amount") or 0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# Amazon referral is a category %. Fees API returns a rounded $ at listing
# price; using referral$/listing$ as the rate (e.g. 0.73/4.89 ≈ 14.93%)
# then re-applying to other sale prices compounds rounding error vs SC.
_KNOWN_REFERRAL_RATES = (0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.45)
_DEFAULT_REFERRAL_RATE = 0.15


def round_money(value: float | Decimal | str) -> float:
    """Penny round with ROUND_HALF_UP (Amazon / Seller Central), not banker's.

    Python's built-in ``round(2.295, 2)`` → ``2.29``; Amazon charges ``2.30``.
    """
    try:
        d = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        return 0.0
    return float(d.quantize(_PENNY, rounding=ROUND_HALF_UP))


def snap_referral_rate(
    listing_price: float | None,
    referral_per_unit: float | None,
    default: float = _DEFAULT_REFERRAL_RATE,
) -> float:
    """Return the category referral rate, snapping past Fees API rounding.

    If ``round_money(listing × 0.15) == referral_per_unit``, use **0.15** — not
    ``referral/listing``. That matches Seller Central (e.g. $5.07 → $0.76,
    $5.99 → $0.90).
    """
    listing = float(listing_price or 0)
    ref = float(referral_per_unit or 0)
    if listing > 0 and ref > 0:
        for rate in _KNOWN_REFERRAL_RATES:
            if round_money(Decimal(str(listing)) * Decimal(str(rate))) == round_money(ref):
                return float(rate)
        return ref / listing
    return float(default)


def line_referral_fee(
    line_revenue: float,
    rate: float = _DEFAULT_REFERRAL_RATE,
    quantity: int = 1,
) -> float:
    """Per-line referral matching Seller Central unit rounding.

    Amazon rounds referral per unit (half-up to the cent) then multiplies by
    quantity. Built-in ``round()`` is banker's rounding and undercounts cases
    like ``$15.30 × 15%`` → ``$2.29`` instead of ``$2.30``.
    """
    rev = max(0.0, float(line_revenue or 0))
    r = float(rate)
    qty = max(1, int(quantity or 1))
    if rev <= 0:
        return 0.0
    if qty <= 1:
        return round_money(Decimal(str(rev)) * Decimal(str(r)))
    unit_rev = Decimal(str(rev)) / Decimal(qty)
    per_unit = round_money(unit_rev * Decimal(str(r)))
    return round_money(Decimal(str(per_unit)) * Decimal(qty))


def line_item_quantity(item: dict) -> int:
    """Amazon line quantity for profitability / velocity.

    Always prefer ``quantityOrdered`` (All Orders `quantity`). Fall back to
    ``quantityShipped`` when ordered is missing. Never treat "one order" as
    one unit — a 3-unit line must count as 3.
    """
    for key in ("quantityOrdered", "quantityShipped", "quantity"):
        raw = item.get(key)
        if raw is None or raw == "":
            continue
        try:
            qty = int(raw)
        except (TypeError, ValueError):
            continue
        if qty > 0:
            return qty
    return 0


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


def recompute_referral_totals(
    order_docs: list[dict],
    sku_data: dict[str, dict],
    rate_by_sku: dict[str, float] | None = None,
    fx: UsdFx | None = None,
) -> None:
    """Overwrite ``referral_total`` using per-line ``round(revenue × rate, 2)``.

    Stored ``orderItems.referralFee`` often used ``referral$/listing$`` as the
    rate (Fees API rounding), which drifts on other sale prices. Pass snapped
    category rates from ``snap_referral_rate`` (default 15%).

    Line revenue is converted to USD first so a CAD sale is not charged
    15% of the CAD face value as if it were dollars.
    """
    rates = rate_by_sku or {}
    rates_lower = {str(k).lower(): float(v) for k, v in rates.items()}
    table = fx if fx is not None else identity_fx()
    for d in sku_data.values():
        d["referral_total"] = 0.0

    for doc in order_docs or []:
        if is_excluded_order_status(doc.get("orderStatus")):
            continue
        on = infer_order_date(doc)
        for it in doc.get("orderItems") or []:
            sku = it.get("sellerSku")
            if not sku or sku not in sku_data:
                continue
            amount = line_item_sales_amount(it)
            if amount <= 0:
                continue
            amount = table.to_usd(amount, referral_fee_currency(doc, it), on)
            qty = line_item_quantity(it) or 1
            rate = rates.get(sku)
            if rate is None:
                rate = rates_lower.get(str(sku).lower(), _DEFAULT_REFERRAL_RATE)
            sku_data[sku]["referral_total"] += line_referral_fee(
                amount, float(rate), qty,
            )


def repair_unconverted_foreign_price_buckets(
    sku_data: dict[str, dict],
    fx: UsdFx | None = None,
) -> None:
    """If a USD price bucket still holds a CAD/MXN face value, convert it.

    Happens when ``infer_line_currency`` returned USD (stamped fee-map) but
    ``marketplace_fee_quotes`` still has the native line (MXN 287.37). Without
    this, Fees API is called at $287 and FBA jumps to the >$50 band.
    No-op when conversion already ran (bucket key is the USD amount).
    """
    table = fx if fx is not None else identity_fx()
    for d in (sku_data or {}).values():
        by_price = d.get("units_by_usd_price")
        quotes = d.get("marketplace_fee_quotes") or {}
        if not isinstance(by_price, dict) or not quotes:
            continue
        for q in quotes.values():
            if not isinstance(q, dict):
                continue
            native = round(float(q.get("amount") or 0), 2)
            ccy = str(q.get("currency") or "").strip().upper()
            if native <= 0 or not ccy or ccy == "USD":
                continue
            qty = int(by_price.get(native) or 0)
            if qty <= 0:
                continue
            on = q.get("on")
            # ``native`` matched a per-unit bucket key, so convert that unit.
            usd_unit = round(float(table.to_usd(native, ccy, on)), 2)
            if usd_unit <= 0 or abs(usd_unit - native) < 0.005:
                fb = FALLBACK_USD_PER_UNIT.get(ccy)
                if not fb or abs(float(fb) - 1.0) < 1e-12:
                    continue
                usd_unit = round_money(native * float(fb))
            if usd_unit <= 0 or abs(usd_unit - native) < 0.005:
                continue
            old_line = round(native * qty, 2)
            new_line = round(usd_unit * qty, 2)
            d["revenue"] = max(0.0, float(d.get("revenue") or 0) - old_line + new_line)
            by_price.pop(native, None)
            by_price[usd_unit] = int(by_price.get(usd_unit) or 0) + qty


def aggregate_sku_metrics_from_orders(
    order_docs: list[dict],
    referral_rate: float = _DEFAULT_REFERRAL_RATE,
    fx: UsdFx | None = None,
) -> tuple[dict[str, dict], list[dict], int]:
    """Build per-SKU units/revenue/fees from Aurora order line items.

    Returns (sku_data, na_price_rows, orders_count).

    All money is converted to USD before summing so CAD/MXN Amazon.ca / .mx
    lines are not added to USD as if they were the same unit.

    Referral is ``round(usd_line_revenue × rate, 2)`` per line (default 15%),
    not the stored ``referralFee`` (which can use a drifted listing ratio, or
    15% of a CAD face value labelled USD). Call ``recompute_referral_totals``
    after product fees load to snap the rate per SKU. Fulfillment uses stored
    line ``fulfillmentFee``, converted when that field is not already USD.
    """
    rate = float(referral_rate or _DEFAULT_REFERRAL_RATE)
    table = fx if fx is not None else identity_fx()
    sku_data: dict[str, dict] = defaultdict(
        lambda: {
            "units": 0,
            "revenue": 0.0,
            "asin": None,
            "referral_total": 0.0,
            "fba_total": 0.0,
            "units_by_marketplace": {},
            "marketplace_fee_quotes": {},
            "units_by_usd_price": {},
        }
    )
    na_price_rows: list[dict] = []
    eligible_orders = 0

    for doc in order_docs:
        if is_excluded_order_status(doc.get("orderStatus")):
            continue
        eligible_orders += 1
        oid = doc.get("amazonOrderId") or ""
        on = infer_order_date(doc)
        for it in doc.get("orderItems") or []:
            sku = it.get("sellerSku")
            if not sku:
                continue
            qty = line_item_quantity(it)
            amount = line_item_sales_amount(it)
            # Skip $0 / missing-price lines entirely for Units, revenue, avg,
            # and line fees — replacements/promos with $0 item-price dilute
            # avg price and confuse Excel comparisons (e.g. B08P3CD3WR).
            if amount <= 0:
                if qty > 0:
                    na_price_rows.append({"order_id": oid, "sku": sku, "qty": qty})
                continue
            line_ccy = infer_line_currency(doc, it)
            usd_rev = table.to_usd(amount, line_ccy, on)
            fba_block = it.get("fulfillmentFee") or {}
            usd_fba = table.to_usd(
                _money_amount(fba_block),
                fba_fee_currency(doc, it),
                on,
            )
            sku_data[sku]["units"] += qty
            sku_data[sku]["revenue"] += usd_rev
            sku_data[sku]["referral_total"] += line_referral_fee(
                usd_rev, rate, qty or 1,
            )
            sku_data[sku]["fba_total"] += usd_fba
            if not sku_data[sku]["asin"] and it.get("asin"):
                sku_data[sku]["asin"] = it["asin"]
            mp = infer_marketplace_id(doc) or US_MARKETPLACE_ID
            by_mp = sku_data[sku]["units_by_marketplace"]
            by_mp[mp] = int(by_mp.get(mp) or 0) + int(qty or 0)
            # Native price for a later Fees API call on CA/MX/etc. (US catalog
            # FBA is stamped on those lines and is the wrong marketplace rate.)
            if not is_us_marketplace(mp):
                quotes = sku_data[sku]["marketplace_fee_quotes"]
                if mp not in quotes:
                    quotes[mp] = {
                        "amount": amount,
                        "currency": line_ccy,
                        "asin": it.get("asin"),
                        "on": on,
                    }
            unit_usd = round(float(usd_rev) / float(qty or 1), 2)
            by_price = sku_data[sku]["units_by_usd_price"]
            by_price[unit_usd] = int(by_price.get(unit_usd) or 0) + int(qty or 0)

    repair_unconverted_foreign_price_buckets(sku_data, table)
    return sku_data, na_price_rows, eligible_orders


def _pack_product_fee_doc(doc: dict) -> dict | None:
    """Normalize one products document into per-unit fee fields."""
    fees = doc.get("fees") or {}
    price = _money_amount((doc.get("price") or {}))
    breakdown = fees.get("breakdown") or []
    stored_fba = _money_amount(fees.get("fbaFee"))
    referral = _money_amount(fees.get("referralFee"))
    total = _money_amount(fees.get("totalFees"))

    # Referral: prefer stored field (Products column); else breakdown / estimate.
    if referral <= 0 and breakdown:
        referral = float(parse_fee_detail_lines(breakdown).get("referral") or 0)
    if referral <= 0 and total > 0 and stored_fba > 0:
        referral = max(total - stored_fba, 0.0)
    elif referral <= 0 and price > 0:
        referral = price * 0.15

    # Fulfillment: Products page displays fees.fbaFee — keep that total.
    # Split into base + fuel for Profitability's two columns.
    fba = 0.0
    fuel = 0.0
    if stored_fba > 0:
        if breakdown:
            parsed = parse_fee_detail_lines(breakdown)
            p_fba = float(parsed.get("fba") or 0)
            p_fuel = float(parsed.get("fuel_surcharge") or 0)
            if p_fba > 0 and abs((p_fba + p_fuel) - stored_fba) <= 0.02:
                fba, fuel = p_fba, p_fuel
            elif p_fba > 0 and p_fuel > 0:
                # Explicit base+fuel lines — trust breakdown even when
                # legacy stored fbaFee was base-only (missing fuel).
                fba, fuel = p_fba, p_fuel
            else:
                fba, fuel = split_bundled_fulfillment_total(stored_fba)
        else:
            fba, fuel = split_bundled_fulfillment_total(stored_fba)
    elif breakdown:
        parsed = parse_fee_detail_lines(breakdown)
        fba = float(parsed.get("fba") or 0)
        fuel = float(parsed.get("fuel_surcharge") or 0)
    if referral <= 0 and fba <= 0 and fuel <= 0:
        return None
    ft = str(doc.get("fulfillmentType") or "").upper()
    is_fba = ft != "FBM"
    return {
        "referral_per_unit": referral,
        "fba_per_unit": fba,
        "fuel_per_unit": fuel,
        # Full Products-page FBA fee (base+fuel) for diagnostics / UI parity.
        "fulfillment_per_unit": round(fba + fuel, 4) if (fba + fuel) > 0 else stored_fba,
        "listing_price": price,
        "asin": doc.get("asin"),
        "is_fba": is_fba,
        "fulfillment_type": doc.get("fulfillmentType") or None,
    }


def _usd_listing_fba_band(price: float) -> str:
    """Same $10 / $50 cuts as ``agent.usd_fba_price_band`` (no import cycle)."""
    p = float(price or 0)
    if p < 10:
        return "lt10"
    if p < 50:
        return "10_50"
    return "gt50"


async def fba_band_rates_for_asins(
    asins: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    """US FBA base+fuel per sale-price band from catalog listings of each ASIN.

    A SKU listed under $10 keeps that listing's FBA card. Units sold in
    the $10–$50 band use that band's rate on the same size-tier. Live
    Fees API quotes the band; when the quote is missing, another listing
    of the same ASIN already priced in that band is the same physical
    product's rate. Applies to every ASIN in the window.
    """
    wanted = sorted({str(a).strip().upper() for a in (asins or []) if str(a).strip()})
    if not wanted:
        return {}
    out: dict[str, dict[str, tuple[float, float]]] = {}
    async for doc in _db().products.find(
        {"asin": {"$in": wanted}},
        {"asin": 1, "fees": 1, "price": 1, "fulfillmentType": 1},
    ):
        packed = _pack_product_fee_doc(doc)
        if not packed:
            continue
        asin = str(packed.get("asin") or doc.get("asin") or "").strip().upper()
        if asin not in wanted:
            continue
        listing = float(packed.get("listing_price") or 0)
        fba = float(packed.get("fba_per_unit") or 0)
        fuel = float(packed.get("fuel_per_unit") or 0)
        if listing <= 0 or fba <= 0:
            continue
        band = _usd_listing_fba_band(listing)
        dest = out.setdefault(asin, {})
        if band not in dest:
            dest[band] = (round(fba, 2), round(fuel, 2))
    return out


def _fulfillment_per_unit(pf: dict | None) -> float:
    if not pf:
        return 0.0
    bundled = float(pf.get("fulfillment_per_unit") or 0)
    if bundled > 0:
        return bundled
    return float(pf.get("fba_per_unit") or 0) + float(pf.get("fuel_per_unit") or 0)


async def product_fee_estimates_by_sku(user: dict, skus: list[str]) -> dict[str, dict]:
    """Per-unit fees from Aurora `products.fees` (same source as Products page).

    Products UI shows ``fees.fbaFee`` as the FBA fee — that field is the source
    of truth for fulfillment. Profitability splits it into base FBA + fuel so
    FBA + Fuel == Products FBA Fee.

    Referral is stored as a per-unit amount at the listing price; callers should
    scale it by revenue when the sale price differs.
    """
    if not skus:
        return {}
    seller_id = ObjectId(str(user["_id"]))
    # Exact match first, then case-insensitive for any miss (order SKUs
    # sometimes differ in casing from products.sku).
    wanted = [str(s) for s in skus if s]
    wanted_lower = {s.lower(): s for s in wanted}
    cursor = _db().products.find(
        {"sellerId": seller_id, "sku": {"$in": wanted}},
        {"sku": 1, "asin": 1, "fees": 1, "price": 1, "fulfillmentType": 1},
    )
    out: dict[str, dict] = {}
    found_lower: set[str] = set()

    async for doc in cursor:
        sku = doc.get("sku")
        if not sku:
            continue
        packed = _pack_product_fee_doc(doc)
        if not packed:
            continue
        # Key by the order SKU spelling when casings differ.
        order_sku = wanted_lower.get(str(sku).lower(), str(sku))
        out[order_sku] = packed
        if order_sku != str(sku):
            out[str(sku)] = packed
        found_lower.add(str(sku).lower())

    missing = [s for s in wanted if s.lower() not in found_lower]
    if missing:
        # Case-insensitive fallback query for remaining SKUs.
        or_clauses = [{"sku": {"$regex": f"^{re.escape(s)}$", "$options": "i"}} for s in missing]
        async for doc in _db().products.find(
            {"sellerId": seller_id, "$or": or_clauses},
            {"sku": 1, "asin": 1, "fees": 1, "price": 1, "fulfillmentType": 1},
        ):
            sku = doc.get("sku")
            if not sku:
                continue
            packed = _pack_product_fee_doc(doc)
            if not packed:
                continue
            order_sku = wanted_lower.get(str(sku).lower(), str(sku))
            out[order_sku] = packed
            out[str(sku)] = packed

    return out


async def enrich_product_fees_from_asin_siblings(
    user: dict,
    sku_data: dict[str, dict],
    fee_map: dict[str, dict],
) -> dict[str, dict]:
    """Fill missing / zero FBA from another product on the same ASIN.

    Accounts like Eleet mint many ``amzn.gr.*`` SKUs. Some have referral but
    ``fees.fbaFee=$0`` (or no products row), while the merchant SKU on the
    same ASIN has a full FBA card. Without this, Profitability shows $0 FBA
    on those rows even though Amazon charged the same size-tier fee.
    """
    if not sku_data:
        return fee_map
    seller_id = ObjectId(str(user["_id"]))
    out = dict(fee_map or {})

    need_asins: set[str] = set()
    sku_asin: dict[str, str] = {}
    for sku, d in sku_data.items():
        asin = (
            (d.get("asin") or (out.get(sku) or {}).get("asin") or "")
        ).strip().upper()
        if not asin:
            continue
        sku_asin[sku] = asin
        if _fulfillment_per_unit(out.get(sku)) <= 0:
            need_asins.add(asin)
    if not need_asins:
        return out

    # asin -> (prefer_merchant: bool score, packed fees)
    donor_by_asin: dict[str, dict] = {}
    donor_is_gr: dict[str, bool] = {}

    def _consider_donor(sku: str, pf: dict) -> None:
        asin = (pf.get("asin") or sku_asin.get(sku) or "").strip().upper()
        if not asin or asin not in need_asins:
            return
        if _fulfillment_per_unit(pf) <= 0:
            return
        is_gr = str(sku).lower().startswith("amzn.gr.")
        prev = donor_by_asin.get(asin)
        if prev is None or (donor_is_gr.get(asin, True) and not is_gr):
            donor_by_asin[asin] = pf
            donor_is_gr[asin] = is_gr

    for sku, pf in list(out.items()):
        _consider_donor(sku, pf)

    still = sorted(a for a in need_asins if a not in donor_by_asin)
    if still:
        async for doc in _db().products.find(
            {"sellerId": seller_id, "asin": {"$in": still}},
            {"sku": 1, "asin": 1, "fees": 1, "price": 1, "fulfillmentType": 1},
        ):
            sku = doc.get("sku")
            if not sku:
                continue
            packed = _pack_product_fee_doc(doc)
            if not packed:
                continue
            _consider_donor(str(sku), packed)

    for sku, asin in sku_asin.items():
        donor = donor_by_asin.get(asin)
        if not donor or _fulfillment_per_unit(donor) <= 0:
            continue
        pf = out.get(sku)
        if pf is None:
            out[sku] = {
                **donor,
                "asin": asin,
            }
            continue
        if _fulfillment_per_unit(pf) > 0:
            continue
        pf["fba_per_unit"] = float(donor.get("fba_per_unit") or 0)
        pf["fuel_per_unit"] = float(donor.get("fuel_per_unit") or 0)
        pf["fulfillment_per_unit"] = float(donor.get("fulfillment_per_unit") or 0)
        if not pf.get("asin"):
            pf["asin"] = asin

    return out


def returned_qty_for_sku(physical: int, refunded: int) -> int:
    """Units to treat as returned for profitability.

    Count physical FBA returns and/or Finances refunds (returnless refunds
    have refund qty with no customerReturns row). When both exist for the
    same SKU, take max so multi-LPN physical returns aren't under-counted
    and the 20% return-processing fee is not double-charged.

    Do not require a refund to count a physical return — Seller Central's
    FBA Customer Returns report (e.g. "Unit returned to inventory") is the
    source of truth for returned units even when Finances refund sync has
    not yet attached ``refunds[]`` to the order.
    """
    physical = int(physical or 0)
    refunded = int(refunded or 0)
    return max(physical, refunded)


async def fba_returns_by_sku(
    user: dict,
    start: datetime,
    end: datetime,
    fx: UsdFx | None = None,
) -> dict[str, dict]:
    """Per-SKU returned/refunded units for orders PURCHASED in [start, end].

    Matches Amazon / Aurora Returned Orders activity:
      - physical FBA returns (`customerReturns` / hasCustomerReturn)
      - Finances refunds (`refunds` / hasRefund), including returnless refunds

    Bound by `purchaseDate` (same sale-window as the rest of Profitability),
    not return/refund date. For an order that has BOTH a physical return and a
    refund for the same SKU, units are counted once (max of the two) so the
    20% return-processing fee is not double-charged.

    Referral / revenue / FBA use the returned order lines (not a SKU-wide
    average). Mixed sale prices make ``revenue × kept/ordered`` drift — e.g.
    Andexports B08P3CD3WR July: proportional $388.26 vs true $387.92 after
    subtracting the $11.76 refunded line.

        refunded_referral += (referralFee.amount / quantityOrdered) × qty
        refunded_revenue  += (line_sales / quantityOrdered) × qty
        refunded_fulfillment += (fulfillmentFee.amount / quantityOrdered) × qty

    Returns {sku: {returned_units, refunded_referral, refunded_revenue,
    refunded_fulfillment, asin}}.
    """
    seller_id = ObjectId(str(user["_id"]))
    cursor = _db().orders.find(
        {
            "sellerId": seller_id,
            "purchaseDate": {"$gte": start, "$lte": end},
            "$or": [
                {"hasCustomerReturn": True},
                {"hasRefund": True},
            ],
        },
        {
            "orderItems": 1,
            "customerReturns": 1,
            "refunds": 1,
            "purchaseDate": 1,
            "salesChannel": 1,
            "marketplaceId": 1,
            "orderTotal": 1,
        },
    )

    out: dict[str, dict] = {}
    table = fx if fx is not None else identity_fx()
    async for order in cursor:
        items = list(order.get("orderItems") or [])
        on = infer_order_date(order)
        # Sum quantityOrdered across split lines for the same SKU — capping
        # returns at a single line's qty under-counts multi-line orders.
        ordered_by_sku: dict[str, float] = {}
        referral_by_sku: dict[str, float] = {}
        revenue_by_sku: dict[str, float] = {}
        fulfillment_by_sku: dict[str, float] = {}
        asin_from_items: dict[str, str] = {}
        primary = items[0] if items else {}
        for it in items:
            sku_key = html.unescape(str(it.get("sellerSku") or "")).strip()
            if not sku_key:
                continue
            line_ccy = infer_line_currency(order, it)
            usd_rev = table.to_usd(line_item_sales_amount(it), line_ccy, on)
            ref_block = it.get("referralFee") or {}
            usd_ref = table.to_usd(
                float(ref_block.get("amount") or 0),
                referral_fee_currency(order, it),
                on,
            )
            fba_block = it.get("fulfillmentFee") or {}
            usd_fba = table.to_usd(
                _money_amount(fba_block),
                fba_fee_currency(order, it),
                on,
            )
            ordered_by_sku[sku_key] = ordered_by_sku.get(sku_key, 0.0) + float(
                line_item_quantity(it)
            )
            referral_by_sku[sku_key] = referral_by_sku.get(sku_key, 0.0) + usd_ref
            revenue_by_sku[sku_key] = revenue_by_sku.get(sku_key, 0.0) + usd_rev
            fulfillment_by_sku[sku_key] = (
                fulfillment_by_sku.get(sku_key, 0.0) + usd_fba
            )
            if it.get("asin") and sku_key not in asin_from_items:
                asin_from_items[sku_key] = str(it["asin"])

        return_units: dict[str, int] = {}
        refund_units: dict[str, int] = {}
        asin_by_sku: dict[str, str] = dict(asin_from_items)

        # One physical unit == one license-plate-number. Count 1 per unique
        # LPN (not 1 per order). Rows without LPN use the row quantity.
        seen_lpn: set[str] = set()
        for row in order.get("customerReturns") or []:
            sku = html.unescape(str(row.get("sku") or "")).strip()
            if not sku:
                sku = html.unescape(str(primary.get("sellerSku") or "")).strip()
            if not sku:
                continue
            lpn = str(
                row.get("licensePlateNumber")
                or row.get("license_plate_number")
                or ""
            ).strip()
            if lpn:
                if lpn in seen_lpn:
                    continue
                seen_lpn.add(lpn)
                qty = 1  # one LPN = one physical returned unit
            else:
                qty = int(row.get("quantity") or 0)
            if qty <= 0:
                continue
            return_units[sku] = return_units.get(sku, 0) + qty
            if row.get("asin"):
                asin_by_sku[sku] = str(row["asin"])

        # listTransactions often stores DEFERRED then RELEASED for the same
        # refundId. Summing both would double returned_units — keep the max
        # qty per (refundId, sku), or sum rows that have no refundId.
        refund_id_qty: dict[str, int] = {}
        for row in order.get("refunds") or []:
            sku = html.unescape(str(row.get("sku") or "")).strip()
            if not sku:
                sku = html.unescape(str(primary.get("sellerSku") or "")).strip()
            if not sku:
                continue
            qty = int(row.get("quantity") or 0)
            if qty <= 0:
                continue
            if row.get("asin") and sku not in asin_by_sku:
                asin_by_sku[sku] = str(row["asin"])

            rid = str(row.get("refundId") or "").strip()
            if rid:
                key = f"{rid}|{sku}"
                prev = refund_id_qty.get(key, 0)
                if qty > prev:
                    refund_units[sku] = refund_units.get(sku, 0) - prev + qty
                    refund_id_qty[key] = qty
            else:
                # Same refund posted twice without refundId (v0 + tx) — keep max
                # qty, do not sum. Summing made 2 real refunds look like 4 units
                # and subtracted revenue/referral/FBA twice.
                key = "|".join(
                    [
                        sku,
                        str(row.get("quantity") or ""),
                        str(row.get("amount") or ""),
                        str(row.get("refundDate") or "")[:10],
                    ]
                )
                prev = refund_id_qty.get(key, 0)
                if qty > prev:
                    refund_units[sku] = refund_units.get(sku, 0) - prev + qty
                    refund_id_qty[key] = qty

        for sku in set(return_units) | set(refund_units):
            qty = returned_qty_for_sku(
                return_units.get(sku, 0),
                refund_units.get(sku, 0),
            )
            if qty <= 0:
                continue

            ordered = float(ordered_by_sku.get(sku) or 0)
            if ordered <= 0 and primary:
                ordered = float(line_item_quantity(primary))
            # Cap at units ordered — multiple Finances refund txs can each repeat
            # ProductContext.quantityShipped (e.g. fee adjustments), which would
            # otherwise double returned_units above what the customer bought.
            if ordered > 0:
                qty = min(qty, int(ordered))
            referral_total = float(referral_by_sku.get(sku) or 0)
            if referral_total <= 0 and primary:
                referral_total = table.to_usd(
                    float((primary.get("referralFee") or {}).get("amount") or 0),
                    referral_fee_currency(order, primary),
                    on,
                )
            revenue_total = float(revenue_by_sku.get(sku) or 0)
            if revenue_total <= 0 and primary:
                revenue_total = table.to_usd(
                    float(line_item_sales_amount(primary)),
                    infer_line_currency(order, primary),
                    on,
                )
            fulfillment_total = float(fulfillment_by_sku.get(sku) or 0)
            if fulfillment_total <= 0 and primary:
                fulfillment_total = table.to_usd(
                    float(
                        (primary.get("fulfillmentFee") or {}).get("amount") or 0
                    ),
                    fba_fee_currency(order, primary),
                    on,
                )
            referral_per_unit = (referral_total / ordered) if ordered > 0 else 0.0
            revenue_per_unit = (revenue_total / ordered) if ordered > 0 else 0.0
            fulfillment_per_unit = (
                (fulfillment_total / ordered) if ordered > 0 else 0.0
            )

            entry = out.setdefault(
                sku,
                {
                    "returned_units": 0,
                    "refunded_referral": 0.0,
                    "refunded_revenue": 0.0,
                    "refunded_fulfillment": 0.0,
                    "returned_units_by_marketplace": {},
                    "returned_units_by_usd_price": {},
                    "asin": None,
                },
            )
            entry["returned_units"] += qty
            entry["refunded_referral"] += referral_per_unit * qty
            entry["refunded_revenue"] += revenue_per_unit * qty
            entry["refunded_fulfillment"] += fulfillment_per_unit * qty
            ret_mp = infer_marketplace_id(order) or US_MARKETPLACE_ID
            by_mp = entry["returned_units_by_marketplace"]
            by_mp[ret_mp] = int(by_mp.get(ret_mp) or 0) + int(qty)
            unit_usd = round(float(revenue_per_unit or 0), 2)
            by_price = entry["returned_units_by_usd_price"]
            by_price[unit_usd] = int(by_price.get(unit_usd) or 0) + int(qty)
            if not entry["asin"]:
                entry["asin"] = (
                    asin_by_sku.get(sku)
                    or None
                )

    for entry in out.values():
        entry["returned_units"] = int(entry["returned_units"] or 0)
        entry["refunded_referral"] = abs(float(entry["refunded_referral"] or 0.0))
        entry["refunded_revenue"] = abs(float(entry["refunded_revenue"] or 0.0))
        entry["refunded_fulfillment"] = abs(
            float(entry["refunded_fulfillment"] or 0.0)
        )

    return out


async def returned_units_by_sku_daily(
    user_id: ObjectId,
    start: datetime,
    end: datetime,
) -> dict[str, dict[str, int]]:
    """Per-day physical FBA customer-return counts per SKU.

    Powers the Restock tab's "Include returns" toggle. Physical
    customerReturns only — returnless refunds are NOT counted here, per
    the product decision that "returns" for restock means units back in
    the warehouse (a returnless refund removes revenue but not a unit).

    Bound by purchaseDate so the return count aligns with the day the
    demand signal (units_ordered) originally landed on — matches how
    velocity/forecast treat orders. Keyed by ISO date string
    (YYYY-MM-DD) so downstream lookups don't fight with datetime/date
    equality edge cases.

    Uses the same LPN-based dedup as `fba_returns_by_sku` so a single
    physical unit that Amazon emitted twice (multiple disposition rows)
    doesn't double-count.

    Known caveat (verified against real Mongo during PR #42 review):
    physical returns typically arrive 30-60 days AFTER the purchase
    date. So for a trailing 7d or 30d window, this helper will return
    very few or zero returns even for a seller with a materially high
    return rate. The 90d and 180d windows show the true return signal,
    and the Prophet forecast (trained on 180d of history) reflects the
    adjustment fully. UI tooltip on the toggle carries the same caveat
    so users don't misread near-zero recent-window shifts as a bug."""
    cursor = _db().orders.find(
        {
            "sellerId": user_id,
            "purchaseDate": {"$gte": start, "$lte": end},
            "hasCustomerReturn": True,
        },
        {
            "orderItems": 1,
            "customerReturns": 1,
            "purchaseDate": 1,
        },
    )

    out: dict[str, dict[str, int]] = {}
    async for order in cursor:
        pd = order.get("purchaseDate")
        if not isinstance(pd, datetime):
            continue
        day_key = pd.date().isoformat()
        items = list(order.get("orderItems") or [])
        primary = items[0] if items else {}

        seen_lpn: set[str] = set()
        for row in order.get("customerReturns") or []:
            sku = html.unescape(str(row.get("sku") or "")).strip()
            if not sku:
                sku = html.unescape(str(primary.get("sellerSku") or "")).strip()
            if not sku:
                continue
            qty = int(row.get("quantity") or 0)
            if qty <= 0:
                continue
            lpn = str(
                row.get("licensePlateNumber")
                or row.get("license_plate_number")
                or ""
            ).strip()
            if lpn:
                if lpn in seen_lpn:
                    continue
                seen_lpn.add(lpn)
            per_day = out.setdefault(sku, {})
            per_day[day_key] = per_day.get(day_key, 0) + qty

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

    Prefer dated `events` when present (Seller Central transaction rows).
    Also returns rolled-up perSku rates for legacy / Finances-join docs.

    Returns the shape used by agent._build_placement_rates, plus optional
    `_events` key on a sentinel… actually returns only per_sku map.
    Use `fba_inbound_placement_charges_for_window` for date-filtered totals.
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
        rate = float(v.get("fee_rate") or v.get("avgFeePerUnit") or 0.0)
        if rate <= 0 and units > 0 and fee > 0:
            rate = fee / units
        if rate <= 0 and fee <= 0:
            continue
        bearing = units if units > 0 else (1 if rate > 0 else 0)
        out[sku] = {
            "fee_total": fee if fee > 0 else round(rate * bearing, 4),
            "units_received": units,
            "fee_bearing_units": bearing,
            "fee_rate": round(rate, 6),
            "asin": v.get("asin"),
            "fnsku": v.get("fnsku"),
        }
    return out or None


def _parse_placement_tx_date(raw: str | None) -> datetime | None:
    """Parse a placement event timestamp.

    Finances / SP-API PostedDate values are UTC (ISO, often without a trailing
    Z). Date-only strings from Seller Central CSV are handled separately by
    `_placement_event_calendar_day` — do not use this for YYYY-MM-DD alone.
    """
    if not raw:
        return None
    s = str(raw).strip().replace("/", "-")
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    # Preserve trailing Z / offset so fromisoformat keeps the correct tz.
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s[:-1] + "+00:00")
        else:
            dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            if len(s) >= 19:
                dt = datetime.fromisoformat(s[:19])
            else:
                dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        # Naive timed values from Finances join are UTC wall clocks.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _placement_event_calendar_day(raw, tz=None) -> date | None:
    """Calendar day matching the Amazon placement-fee report Transaction date.

    Seller Central's FBA Inbound Placement Fee report / CSV filters on
    Transaction date using the UTC calendar day of the charge (the date
    digits of PostedDate / report Transaction date) — NOT marketplace
    Event Date. Example: ``2026-02-01T05:13:42Z`` is still Feb 1 on the
    report even though it is Jan 31 evening in America/Los_Angeles.

    - Date-only (YYYY-MM-DD): use as written (CSV Transaction date).
    - Midnight wall clock without Z/offset: calendar day as written.
    - Timed ISO (Finances PostedDate): UTC calendar day.
    """
    if raw is None:
        return None
    s = str(raw).strip().replace("/", "-")
    if not s:
        return None
    # Pure calendar day from Seller Central CSV — do not shift by timezone.
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    # Local midnight without explicit offset → calendar day as written.
    has_explicit_tz = (
        s.endswith("Z") or s.endswith("z")
        or (len(s) > 19 and ("+" in s[19:] or s[19:].startswith("-")))
    )
    midnight = re.match(
        r"^(\d{4}-\d{2}-\d{2})[T ]00:00:00(\.\d+)?$", s,
    )
    if midnight and not has_explicit_tz:
        try:
            return date.fromisoformat(midnight.group(1))
        except ValueError:
            return None

    tx = _parse_placement_tx_date(s)
    if tx is None:
        return None
    # Match Amazon report Transaction date (= UTC date of the charge).
    _ = tz  # retained for call-site compat; report does not use marketplace TZ
    return tx.astimezone(timezone.utc).date()


async def fba_inbound_placement_charges_for_window(
    user: dict,
    start_dt: datetime,
    end_dt: datetime,
    marketplace_tz: str | None = None,
    display_start: str | None = None,
    display_end: str | None = None,
) -> tuple[dict[str, float], dict[str, float], dict]:
    """Sum placement charges whose report Transaction date falls in the
    profitability filter — same filter as the Amazon placement fee report.

    Attribute each charge by the report Transaction date (UTC calendar day
    of PostedDate / CSV Transaction date). Do NOT convert to marketplace
    Event Date — that under-counts months vs the SC Excel Total charge sum
    (e.g. andexports Feb: report $302.22 vs Pacific Event Date $248.22).

    When `display_start` / `display_end` (YYYY-MM-DD from the date picker)
    are provided, those exact calendar days are the inclusive bounds.

    Returns (by_sku_fee_total, by_asin_fee_total, meta).
    ASIN map only includes events that could not be resolved to a seller SKU
    (avoids double-counting when both maps are used).
    """
    from marketplace_timezone import inclusive_ymd_bounds

    seller_id = ObjectId(str(user["_id"]))
    doc = await _db().fbainboundplacementfees.find_one({"sellerId": seller_id})
    meta: dict = {"source": None, "event_count": 0, "matched_events": 0}
    if not doc:
        return {}, {}, meta

    meta["source"] = doc.get("source") or "unknown"
    # Prefer Amazon-authoritative `charges` (SC Total charge / Finances tx
    # amounts) for BLENDED window totals so Aurora matches Seller Central.
    # Fall back to SKU-split `events` for legacy docs.
    charge_rows = doc.get("charges") or []
    event_rows = doc.get("events") or []
    if charge_rows:
        total_rows = charge_rows
        meta["totals_from"] = "amazon_charges"
    else:
        total_rows = event_rows
        meta["totals_from"] = "events_fallback"
    events = event_rows  # still used for optional per-SKU maps
    if not total_rows and not events:
        return {}, {}, meta

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    by_sku: dict[str, float] = defaultdict(float)
    by_asin: dict[str, float] = defaultdict(float)
    meta["event_count"] = len(events)
    meta["charge_count"] = len(charge_rows)
    window_total = 0.0
    matched_charges = 0

    # Inclusive picker calendar days (= report Transaction date filter).
    start_day, end_day = inclusive_ymd_bounds(
        start_dt, end_dt, marketplace_tz or "UTC", display_start, display_end,
    )
    meta["window_start"] = start_day.isoformat()
    meta["window_end"] = end_day.isoformat()
    meta["date_basis"] = "report_transaction_date_utc"
    meta["bounds_from_display"] = bool(
        display_start and display_end
        and len(str(display_start).strip()) == 10
        and len(str(display_end).strip()) == 10
    )

    # BLENDED Totals: sum Amazon charge rows in the report Transaction window.
    # Collapse SKU-split rows that share shipment_id + transaction_date into one
    # charge first (seed/legacy docs stored per-SKU event rows as `charges`).
    # Sum of shares == Amazon shipment charge; collapsing prevents double-count
    # if shipment-level and SKU-level rows ever coexist.
    collapsed: dict[tuple, float] = {}
    for row in total_rows:
        if not isinstance(row, dict):
            continue
        tx_raw = row.get("transaction_date")
        tx_day = _placement_event_calendar_day(tx_raw)
        if tx_day is None:
            continue
        if tx_day < start_day or tx_day > end_day:
            continue
        fee = round(float(row.get("fee_total") or 0), 2)
        if fee <= 0:
            continue
        ship = (row.get("shipment_id") or row.get("amazon_shipment_id") or "").strip()
        key = (ship or id(row), str(tx_raw or ""), tx_day.isoformat())
        collapsed[key] = round(collapsed.get(key, 0.0) + fee, 2)

    for fee in collapsed.values():
        matched_charges += 1
        window_total = round(window_total + fee, 2)

    meta["matched_events"] = matched_charges
    meta["window_total"] = round(window_total, 2)
    meta["collapsed_charge_keys"] = len(collapsed)

    # Optional per-SKU map from split events (not used for BLENDED Totals).
    for ev in events:
        if not isinstance(ev, dict):
            continue
        tx_day = _placement_event_calendar_day(ev.get("transaction_date"))
        if tx_day is None:
            continue
        if tx_day < start_day or tx_day > end_day:
            continue
        fee = round(float(ev.get("fee_total") or 0), 2)
        if fee <= 0:
            continue
        sku = (ev.get("sku") or "").strip()
        asin = (ev.get("asin") or "").strip().upper()
        if sku:
            by_sku[sku] = round(by_sku[sku] + fee, 2)
        elif asin:
            by_asin[asin] = round(by_asin[asin] + fee, 2)

    return (
        {k: round(v, 2) for k, v in by_sku.items()},
        {k: round(v, 2) for k, v in by_asin.items()},
        meta,
    )


async def resolve_placement_events_to_skus(
    user: dict,
    events: list[dict],
) -> list[dict]:
    """Attach seller SKU to each dated placement event via FNSKU/ASIN."""
    if not events:
        return []
    # Reuse resolve on a synthetic per-key map, then stamp sku onto events.
    synthetic: dict[str, dict] = {}
    for ev in events:
        fn = (ev.get("fnsku") or "").strip().upper()
        asin = (ev.get("asin") or "").strip().upper()
        key = fn or asin or (ev.get("sku") or "")
        if not key:
            continue
        synthetic[key] = {
            "fee_total": float(ev.get("fee_total") or 0),
            "units_received": int(ev.get("units") or 0),
            "fee_bearing_units": int(ev.get("units") or 0),
            "fee_rate": float(ev.get("fee_rate") or 0),
            "asin": asin or None,
            "fnsku": fn or None,
            "sku": (ev.get("sku") or None),
        }
    resolved = await resolve_placement_report_to_skus(user, synthetic)
    # Build fnsku→sku and asin→sku from resolved
    fn_to_sku: dict[str, str] = {}
    asin_to_sku: dict[str, str] = {}
    for sku, b in resolved.items():
        if b.get("fnsku"):
            fn_to_sku[str(b["fnsku"]).upper()] = sku
        if b.get("asin"):
            asin_to_sku[str(b["asin"]).upper()] = sku

    out: list[dict] = []
    for ev in events:
        row = dict(ev)
        fn = (row.get("fnsku") or "").strip().upper()
        asin = (row.get("asin") or "").strip().upper()
        sku = (row.get("sku") or "").strip()
        if not sku and fn and fn in fn_to_sku:
            sku = fn_to_sku[fn]
        if not sku and asin and asin in asin_to_sku:
            sku = asin_to_sku[asin]
        row["sku"] = sku or None
        row["fnsku"] = fn or None
        row["asin"] = asin or None
        out.append(row)
    return out


async def resolve_placement_report_to_skus(
    user: dict,
    report_rows: dict[str, dict],
) -> dict[str, dict]:
    """Map placement report rows (often keyed by FNSKU) onto seller SKUs.

    Seller Central's FBA inbound placement service fees CSV is keyed by
    FNSKU + ASIN. Profitability is SKU-keyed, so resolve via Aurora products.
    """
    if not report_rows:
        return {}
    seller_id = ObjectId(str(user["_id"]))
    fnskus = sorted(
        {
            str(b.get("fnsku") or key).strip().upper()
            for key, b in report_rows.items()
            if isinstance(b, dict)
            and (
                b.get("fnsku")
                or (str(key).upper().startswith("X") and len(str(key)) >= 10)
            )
        }
    )
    asins = sorted(
        {
            str(b.get("asin") or "").strip().upper()
            for b in report_rows.values()
            if isinstance(b, dict) and b.get("asin")
        }
    )
    fnsku_to_sku: dict[str, str] = {}
    asin_to_skus: dict[str, list[str]] = {}
    if fnskus or asins:
        query: dict[str, Any] = {"sellerId": seller_id}
        ors = []
        if fnskus:
            ors.append({"fnSku": {"$in": fnskus}})
        if asins:
            ors.append({"asin": {"$in": asins}})
        if ors:
            query["$or"] = ors
        cursor = _db().products.find(query, {"sku": 1, "asin": 1, "fnSku": 1})
        async for doc in cursor:
            sku = (doc.get("sku") or "").strip()
            if not sku:
                continue
            fn = (doc.get("fnSku") or "").strip().upper()
            asin = (doc.get("asin") or "").strip().upper()
            if fn:
                fnsku_to_sku[fn] = sku
            if asin:
                asin_to_skus.setdefault(asin, []).append(sku)

    per_sku: dict[str, dict] = {}

    def _merge(sku: str, b: dict) -> None:
        if not sku:
            return
        dst = per_sku.setdefault(
            sku,
            {
                "fee_total": 0.0,
                "units_received": 0,
                "fee_bearing_units": 0,
                "rate_weight": 0.0,
                "rate_weighted_sum": 0.0,
                "asin": None,
                "fnsku": None,
            },
        )
        fee = float(b.get("fee_total") or 0)
        units = int(b.get("units_received") or 0)
        bearing = int(b.get("fee_bearing_units") or 0)
        rate = float(b.get("fee_rate") or 0)
        weight = bearing if bearing > 0 else (units if units > 0 else (1 if rate > 0 else 0))
        dst["fee_total"] += fee
        dst["units_received"] += units
        dst["fee_bearing_units"] += bearing if bearing > 0 else weight
        if rate > 0 and weight > 0:
            dst["rate_weight"] += weight
            dst["rate_weighted_sum"] += rate * weight
        if b.get("asin") and not dst.get("asin"):
            dst["asin"] = str(b["asin"]).upper()
        if b.get("fnsku") and not dst.get("fnsku"):
            dst["fnsku"] = str(b["fnsku"]).upper()

    for key, b in report_rows.items():
        if not isinstance(b, dict):
            continue
        explicit_sku = (b.get("sku") or "").strip()
        fnsku = (b.get("fnsku") or "").strip().upper()
        asin = (b.get("asin") or "").strip().upper()
        key_s = str(key).strip()
        key_u = key_s.upper()

        sku = explicit_sku
        if sku and (
            sku.upper() == (fnsku or "").upper()
            or sku.upper() == (asin or "").upper()
            or (sku.upper().startswith("X") and len(sku) >= 10)
        ):
            sku = ""
        if not sku and key_s and not key_u.startswith("X") and not (
            key_u.startswith("B") and len(key_u) == 10
        ):
            sku = key_s
        if not sku and fnsku and fnsku in fnsku_to_sku:
            sku = fnsku_to_sku[fnsku]
        if not sku and key_u.startswith("X") and key_u in fnsku_to_sku:
            sku = fnsku_to_sku[key_u]
            fnsku = fnsku or key_u
        if not sku and asin:
            cands = asin_to_skus.get(asin) or []
            if len(cands) == 1:
                sku = cands[0]
        if sku:
            _merge(sku, {**b, "fnsku": fnsku or b.get("fnsku"), "asin": asin or b.get("asin")})
        elif asin:
            # Keep ASIN-keyed row so _build_placement_rates can still match.
            _merge(asin, {**b, "fnsku": fnsku or b.get("fnsku"), "asin": asin})

    out: dict[str, dict] = {}
    for sku, v in per_sku.items():
        if v.get("rate_weight", 0) > 0:
            fee_rate = v["rate_weighted_sum"] / v["rate_weight"]
        elif v.get("fee_bearing_units", 0) > 0:
            fee_rate = v["fee_total"] / v["fee_bearing_units"]
        elif v.get("units_received", 0) > 0:
            fee_rate = v["fee_total"] / v["units_received"]
        else:
            continue
        if fee_rate <= 0:
            continue
        out[sku] = {
            "fee_total": round(v["fee_total"], 4),
            "units_received": int(v["units_received"]),
            "fee_bearing_units": int(v["fee_bearing_units"]),
            "fee_rate": round(fee_rate, 6),
            "asin": v.get("asin"),
            "fnsku": v.get("fnsku"),
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


async def fetch_ad_spend_by_campaign(
    user: dict, start_ymd: str, end_ymd: str,
) -> dict[str, float]:
    """Per-campaign spend in [start_ymd, end_ymd] from Aurora's
    `admetricsdailies` collection — the exact source Aurora's own
    /api/ads dashboard aggregates when a date filter is applied
    (see auroraBackend/src/services/adsMetricsService.js:124).

    YMD strings match Aurora's `date` field format. Returns
    {campaignId: total_spend_in_range}.
    """
    seller_id = ObjectId(str(user["_id"]))
    cursor = _db().admetricsdailies.aggregate([
        {"$match": {
            "sellerId": seller_id,
            "source": "DAILY",
            "date": {"$gte": start_ymd, "$lte": end_ymd},
        }},
        {"$group": {
            "_id": "$campaignId",
            "spend": {"$sum": {"$ifNull": ["$spend", 0]}},
        }},
    ])
    rows = await cursor.to_list(length=None)
    return {str(r["_id"]): float(r.get("spend") or 0) for r in rows if r.get("_id")}


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
            qty = line_item_quantity(it)
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
