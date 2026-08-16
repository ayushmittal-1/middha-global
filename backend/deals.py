"""Amazon Deals — active + historical view for the Deals tab.

Three data sources, three levels of coverage:

  1. HISTORICAL "deals that ran" — computed from Aurora's own `orders`
     collection using the per-line `promotionIds` + `promotionDiscount`
     fields Amazon fills in whenever a promo triggered a discount. Fully
     covers Lightning Deals, 7-Day Deals, Best Deals, coupons, Prime
     Exclusive Discounts, subscribe-and-save — anything Amazon
     attributed to a promotion on the sale. Pure DB read, zero external
     API calls.

     Known caveat (verified against real Mongo during PR #43 build):
     Subscribe & Save orders arrive with `promotionDiscount.amount = 0`
     because Amazon accounts for the SNS discount in the Finances
     events feed (listFinancialEvents), not on the order line item.
     So an "SNS Promotion V2" row in this report will honestly show
     units + orders but `$0` discount. Non-SNS deals (Lightning, Best
     Deal, coupons, PED) carry the real discount amount on the line.
     Wiring SNS discounts through would require joining against
     Finances — deferred as a Phase 2 refinement.

  2. ACTIVE COUPONS — via SP-API `/promotions/v1/promotions` (best-
     effort). Amazon's API surface here is narrow and per-region; a 403
     / 404 for a specific seller yields `{"unavailable": <reason>}`
     rather than failing the whole endpoint.

  3. PRIME EXCLUSIVE DISCOUNTS — also best-effort via SP-API. Same
     defensive treatment.

Explicitly NOT covered (return an informational placeholder in the
response so the FE can show what's missing and why):

  - Lightning Deals list ("my active/upcoming lightning deals")
  - Deal opportunities ("deals I could run", eligible ASINs)
  - Amazon's "suggested inventory" for a deal being created

These are Seller-Central-UI features with no known SP-API endpoint. A
scraping-based path would need a separate design conversation about
seller credential storage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

from bson import ObjectId

log = logging.getLogger("deals")


UNSUPPORTED_DEAL_TYPES: tuple[dict, ...] = (
    {
        "name": "Lightning Deals",
        "reason": (
            "Amazon SP-API doesn't expose a list of active or upcoming "
            "Lightning Deals for a seller. This is a Seller Central-only "
            "view. Check Seller Central → Advertising → Deals."
        ),
    },
    {
        "name": "7-Day Deals / Best Deals",
        "reason": (
            "Same as Lightning Deals — no SP-API endpoint. Managed in "
            "Seller Central → Advertising → Deals."
        ),
    },
    {
        "name": "Deal opportunities (eligible ASINs)",
        "reason": (
            "Amazon calculates deal eligibility internally per ASIN. No "
            "SP-API endpoint exposes 'which of my products qualify for a "
            "deal right now'. Visible only in the Deals dashboard in "
            "Seller Central."
        ),
    },
    {
        "name": "Amazon-suggested inventory for a deal",
        "reason": (
            "The 'recommended inventory' number Amazon shows while you're "
            "creating a deal is calculated on the fly and not returned by "
            "any SP-API endpoint. Compare against our Restock tab for a "
            "coverage estimate based on your own velocity."
        ),
    },
)


# ── Historical: rollup of past promotional order lines ───────────────────


def aggregate_historical_promotions(order_docs: Iterable[dict]) -> dict:
    """Group Aurora orders by promotionId and return per-promotion rollups.

    Pure function — takes a list of order docs (as returned by
    `aurora_data.fetch_orders_with_items`) and returns a dict shaped for
    the /deals response. Any order-line with a non-empty `promotionIds`
    array counts.

    Attribution when a line has multiple promotion IDs: credit the
    discount + units to the FIRST id only. Amazon's own Deals dashboard
    treats the first as the primary attribution; secondary IDs are
    typically stackable coupons / SNS which shouldn't inflate the
    primary deal's numbers.

    Returned shape:
      {
        "promotions": [
          {
            "promotion_id": str,
            "units_sold": int,
            "orders": int,
            "gross_revenue": float,
            "discount_amount": float,     # always POSITIVE (money returned to buyer)
            "net_revenue": float,          # gross - discount
            "avg_discount_per_unit": float,
            "first_seen": ISO date,
            "last_seen": ISO date,
            "skus": [top 10 SKUs by units affected],
            "sku_count": int,
          }, ...
        ],
        "totals": {
          "unique_promotions": int,
          "units_sold": int,
          "gross_revenue": float,
          "discount_amount": float,
          "net_revenue": float,
        }
      }
    """
    per_promo: dict[str, dict] = {}
    order_ids_by_promo: dict[str, set[str]] = {}
    skus_by_promo: dict[str, dict[str, int]] = {}

    for doc in order_docs or []:
        if not isinstance(doc, dict):
            continue
        order_id = str(doc.get("amazonOrderId") or "") or None
        purchase_date = doc.get("purchaseDate")
        if not isinstance(purchase_date, datetime):
            purchase_date = None
        for item in doc.get("orderItems") or []:
            if not isinstance(item, dict):
                continue
            promo_ids = item.get("promotionIds") or []
            if not promo_ids:
                continue
            primary = str(promo_ids[0] or "").strip()
            if not primary:
                continue
            # Money fields — Aurora order sync shape.
            def _money(block) -> float:
                if not isinstance(block, dict):
                    return 0.0
                try:
                    return float(block.get("amount") or 0)
                except (TypeError, ValueError):
                    return 0.0
            gross = _money(item.get("itemSubtotal")) or _money(item.get("itemPrice"))
            discount = _money(item.get("promotionDiscount"))
            qty = int(item.get("quantityOrdered") or 0)
            sku = (item.get("sellerSku") or "").strip()

            entry = per_promo.setdefault(primary, {
                "promotion_id": primary,
                "units_sold": 0,
                "gross_revenue": 0.0,
                "discount_amount": 0.0,
                "first_seen": None,
                "last_seen": None,
            })
            entry["units_sold"] += qty
            entry["gross_revenue"] += gross
            entry["discount_amount"] += abs(discount)
            if purchase_date:
                if entry["first_seen"] is None or purchase_date < entry["first_seen"]:
                    entry["first_seen"] = purchase_date
                if entry["last_seen"] is None or purchase_date > entry["last_seen"]:
                    entry["last_seen"] = purchase_date

            if order_id:
                order_ids_by_promo.setdefault(primary, set()).add(order_id)
            if sku:
                sku_counts = skus_by_promo.setdefault(primary, {})
                sku_counts[sku] = sku_counts.get(sku, 0) + qty

    # Finalize per-promotion entries.
    promotions: list[dict] = []
    for pid, entry in per_promo.items():
        gross = round(entry["gross_revenue"], 2)
        discount = round(entry["discount_amount"], 2)
        units = int(entry["units_sold"])
        sku_counts = skus_by_promo.get(pid, {})
        top_skus = [
            sku for sku, _ in sorted(sku_counts.items(), key=lambda kv: -kv[1])
        ][:10]
        promotions.append({
            "promotion_id": pid,
            "units_sold": units,
            "orders": len(order_ids_by_promo.get(pid, set())),
            "gross_revenue": gross,
            "discount_amount": discount,
            "net_revenue": round(gross - discount, 2),
            "avg_discount_per_unit": round(discount / units, 2) if units > 0 else 0.0,
            "first_seen": entry["first_seen"].date().isoformat() if entry["first_seen"] else None,
            "last_seen": entry["last_seen"].date().isoformat() if entry["last_seen"] else None,
            "skus": top_skus,
            "sku_count": len(sku_counts),
        })
    # Sort: biggest discount spend first (that's what a seller thinks
    # about — "which deal cost me the most?").
    promotions.sort(key=lambda p: -p["discount_amount"])

    totals = {
        "unique_promotions": len(promotions),
        "units_sold": sum(p["units_sold"] for p in promotions),
        "gross_revenue": round(sum(p["gross_revenue"] for p in promotions), 2),
        "discount_amount": round(sum(p["discount_amount"] for p in promotions), 2),
        "net_revenue": round(sum(p["net_revenue"] for p in promotions), 2),
    }
    return {"promotions": promotions, "totals": totals}


# ── SP-API best-effort fetches (defensive) ───────────────────────────────


def _sp_api_unavailable(reason: str) -> dict:
    """Shape returned when an SP-API endpoint isn't accessible for this
    seller (missing scope, 403, 404, region-specific). Kept as a dict
    (not raised as an error) so the /deals endpoint can still return
    the historical view — which is the most useful part."""
    return {"unavailable": reason, "items": []}


async def fetch_active_coupons(user: dict) -> dict:
    """Best-effort fetch of the seller's active coupons via SP-API.

    Amazon's coupons endpoint isn't universally available and its exact
    path has changed over the years. This wrapper tries the current path
    and returns `_sp_api_unavailable(...)` on 4xx so the Deals page
    still loads with the historical view."""
    try:
        import amazon_sp
    except ImportError:
        return _sp_api_unavailable("amazon_sp module not loaded")

    try:
        # SP-API coupons endpoint (best-effort — path may vary by
        # region / seller-app scope permissions). If the seller's app
        # doesn't have the promotion scope granted, this 403s.
        resp = await amazon_sp._sp_request(  # type: ignore[attr-defined]
            "GET", "/promotions/2024-06-11/coupons", params={},
        )
        items = resp.get("coupons") if isinstance(resp, dict) else None
        return {"items": items or [], "raw_source": "/promotions/2024-06-11/coupons"}
    except Exception as e:  # noqa: BLE001 — defensive
        msg = str(e)
        if "403" in msg or "AccessDenied" in msg or "Unauthorized" in msg:
            return _sp_api_unavailable(
                "Coupons endpoint returned 403 — the seller-app "
                "authorization is missing the promotions scope. Ask the "
                "seller to re-authorize with the coupons permission."
            )
        if "404" in msg or "NotFound" in msg:
            return _sp_api_unavailable(
                "Coupons endpoint returned 404 — Amazon hasn't exposed "
                "this API for the seller's marketplace/region."
            )
        return _sp_api_unavailable(f"Coupons fetch failed: {msg[:200]}")


async def fetch_prime_exclusive_discounts(user: dict) -> dict:
    """Best-effort fetch of the seller's Prime Exclusive Discounts.

    Same defensive treatment as `fetch_active_coupons` — most sellers
    don't have this scope; a 403 shouldn't break the Deals page."""
    try:
        import amazon_sp
    except ImportError:
        return _sp_api_unavailable("amazon_sp module not loaded")
    try:
        resp = await amazon_sp._sp_request(  # type: ignore[attr-defined]
            "GET", "/promotions/2024-06-11/primeExclusiveDiscounts", params={},
        )
        items = resp.get("primeExclusiveDiscounts") if isinstance(resp, dict) else None
        return {"items": items or [], "raw_source": "/promotions/2024-06-11/primeExclusiveDiscounts"}
    except Exception as e:  # noqa: BLE001 — defensive
        msg = str(e)
        if "403" in msg or "AccessDenied" in msg or "Unauthorized" in msg:
            return _sp_api_unavailable(
                "Prime Exclusive Discounts endpoint returned 403 — "
                "seller-app authorization missing the promotions scope."
            )
        if "404" in msg or "NotFound" in msg:
            return _sp_api_unavailable(
                "Prime Exclusive Discounts endpoint returned 404 — "
                "not exposed for the seller's marketplace."
            )
        return _sp_api_unavailable(f"PED fetch failed: {msg[:200]}")
