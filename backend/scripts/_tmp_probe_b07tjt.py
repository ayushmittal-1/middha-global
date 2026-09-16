"""January 2026 B07TJT135V — units, FX, returns, expected ref 46.08 / FBA 88.20."""
from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from dotenv import load_dotenv

load_dotenv(BACKEND / ".env")
load_dotenv(BACKEND.parent / ".env")

ASIN = "B07TJT135V"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _amt(block):
    if not isinstance(block, dict):
        return 0.0, ""
    return float(block.get("amount") or 0), str(block.get("currencyCode") or "")


async def main() -> None:
    from auth import _db
    from aurora_data import (
        aggregate_sku_metrics_from_orders,
        fba_returns_by_sku,
        is_excluded_order_status,
        line_item_quantity,
        line_item_sales_amount,
        line_referral_fee,
        recompute_referral_totals,
        _pack_product_fee_doc,
    )
    from currency_fx import infer_line_currency, infer_marketplace_id, load_usd_fx
    from amazon_sp import split_bundled_fulfillment_total
    from agent import (
        _apply_returns_to_sku_data,
        apply_sale_price_fba,
        resolve_sku_referral_fba_fuel,
    )

    db = _db()
    print("=== products ===")
    products = []
    async for p in db.products.find({"asin": ASIN}):
        products.append(p)
        packed = _pack_product_fee_doc(p)
        print(
            {
                "sku": p.get("sku"),
                "sellerId": str(p.get("sellerId")),
                "price": p.get("price"),
                "fees": p.get("fees"),
                "status": p.get("status"),
                "packed": packed,
            }
        )

    print("\n=== order months / channels ===")
    pipeline = [
        {"$match": {"orderItems.asin": ASIN}},
        {"$unwind": "$orderItems"},
        {"$match": {"orderItems.asin": ASIN}},
        {
            "$group": {
                "_id": {
                    "seller": "$sellerId",
                    "ym": {"$dateToString": {"format": "%Y-%m", "date": "$purchaseDate"}},
                    "channel": "$salesChannel",
                    "mp": "$marketplaceId",
                    "status": "$orderStatus",
                    "sku": "$orderItems.sellerSku",
                },
                "orders": {"$sum": 1},
                "qty": {"$sum": {"$ifNull": ["$orderItems.quantityOrdered", 1]}},
                "rev": {
                    "$sum": {
                        "$ifNull": [
                            "$orderItems.itemSubtotal.amount",
                            "$orderItems.itemPrice.amount",
                        ]
                    }
                },
                "revCcy": {"$addToSet": "$orderItems.itemSubtotal.currencyCode"},
                "priceCcy": {"$addToSet": "$orderItems.itemPrice.currencyCode"},
                "fba": {"$sum": {"$ifNull": ["$orderItems.fulfillmentFee.amount", 0]}},
                "fbaCcy": {"$addToSet": "$orderItems.fulfillmentFee.currencyCode"},
                "ref": {"$sum": {"$ifNull": ["$orderItems.referralFee.amount", 0]}},
                "refCcy": {"$addToSet": "$orderItems.referralFee.currencyCode"},
                "prices": {"$addToSet": "$orderItems.itemPrice.amount"},
            }
        },
        {"$sort": {"_id.ym": 1, "_id.channel": 1}},
    ]
    async for row in db.orders.aggregate(pipeline):
        print(row)

    jan_sellers = set()
    async for o in db.orders.find(
        {"purchaseDate": {"$gte": START, "$lt": END}, "orderItems.asin": ASIN}
    ):
        jan_sellers.add(str(o.get("sellerId")))
    print("\njan sellers", jan_sellers)

    for seller_s in sorted(jan_sellers):
        from bson import ObjectId

        seller = ObjectId(seller_s)
        orders = []
        async for o in db.orders.find(
            {
                "sellerId": seller,
                "purchaseDate": {"$gte": START, "$lt": END},
                "orderItems.asin": ASIN,
            }
        ):
            orders.append(o)
        print(f"\n===== seller {seller_s} orders {len(orders)} =====")
        paid = 0
        zero = 0
        by_ch = defaultdict(
            lambda: {
                "qty": 0,
                "rev_native": 0.0,
                "ccy": set(),
                "fba": 0.0,
                "ref": 0.0,
                "skus": set(),
            }
        )
        currencies = set()
        for o in orders:
            if is_excluded_order_status(o.get("orderStatus")):
                continue
            for it in o.get("orderItems") or []:
                if it.get("asin") != ASIN:
                    continue
                qty = line_item_quantity(it)
                amt = line_item_sales_amount(it)
                ch = o.get("salesChannel") or "?"
                ccy = infer_line_currency(o, it)
                currencies.add(ccy)
                if amt <= 0:
                    zero += qty
                    print("ZERO", o.get("amazonOrderId"), ch, qty, it.get("sellerSku"))
                    continue
                paid += qty
                b = by_ch[ch]
                b["qty"] += qty
                b["rev_native"] += amt
                b["ccy"].add(ccy)
                b["skus"].add(it.get("sellerSku"))
                fba, _ = _amt(it.get("fulfillmentFee"))
                ref, _ = _amt(it.get("referralFee"))
                b["fba"] += fba
                b["ref"] += ref
                print(
                    "LINE",
                    o.get("amazonOrderId"),
                    ch,
                    infer_marketplace_id(o),
                    it.get("sellerSku"),
                    qty,
                    round(amt, 4),
                    ccy,
                    "fba",
                    it.get("fulfillmentFee"),
                    "ref",
                    it.get("referralFee"),
                    "ret",
                    o.get("hasCustomerReturn"),
                    o.get("hasRefund"),
                )
        print("paid qty", paid, "zero qty", zero)
        for ch, b in by_ch.items():
            print("CH", ch, {k: (sorted(v) if isinstance(v, set) else round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})

        fx = await load_usd_fx(currencies, START, END)
        print("fx", fx.source, dict(fx.rates) if hasattr(fx, "rates") else fx)
        sku_data, na, n = aggregate_sku_metrics_from_orders(orders, fx=fx)
        print("skus in window", list(sku_data.keys()), "orders_count", n, "na", na)
        user = {"_id": seller}
        returns = await fba_returns_by_sku(
            user, START, datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc), fx=fx
        )
        for sku, d0 in sku_data.items():
            if (d0.get("asin") or "").upper() != ASIN:
                continue
            print("BEFORE returns", sku, {
                "units": d0.get("units"),
                "rev": round(float(d0.get("revenue") or 0), 4),
                "ref": round(float(d0.get("referral_total") or 0), 4),
                "fba": round(float(d0.get("fba_total") or 0), 4),
                "by_price": d0.get("units_by_usd_price"),
                "by_mp": d0.get("units_by_marketplace"),
            })
            print("returns map", returns.get(sku))
            recompute_referral_totals(orders, sku_data, {sku: 0.15}, fx=fx)
            print("after recompute ref", round(float(sku_data[sku].get("referral_total") or 0), 4))
            _apply_returns_to_sku_data(sku_data, returns)
            d = sku_data[sku]
            print("AFTER returns", {
                "units": d.get("units"),
                "ordered": d.get("ordered_units"),
                "returned": d.get("returned_units"),
                "net": d.get("net_units"),
                "rev": round(float(d.get("revenue") or 0), 2),
                "ordered_rev": round(float(d.get("ordered_revenue") or 0), 2),
                "ref": round(float(d.get("referral_total") or 0), 2),
                "fba_stamped": round(float(d.get("fba_total") or 0), 2),
                "by_price_net": d.get("units_by_usd_price"),
                "by_price_ord": d.get("ordered_units_by_usd_price"),
            })
            pf = None
            for p in products:
                if str(p.get("sellerId")) == seller_s and p.get("sku") == sku:
                    pf = _pack_product_fee_doc(p)
                    break
            if not pf:
                for p in products:
                    if str(p.get("sellerId")) == seller_s:
                        pf = _pack_product_fee_doc(p)
                        break
            print("product fees packed", pf)
            fee_units = int(d.get("ordered_units") or 0)
            fee_rev = float(d.get("ordered_revenue") or d.get("revenue") or 0)
            ref, fba, fuel, src = resolve_sku_referral_fba_fuel(
                line_referral=float(d.get("referral_total") or 0),
                line_fba=float(d.get("fba_total") or 0),
                bill_units=fee_units,
                revenue=fee_rev,
                product_fees=pf,
                fee_estimate=None,
                include_fuel=False,
            )
            print("resolve", {"ref": ref, "fba": fba, "fuel": fuel, "src": src})
            if pf:
                catalog = float(pf.get("fba_per_unit") or 0)
                fuel_u = float(pf.get("fuel_per_unit") or 0)
                ful = float(pf.get("fulfillment_per_unit") or 0)
                print("catalog fba/fuel/ful", catalog, fuel_u, ful)
                if fuel_u <= 0 and ful > 0:
                    base, peeled = split_bundled_fulfillment_total(ful)
                    print("peeled bundled", ful, "->", base, peeled, "x units", round(base * fee_units, 2))
                rebuilt = apply_sale_price_fba(
                    bill_units=fee_units,
                    units_by_usd_price=d.get("ordered_units_by_usd_price"),
                    fba_per_usd_price={},
                    catalog_fba_per_unit=catalog if catalog else 0,
                    include_fuel=False,
                )
                print("apply_sale_price_fba catalog", rebuilt)
            print("EXPECTED ref 46.08 fba 88.20")
            print("15% of ordered rev", round(fee_rev * 0.15, 2))
            print("15% of net rev", round(float(d.get("revenue") or 0) * 0.15, 2))


if __name__ == "__main__":
    asyncio.run(main())
