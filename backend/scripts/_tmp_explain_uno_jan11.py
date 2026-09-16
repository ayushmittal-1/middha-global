"""Explain ASG UNO B07TJT135V fees for 2026-01-01..2026-01-11."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
from dotenv import load_dotenv

load_dotenv(BACKEND / ".env")
load_dotenv(BACKEND.parent / ".env")

ASIN = "B07TJT135V"
SKU_HINT = "UNO+UNO FLIP"


async def main() -> None:
    from auth import _db
    from aurora_data import (
        _pack_product_fee_doc,
        aggregate_sku_metrics_from_orders,
        fba_returns_by_sku,
        is_excluded_order_status,
        line_item_quantity,
        line_item_sales_amount,
        line_referral_fee,
        recompute_referral_totals,
        snap_referral_rate,
    )
    from currency_fx import infer_line_currency, infer_marketplace_id, load_usd_fx
    from agent import (
        _apply_returns_to_sku_data,
        apply_sale_price_fba,
        merge_asin_catalog_fba_bands,
        resolve_sku_referral_fba_fuel,
        usd_fba_price_band,
    )
    from marketplace_timezone import parse_date_range_for_query

    db = _db()
    start_dt, end_dt = parse_date_range_for_query(
        "2026-01-01", "2026-01-11", "America/Los_Angeles",
    )
    print("window", start_dt, end_dt)

    print("\n=== products ===")
    async for p in db.products.find({"asin": ASIN}):
        packed = _pack_product_fee_doc(p)
        print(
            {
                "sku": p.get("sku"),
                "seller": str(p.get("sellerId"))[-6:],
                "listing": (packed or {}).get("listing_price"),
                "fba": (packed or {}).get("fba_per_unit"),
                "fuel": (packed or {}).get("fuel_per_unit"),
                "ful": (packed or {}).get("fulfillment_per_unit"),
                "ref": (packed or {}).get("referral_per_unit"),
            }
        )

    sellers = set()
    async for o in db.orders.find(
        {
            "purchaseDate": {"$gte": start_dt, "$lte": end_dt},
            "orderItems.asin": ASIN,
        },
        {"sellerId": 1},
    ):
        sellers.add(o.get("sellerId"))
    print("sellers", [str(s)[-6:] for s in sellers])

    for seller in sellers:
        orders = [
            o
            async for o in db.orders.find(
                {
                    "sellerId": seller,
                    "purchaseDate": {"$gte": start_dt, "$lte": end_dt},
                    "orderItems.asin": ASIN,
                }
            )
        ]
        ccy = set()
        lines = []
        for o in orders:
            if is_excluded_order_status(o.get("orderStatus")):
                continue
            for it in o.get("orderItems") or []:
                if str(it.get("asin") or "").upper() != ASIN:
                    continue
                qty = line_item_quantity(it)
                amt = line_item_sales_amount(it)
                if amt <= 0:
                    continue
                cur = infer_line_currency(o, it)
                ccy.add(cur)
                mp = infer_marketplace_id(o)
                lines.append(
                    {
                        "sku": it.get("sellerSku"),
                        "qty": qty,
                        "amt": round(amt, 2),
                        "ccy": cur,
                        "mp": mp[-4:] if mp else "",
                        "oid": o.get("amazonOrderId"),
                    }
                )
        fx = await load_usd_fx(ccy, start_dt, end_dt)
        print("\nseller", str(seller)[-6:], "lines", len(lines), "fx", fx.source)
        us = mx = 0
        mx_native = 0.0
        by_sku_qty = {}
        for ln in lines:
            sku = ln["sku"]
            by_sku_qty[sku] = by_sku_qty.get(sku, 0) + ln["qty"]
            if ln["ccy"] == "MXN" or ln["mp"] == "Y8":
                mx += ln["qty"]
                mx_native += ln["amt"]
            else:
                us += ln["qty"]
        print("us", us, "mx", mx, "mx_native", round(mx_native, 2), "by_sku", by_sku_qty)
        for ln in lines:
            usd = fx.to_usd(ln["amt"], ln["ccy"], start_dt.date())
            print(
                " ",
                ln["sku"],
                "x",
                ln["qty"],
                ln["amt"],
                ln["ccy"],
                "->",
                round(usd, 2),
                "usd/u",
                round(usd / max(ln["qty"], 1), 2),
                "ref15",
                round(line_referral_fee(usd, 0.15, ln["qty"]), 2),
            )

        sku_data, _, _ = aggregate_sku_metrics_from_orders(orders, fx=fx)
        returns = await fba_returns_by_sku({"_id": seller}, start_dt, end_dt, fx=fx)
        for sku, d0 in list(sku_data.items()):
            if (d0.get("asin") or "").upper() != ASIN:
                continue
            if SKU_HINT not in str(sku).upper() and "UNO" not in str(sku).upper():
                continue
            recompute_referral_totals(orders, sku_data, {sku: 0.15}, fx=fx)
            _apply_returns_to_sku_data(sku_data, returns, {sku: 0.15})
            d = sku_data[sku]
            print("\nSKU", sku)
            print(
                {
                    "ordered": d.get("ordered_units"),
                    "returned": d.get("returned_units"),
                    "net": d.get("net_units"),
                    "rev": round(float(d.get("revenue") or 0), 2),
                    "ref_after_returns": round(float(d.get("referral_total") or 0), 2),
                    "by_price": d.get("units_by_usd_price"),
                    "ordered_by_price": d.get("ordered_units_by_usd_price"),
                    "mp": d.get("units_by_marketplace"),
                    "15pct_rev": round(float(d.get("revenue") or 0) * 0.15, 2),
                }
            )
            ret = returns.get(sku) or {}
            print("returns", {k: ret.get(k) for k in ("returned_units", "refunded_revenue", "refunded_referral", "refunded_fulfillment")})

            pf = {
                "listing_price": 9.92,
                "referral_per_unit": 1.49,
                "fba_per_unit": 3.38,
                "fuel_per_unit": 0.12,
                "fulfillment_per_unit": 3.50,
            }
            net = int(d.get("net_units") or 0)
            ref, fba, fuel, src = resolve_sku_referral_fba_fuel(
                line_referral=float(d.get("referral_total") or 0),
                line_fba=float(d.get("fba_total") or 0),
                bill_units=net,
                revenue=float(d.get("revenue") or 0),
                product_fees=pf,
                include_fuel=False,
            )
            print("resolve catalog", src, "ref", ref, "fba", fba, "fuel", fuel, "fba/u", round(fba / net, 4) if net else None)
            listing_bands = {"B07TJT135V": {"lt10": (3.38, 0.12), "10_50": (4.20, 0.15)}}
            rates = merge_asin_catalog_fba_bands(
                {}, asin=ASIN, listing_bands=listing_bands,
            )
            rates.pop("lt10", None)
            rebuilt = apply_sale_price_fba(
                bill_units=net,
                units_by_usd_price=d.get("units_by_usd_price"),
                fba_per_usd_price={},
                catalog_fba_per_unit=3.38,
                include_fuel=False,
                fba_per_band={k: v[0] for k, v in rates.items()},
                fuel_per_band={k: v[1] for k, v in rates.items()},
            )
            print("rebuild 4.20-band", rebuilt, "expect 4.20*net", round(4.20 * net, 2))
            print("screen referral 45.99 fba 96.6 | 96.6/net", round(96.6 / net, 4) if net else None)
            print("96.6/4.20", round(96.6 / 4.20, 4), "45.99/0.15", round(45.99 / 0.15, 2))


if __name__ == "__main__":
    asyncio.run(main())
