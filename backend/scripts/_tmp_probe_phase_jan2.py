"""January 2026 B07H4S83D8 aggregates per seller — no PII."""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")
load_dotenv(BACKEND.parent / ".env")

ASIN = "B07H4S83D8"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 2, 1, tzinfo=timezone.utc)


async def main() -> None:
    from auth import _db
    from aurora_data import (
        aggregate_sku_metrics_from_orders,
        fba_returns_by_sku,
        is_excluded_order_status,
        line_item_quantity,
        line_item_sales_amount,
        recompute_referral_totals,
    )
    from currency_fx import infer_line_currency, infer_marketplace_id, load_usd_fx
    from agent import _apply_returns_to_sku_data

    db = _db()
    sellers: set[str] = set()
    async for o in db.orders.find(
        {"purchaseDate": {"$gte": START, "$lt": END}, "orderItems.asin": ASIN},
        {"sellerId": 1},
    ):
        sellers.add(str(o.get("sellerId")))
    print("sellers", len(sellers), sorted(s[-6:] for s in sellers))

    from bson import ObjectId

    for seller_s in sorted(sellers):
        seller = ObjectId(seller_s)
        orders = [
            o
            async for o in db.orders.find(
                {
                    "sellerId": seller,
                    "purchaseDate": {"$gte": START, "$lt": END},
                    "orderItems.asin": ASIN,
                }
            )
        ]
        ccy = set()
        us_qty = mx_qty = 0
        mx_native = 0.0
        for o in orders:
            if is_excluded_order_status(o.get("orderStatus")):
                continue
            for it in o.get("orderItems") or []:
                if it.get("asin") != ASIN:
                    continue
                qty = line_item_quantity(it)
                amt = line_item_sales_amount(it)
                if amt <= 0:
                    continue
                cur = infer_line_currency(o, it)
                ccy.add(cur)
                mp = infer_marketplace_id(o)
                if mp == "A1AM78C64UM0Y8" or cur == "MXN":
                    mx_qty += qty
                    mx_native += amt
                else:
                    us_qty += qty
        fx = await load_usd_fx(ccy, START, END)
        sku_data, _, _ = aggregate_sku_metrics_from_orders(orders, fx=fx)
        returns = await fba_returns_by_sku(
            {"_id": seller},
            START,
            datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
            fx=fx,
        )
        for sku, d0 in list(sku_data.items()):
            if (d0.get("asin") or "").upper() != ASIN:
                continue
            recompute_referral_totals(orders, sku_data, {sku: 0.15}, fx=fx)
            _apply_returns_to_sku_data(sku_data, returns)
            d = sku_data[sku]
            ordered = int(d.get("ordered_units") or 0)
            net = int(d.get("net_units") or 0)
            ret = int(d.get("returned_units") or 0)
            ordered_rev = float(d.get("ordered_revenue") or 0)
            net_rev = float(d.get("revenue") or 0)
            ref_ship = round(float(d.get("referral_total") or 0), 2)
            # net referral = 15% of net rev (per-line already on gross)
            mx_usd = ordered_rev
            print(
                {
                    "seller": seller_s[-6:],
                    "sku": sku,
                    "us_qty": us_qty,
                    "mx_qty": mx_qty,
                    "mx_native": round(mx_native, 2),
                    "ordered": ordered,
                    "returned": ret,
                    "net": net,
                    "ordered_rev": round(ordered_rev, 2),
                    "net_rev": round(net_rev, 2),
                    "ref_shipped": ref_ship,
                    "ref_15_net": round(net_rev * 0.15, 2),
                    "fba_4.20_ordered": round(4.20 * ordered, 2),
                    "fba_4.20_net": round(4.20 * net, 2),
                    "fx": fx.source,
                }
            )
            print("  expected now 201.26 / 315 | earlier expected 206.79 / 323.40")


if __name__ == "__main__":
    asyncio.run(main())
