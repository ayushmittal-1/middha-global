"""January 2026 B07H4S83D8 for ASG seller — units, FX, returns."""
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

ASIN = "B07H4S83D8"
SELLER = "6a2bd7b9f111acb5e290ec5b"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 2, 1, tzinfo=timezone.utc)


async def main() -> None:
    from auth import _db
    from bson import ObjectId
    from aurora_data import (
        aggregate_sku_metrics_from_orders,
        fba_returns_by_sku,
        line_item_quantity,
        line_item_sales_amount,
        is_excluded_order_status,
    )
    from currency_fx import UsdFx, infer_line_currency, infer_marketplace_id, load_usd_fx, infer_order_date
    from agent import _apply_returns_to_sku_data

    db = _db()
    seller = ObjectId(SELLER)
    orders = []
    async for o in db.orders.find(
        {
            "sellerId": seller,
            "purchaseDate": {"$gte": START, "$lt": END},
            "orderItems.asin": ASIN,
        }
    ):
        orders.append(o)
    print("order docs", len(orders))

    paid = 0
    zero = 0
    by_ch = defaultdict(lambda: {"qty": 0, "rev_native": 0.0, "ccy": set(), "fba": 0.0, "fba_ccy": set(), "ref": 0.0, "ref_ccy": set()})
    for o in orders:
        if is_excluded_order_status(o.get("orderStatus")):
            continue
        for it in o.get("orderItems") or []:
            if it.get("asin") != ASIN:
                continue
            qty = line_item_quantity(it)
            amt = line_item_sales_amount(it)
            ch = o.get("salesChannel") or "?"
            if amt <= 0:
                zero += qty
                continue
            paid += qty
            b = by_ch[ch]
            b["qty"] += qty
            b["rev_native"] += amt
            b["ccy"].add(infer_line_currency(o, it))
            fba = it.get("fulfillmentFee") or {}
            b["fba"] += float(fba.get("amount") or 0)
            b["fba_ccy"].add(str(fba.get("currencyCode") or ""))
            ref = it.get("referralFee") or {}
            b["ref"] += float(ref.get("amount") or 0)
            b["ref_ccy"].add(str(ref.get("currencyCode") or ""))
            print(
                "LINE",
                o.get("amazonOrderId"),
                ch,
                infer_marketplace_id(o),
                qty,
                amt,
                infer_line_currency(o, it),
                "fba",
                fba,
                "ref",
                ref,
                "ret",
                o.get("hasCustomerReturn"),
                o.get("hasRefund"),
            )
    print("paid qty", paid, "zero qty", zero)
    for ch, b in by_ch.items():
        print("CH", ch, {k: (sorted(v) if isinstance(v, set) else v) for k, v in b.items()})

    fx = await load_usd_fx(START, END)
    print("fx source", fx.source)
    sku_data, na, n = aggregate_sku_metrics_from_orders(orders, fx=fx)
    sku = "ASG - PHASE CARD GAME PO2"
    print("agg before returns", sku_data.get(sku))
    print("na zero rows", na)

    user = {"_id": seller}
    returns = await fba_returns_by_sku(user, START, datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc), fx=fx)
    print("returns map", returns.get(sku))
    _apply_returns_to_sku_data(sku_data, returns)
    d = sku_data.get(sku) or {}
    print("after returns units/ordered/returned/net/rev/ref/fba", {
        "units": d.get("units"),
        "ordered": d.get("ordered_units"),
        "returned": d.get("returned_units"),
        "net": d.get("net_units"),
        "rev": round(float(d.get("revenue") or 0), 2),
        "ref": round(float(d.get("referral_total") or 0), 2),
        "fba": round(float(d.get("fba_total") or 0), 2),
        "by_price": d.get("units_by_usd_price"),
        "by_mp": d.get("units_by_marketplace"),
    })
    print("expected ref 206.79 fba 323.40")


if __name__ == "__main__":
    asyncio.run(main())
