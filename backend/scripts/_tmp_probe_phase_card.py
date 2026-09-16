"""Probe B07H4S83D8 / ASG PHASE CARD GAME profitability mix."""
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


def _amt(block):
    if not isinstance(block, dict):
        return 0.0, ""
    return float(block.get("amount") or 0), str(block.get("currencyCode") or "")


async def main() -> None:
    from auth import _db

    db = _db()
    print("=== products ===")
    async for p in db.products.find({"asin": ASIN}):
        print(
            {
                "sku": p.get("sku"),
                "sellerId": str(p.get("sellerId")),
                "price": p.get("price"),
                "fees": p.get("fees"),
                "status": p.get("status"),
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
                },
                "orders": {"$sum": 1},
                "qty": {"$sum": {"$ifNull": ["$orderItems.quantityOrdered", 1]}},
                "rev": {"$sum": {"$ifNull": ["$orderItems.itemSubtotal.amount", "$orderItems.itemPrice.amount"]}},
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

    print("\n=== sample lines (latest 8) ===")
    async for o in (
        db.orders.find({"orderItems.asin": ASIN}).sort("purchaseDate", -1).limit(8)
    ):
        for it in o.get("orderItems") or []:
            if it.get("asin") != ASIN:
                continue
            print(
                o.get("amazonOrderId"),
                o.get("purchaseDate"),
                o.get("salesChannel"),
                o.get("marketplaceId"),
                o.get("orderStatus"),
                {
                    "sku": it.get("sellerSku"),
                    "qty": it.get("quantityOrdered"),
                    "itemPrice": it.get("itemPrice"),
                    "itemSubtotal": it.get("itemSubtotal"),
                    "promo": it.get("promotionDiscount"),
                    "referral": it.get("referralFee"),
                    "fba": it.get("fulfillmentFee"),
                    "hasReturn": o.get("hasCustomerReturn"),
                    "hasRefund": o.get("hasRefund"),
                },
            )


if __name__ == "__main__":
    asyncio.run(main())
