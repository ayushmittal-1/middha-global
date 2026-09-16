"""Dump one MX January order for B07TJT135V and user emails."""
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

ASIN = "B07TJT135V"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 2, 1, tzinfo=timezone.utc)


async def main() -> None:
    from auth import _db
    from bson import ObjectId

    db = _db()
    o = await db.orders.find_one(
        {
            "salesChannel": "Amazon.com.mx",
            "purchaseDate": {"$gte": START, "$lt": END},
            "orderItems.asin": ASIN,
            "orderStatus": {"$nin": ["Canceled", "Cancelled"]},
        }
    )
    if not o:
        print("no mx order")
        return
    o["_id"] = str(o["_id"])
    o["sellerId"] = str(o.get("sellerId"))
    keys = [
        "amazonOrderId", "sellerId", "salesChannel", "marketplaceId",
        "orderTotal", "purchaseDate", "orderStatus", "orderItems",
    ]
    for k in keys:
        print(k, o.get(k))

    print("\n=== users ===")
    async for u in db.users.find(
        {"_id": {"$in": [
            ObjectId("6a48bd8172ff044375386e71"),
            ObjectId("6a2bd7b9f111acb5e290ec5b"),
            ObjectId("6a3fa17bbfbe2c60e4207a68"),
        ]}},
        {"email": 1, "name": 1},
    ):
        print(u)


if __name__ == "__main__":
    asyncio.run(main())
