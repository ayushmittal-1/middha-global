"""Find B09JZL4J8S / Eleet across sellers and January orders."""
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


async def main() -> None:
    from auth import _db

    db = _db()
    print("products by asin", await db.products.count_documents({"asin": "B09JZL4J8S"}))
    async for p in db.products.find({"asin": "B09JZL4J8S"}).limit(10):
        print("product", p.get("sku"), p.get("sellerId"), p.get("price"), p.get("fees"))
    print("products sku regex", await db.products.count_documents({"sku": {"$regex": "Eleet", "$options": "i"}}))
    async for p in db.products.find({"sku": {"$regex": "Eleet", "$options": "i"}}).limit(10):
        print("eleet sku", p.get("sku"), p.get("asin"), p.get("sellerId"), p.get("fees"))
    print("orders asin", await db.orders.count_documents({"orderItems.asin": "B09JZL4J8S"}))
    print("orders sku eleet", await db.orders.count_documents({"orderItems.sellerSku": {"$regex": "Eleet", "$options": "i"}}))
    async for o in db.orders.find({"orderItems.sellerSku": {"$regex": "Eleet", "$options": "i"}}).limit(3):
        print("order", o.get("amazonOrderId"), o.get("sellerId"), o.get("purchaseDate"), o.get("salesChannel"), o.get("orderItems"))


if __name__ == "__main__":
    asyncio.run(main())
