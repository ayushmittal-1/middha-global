"""Pull raw productSettings doc + cache row for a SKU."""
from __future__ import annotations
import asyncio, sys, json
from pathlib import Path
from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "backend"))

from auth import current_user  # noqa: E402
from database import (  # noqa: E402
    _product_settings, _forecast_cache,
    DEFAULT_PRODUCT_SETTINGS,
    get_product_settings_for_user,
    latest_inventory_for_user,
)


async def main() -> None:
    user_id = ObjectId(sys.argv[1])
    asin = sys.argv[2]
    current_user.set({"_id": user_id, "email": "probe@local"})

    inv = await latest_inventory_for_user(user_id)
    match = [(sku, row) for sku, row in inv.items() if row.get("asin") == asin]
    if not match:
        print(f"[probe] no inventory for asin={asin}")
        return
    sku, row = match[0]
    print(f"[probe] sku={sku!r}")
    print(f"[probe] DEFAULT velocity_weights = {DEFAULT_PRODUCT_SETTINGS['velocity_weights']}")

    raw = await _product_settings().find_one(
        {"userId": user_id, "sku": (sku or "").strip()},
        {"_id": 0},
    )
    print(f"[probe] RAW productSettings doc for this SKU: {raw}")
    if raw and raw.get("velocity_weights"):
        print(f"[probe]   → per-SKU override stored: {raw['velocity_weights']}")
    else:
        print(f"[probe]   → no per-SKU override; global default used")

    resolved = await get_product_settings_for_user(user_id, sku)
    print(f"[probe] RESOLVED (what code actually uses) velocity_weights = {resolved['velocity_weights']}")
    print(f"[probe] NOTE: database.py:1337 hard-overrides velocity_weights to the default,")
    print(f"[probe]       so any stored per-SKU override is IGNORED right now.")

    cache = await _forecast_cache().find_one({"userId": user_id, "sku": sku})
    if cache:
        r = cache.get("reorder") or {}
        print(f"[probe] cache.reorder.days_of_cover           = {r.get('days_of_cover')}")
        print(f"[probe] cache.reorder.stockout_date           = {r.get('stockout_date')}")
        print(f"[probe] cache.reorder.avg_daily_demand        = {r.get('avg_daily_demand')}")
        print(f"[probe] cache.reorder.recommended_po_qty      = {r.get('recommended_po_qty')}")
        print(f"[probe] cache.weighted_velocity               = {cache.get('weighted_velocity')}")


if __name__ == "__main__":
    asyncio.run(main())
