"""Probe days-of-cover math for a single SKU by ASIN."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "backend"))

from auth import current_user  # noqa: E402
from database import get_sales_daily, latest_inventory_for_user, _forecast_cache  # noqa: E402
from forecasting.model import compute_velocity_windows, weighted_velocity  # noqa: E402
from datetime import datetime, timezone, timedelta


async def main() -> None:
    user_id = ObjectId(sys.argv[1])
    asin = sys.argv[2]
    current_user.set({"_id": user_id, "email": "probe@local"})
    inv = await latest_inventory_for_user(user_id)
    match = [(sku, row) for sku, row in inv.items() if row.get("asin") == asin]
    if not match:
        print(f"[cover] no inventory row for asin={asin}")
        return
    sku, row = match[0]
    print(f"[cover] sku={sku!r} asin={asin}")
    for k in ("total", "available", "reserved", "reserved_customer_order",
             "reserved_fc_processing", "sent_to_fba", "inbound_working",
             "unfulfillable", "is_buyable"):
        print(f"[cover]   inv.{k} = {row.get(k)}")

    # Pull sales + compute per-window velocity like the drawer does
    now = datetime.now(timezone.utc)
    rows = await get_sales_daily(sku=sku, since=now - timedelta(days=180))
    windows = compute_velocity_windows(rows, now)
    print(f"[cover] windows (period, units_sold, days_in_stock, velocity):")
    for w in windows:
        print(f"[cover]   {w['period_days']:>4}d: units={w['units_sold']:>4} "
              f"in_stock_days={w['days_in_stock']:>3} vel={w['velocity']}/day")

    # Default velocity weights (from DEFAULT_PRODUCT_SETTINGS)
    default_weights = {"d3": 0, "d7": 5, "d14": 3, "d30": 2, "d60": 1, "d180": 1}
    wv = weighted_velocity(windows, default_weights)
    print(f"[cover] weighted_velocity (d7=5,d14=3,d30=2,d60=1,d180=1) = {wv}/day")

    stock_forward = (int(row.get("available") or 0)
                     + int(row.get("reserved") or 0)
                     + int(row.get("sent_to_fba") or 0)
                     + int(row.get("inbound_working") or 0))
    print(f"[cover] stock_forward (available+reserved+sent_to_fba+inbound_working) = {stock_forward}")
    if wv and wv > 0:
        print(f"[cover] days_of_cover = stock_forward / wv = {stock_forward}/{wv} = {stock_forward/wv:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
