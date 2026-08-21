"""Read one SKU's forecast_cache row and print the fields the drawer uses."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "backend"))

from auth import current_user  # noqa: E402
from database import _forecast_cache  # noqa: E402


async def main() -> None:
    user_id = ObjectId(sys.argv[1])
    sku = sys.argv[2]
    current_user.set({"_id": user_id, "email": "probe@local"})
    row = await _forecast_cache().find_one({"userId": user_id, "sku": sku})
    if not row:
        print(f"[probe] no cache row for sku={sku!r}")
        return
    bt = row.get("backtest") or {}
    m = bt.get("metrics") or {}
    picker = row.get("picker") or {}
    fc = row.get("forecast") or []
    p50_30d = sum(float(r.get("p50", 0)) for r in fc[:30])
    print(f"[probe] sku={sku}")
    print(f"[probe] generated_at={row.get('generated_at')}")
    print(f"[probe] method={row.get('method')}")
    print(f"[probe] picker={picker}")
    print(f"[probe] bt.method={bt.get('method')}")
    print(f"[probe] bt.train_start→end = {bt.get('train_start')} → {bt.get('train_end')}")
    print(f"[probe] bt.holdout_start→end = {bt.get('holdout_start')} → {bt.get('holdout_end')}")
    print(f"[probe] metrics.actual_total={m.get('actual_total')}")
    print(f"[probe] metrics.predicted_total={m.get('predicted_total')}")
    print(f"[probe] metrics.accuracy_pct={m.get('accuracy_pct')}")
    print(f"[probe] metrics.low_volume={m.get('low_volume')}")
    print(f"[probe] forward p50 sum (first 30 days) = {p50_30d:.1f}")
    print(f"[probe] drivers.recovery_bump={row.get('drivers', {}).get('recovery_bump')}")


if __name__ == "__main__":
    asyncio.run(main())
