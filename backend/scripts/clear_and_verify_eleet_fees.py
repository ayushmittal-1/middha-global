"""
Clear Profitability fee-report caches for a seller, then print fee totals
from a local profitability rebuild.

Usage (from aiModel/backend):
  python scripts/clear_and_verify_eleet_fees.py
  python scripts/clear_and_verify_eleet_fees.py eleet@gmail.com 2026-07-01 2026-08-01
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BACKEND_DIR.parent / ".env")


async def main() -> None:
    email = (sys.argv[1] if len(sys.argv) > 1 else "eleet@gmail.com").strip()
    start_s = sys.argv[2] if len(sys.argv) > 2 else "2026-07-01"
    end_s = sys.argv[3] if len(sys.argv) > 3 else "2026-08-01"

    from auth import _db, current_user
    from token_encryption import hydrate_user_tokens
    from database import clear_profitability_fee_caches
    from marketplace_timezone import resolve_dashboard_timezone
    from agent import compute_profitability_data

    db = _db()
    user = await db.users.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})
    if not user:
        raise SystemExit(f"User not found: {email}")
    user = hydrate_user_tokens(user)
    current_user.set(user)

    skip_clear = os.environ.get("SKIP_CLEAR", "").strip() in ("1", "true", "yes")
    if not skip_clear:
        deleted = await clear_profitability_fee_caches(user["_id"])
        print(f"[clear] {email} caches deleted: {deleted}")
    else:
        print(f"[clear] skipped for {email}")

    mp_tz = resolve_dashboard_timezone(user) or "America/Los_Angeles"
    print(f"[verify] profitability {start_s} -> {end_s} ({mp_tz})")
    result = await compute_profitability_data(
        start=start_s,
        end=end_s,
        paginate=True,
        time_zone=mp_tz,
    )
    if isinstance(result, dict) and result.get("error"):
        print("[verify] ERROR:", result["error"])
        raise SystemExit(1)

    totals = (result or {}).get("totals") or {}
    warnings = (result or {}).get("warnings") or []
    keys = [
        "referral_fee",
        "fba_fee",
        "fuel_surcharge",
        "storage_fee",
        "aged_inventory_fee",
        "removal_fee",
        "revenue",
        "units",
    ]
    print("[verify] totals:")
    for k in keys:
        print(f"  {k}: {totals.get(k)}")
    meta_bits = {
        "storage_report_total": totals.get("storage_report_total"),
        "aged_report_total": totals.get("aged_report_total"),
        "removal_report_total": totals.get("removal_report_total"),
        "removal_blended": totals.get("removal_blended"),
        "storage_meta": (result or {}).get("storage_meta"),
        "aged_charges_meta": (result or {}).get("aged_charges_meta")
        or (result or {}).get("aged_meta"),
        "removal_meta": (result or {}).get("removal_meta"),
    }
    print("[verify] report meta:", meta_bits)
    if warnings:
        print("[verify] warnings:")
        for w in warnings[:20]:
            print(" -", w)


if __name__ == "__main__":
    asyncio.run(main())
