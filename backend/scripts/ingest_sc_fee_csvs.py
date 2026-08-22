"""
Ingest Seller Central CSV downloads into Profitability fee caches.

Use when Amazon Reports quota is exhausted (429) but you already have the
Monthly Storage Fees / Aged Inventory Surcharge CSVs from Seller Central.

Usage (from aiModel/backend):
  python scripts/ingest_sc_fee_csvs.py eleet@gmail.com ^
    --storage "C:\\Users\\Desktop\\Downloads\\410892020687 (1).csv" ^
    --aged "C:\\Users\\Desktop\\Downloads\\410930020687.csv" ^
    --aged-start 2026-03-01 --aged-end 2026-03-31
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("email", nargs="?", default="eleet@gmail.com")
    parser.add_argument("--storage", help="Monthly Storage Fees CSV/TSV path")
    parser.add_argument("--aged", help="Aged Inventory Surcharge charges CSV/TSV path")
    parser.add_argument("--aged-start", default="2026-03-01")
    parser.add_argument("--aged-end", default="2026-03-31")
    args = parser.parse_args()

    from auth import _db, current_user
    from token_encryption import hydrate_user_tokens
    from marketplace_timezone import (
        parse_date_range_for_query,
        resolve_dashboard_timezone,
        utc_instant_to_iso_z,
    )
    from amazon_sp import (
        merge_storage_by_asin_month,
        parse_aged_surcharge_charges_report,
        parse_storage_fee_report,
    )
    from database import (
        get_storage_cache,
        merge_storage_cache,
        put_aged_surcharge_charges_cache,
    )

    db = _db()
    user = await db.users.find_one(
        {"email": {"$regex": f"^{args.email}$", "$options": "i"}},
    )
    if not user:
        raise SystemExit(f"User not found: {args.email}")
    user = hydrate_user_tokens(user)
    current_user.set(user)
    mp_tz = resolve_dashboard_timezone(user) or "America/Los_Angeles"

    if args.storage:
        path = Path(args.storage)
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        per_asin, months = parse_storage_fee_report(text)
        total = round(
            sum(
                float(b.get("monthly_fee") or 0)
                for by_m in per_asin.values()
                for b in by_m.values()
            ),
            2,
        )
        print(f"[storage] parsed months={months} asins={len(per_asin)} total=${total}")
        existing = await get_storage_cache(max_age_hours=24 * 45)
        base = (existing or {}).get("per_sku_monthly") or {}
        merged = merge_storage_by_asin_month(base, per_asin)
        out = await merge_storage_cache(merged, months, empty_months=[])
        print(
            f"[storage] cache months_covered={out.get('months_covered')} "
            f"updated_at={out.get('updated_at')}"
        )

    if args.aged:
        path = Path(args.aged)
        raw = path.read_bytes()
        for enc in ("utf-8-sig", "latin-1", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                text = raw.decode("latin-1", errors="replace")
        start_dt, end_dt = parse_date_range_for_query(
            args.aged_start, args.aged_end, mp_tz,
        )
        month_keys = sorted({
            start_dt.astimezone(__import__("zoneinfo").ZoneInfo(mp_tz)).strftime("%Y-%m"),
        })
        # Prefer months present in the file.
        per_sku, seen = parse_aged_surcharge_charges_report(text, months_filter=None)
        if seen:
            month_keys = seen
            per_sku, _ = parse_aged_surcharge_charges_report(text, months_filter=seen)
        total = round(
            sum(float(v.get("charged_total") or 0) for v in per_sku.values()), 2,
        )
        print(
            f"[aged] parsed months={seen or month_keys} skus={len(per_sku)} "
            f"total=${total}"
        )
        start_iso = utc_instant_to_iso_z(start_dt)
        end_iso = utc_instant_to_iso_z(end_dt)
        await put_aged_surcharge_charges_cache(
            per_sku, start_iso, end_iso, months=list(seen or month_keys),
        )
        print(f"[aged] cached window {start_iso} -> {end_iso} months={seen or month_keys}")

    print("[done]")


if __name__ == "__main__":
    asyncio.run(main())
