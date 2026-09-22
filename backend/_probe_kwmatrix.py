"""Run the keyword-matrix pipeline locally against real ASINs and dump a
relevance report, without going through HTTP or the job-polling UX.

Usage:
    python backend/_probe_kwmatrix.py <email> <asin> [<asin> ...]

Prints, per source: the raw sourced keywords, Brand Analytics coverage, and
the final 3x3 cells with their scores — which is what you need to eyeball
whether the matrix is actually about the product.
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

import keyword_matrix  # noqa: E402
from auth import current_user  # noqa: E402


async def run(email: str, asins: list[str]) -> dict:
    cli = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = cli[os.getenv("MONGO_DB_NAME") or "test"]
    u = await db.users.find_one({"email": email})
    if not u:
        raise SystemExit(f"user not found for email={email!r}")
    u["_token"] = "n/a"
    current_user.set(u)

    now = datetime.now(timezone.utc)
    job = {
        "job_id": "probe" + uuid.uuid4().hex[:7],
        "user_id": str(u["_id"]),
        "status": "running",
        "step": "sourcing",
        "asins": [a.strip().upper() for a in asins],
        "ad_group_id": None,
        "campaign_id": None,
        "ad_group_source": None,
        "started_at": now.isoformat(),
        "created_at": now,
        "sources": {s: {"keywords": [], "notes": []} for s in keyword_matrix.SOURCES},
        "titles": {},
        "matrix": None,
        "error": None,
    }
    await keyword_matrix._run_job(job)
    return job


def report(job: dict) -> None:
    print("=" * 100)
    for asin, title in (job.get("titles") or {}).items():
        print(f"ASIN {asin}: {title}")
    print(f"status={job['status']} error={job.get('error')}")
    for asin, prod in (job.get("products") or {}).items():
        print(f"  profile {asin}: head_noun={prod.get('head_noun')!r} "
              f"item_type={prod.get('item_type')!r} browse={prod.get('browse_node')!r} "
              f"product_type={prod.get('product_type')!r}")
    print("  vocabulary:", json.dumps(job.get("vocabulary") or []))
    for name, src in (job.get("sources") or {}).items():
        kws = src.get("keywords") or []
        cov = src.get("ba_coverage") or {}
        print(f"\n--- {name}: raw={len(kws)} ba={cov.get('in_report')}/{cov.get('total')}")
        for n in src.get("notes") or []:
            print(f"    note: {n}")
        print("    " + json.dumps(kws[:60]))
    m = job.get("matrix") or {}
    print("\n--- MATRIX")
    for tier in ("top", "medium", "low"):
        for col in m.get("cols") or []:
            cell = (m.get("cells") or {}).get(tier, {}).get(col) or []
            rendered = ", ".join(
                f"{e['keyword']} [{(e.get('composite') or 0):.2f}]" for e in cell
            )
            print(f"  {tier:6s} | {col:17s} | {rendered}")


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    email, asins = sys.argv[1], sys.argv[2:]
    for asin in asins:
        job = await run(email, [asin])
        report(job)


if __name__ == "__main__":
    asyncio.run(main())
