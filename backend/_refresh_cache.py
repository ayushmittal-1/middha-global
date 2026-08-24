"""One-shot: refresh forecast_cache for a user so the drawer picks up
the new picker's output. Bypasses HTTP auth by setting the ContextVar
directly."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "backend"))

from auth import current_user  # noqa: E402
from forecasting.model import refresh_forecasts_for_user  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", required=True)
    args = p.parse_args()
    user_id = ObjectId(args.user_id)
    current_user.set({"_id": user_id, "email": "refresh@local"})
    print(f"[refresh] refit for user={args.user_id} ...")
    out = await refresh_forecasts_for_user(user_id)
    print(f"[refresh] wrote {out.get('skus')} SKUs")
    print(f"[refresh] winners: {out.get('methods')}")


if __name__ == "__main__":
    asyncio.run(main())
