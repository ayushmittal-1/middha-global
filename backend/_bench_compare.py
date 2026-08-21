"""Ad-hoc harness: measure forecast accuracy across a user's real SKUs.

Runs the same multi-model picker the production nightly job uses
(`_multimodel_forecast`) against every SKU with enough history, then
prints per-SKU accuracy_pct and the fleet median (excluding low-volume
SKUs). This is what we iterate against to hit the 75% target.

Usage:
    .venv/bin/python -m backend._bench_compare \
        --user-id 6a48bd8172ff044375386e71 --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "backend"))

from auth import current_user  # noqa: E402
from database import get_sales_daily_for_user  # noqa: E402
from forecasting.model import (  # noqa: E402
    BACKTEST_HOLDOUT_DAYS,
    _build_series_imputed,
    _is_real_sku,
    _multimodel_forecast,
    _row_day_naive,
    _try_deepts_train,
    _try_lgbm_train,
)
import pandas as pd  # noqa: E402


async def run(user_id_hex: str, limit: int, train_days: int) -> None:
    user_id = ObjectId(user_id_hex)
    current_user.set({"_id": user_id, "email": "bench@local"})

    today = datetime.now(timezone.utc)
    since = today - timedelta(days=train_days)
    cutoff = (today - timedelta(days=BACKTEST_HOLDOUT_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    cutoff_naive = cutoff.replace(tzinfo=None)

    print(f"[bench] fetching {train_days}d of sales for user={user_id_hex}")
    all_rows = await get_sales_daily_for_user(user_id, sku=None, since=since)
    print(f"[bench] {len(all_rows):,} sales rows total")

    rows_by_sku: dict[str, list[dict]] = {}
    for r in all_rows:
        s = r.get("sku")
        if not _is_real_sku(s):
            continue
        rows_by_sku.setdefault(s, []).append(r)
    print(f"[bench] {len(rows_by_sku)} real SKUs (post amzn.gr filter)")

    # Rank SKUs by trailing 90-day volume so we test the ones sellers
    # actually care about. Low-volume tail SKUs will show up but get
    # flagged low_volume=True and excluded from the median.
    ranked: list[tuple[str, int]] = []
    ninety_ago = today - timedelta(days=90)
    ninety_naive = ninety_ago.replace(tzinfo=None)
    for sku, rows in rows_by_sku.items():
        vol = sum(
            int(r.get("units_ordered") or 0)
            for r in rows
            if (d := _row_day_naive(r)) is not None and d >= ninety_naive
        )
        ranked.append((sku, vol))
    ranked.sort(key=lambda x: -x[1])
    picks = [s for s, _ in ranked[:limit]]
    print(f"[bench] running picker on top {len(picks)} SKUs by 90d volume")

    # Pre-train LGBM once on the whole user catalog — matches production.
    train_series_by_sku: dict = {}
    for sku_key, sku_rows in rows_by_sku.items():
        train_rows_for_lgbm = [
            r for r in sku_rows
            if (d := _row_day_naive(r)) is not None and d < cutoff_naive
        ]
        s = _build_series_imputed(train_rows_for_lgbm, cutoff)
        if not s.empty:
            train_series_by_sku[sku_key] = s
    print(f"[bench] training LGBM on {len(train_series_by_sku)} SKU panels...")
    lgbm_state, lgbm_module = _try_lgbm_train(
        train_series_by_sku,
        pd.Timestamp(cutoff.date()) - pd.Timedelta(days=1),
    )
    print(f"[bench] lgbm_state={'ok' if lgbm_state else 'skipped'}")
    print(f"[bench] training DeepAR + TFT (if DEEPTS_ENABLED)...")
    deepts_state = _try_deepts_train(
        train_series_by_sku,
        pd.Timestamp(cutoff.date()) - pd.Timedelta(days=1),
    )
    print(
        f"[bench] deepar={'ok' if deepts_state.get('deepar') else 'skipped'}"
        f" tft={'ok' if deepts_state.get('tft') else 'skipped'}"
    )

    results = []
    for i, sku in enumerate(picks, 1):
        rows = rows_by_sku[sku]
        try:
            r = _multimodel_forecast(
                sku, rows, train_series_by_sku,
                lgbm_state, lgbm_module,
                horizon=30, today=today, cutoff=cutoff,
                deepts_state=deepts_state,
            )
        except Exception as e:
            print(f"[bench] {i:>3}/{len(picks)} {sku[:40]:<40} ERROR {e}")
            continue
        bt = r.get("backtest") or {}
        m = bt.get("metrics") or {}
        picker = r.get("picker") or {}
        winner = picker.get("winner") or bt.get("method") or "?"
        acc = m.get("accuracy_pct")
        acc_d = m.get("accuracy_daily_pct")
        actual_total = m.get("actual_total")
        low = m.get("low_volume")
        cands = picker.get("candidates") or {}
        cand_str = ", ".join(
            f"{k}={v if v is not None else '-'}" for k, v in cands.items()
        )
        print(
            f"[bench] {i:>3}/{len(picks)} {sku[:40]:<40} "
            f"win={winner:<7} acc={str(acc):>6} daily={str(acc_d):>6} "
            f"actual={str(actual_total):>4} {'LOW' if low else '   '} "
            f"[{cand_str}]"
        )
        results.append({
            "sku": sku, "winner": winner, "accuracy_pct": acc,
            "accuracy_daily_pct": acc_d,
            "actual_total": actual_total, "low_volume": bool(low),
            "candidates": cands,
        })

    scored = [r for r in results if r["accuracy_pct"] is not None]
    real = [r for r in scored if not r["low_volume"]]
    print()
    print(f"[bench] scored SKUs: {len(scored)}  (excluding low-volume: {len(real)})")
    if real:
        accs = [r["accuracy_pct"] for r in real]
        daily_accs = [
            r["accuracy_daily_pct"] for r in real
            if r["accuracy_daily_pct"] is not None
        ]
        print(f"[bench] accuracy_pct         median={statistics.median(accs):5.1f}  "
              f"mean={statistics.mean(accs):5.1f}  "
              f"p25={statistics.quantiles(accs, n=4)[0] if len(accs) >= 4 else accs[0]:5.1f}  "
              f"p75={statistics.quantiles(accs, n=4)[2] if len(accs) >= 4 else accs[-1]:5.1f}")
        if daily_accs:
            print(f"[bench] accuracy_daily_pct   median={statistics.median(daily_accs):5.1f}  "
                  f"mean={statistics.mean(daily_accs):5.1f}")
        winners: dict[str, int] = {}
        for r in real:
            winners[r["winner"]] = winners.get(r["winner"], 0) + 1
        print(f"[bench] winners: {winners}")
        hit75 = sum(1 for a in accs if a >= 75)
        print(f"[bench] SKUs ≥ 75%: {hit75}/{len(accs)} ({hit75/len(accs)*100:.0f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", required=True)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--train-days", type=int, default=540)
    args = p.parse_args()
    asyncio.run(run(args.user_id, args.limit, args.train_days))


if __name__ == "__main__":
    main()
