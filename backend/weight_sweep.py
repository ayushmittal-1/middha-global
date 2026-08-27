"""Weight-config sweep: score 5 candidate velocity_weights configs
against a rolling 30-day holdout, per SKU, and persist the results.

Purpose: testers give us a handful of candidate weight profiles and
want to know which one produces the most accurate demand predictions
across the whole catalog. We treat each config's weighted_velocity ×
30 as a *naive* 30-day forecast and score it against the SKU's actual
30-day sales.

Storage: `weightSweepResults` collection. One document per sweep run
(userId + computedAt + per-config aggregates + per-SKU × per-config
predictions). Trigger via POST /forecasting/weight-sweep. Latest run
readable via GET /forecasting/weight-sweep.

Not to be confused with the picker's model-selection backtest — that
one compares prophet/lgbm/xgb/deepar/tft/ensemble/naive against each
other and is independent of user weight config. This sweep is
specifically about tuning the weighted-velocity blend.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId

from database import (
    _db,
    _forecast_cache,
    get_sales_daily_for_user,
)
from forecasting.model import weighted_velocity

log = logging.getLogger("weight_sweep")


# The five candidate configs the tester wants swept. Names are
# user-visible; weights are the raw values (they don't need to sum to
# 100 — weighted_velocity normalizes internally).
CANDIDATE_WEIGHT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Mid-balanced",
        "weights": {"d7": 15, "d14": 20, "d30": 30, "d60": 20, "d90": 15},
        "description": (
            "Peaks at 30-day, tapering out to 7d and 90d. Good default "
            "for SKUs with steady demand and no strong recency signal."
        ),
    },
    {
        "name": "Recent-heavy",
        "weights": {"d7": 30, "d14": 25, "d30": 20, "d60": 15, "d90": 10},
        "description": (
            "Monotonic decay from 7d down to 90d. Best when demand is "
            "accelerating or you want the forecast to react quickly to "
            "recent momentum."
        ),
    },
    {
        "name": "Ultra-recent",
        "weights": {"d7": 50, "d14": 30, "d30": 15, "d60": 5, "d90": 0},
        "description": (
            "Almost all weight on the trailing 2 weeks. Reacts fastest "
            "to shifts but noisy on low-volume SKUs."
        ),
    },
    {
        "name": "Long-tail",
        "weights": {"d7": 0, "d14": 5, "d30": 15, "d60": 30, "d90": 50},
        "description": (
            "Mirror image of Recent-heavy — inverse decay. Best when "
            "recent sales look noisy and the 60-90d trend is more "
            "representative of true demand."
        ),
    },
    {
        "name": "Uniform",
        "weights": {"d7": 20, "d14": 20, "d30": 20, "d60": 20, "d90": 20},
        "description": (
            "Equal weight across all windows. Baseline to measure the "
            "other configs against — no bias in either direction."
        ),
    },
]


HOLDOUT_DAYS = 30
LOW_VOLUME_ACTUAL = 30  # skip aggregate scoring on SKUs below this


def _score(predicted: float, actual: int) -> float | None:
    """Volume accuracy metric matching the picker's backtest formula
    (see forecasting/model.py::_score_forecast_days) so numbers are
    directly comparable to the "Best model accuracy" column.

    Returns None for zero-actual holdouts (no way to score meaningfully).
    """
    if actual <= 0:
        return None
    return round(max(0.0, (1 - abs(predicted - actual) / actual)) * 100, 1)


def _compute_windows_from_cache(cache_row: dict) -> list[dict] | None:
    """Cached velocity_windows are what the picker wrote at last
    nightly refresh — reuse rather than re-aggregating sales."""
    windows = cache_row.get("velocity_windows")
    if not windows:
        return None
    return [
        {
            "period_days": int(w.get("period_days") or 0),
            "days_in_stock": int(w.get("days_in_stock") or 0),
            "units_sold": int(w.get("units_sold") or 0),
            "velocity": float(w.get("velocity") or 0.0),
        }
        for w in windows
    ]


async def _actual_units_in_window(
    user_id: ObjectId, sku: str, cutoff: datetime,
) -> int:
    """Sum units_ordered for `sku` between `cutoff` and yesterday
    (inclusive). This is the "ground truth" the sweep scores against."""
    since = cutoff
    rows = await get_sales_daily_for_user(user_id, sku=sku, since=since)
    cutoff_naive = cutoff.replace(tzinfo=None) if cutoff.tzinfo else cutoff
    total = 0
    for r in rows:
        d = r.get("date")
        if not isinstance(d, datetime):
            continue
        d_naive = d.replace(tzinfo=None) if d.tzinfo else d
        if d_naive >= cutoff_naive:
            total += int(r.get("units_ordered") or 0)
    return total


async def run_weight_sweep(user_id: ObjectId) -> dict:
    """Run the 5-config sweep for every SKU in the user's forecast_cache.

    Writes one row to `weightSweepResults` and returns it. Cheap: pulls
    cached velocity_windows (no live sales aggregation), computes 5
    weighted-velocity values per SKU, fetches per-SKU 30-day actuals,
    scores each config. Whole thing runs in ~5-15s on a 45-SKU catalog.
    """
    computed_at = datetime.now(timezone.utc)
    cutoff = (computed_at - timedelta(days=HOLDOUT_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )

    # Load every cached SKU row in one shot.
    cache_cursor = _forecast_cache().find(
        {"userId": user_id},
        {"sku": 1, "velocity_windows": 1, "_id": 0},
    )
    cache_rows = [c async for c in cache_cursor]
    log.info("weight_sweep: %d cached SKUs for user=%s", len(cache_rows), user_id)

    # Per-config aggregators — flat list per SKU, so we can compute
    # median/mean/hit-rate after the loop.
    per_config: dict[str, dict] = {
        cfg["name"]: {
            "name": cfg["name"],
            "weights": cfg["weights"],
            "description": cfg["description"],
            "per_sku": [],  # {sku, predicted, actual, accuracy_pct, low_volume}
        }
        for cfg in CANDIDATE_WEIGHT_CONFIGS
    }

    scored = 0
    for row in cache_rows:
        sku = row.get("sku")
        if not sku:
            continue
        windows = _compute_windows_from_cache(row)
        if not windows:
            continue
        actual = await _actual_units_in_window(user_id, sku, cutoff)
        for cfg in CANDIDATE_WEIGHT_CONFIGS:
            wv = weighted_velocity(windows, cfg["weights"]) or 0.0
            predicted = round(wv * HOLDOUT_DAYS, 2)
            acc = _score(predicted, actual)
            per_config[cfg["name"]]["per_sku"].append({
                "sku": sku,
                "predicted": predicted,
                "actual": actual,
                "accuracy_pct": acc,
                "low_volume": actual < LOW_VOLUME_ACTUAL,
            })
        scored += 1

    # Aggregate — for each config, compute median/mean/p75 accuracy
    # over the SKUs that had non-low-volume actuals so tiny-count SKUs
    # don't dominate the ranking.
    for cfg_name, block in per_config.items():
        scored_rows = [
            r for r in block["per_sku"]
            if r["accuracy_pct"] is not None and not r["low_volume"]
        ]
        block["n_scored"] = len(scored_rows)
        if scored_rows:
            accs = [r["accuracy_pct"] for r in scored_rows]
            block["median_accuracy_pct"] = round(statistics.median(accs), 1)
            block["mean_accuracy_pct"] = round(statistics.mean(accs), 1)
            block["p25_accuracy_pct"] = round(
                statistics.quantiles(accs, n=4)[0] if len(accs) >= 4 else accs[0], 1,
            )
            block["p75_accuracy_pct"] = round(
                statistics.quantiles(accs, n=4)[2] if len(accs) >= 4 else accs[-1], 1,
            )
            block["skus_at_or_above_75pct"] = sum(1 for a in accs if a >= 75)
        else:
            block["median_accuracy_pct"] = None
            block["mean_accuracy_pct"] = None
            block["p25_accuracy_pct"] = None
            block["p75_accuracy_pct"] = None
            block["skus_at_or_above_75pct"] = 0

    # Pick winner: highest median (ties → higher mean → higher hit-rate).
    ranked = sorted(
        per_config.values(),
        key=lambda b: (
            -(b["median_accuracy_pct"] or -1),
            -(b["mean_accuracy_pct"] or -1),
            -b["skus_at_or_above_75pct"],
        ),
    )
    winner = ranked[0]["name"] if ranked and ranked[0]["median_accuracy_pct"] is not None else None

    doc = {
        "userId": user_id,
        "computedAt": computed_at,
        "holdout_start": cutoff.date().isoformat(),
        "holdout_end": (computed_at - timedelta(days=1)).date().isoformat(),
        "n_skus_total": len(cache_rows),
        "n_skus_scored": scored,
        "low_volume_threshold": LOW_VOLUME_ACTUAL,
        "configs": list(per_config.values()),
        "winner": winner,
        "ranking": [b["name"] for b in ranked],
    }

    await _db()["weightSweepResults"].insert_one(dict(doc))
    log.info(
        "weight_sweep done for user=%s winner=%s ranking=%s",
        user_id, winner, doc["ranking"],
    )
    # insert_one mutates doc with `_id` — strip it so the return value
    # is JSON-serializable by FastAPI's default encoder.
    doc.pop("_id", None)
    doc["userId"] = str(user_id)
    doc["computedAt"] = computed_at.isoformat()
    return doc


async def latest_weight_sweep(user_id: ObjectId) -> dict | None:
    """Return the most recent sweep document for the user, or None if
    they've never run one. Used by GET /forecasting/weight-sweep."""
    doc = await _db()["weightSweepResults"].find_one(
        {"userId": user_id}, sort=[("computedAt", -1)],
    )
    if not doc:
        return None
    doc.pop("_id", None)
    doc["userId"] = str(doc["userId"])
    if isinstance(doc.get("computedAt"), datetime):
        doc["computedAt"] = doc["computedAt"].isoformat()
    return doc
