"""Weight-config × model sweep: for every SKU, score every combination
of (velocity_weights config, forecast model) against a 30-day holdout
and rank the 5 × 7 = 35 combos. Persist the whole matrix so testers
can drill in.

Design in one sentence: reuse cached per-model backtest predictions
(bt.all[model].metrics.predicted_total) and cached velocity_windows,
apply each config's rescale (weighted_velocity / recent_avg) to each
model's predicted total, score against actual holdout.

Why no retraining: models don't consume velocity_weights at any stage
of training in the current pipeline (verified in refresh_forecasts_for_user).
Weight config only rescales the winner's forecast level AFTER the
picker. That means every possible (config, model) accuracy we could
observe is computable from data already sitting in forecast_cache. No
Prophet/LGBM/XGBoost/DeepAR/TFT refit is required for this sweep.

Storage: `weightSweepResults` collection. One document per run.
Full schema documented in the endpoint docstring; convenience field
`combo_ranking` at the top gives the 35 combos ranked by median
accuracy across the catalog. `per_sku_winner[i].matrix` gives every
combo's accuracy for that individual SKU (useful for spotting SKUs
where the fleet winner differs from the per-SKU winner).
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

# All model keys the picker can produce. Any model missing from bt.all
# for a given SKU (e.g. DeepAR/TFT before DEEPTS_ENABLED was on) is
# silently skipped for that SKU.
ALL_MODELS = ["prophet", "naive", "lgbm", "xgb", "ensemble", "deepar", "tft"]

HOLDOUT_DAYS = 30
LOW_VOLUME_ACTUAL = 30  # skip aggregate scoring on SKUs below this


def _score(predicted: float, actual: int) -> float | None:
    """Same formula as the picker's _score_forecast_days so numbers
    are directly comparable to the drawer's Prediction accuracy card."""
    if actual <= 0:
        return None
    return round(max(0.0, (1 - abs(predicted - actual) / actual)) * 100, 1)


def _rescale_predicted_total(
    predicted_total: float,
    recent_avg: float,
    wv_config: float,
) -> float:
    """Mirror the production reblend at forecasting/model.py:1263-1284.

    Prod formula for each forecast day: p50_new = p50_raw × (wv /
    recent_avg). Summing across the horizon: predicted_total_new =
    predicted_total_raw × scale. Edge case: recent_avg == 0 means the
    model saw a flat-zero series, in which case prod falls back to a
    flat forecast at wv_config × 1 unit/day, i.e. predicted_total =
    wv × horizon_days.
    """
    if recent_avg > 0:
        scale = wv_config / recent_avg
        return predicted_total * scale
    return wv_config * HOLDOUT_DAYS


async def _fetch_actuals_batch(
    user_id: ObjectId, cutoff: datetime,
) -> dict[str, int]:
    """One Mongo aggregation for the whole catalog, grouped by SKU in
    memory. Avoids N per-SKU queries (~2-9 sec) — pulls the last-30d
    rows once and buckets them locally (<300ms).
    """
    rows = await get_sales_daily_for_user(user_id, sku=None, since=cutoff)
    cutoff_naive = cutoff.replace(tzinfo=None) if cutoff.tzinfo else cutoff
    actuals: dict[str, int] = {}
    for r in rows:
        sku = r.get("sku")
        if not sku:
            continue
        d = r.get("date")
        if not isinstance(d, datetime):
            continue
        d_naive = d.replace(tzinfo=None) if d.tzinfo else d
        if d_naive < cutoff_naive:
            continue
        actuals[sku] = actuals.get(sku, 0) + int(r.get("units_ordered") or 0)
    return actuals


def _extract_predicted_total(bt_all_slice: dict) -> float | None:
    """Pull predicted_total for one model from the cache's bt.all
    slice. Handles the varied shapes different picker code paths write
    (some populate metrics directly, others compute later from days)."""
    metrics = (bt_all_slice or {}).get("metrics") or {}
    pt = metrics.get("predicted_total")
    if pt is not None:
        return float(pt)
    # Fallback: sum p50s from the days list if predicted_total wasn't stored.
    days = (bt_all_slice or {}).get("days") or []
    if not days:
        return None
    return round(sum(float(d.get("p50") or 0) for d in days), 2)


def _extract_recent_avg(cache_row: dict) -> float:
    """Reference recent-avg for the rescale denominator.

    The nightly picker stores drivers.recent_avg for the WINNER model
    on the cache row's top-level `drivers` field (see
    _multimodel_forecast). Different models compute recent_avg
    slightly differently (Prophet uses mean, naive uses trimmed mean),
    but they're all trailing-28-day means on the same input series and
    close in practice. Using the cached top-level value as the common
    denominator is the same choice production makes when reblending.
    """
    drivers = cache_row.get("drivers") or {}
    return float(drivers.get("recent_avg") or 0.0)


async def run_weight_sweep(user_id: ObjectId) -> dict:
    """Run the joint config × model sweep for every SKU in the user's
    forecast_cache. See module docstring for design rationale and
    format documentation."""
    computed_at = datetime.now(timezone.utc)
    cutoff = (computed_at - timedelta(days=HOLDOUT_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )

    # 1. Load every cached SKU row in one shot (bt.all + windows + drivers).
    cache_cursor = _forecast_cache().find(
        {"userId": user_id},
        {"sku": 1, "velocity_windows": 1, "drivers": 1, "backtest": 1, "_id": 0},
    )
    cache_rows = [c async for c in cache_cursor]
    n_total = len(cache_rows)
    log.info("weight_sweep: %d cached SKUs for user=%s", n_total, user_id)

    # 2. Batch-fetch 30-day actuals for every SKU (one Mongo query).
    actuals_by_sku = await _fetch_actuals_batch(user_id, cutoff)

    # 3. For each SKU, build a 5×7 matrix. Also fill accumulators for
    # the per-combo aggregates.
    per_sku_winners: list[dict] = []
    # combo_accuracies[(config_name, model_name)] = [acc, acc, ...] across
    # non-low-volume SKUs; used to compute median/mean/p75 at the end.
    combo_accuracies: dict[tuple[str, str], list[float]] = {
        (c["name"], m): [] for c in CANDIDATE_WEIGHT_CONFIGS for m in ALL_MODELS
    }
    combo_hitrate: dict[tuple[str, str], int] = {k: 0 for k in combo_accuracies}
    combo_scored_count: dict[tuple[str, str], int] = {k: 0 for k in combo_accuracies}

    n_scored = 0
    for row in cache_rows:
        sku = row.get("sku")
        if not sku:
            continue
        windows = row.get("velocity_windows") or []
        if not windows:
            continue
        recent_avg = _extract_recent_avg(row)
        bt_all = ((row.get("backtest") or {}).get("all")) or {}
        actual_total = actuals_by_sku.get(sku, 0)
        low_volume = actual_total < LOW_VOLUME_ACTUAL

        # Pre-compute wv per config so we don't re-do it per model.
        wv_by_config = {
            c["name"]: (weighted_velocity(windows, c["weights"]) or 0.0)
            for c in CANDIDATE_WEIGHT_CONFIGS
        }

        # Build the 5×7 accuracy matrix for this SKU.
        matrix: dict[str, dict[str, float | None]] = {}
        best_cell = None  # (config, model, accuracy, predicted_total)
        for cfg in CANDIDATE_WEIGHT_CONFIGS:
            row_out: dict[str, float | None] = {}
            wv = wv_by_config[cfg["name"]]
            for model_name in ALL_MODELS:
                slice_ = bt_all.get(model_name)
                if not slice_:
                    row_out[model_name] = None
                    continue
                raw_predicted = _extract_predicted_total(slice_)
                if raw_predicted is None:
                    row_out[model_name] = None
                    continue
                rescaled = _rescale_predicted_total(raw_predicted, recent_avg, wv)
                acc = _score(rescaled, actual_total)
                row_out[model_name] = acc
                # Aggregate — only non-low-volume rows count in the medians.
                if acc is not None and not low_volume:
                    key = (cfg["name"], model_name)
                    combo_accuracies[key].append(acc)
                    combo_scored_count[key] += 1
                    if acc >= 75:
                        combo_hitrate[key] += 1
                # Track best cell for this SKU.
                if acc is not None:
                    if best_cell is None or acc > best_cell["accuracy_pct"]:
                        best_cell = {
                            "config": cfg["name"],
                            "model": model_name,
                            "accuracy_pct": acc,
                            "predicted_total": round(rescaled, 1),
                        }
            matrix[cfg["name"]] = row_out

        per_sku_winners.append({
            "sku": sku,
            "actual_total": actual_total,
            "low_volume": low_volume,
            "winner": best_cell,
            "matrix": matrix,
        })
        n_scored += 1

    # 4. Aggregate per (config, model) combo.
    combo_ranking: list[dict] = []
    for cfg in CANDIDATE_WEIGHT_CONFIGS:
        for model_name in ALL_MODELS:
            accs = combo_accuracies[(cfg["name"], model_name)]
            n = len(accs)
            if n == 0:
                # Combo has no scored SKUs — skip from ranking.
                continue
            combo_ranking.append({
                "config": cfg["name"],
                "model": model_name,
                "median_accuracy_pct": round(statistics.median(accs), 1),
                "mean_accuracy_pct": round(statistics.mean(accs), 1),
                "p25_accuracy_pct": round(
                    statistics.quantiles(accs, n=4)[0] if n >= 4 else accs[0], 1,
                ),
                "p75_accuracy_pct": round(
                    statistics.quantiles(accs, n=4)[2] if n >= 4 else accs[-1], 1,
                ),
                "skus_at_or_above_75pct": combo_hitrate[(cfg["name"], model_name)],
                "n_scored": n,
            })
    # Rank best → worst by median (ties: higher mean → higher hit-rate).
    combo_ranking.sort(
        key=lambda r: (
            -r["median_accuracy_pct"],
            -r["mean_accuracy_pct"],
            -r["skus_at_or_above_75pct"],
        ),
    )
    for i, r in enumerate(combo_ranking, start=1):
        r["rank"] = i

    winner_combo = None
    if combo_ranking:
        top = combo_ranking[0]
        winner_combo = {
            "config": top["config"],
            "model": top["model"],
            "median_accuracy_pct": top["median_accuracy_pct"],
        }

    # 5. Convenience marginal aggregates — collapse each axis alone.
    config_ranking: list[dict] = []
    for cfg in CANDIDATE_WEIGHT_CONFIGS:
        # Best model within this config = highest median across the models.
        rows = [r for r in combo_ranking if r["config"] == cfg["name"]]
        if not rows:
            continue
        best_row = max(rows, key=lambda r: r["median_accuracy_pct"])
        avg_across = round(
            statistics.mean(r["median_accuracy_pct"] for r in rows), 1,
        )
        config_ranking.append({
            "config": cfg["name"],
            "best_model": best_row["model"],
            "best_model_median_accuracy_pct": best_row["median_accuracy_pct"],
            "avg_across_models": avg_across,
        })
    config_ranking.sort(key=lambda r: -r["best_model_median_accuracy_pct"])

    model_ranking: list[dict] = []
    for model_name in ALL_MODELS:
        rows = [r for r in combo_ranking if r["model"] == model_name]
        if not rows:
            continue
        best_row = max(rows, key=lambda r: r["median_accuracy_pct"])
        avg_across = round(
            statistics.mean(r["median_accuracy_pct"] for r in rows), 1,
        )
        model_ranking.append({
            "model": model_name,
            "best_config": best_row["config"],
            "best_config_median_accuracy_pct": best_row["median_accuracy_pct"],
            "avg_across_configs": avg_across,
        })
    model_ranking.sort(key=lambda r: -r["best_config_median_accuracy_pct"])

    doc = {
        "userId": user_id,
        "computedAt": computed_at,
        "holdout_start": cutoff.date().isoformat(),
        "holdout_end": (computed_at - timedelta(days=1)).date().isoformat(),
        "n_skus_total": n_total,
        "n_skus_scored": n_scored,
        "low_volume_threshold": LOW_VOLUME_ACTUAL,
        "configs": CANDIDATE_WEIGHT_CONFIGS,
        "models": ALL_MODELS,
        "combo_ranking": combo_ranking,
        "winner_combo": winner_combo,
        "per_sku_winner": per_sku_winners,
        "config_ranking": config_ranking,
        "model_ranking": model_ranking,
    }

    await _db()["weightSweepResults"].insert_one(dict(doc))
    log.info(
        "weight_sweep done for user=%s winner=%s (combos=%d, skus_scored=%d)",
        user_id, winner_combo, len(combo_ranking), n_scored,
    )
    # insert_one mutates the dict with _id — strip so the return is
    # JSON-serializable by FastAPI's default encoder.
    doc.pop("_id", None)
    doc["userId"] = str(user_id)
    doc["computedAt"] = computed_at.isoformat()
    return doc


async def latest_weight_sweep(user_id: ObjectId) -> dict | None:
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
