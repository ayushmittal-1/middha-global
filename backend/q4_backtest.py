"""Q4 backtest: train models with cutoff at Sep 30 of the most recent
completed year, score them against actual Oct-Dec sales.

Why a separate module (not just calling _multimodel_forecast with a
different cutoff): the standard picker's backtest window is hard-coded
to BACKTEST_HOLDOUT_DAYS=30 for scoring, but Q4 is 92 days. This
module reuses the same primitive helpers (_prophet_forecast,
_naive_forecast, _score_forecast_days, etc.) but wires them for the
Q4-specific holdout length and target year.

Async job pattern:
  1. POST /forecasting/q4-backtest[/{sku}] creates a job doc in
     `q4BacktestJobs`, kicks off asyncio.create_task, returns job_id.
  2. Task updates status queued → running → done|failed.
  3. Per-SKU results land in `q4BacktestResults` collection.
  4. GET /forecasting/q4-backtest/job/{id} polls status.
  5. GET /forecasting/q4-backtest/results returns per-SKU results.

Restock table joins q4BacktestResults on load for the Q4 accuracy column.

Cost note: fleet mode retrains LGBM/XGBoost user-global on the pre-Q4
cutoff (~30-60s each) plus Prophet + Naive per-SKU (~1-2s × N). DeepAR
and TFT are supported but off by default (Q4_INCLUDE_DEEPTS env flag)
because retraining them adds 5-10 min each. Per-SKU mode skips
retraining user-global models — trains only Prophet + Naive for the
target SKU and reuses cached fleet bundles if any exist.
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from bson import ObjectId

from database import (
    _db,
    get_sales_daily_for_user,
    get_product_settings_for_user,
    DEFAULT_PRODUCT_SETTINGS,
)
from forecasting.model import (
    _apply_recovery_bump,
    _build_series,
    _build_series_imputed,
    _is_real_sku,
    _naive_forecast,
    _prophet_forecast,
    _score_forecast_days,
    _split_train_holdout,
    _try_deepts_train,
    _try_lgbm_train,
    _try_xgb_train,
    compute_velocity_windows,
    weighted_velocity,
    MIN_HISTORY_DAYS,
)
# CANDIDATE_WEIGHT_CONFIGS shared with weight_sweep so the two sweeps
# rank the same set of configs — testers compare like with like.
from weight_sweep import CANDIDATE_WEIGHT_CONFIGS

log = logging.getLogger("q4_backtest")


# ── Constants ─────────────────────────────────────────────────────────────

# Q4 = Oct 1 - Dec 31. Length is fixed at 92 days.
Q4_START_MONTH = 10
Q4_START_DAY = 1
Q4_END_MONTH = 12
Q4_END_DAY = 31
Q4_LENGTH_DAYS = 92

# Skip Q4 aggregate scoring on SKUs with under this many actual units
# in the whole quarter — matches the picker's low-volume threshold so
# tail SKUs don't drown out the ranking signal.
LOW_VOLUME_ACTUAL = 30


def _most_recent_completed_q4(today: datetime) -> tuple[datetime, datetime, datetime, int]:
    """Return (train_cutoff, q4_start, q4_end, year).

    "Most recent completed" means the last Q4 that has fully ended
    before `today`. If today is Feb 2027 → returns Q4 2026. If today
    is Aug 2026 → returns Q4 2025. If today is Nov 2026 → still Q4
    2025 because Q4 2026 hasn't ended yet.
    """
    year = today.year - 1 if today.month <= Q4_END_MONTH else today.year
    # Additional guard: if today is Jan-Sep, Q4 of (today.year - 1)
    # ended in Dec of (today.year - 1), which is fully past — pick it.
    # If today is Oct-Nov, Q4 of today.year is in progress; use Q4 of
    # (today.year - 1). If today is Dec, same — Q4 of today.year isn't
    # finished until Dec 31.
    if today.month == Q4_END_MONTH and today.day > Q4_END_DAY:
        year = today.year
    q4_start = datetime(year, Q4_START_MONTH, Q4_START_DAY, tzinfo=timezone.utc)
    q4_end = datetime(year, Q4_END_MONTH, Q4_END_DAY, tzinfo=timezone.utc)
    train_cutoff = q4_start  # cutoff for _split_train_holdout — anything ≥ cutoff = holdout
    return train_cutoff, q4_start, q4_end, year


# ── Per-SKU Q4 backtest (fast path) ───────────────────────────────────────

async def _q4_backtest_one_sku(
    user_id: ObjectId,
    sku: str,
    train_cutoff: datetime,
    q4_start: datetime,
    q4_end: datetime,
    year: int,
    lgbm_state=None,
    lgbm_module=None,
    xgb_state=None,
    xgb_module=None,
    deepts_state: dict | None = None,
) -> dict:
    """Score every model + every weight config for one SKU on the
    Q4 holdout. Optional pre-trained user-global bundles (LGBM/XGB)
    can be passed in — otherwise those models are skipped for this SKU.

    Returns the shape persisted to q4BacktestResults (see module
    docstring). Safe to call for many SKUs in a loop — no shared state.
    """
    # Wide enough sales window to cover pre-cutoff training + Q4 holdout.
    since = train_cutoff - timedelta(days=540)
    rows = await get_sales_daily_for_user(user_id, sku=sku, since=since)
    if not rows:
        return {
            "sku": sku, "year": year,
            "actual_q4_units": 0,
            "n_history_days": 0,
            "skipped": "no sales history",
        }

    train_rows, holdout_by_day = _split_train_holdout(rows, train_cutoff)
    train_fit = _build_series_imputed(train_rows, train_cutoff)
    train_series = _build_series(train_rows, train_cutoff)

    # Clip holdout to the Q4 window (Oct 1 - Dec 31 of the target year).
    q4_days = {
        d: units for d, units in holdout_by_day.items()
        if q4_start.date().isoformat() <= d <= q4_end.date().isoformat()
    }
    actual_q4_units = sum(q4_days.values())

    candidates: dict[str, dict] = {}

    # ── Prophet ──
    if len(train_series) >= MIN_HISTORY_DAYS:
        try:
            p_result = _prophet_forecast(
                train_fit, horizon=Q4_LENGTH_DAYS, today=q4_start,
            )
            _apply_recovery_bump(p_result, train_rows, q4_start)
            _, p_metrics = _score_forecast_days(
                p_result.get("forecast") or [], q4_days,
            )
            candidates["prophet"] = p_metrics or {}
        except Exception as e:
            log.warning("q4 prophet failed sku=%s: %s", sku, e)

    # ── Naive ──
    try:
        n_result = _naive_forecast(
            train_fit, horizon=Q4_LENGTH_DAYS, today=q4_start,
        )
        _apply_recovery_bump(n_result, train_rows, q4_start)
        _, n_metrics = _score_forecast_days(
            n_result.get("forecast") or [], q4_days,
        )
        candidates["naive"] = n_metrics or {}
    except Exception as e:
        log.warning("q4 naive failed sku=%s: %s", sku, e)

    # ── LGBM (only when the caller trained a user-global bundle) ──
    if lgbm_state is not None and lgbm_module is not None and not train_fit.empty:
        try:
            l_result = lgbm_module.forecast_sku(
                lgbm_state, train_fit, sku,
                horizon=Q4_LENGTH_DAYS, today=q4_start,
            )
            _apply_recovery_bump(l_result, train_rows, q4_start)
            _, l_metrics = _score_forecast_days(
                l_result.get("forecast") or [], q4_days,
            )
            candidates["lgbm"] = l_metrics or {}
        except Exception as e:
            log.warning("q4 lgbm failed sku=%s: %s", sku, e)

    # ── XGBoost ──
    if xgb_state is not None and xgb_module is not None and not train_fit.empty:
        try:
            x_result = xgb_module.forecast_sku(
                xgb_state, train_fit, sku,
                horizon=Q4_LENGTH_DAYS, today=q4_start,
            )
            _apply_recovery_bump(x_result, train_rows, q4_start)
            _, x_metrics = _score_forecast_days(
                x_result.get("forecast") or [], q4_days,
            )
            candidates["xgb"] = x_metrics or {}
        except Exception as e:
            log.warning("q4 xgb failed sku=%s: %s", sku, e)

    # ── DeepAR + TFT ──
    # Same pattern as the picker: reuse the pre-trained user-global
    # bundles the fleet orchestrator built. If they weren't trained
    # (per-SKU call, or DEEPTS_ENABLED unset) these branches no-op.
    if deepts_state and deepts_state.get("module") is not None and not train_fit.empty:
        deepts_mod = deepts_state["module"]
        for kind_key in ("deepar", "tft"):
            fc_state = deepts_state.get(kind_key)
            if fc_state is None:
                continue
            try:
                d_result = deepts_mod.forecast_sku(
                    fc_state, train_fit, sku,
                    horizon=Q4_LENGTH_DAYS, today=q4_start,
                )
                _apply_recovery_bump(d_result, train_rows, q4_start)
                _, d_metrics = _score_forecast_days(
                    d_result.get("forecast") or [], q4_days,
                )
                candidates[kind_key] = d_metrics or {}
            except Exception as e:
                log.warning("q4 %s failed sku=%s: %s", kind_key, sku, e)

    # ── Ensemble (per-day mean of positive p50s from prophet/naive/lgbm/xgb/deepar/tft) ──
    # Rebuild from the per-model forecasts we just computed.
    # For simplicity we score on total: mean of positive predicted_totals.
    positives = [
        m.get("predicted_total") for m in candidates.values()
        if m and m.get("predicted_total") and m.get("predicted_total") > 0
    ]
    if positives and actual_q4_units > 0:
        ens_predicted = sum(positives) / len(positives)
        ens_acc = round(
            max(0.0, (1 - abs(ens_predicted - actual_q4_units) / actual_q4_units)) * 100, 1,
        )
        candidates["ensemble"] = {
            "predicted_total": round(ens_predicted, 1),
            "actual_total": actual_q4_units,
            "accuracy_pct": ens_acc,
            "low_volume": actual_q4_units < LOW_VOLUME_ACTUAL,
        }

    # ── Weight-config-only predictions (each config as its own naive) ──
    # Fetch product settings — use the SKU's saved weights AND run
    # every candidate config so testers see the full picture. Score
    # each as `weighted_velocity × 92` vs Q4 actual.
    ps = await get_product_settings_for_user(user_id, sku)
    # Windows must be computed from data BEFORE the Q4 cutoff — we're
    # simulating "as if we were standing at Sep 30 of the target year".
    pre_q4_rows = [
        r for r in rows
        if isinstance(r.get("date"), datetime)
        and (r["date"].replace(tzinfo=None) if r["date"].tzinfo else r["date"]) < train_cutoff.replace(tzinfo=None)
    ]
    windows = compute_velocity_windows(pre_q4_rows, train_cutoff)
    per_config = []
    for cfg in CANDIDATE_WEIGHT_CONFIGS:
        wv = weighted_velocity(windows, cfg["weights"]) or 0.0
        predicted = round(wv * Q4_LENGTH_DAYS, 1)
        acc = None
        if actual_q4_units > 0:
            acc = round(
                max(0.0, (1 - abs(predicted - actual_q4_units) / actual_q4_units)) * 100, 1,
            )
        per_config.append({
            "name": cfg["name"],
            "weights": cfg["weights"],
            "predicted_q4": predicted,
            "accuracy_pct": acc,
        })

    # Winner across all sources (models + configs).
    all_scored = []
    for name, m in candidates.items():
        if m and m.get("accuracy_pct") is not None:
            all_scored.append({
                "source": "model",
                "name": name,
                "predicted": m.get("predicted_total"),
                "accuracy_pct": m["accuracy_pct"],
            })
    for cfg in per_config:
        if cfg.get("accuracy_pct") is not None:
            all_scored.append({
                "source": "config",
                "name": cfg["name"],
                "predicted": cfg["predicted_q4"],
                "accuracy_pct": cfg["accuracy_pct"],
            })
    all_scored.sort(key=lambda r: -r["accuracy_pct"])
    winner = all_scored[0] if all_scored else None

    return {
        "sku": sku,
        "year": year,
        "train_end": (train_cutoff - timedelta(days=1)).date().isoformat(),
        "holdout_start": q4_start.date().isoformat(),
        "holdout_end": q4_end.date().isoformat(),
        "actual_q4_units": actual_q4_units,
        "low_volume": actual_q4_units < LOW_VOLUME_ACTUAL,
        "models": {
            name: {
                "predicted_q4": m.get("predicted_total"),
                "accuracy_pct": m.get("accuracy_pct"),
            }
            for name, m in candidates.items()
        },
        "per_config": per_config,
        "winner": winner,
    }


# ── Fleet orchestrator (trains user-global bundles once, then loops) ──────

async def run_q4_backtest_fleet(user_id: ObjectId, job_id: str) -> dict:
    """Fleet-wide Q4 backtest. Trains user-global LGBM + XGBoost bundles
    on data ≤ train_cutoff, then loops every SKU calling
    _q4_backtest_one_sku. Writes each SKU's result to q4BacktestResults
    and returns aggregate summary.

    Runtime: ~1-3 min on a 45-SKU catalog without deep models.
    ~10-20 min with Q4_INCLUDE_DEEPTS=1 (deep models retrain user-global).
    """
    today = datetime.now(timezone.utc)
    train_cutoff, q4_start, q4_end, year = _most_recent_completed_q4(today)

    async def update(**fields):
        fields["updatedAt"] = datetime.now(timezone.utc)
        await _db()["q4BacktestJobs"].update_one(
            {"_id": job_id}, {"$set": fields}, upsert=True,
        )

    await update(
        status="running", stage="fetching sales",
        progress=1, total_skus=0, done_skus=0,
        year=year, holdout_start=q4_start.date().isoformat(),
        holdout_end=q4_end.date().isoformat(),
    )

    # 1. Load all sales rows once to build the user-global training panel.
    since = train_cutoff - timedelta(days=540)
    all_rows = await get_sales_daily_for_user(user_id, sku=None, since=since)
    rows_by_sku: dict[str, list[dict]] = {}
    for r in all_rows:
        s = r.get("sku")
        if not _is_real_sku(s):
            continue
        rows_by_sku.setdefault(s, []).append(r)
    total_skus = len(rows_by_sku)
    await update(total_skus=total_skus, stage="training LGBM + XGBoost panels")

    # 2. Train user-global bundles on pre-Q4 data.
    train_series_by_sku: dict = {}
    for sku_key, sku_rows in rows_by_sku.items():
        train_rows = [
            r for r in sku_rows
            if (d := (r["date"].replace(tzinfo=None) if isinstance(r.get("date"), datetime) and r["date"].tzinfo else r.get("date"))) is not None
            and d < train_cutoff.replace(tzinfo=None)
        ]
        s = _build_series_imputed(train_rows, train_cutoff)
        if not s.empty:
            train_series_by_sku[sku_key] = s

    train_end_ts = pd.Timestamp(train_cutoff.date()) - pd.Timedelta(days=1)
    lgbm_state, lgbm_module = await asyncio.to_thread(
        _try_lgbm_train, train_series_by_sku, train_end_ts,
    )
    xgb_state, xgb_module = await asyncio.to_thread(
        _try_xgb_train, train_series_by_sku, train_end_ts,
    )
    # DeepAR + TFT (gated by DEEPTS_ENABLED). Training takes 2-5 min
    # each on Render Pro Plus CPU; total fleet Q4 run goes from ~3 min
    # to ~15-25 min when they're on. `_try_deepts_train` returns a
    # dict with `deepar`/`tft`/`module` keys, or empty dict when
    # DEEPTS_ENABLED is unset.
    await update(stage="training DeepAR + TFT (this can take 10+ min)")
    deepts_state = await asyncio.to_thread(
        _try_deepts_train, train_series_by_sku, train_end_ts,
    )
    await update(stage="running per-SKU backtests")

    # 3. Loop every SKU, storing results as we go.
    results_coll = _db()["q4BacktestResults"]
    computed_at = datetime.now(timezone.utc)
    for i, sku in enumerate(rows_by_sku.keys(), start=1):
        try:
            result = await _q4_backtest_one_sku(
                user_id, sku,
                train_cutoff, q4_start, q4_end, year,
                lgbm_state, lgbm_module, xgb_state, xgb_module,
                deepts_state,
            )
            result["userId"] = user_id
            result["computedAt"] = computed_at
            result["jobId"] = job_id
            await results_coll.update_one(
                {"userId": user_id, "sku": sku, "year": year},
                {"$set": result},
                upsert=True,
            )
        except Exception as e:
            log.exception("q4 fleet: sku=%s failed: %s", sku, e)
        await update(done_skus=i, progress=int(100 * i / max(total_skus, 1)))

    # 4. Aggregate across the fleet.
    # Two views the FE + tester care about:
    #   a) `winner_leaderboard` — how many SKUs each candidate WON,
    #      sorted by wins. Answers "which candidate won most often?"
    #   b) `model_leaderboard` — for each model, median Q4 accuracy
    #      across every non-low-volume SKU (regardless of whether it
    #      won that SKU). Answers "which model is most accurate for
    #      Q4 overall?" — the metric the client explicitly asked for.
    #   c) `config_leaderboard` — same as (b) but for weight configs.
    scored_docs = await results_coll.find(
        {"userId": user_id, "year": year, "low_volume": False},
        {"winner": 1, "models": 1, "per_config": 1},
    ).to_list(length=None)

    winner_by_source: dict[str, list[float]] = {}
    model_accuracies: dict[str, list[float]] = {}
    config_accuracies: dict[str, list[float]] = {}
    for d in scored_docs:
        w = d.get("winner")
        if w and w.get("accuracy_pct") is not None:
            key = f"{w['source']}:{w['name']}"
            winner_by_source.setdefault(key, []).append(w["accuracy_pct"])
        for model_name, m in (d.get("models") or {}).items():
            acc = (m or {}).get("accuracy_pct")
            if acc is not None:
                model_accuracies.setdefault(model_name, []).append(float(acc))
        for cfg in (d.get("per_config") or []):
            acc = cfg.get("accuracy_pct")
            name = cfg.get("name")
            if acc is not None and name:
                config_accuracies.setdefault(name, []).append(float(acc))

    winner_leaderboard = [
        {
            "source_name": k,
            "n_wins": len(v),
            "median_accuracy_when_winning": round(statistics.median(v), 1),
        }
        for k, v in sorted(
            winner_by_source.items(), key=lambda kv: -len(kv[1]),
        )
    ]

    def _rank(accs_by_key: dict[str, list[float]]) -> list[dict]:
        out = []
        for k, v in accs_by_key.items():
            if not v:
                continue
            out.append({
                "name": k,
                "median_accuracy_pct": round(statistics.median(v), 1),
                "mean_accuracy_pct": round(statistics.mean(v), 1),
                "skus_at_or_above_75pct": sum(1 for a in v if a >= 75),
                "n_scored": len(v),
            })
        out.sort(key=lambda r: (
            -r["median_accuracy_pct"],
            -r["mean_accuracy_pct"],
            -r["skus_at_or_above_75pct"],
        ))
        return out

    model_leaderboard = _rank(model_accuracies)
    config_leaderboard = _rank(config_accuracies)
    best_q4_model = model_leaderboard[0] if model_leaderboard else None
    best_q4_config = config_leaderboard[0] if config_leaderboard else None
    fleet_summary = {
        "winner_leaderboard": winner_leaderboard,
        "model_leaderboard": model_leaderboard,
        "config_leaderboard": config_leaderboard,
        "best_q4_model": best_q4_model,
        "best_q4_config": best_q4_config,
    }

    finished_at = datetime.now(timezone.utc)
    await update(
        status="done",
        stage="complete",
        progress=100,
        finishedAt=finished_at,
        elapsed_sec=round((finished_at - computed_at).total_seconds(), 1),
        fleet_summary=fleet_summary,
    )
    return {
        "job_id": job_id,
        "status": "done",
        "year": year,
        "n_skus_scored": total_skus,
        "fleet_summary": fleet_summary,
        "best_q4_model": best_q4_model,
        "best_q4_config": best_q4_config,
    }


# ── Per-SKU orchestrator (fast path — no user-global retrain) ─────────────

async def run_q4_backtest_one(
    user_id: ObjectId, sku: str, job_id: str,
) -> dict:
    """Per-SKU Q4 backtest. Skips user-global model retraining — only
    runs Prophet + Naive + weight-config sweep for this SKU. ~10-30 sec.

    If the caller wants LGBM/XGB/DeepAR/TFT results, they need to
    trigger the fleet backtest instead.
    """
    today = datetime.now(timezone.utc)
    train_cutoff, q4_start, q4_end, year = _most_recent_completed_q4(today)

    async def update(**fields):
        fields["updatedAt"] = datetime.now(timezone.utc)
        await _db()["q4BacktestJobs"].update_one(
            {"_id": job_id}, {"$set": fields}, upsert=True,
        )

    await update(
        status="running", stage=f"training Prophet + Naive for {sku}",
        year=year, holdout_start=q4_start.date().isoformat(),
        holdout_end=q4_end.date().isoformat(),
        total_skus=1, done_skus=0, progress=10,
    )

    started_at = datetime.now(timezone.utc)
    try:
        result = await _q4_backtest_one_sku(
            user_id, sku, train_cutoff, q4_start, q4_end, year,
        )
        result["userId"] = user_id
        result["computedAt"] = started_at
        result["jobId"] = job_id
        await _db()["q4BacktestResults"].update_one(
            {"userId": user_id, "sku": sku, "year": year},
            {"$set": result},
            upsert=True,
        )
    except Exception as e:
        log.exception("q4 per-sku failed: %s", e)
        await update(
            status="failed",
            error=str(e),
            traceback=traceback.format_exc(),
            done_skus=0, progress=0,
        )
        return {"job_id": job_id, "status": "failed", "error": str(e)}

    finished_at = datetime.now(timezone.utc)
    await update(
        status="done", stage="complete", done_skus=1, progress=100,
        finishedAt=finished_at,
        elapsed_sec=round((finished_at - started_at).total_seconds(), 1),
        winner=result.get("winner"),
    )
    return {
        "job_id": job_id,
        "status": "done",
        "sku": sku,
        "year": year,
        "winner": result.get("winner"),
    }


# ── Job spawning + status polling ─────────────────────────────────────────

async def start_q4_job(
    user_id: ObjectId, sku: str | None = None,
) -> str:
    """Create a job doc + kick off the coroutine in the background.
    Returns the job_id for polling."""
    job_id = uuid.uuid4().hex[:16]
    scope = "sku" if sku else "fleet"
    await _db()["q4BacktestJobs"].insert_one({
        "_id": job_id,
        "userId": user_id,
        "sku": sku,
        "scope": scope,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "createdAt": datetime.now(timezone.utc),
    })
    coro = (
        run_q4_backtest_one(user_id, sku, job_id)
        if sku else run_q4_backtest_fleet(user_id, job_id)
    )
    asyncio.create_task(coro)
    return job_id


async def get_q4_job(job_id: str) -> dict | None:
    """Return the current status of a job for polling. userId is
    included so the caller can enforce ownership."""
    doc = await _db()["q4BacktestJobs"].find_one({"_id": job_id})
    if not doc:
        return None
    if isinstance(doc.get("userId"), ObjectId):
        doc["userId"] = str(doc["userId"])
    for k in ("createdAt", "updatedAt", "finishedAt"):
        if isinstance(doc.get(k), datetime):
            doc[k] = doc[k].isoformat()
    return doc


async def get_q4_results(
    user_id: ObjectId, sku: str | None = None, year: int | None = None,
) -> list[dict]:
    """Fetch persisted Q4 backtest results for the user. Filter by
    sku and/or year when supplied."""
    query: dict = {"userId": user_id}
    if sku:
        query["sku"] = sku
    if year:
        query["year"] = year
    cursor = _db()["q4BacktestResults"].find(query, {"userId": 0})
    out = []
    async for d in cursor:
        if isinstance(d.get("_id"), ObjectId):
            d["_id"] = str(d["_id"])
        for k in ("computedAt",):
            if isinstance(d.get(k), datetime):
                d[k] = d[k].isoformat()
        out.append(d)
    return out
