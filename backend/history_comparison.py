"""All-history vs 540d training-window comparison backtest.

Answers "does training on all available sales history give better
accuracy than the current 540-day window?" — the question a tester
asks when they want to know if we're leaving accuracy on the table by
capping training data.

Design:
- Reads the current 540d picker accuracy per SKU straight from
  `forecast_cache.backtest.all[model].metrics` — no need to retrain
  540d, the last nightly refresh already scored it.
- Trains a fresh picker with unbounded history (no since= limit) using
  the same 30-day recent holdout as the standard picker so numbers
  are directly comparable.
- Per-SKU deltas + fleet-median summary + net winner ("all_history"
  vs "540d") persisted to `historyComparisonResults`.
- Async job pattern (queued → running → done) with polling, same as
  q4_backtest.

Cost note: fleet mode retrains LGBM + XGBoost user-global on all
history (~30-60s each — longer than 540d because more rows). DeepAR +
TFT retrain user-global too when DEEPTS_ENABLED is on (~5-10 min
extra). Per-SKU mode skips user-global retraining and only re-runs
Prophet + Naive for the target SKU.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from bson import ObjectId

from database import (
    _db,
    _forecast_cache,
    get_sales_daily_for_user,
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
    MIN_HISTORY_DAYS,
)

log = logging.getLogger("history_comparison")


HOLDOUT_DAYS = 30
LOW_VOLUME_ACTUAL = 30
BASELINE_WINDOW_LABEL = "540d"
CANDIDATE_WINDOW_LABEL = "all_history"


def _baseline_accuracy_from_cache(cache_row: dict) -> dict[str, float | None]:
    """Read the current 540d picker's per-model accuracy from the SKU's
    forecast_cache row. Returns {model_name: accuracy_pct} — the FE
    doesn't need us to recompute what the nightly job already scored.
    """
    bt_all = ((cache_row.get("backtest") or {}).get("all")) or {}
    out: dict[str, float | None] = {}
    for name, slice_ in bt_all.items():
        metrics = (slice_ or {}).get("metrics") or {}
        acc = metrics.get("accuracy_pct")
        out[name] = float(acc) if acc is not None else None
    return out


async def _backtest_all_history_one_sku(
    user_id: ObjectId,
    sku: str,
    all_rows: list[dict],
    lgbm_state=None,
    lgbm_module=None,
    xgb_state=None,
    xgb_module=None,
    deepts_state: dict | None = None,
) -> dict[str, float | None]:
    """Train Prophet + Naive per-SKU on ALL history, score every model
    on the last-30-day holdout. LGBM/XGBoost/DeepAR/TFT are optional —
    pass their user-global bundles when the caller has them.

    Returns {model_name: accuracy_pct}. Models missing from the caller's
    bundles are silently absent from the dict.
    """
    today = datetime.now(timezone.utc)
    cutoff = (today - timedelta(days=HOLDOUT_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    train_rows, holdout_by_day = _split_train_holdout(all_rows, cutoff)
    train_fit = _build_series_imputed(train_rows, cutoff)
    train_series = _build_series(train_rows, cutoff)

    accuracies: dict[str, float | None] = {}

    if len(train_series) >= MIN_HISTORY_DAYS:
        try:
            p = _prophet_forecast(train_fit, horizon=HOLDOUT_DAYS, today=cutoff)
            _apply_recovery_bump(p, train_rows, cutoff)
            _, pm = _score_forecast_days(p.get("forecast") or [], holdout_by_day)
            accuracies["prophet"] = (pm or {}).get("accuracy_pct")
        except Exception as e:
            log.warning("all-history prophet failed sku=%s: %s", sku, e)

    try:
        n = _naive_forecast(train_fit, horizon=HOLDOUT_DAYS, today=cutoff)
        _apply_recovery_bump(n, train_rows, cutoff)
        _, nm = _score_forecast_days(n.get("forecast") or [], holdout_by_day)
        accuracies["naive"] = (nm or {}).get("accuracy_pct")
    except Exception as e:
        log.warning("all-history naive failed sku=%s: %s", sku, e)

    if lgbm_state is not None and lgbm_module is not None and not train_fit.empty:
        try:
            l = lgbm_module.forecast_sku(
                lgbm_state, train_fit, sku, horizon=HOLDOUT_DAYS, today=cutoff,
            )
            _apply_recovery_bump(l, train_rows, cutoff)
            _, lm = _score_forecast_days(l.get("forecast") or [], holdout_by_day)
            accuracies["lgbm"] = (lm or {}).get("accuracy_pct")
        except Exception as e:
            log.warning("all-history lgbm failed sku=%s: %s", sku, e)

    if xgb_state is not None and xgb_module is not None and not train_fit.empty:
        try:
            x = xgb_module.forecast_sku(
                xgb_state, train_fit, sku, horizon=HOLDOUT_DAYS, today=cutoff,
            )
            _apply_recovery_bump(x, train_rows, cutoff)
            _, xm = _score_forecast_days(x.get("forecast") or [], holdout_by_day)
            accuracies["xgb"] = (xm or {}).get("accuracy_pct")
        except Exception as e:
            log.warning("all-history xgb failed sku=%s: %s", sku, e)

    if deepts_state and deepts_state.get("module") is not None and not train_fit.empty:
        for kind_key in ("deepar", "tft"):
            fc_state = deepts_state.get(kind_key)
            if fc_state is None:
                continue
            try:
                d = deepts_state["module"].forecast_sku(
                    fc_state, train_fit, sku, horizon=HOLDOUT_DAYS, today=cutoff,
                )
                _apply_recovery_bump(d, train_rows, cutoff)
                _, dm = _score_forecast_days(d.get("forecast") or [], holdout_by_day)
                accuracies[kind_key] = (dm or {}).get("accuracy_pct")
            except Exception as e:
                log.warning("all-history %s failed sku=%s: %s", kind_key, sku, e)

    return accuracies


def _summarize_deltas(per_sku: list[dict]) -> dict:
    """Roll per-SKU deltas up into fleet-level statistics."""
    # median delta per model
    model_deltas: dict[str, list[float]] = {}
    for r in per_sku:
        for name, m in (r.get("models") or {}).items():
            d = m.get("delta")
            if d is not None:
                model_deltas.setdefault(name, []).append(float(d))
    median_delta_by_model = {
        name: {
            "median_delta_pct": round(statistics.median(v), 1),
            "mean_delta_pct": round(statistics.mean(v), 1),
            "n_scored": len(v),
            "n_positive": sum(1 for x in v if x > 0),
            "n_negative": sum(1 for x in v if x < 0),
        }
        for name, v in model_deltas.items()
    }

    # winner shift
    a_win_deltas = [
        r.get("winner_delta") for r in per_sku
        if r.get("winner_delta") is not None
    ]
    n_improved = sum(1 for d in a_win_deltas if d > 0)
    n_regressed = sum(1 for d in a_win_deltas if d < 0)
    n_no_change = sum(1 for d in a_win_deltas if d == 0)
    if a_win_deltas:
        median_winner_delta = round(statistics.median(a_win_deltas), 1)
        mean_winner_delta = round(statistics.mean(a_win_deltas), 1)
    else:
        median_winner_delta = None
        mean_winner_delta = None
    net_winner_window = (
        CANDIDATE_WINDOW_LABEL if (median_winner_delta or 0) > 0
        else BASELINE_WINDOW_LABEL if (median_winner_delta or 0) < 0
        else "tie"
    )
    return {
        "median_delta_by_model": median_delta_by_model,
        "n_skus_scored": len(per_sku),
        "n_skus_improved": n_improved,
        "n_skus_regressed": n_regressed,
        "n_skus_no_change": n_no_change,
        "median_winner_delta_pct": median_winner_delta,
        "mean_winner_delta_pct": mean_winner_delta,
        "net_winner_window": net_winner_window,
    }


# ── Fleet orchestrator ────────────────────────────────────────────────────

async def run_history_comparison_fleet(user_id: ObjectId, job_id: str) -> dict:
    """Fleet-wide comparison. Reads 540d baseline from cache, trains
    all-history user-global bundles, per-SKU scores + deltas, aggregates."""
    async def update(**fields):
        fields["updatedAt"] = datetime.now(timezone.utc)
        await _db()["historyComparisonJobs"].update_one(
            {"_id": job_id}, {"$set": fields}, upsert=True,
        )

    await update(status="running", stage="fetching all-history sales",
                 progress=1, total_skus=0, done_skus=0)

    # 1. Fetch ALL sales — no since= filter. Big query but only done once.
    since_all = datetime(1970, 1, 1, tzinfo=timezone.utc)
    all_rows = await get_sales_daily_for_user(user_id, sku=None, since=since_all)
    rows_by_sku: dict[str, list[dict]] = {}
    for r in all_rows:
        s = r.get("sku")
        if not _is_real_sku(s):
            continue
        rows_by_sku.setdefault(s, []).append(r)
    total_skus = len(rows_by_sku)
    await update(total_skus=total_skus, stage="training user-global panels")

    # 2. Train user-global bundles on all-history data.
    today = datetime.now(timezone.utc)
    cutoff = (today - timedelta(days=HOLDOUT_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    train_series_by_sku: dict = {}
    cutoff_naive = cutoff.replace(tzinfo=None)
    for sku_key, sku_rows in rows_by_sku.items():
        train_rows = [
            r for r in sku_rows
            if isinstance(r.get("date"), datetime)
            and (r["date"].replace(tzinfo=None) if r["date"].tzinfo else r["date"]) < cutoff_naive
        ]
        s = _build_series_imputed(train_rows, cutoff)
        if not s.empty:
            train_series_by_sku[sku_key] = s

    train_end_ts = pd.Timestamp(cutoff.date()) - pd.Timedelta(days=1)
    lgbm_state, lgbm_module = await asyncio.to_thread(
        _try_lgbm_train, train_series_by_sku, train_end_ts,
    )
    xgb_state, xgb_module = await asyncio.to_thread(
        _try_xgb_train, train_series_by_sku, train_end_ts,
    )
    await update(stage="training DeepAR + TFT (can take 10+ min)")
    deepts_state = await asyncio.to_thread(
        _try_deepts_train, train_series_by_sku, train_end_ts,
    )
    await update(stage="running per-SKU scoring")

    # 3. Per-SKU: baseline (cache) + candidate (all-history) + delta.
    results_coll = _db()["historyComparisonResults"]
    computed_at = datetime.now(timezone.utc)
    per_sku_results: list[dict] = []

    # Pre-fetch every SKU's cache row in one query so we don't hit
    # Mongo N times inside the loop.
    cache_by_sku: dict[str, dict] = {}
    async for c in _forecast_cache().find({"userId": user_id}):
        cache_by_sku[c.get("sku")] = c

    for i, (sku, sku_rows) in enumerate(rows_by_sku.items(), start=1):
        try:
            baseline_accs = _baseline_accuracy_from_cache(cache_by_sku.get(sku) or {})
            candidate_accs = await _backtest_all_history_one_sku(
                user_id, sku, sku_rows,
                lgbm_state, lgbm_module, xgb_state, xgb_module, deepts_state,
            )

            per_model = {}
            for name in set(baseline_accs) | set(candidate_accs):
                a = baseline_accs.get(name)
                b = candidate_accs.get(name)
                delta = None
                if a is not None and b is not None:
                    delta = round(b - a, 1)
                per_model[name] = {
                    "baseline_accuracy_pct": a,
                    "all_history_accuracy_pct": b,
                    "delta": delta,
                }

            # Winner under each window.
            def _winner(accs: dict[str, float | None]) -> dict | None:
                scored = [(n, a) for n, a in accs.items() if a is not None]
                if not scored:
                    return None
                scored.sort(key=lambda x: -x[1])
                return {"name": scored[0][0], "accuracy_pct": scored[0][1]}

            winner_a = _winner(baseline_accs)
            winner_b = _winner(candidate_accs)
            winner_delta = None
            if winner_a and winner_b:
                winner_delta = round(winner_b["accuracy_pct"] - winner_a["accuracy_pct"], 1)

            actual_units = sum(
                int(r.get("units_ordered") or 0) for r in sku_rows
                if isinstance(r.get("date"), datetime)
                and (r["date"].replace(tzinfo=None) if r["date"].tzinfo else r["date"]) >= cutoff_naive
            )

            result = {
                "userId": user_id,
                "sku": sku,
                "computedAt": computed_at,
                "jobId": job_id,
                "holdout_start": cutoff.date().isoformat(),
                "holdout_end": (today - timedelta(days=1)).date().isoformat(),
                "actual_units_30d": actual_units,
                "low_volume": actual_units < LOW_VOLUME_ACTUAL,
                "models": per_model,
                "winner_baseline": winner_a,
                "winner_all_history": winner_b,
                "winner_delta": winner_delta,
            }
            await results_coll.update_one(
                {"userId": user_id, "sku": sku},
                {"$set": result},
                upsert=True,
            )
            per_sku_results.append(result)
        except Exception as e:
            log.exception("history-compare fleet: sku=%s failed", sku)
        await update(done_skus=i, progress=int(100 * i / max(total_skus, 1)))

    # 4. Fleet aggregate.
    summary = _summarize_deltas([
        r for r in per_sku_results if not r.get("low_volume")
    ])
    finished_at = datetime.now(timezone.utc)
    await update(
        status="done", stage="complete", progress=100,
        finishedAt=finished_at,
        elapsed_sec=round((finished_at - computed_at).total_seconds(), 1),
        fleet_summary=summary,
    )
    return {
        "job_id": job_id,
        "status": "done",
        "n_skus_scored": total_skus,
        "fleet_summary": summary,
    }


# ── Per-SKU orchestrator ──────────────────────────────────────────────────

async def run_history_comparison_one(
    user_id: ObjectId, sku: str, job_id: str,
) -> dict:
    """Per-SKU comparison. Uses Prophet + Naive only (no user-global
    retraining) plus whatever the SKU's cache has for other models.
    Fast (~10-30 sec)."""
    async def update(**fields):
        fields["updatedAt"] = datetime.now(timezone.utc)
        await _db()["historyComparisonJobs"].update_one(
            {"_id": job_id}, {"$set": fields}, upsert=True,
        )

    await update(status="running", stage=f"training on all history for {sku}",
                 total_skus=1, done_skus=0, progress=10)

    started_at = datetime.now(timezone.utc)
    try:
        cache = await _forecast_cache().find_one({"userId": user_id, "sku": sku})
        baseline_accs = _baseline_accuracy_from_cache(cache or {})
        since_all = datetime(1970, 1, 1, tzinfo=timezone.utc)
        rows = await get_sales_daily_for_user(user_id, sku=sku, since=since_all)
        candidate_accs = await _backtest_all_history_one_sku(user_id, sku, rows)

        per_model = {}
        for name in set(baseline_accs) | set(candidate_accs):
            a = baseline_accs.get(name)
            b = candidate_accs.get(name)
            delta = round(b - a, 1) if (a is not None and b is not None) else None
            per_model[name] = {
                "baseline_accuracy_pct": a,
                "all_history_accuracy_pct": b,
                "delta": delta,
            }

        def _winner(accs):
            scored = [(n, a) for n, a in accs.items() if a is not None]
            if not scored:
                return None
            scored.sort(key=lambda x: -x[1])
            return {"name": scored[0][0], "accuracy_pct": scored[0][1]}

        winner_a = _winner(baseline_accs)
        winner_b = _winner(candidate_accs)
        winner_delta = None
        if winner_a and winner_b:
            winner_delta = round(winner_b["accuracy_pct"] - winner_a["accuracy_pct"], 1)

        result = {
            "userId": user_id,
            "sku": sku,
            "computedAt": started_at,
            "jobId": job_id,
            "models": per_model,
            "winner_baseline": winner_a,
            "winner_all_history": winner_b,
            "winner_delta": winner_delta,
        }
        await _db()["historyComparisonResults"].update_one(
            {"userId": user_id, "sku": sku},
            {"$set": result},
            upsert=True,
        )
    except Exception as e:
        log.exception("history-compare per-sku failed")
        await update(
            status="failed", error=str(e), traceback=traceback.format_exc(),
            progress=0,
        )
        return {"job_id": job_id, "status": "failed", "error": str(e)}

    finished_at = datetime.now(timezone.utc)
    await update(
        status="done", stage="complete", done_skus=1, progress=100,
        finishedAt=finished_at,
        elapsed_sec=round((finished_at - started_at).total_seconds(), 1),
        winner_delta=winner_delta,
    )
    return {
        "job_id": job_id,
        "status": "done",
        "sku": sku,
        "winner_baseline": winner_a,
        "winner_all_history": winner_b,
        "winner_delta": winner_delta,
    }


# ── Job orchestration ─────────────────────────────────────────────────────

async def start_history_comparison_job(
    user_id: ObjectId, sku: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex[:16]
    scope = "sku" if sku else "fleet"
    await _db()["historyComparisonJobs"].insert_one({
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
        run_history_comparison_one(user_id, sku, job_id)
        if sku else run_history_comparison_fleet(user_id, job_id)
    )
    asyncio.create_task(coro)
    return job_id


async def get_history_comparison_job(job_id: str) -> dict | None:
    doc = await _db()["historyComparisonJobs"].find_one({"_id": job_id})
    if not doc:
        return None
    if isinstance(doc.get("userId"), ObjectId):
        doc["userId"] = str(doc["userId"])
    for k in ("createdAt", "updatedAt", "finishedAt"):
        if isinstance(doc.get(k), datetime):
            doc[k] = doc[k].isoformat()
    return doc


async def get_history_comparison_results(
    user_id: ObjectId, sku: str | None = None,
) -> list[dict]:
    query: dict = {"userId": user_id}
    if sku:
        query["sku"] = sku
    cursor = _db()["historyComparisonResults"].find(query, {"userId": 0})
    out = []
    async for d in cursor:
        if isinstance(d.get("_id"), ObjectId):
            d["_id"] = str(d["_id"])
        if isinstance(d.get("computedAt"), datetime):
            d["computedAt"] = d["computedAt"].isoformat()
        out.append(d)
    return out
