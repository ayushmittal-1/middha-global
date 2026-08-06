"""Dev-only benchmark endpoint: /forecasting/sku/{sku}/compare.

Runs Prophet AND LightGBM on the exact same 30-day holdout for one SKU
and returns both sets of accuracy metrics side-by-side. Mounted onto the
FastAPI app only when the LGBM_BENCHMARK env flag is set, so a normal
deploy is unaffected even if this file ships.

Metric formulas mirror the production backtest at main.py:761-804 so the
numbers are directly comparable to what the FE already displays under
"Prediction accuracy".
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, Depends

from auth import protect
from database import get_sales_daily
from forecasting.model import (
    _build_series,
    _forecast_one,
    _is_real_sku,
    _naive_forecast,
)
from forecasting.lgbm.model import forecast_sku as lgbm_forecast_sku
from forecasting.lgbm.model import train_user_global

log = logging.getLogger("forecasting.lgbm.routes")

router = APIRouter(prefix="/forecasting", tags=["forecasting-benchmark"])

# Backtest window — matches main.py:713-820 exactly so the accuracy_pct
# reported here is on the same footing as what the SKU drawer shows.
TRAIN_DAYS = 120
HOLDOUT_DAYS = 30


def _compute_metrics(bt_days: list[dict]) -> dict | None:
    """Same metric formulas as main.py:761-804. Duplicated (not imported)
    so the production endpoint stays a single self-contained function."""
    n = len(bt_days)
    if n == 0:
        return None
    errors = [d["p50"] - d["actual"] for d in bt_days]
    abs_errors = [abs(e) for e in errors]
    squared = [e * e for e in errors]
    actual_total = sum(d["actual"] for d in bt_days)
    pred_total = sum(d["p50"] for d in bt_days)
    pct_errors = [
        abs(d["p50"] - d["actual"]) / d["actual"]
        for d in bt_days if d["actual"] > 0
    ]
    coverage_hits = sum(1 for d in bt_days if 0 <= d["actual"] <= d["p90"])
    volume_error = abs(pred_total - actual_total)
    accuracy_pct = (
        round(max(0.0, (1 - volume_error / actual_total) * 100), 1)
        if actual_total > 0 else None
    )
    wape = sum(abs_errors) / actual_total if actual_total > 0 else None
    accuracy_daily_pct = (
        round(max(0.0, (1 - wape) * 100), 1) if wape is not None else None
    )
    return {
        "mae": round(sum(abs_errors) / n, 2),
        "rmse": round((sum(squared) / n) ** 0.5, 2),
        "bias": round(sum(errors) / n, 2),
        "mape_pct": (
            round((sum(pct_errors) / len(pct_errors)) * 100, 1)
            if pct_errors else None
        ),
        "accuracy_pct": accuracy_pct,
        "accuracy_daily_pct": accuracy_daily_pct,
        "actual_total": actual_total,
        "predicted_total": round(pred_total, 1),
        "coverage_pct": round((coverage_hits / n) * 100, 1),
        "days_evaluated": n,
    }


def _prophet_backtest(sku_rows: list[dict], cutoff: datetime,
                      holdout_by_day: dict[str, int]) -> dict:
    """Run the existing Prophet/naive path on the training slice and
    score against the same holdout dict."""
    train_rows = [r for r in sku_rows if _row_day_naive(r) < cutoff.replace(tzinfo=None)]
    result = _forecast_one(train_rows, horizon=HOLDOUT_DAYS, today=cutoff)
    return _score_forecast(result, holdout_by_day)


def _naive_backtest(sku_rows: list[dict], cutoff: datetime,
                    holdout_by_day: dict[str, int]) -> dict:
    """Force-run the naive fallback so it can be compared directly. In
    production it only fires when history < MIN_HISTORY_DAYS — here we
    always invoke it so the 3-way comparison has a baseline row."""
    train_rows = [r for r in sku_rows if _row_day_naive(r) < cutoff.replace(tzinfo=None)]
    series = _build_series(train_rows, cutoff)
    result = _naive_forecast(series, horizon=HOLDOUT_DAYS, today=cutoff)
    return _score_forecast(result, holdout_by_day)


def _lgbm_backtest(
    all_user_rows_by_sku: dict[str, list[dict]],
    target_sku: str,
    cutoff: datetime,
    holdout_by_day: dict[str, int],
) -> dict:
    """Train the user-global LightGBM on every SKU's data up to `cutoff`,
    then score its target-SKU forecast on the same holdout dict."""
    series_by_sku: dict[str, pd.DataFrame] = {}
    for sku, rows in all_user_rows_by_sku.items():
        train_rows = [r for r in rows if _row_day_naive(r) < cutoff.replace(tzinfo=None)]
        s = _build_series(train_rows, cutoff)
        if not s.empty:
            series_by_sku[sku] = s
    if not series_by_sku:
        return {"method": "lgbm_empty", "days": [], "metrics": None}

    train_end_ts = pd.Timestamp(cutoff.date()) - pd.Timedelta(days=1)
    fc = train_user_global(series_by_sku, train_end=train_end_ts)
    if fc is None:
        return {"method": "lgbm_skipped", "days": [], "metrics": None}

    target_series = series_by_sku.get(target_sku)
    if target_series is None or target_series.empty:
        return {"method": "lgbm_no_history", "days": [], "metrics": None}

    result = lgbm_forecast_sku(fc, target_series, target_sku,
                               horizon=HOLDOUT_DAYS, today=cutoff)
    return _score_forecast(result, holdout_by_day)


def _row_day_naive(r: dict) -> datetime | None:
    """Strip tz so date comparisons with `cutoff_naive` work regardless of
    whether Mongo returned naive or aware datetimes."""
    d = r.get("date")
    if not isinstance(d, datetime):
        return None
    return d.replace(tzinfo=None) if d.tzinfo is not None else d


def _score_forecast(result: dict, holdout_by_day: dict[str, int]) -> dict:
    bt_days = []
    for r in result.get("forecast") or []:
        d_key = r["date"][:10]
        actual = int(holdout_by_day.get(d_key, 0))
        bt_days.append({
            "date": d_key,
            "actual": actual,
            "p50": round(float(r.get("p50", 0)), 2),
            "p90": round(float(r.get("p90", 0)), 2),
        })
    return {
        "method": result.get("method"),
        "days": bt_days,
        "metrics": _compute_metrics(bt_days),
        "drivers": result.get("drivers"),
    }


@router.get("/sku/{sku}/compare")
async def compare_models(sku: str, user: dict = Depends(protect)):
    """Side-by-side Prophet vs LightGBM backtest for one SKU.

    Same 30-day holdout, same training window, same metric formulas as
    the production /forecasting/sku/{sku} endpoint — the only difference
    is that LightGBM is trained on ALL of this user's SKUs together
    (user-global) while Prophet is fit only on this SKU's history.
    """
    now_utc = datetime.now(timezone.utc)
    since = now_utc - timedelta(days=TRAIN_DAYS)
    cutoff = (now_utc - timedelta(days=HOLDOUT_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    cutoff_naive = cutoff.replace(tzinfo=None)

    # Prophet path: only needs this SKU's rows.
    sku_rows = await get_sales_daily(sku=sku, since=since)

    # LightGBM path: needs the whole user catalog since the model is
    # user-global. Excludes promo/giveaway SKUs to match production.
    all_rows = await get_sales_daily(sku=None, since=since)
    rows_by_sku: dict[str, list[dict]] = {}
    for r in all_rows:
        s = r.get("sku")
        if not _is_real_sku(s):
            continue
        rows_by_sku.setdefault(s, []).append(r)

    # Holdout: sum of actual units per day for the target SKU only.
    holdout_by_day: dict[str, int] = {}
    for r in sku_rows:
        d = _row_day_naive(r)
        if d is None or d < cutoff_naive:
            continue
        key = d.date().isoformat()
        holdout_by_day[key] = holdout_by_day.get(key, 0) + int(
            r.get("units_ordered") or 0
        )

    prophet_result = _prophet_backtest(sku_rows, cutoff, holdout_by_day)
    naive_result = _naive_backtest(sku_rows, cutoff, holdout_by_day)
    try:
        lgbm_result = _lgbm_backtest(rows_by_sku, sku, cutoff, holdout_by_day)
    except Exception as e:
        log.exception("lgbm backtest failed for sku=%s", sku)
        lgbm_result = {
            "method": "lgbm_error",
            "days": [],
            "metrics": None,
            "error": str(e),
        }

    return {
        "sku": sku,
        "train_start": since.date().isoformat(),
        "train_end": (cutoff - timedelta(days=1)).date().isoformat(),
        "holdout_start": cutoff.date().isoformat(),
        "holdout_end": (now_utc - timedelta(days=1)).date().isoformat(),
        "n_user_skus": len(rows_by_sku),
        "prophet": prophet_result,
        "lgbm": lgbm_result,
        "naive": naive_result,
    }


def mount_if_enabled(app) -> None:
    """Mount the benchmark router only when LGBM_BENCHMARK env is truthy.

    Called from main.py right after `app = FastAPI(...)`. In production
    the flag is unset, the router isn't mounted, and lightgbm is never
    imported — so a missing dependency in the prod image is harmless.
    """
    if os.getenv("LGBM_BENCHMARK", "").lower() not in ("1", "true", "yes"):
        return
    try:
        app.include_router(router)
        log.info("LGBM benchmark endpoint mounted at /forecasting/sku/{sku}/compare")
    except Exception as e:
        log.warning("failed to mount LGBM benchmark router: %s", e)
