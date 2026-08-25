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
from bson import ObjectId
from database import get_sales_daily, _forecast_cache
from forecasting.model import (
    _apply_recovery_bump,
    _build_series_imputed,
    _forecast_one,
    _is_real_sku,
    _naive_forecast,
)
from forecasting.lgbm.model import forecast_sku as lgbm_forecast_sku
from forecasting.lgbm.model import train_user_global

log = logging.getLogger("forecasting.lgbm.routes")

router = APIRouter(prefix="/forecasting", tags=["forecasting-benchmark"])

# Backtest window. 540 days of history gives Prophet a full annual cycle
# for yearly seasonality and populates LGBM's lag_28/roll_56 features
# properly — 120 days (the earlier value that matched main.py's live-
# fallback backtest) meant Prophet's yearly term was unfit and LGBM
# threw out ~56 rows of usable training per SKU. Holdout is still 30
# days so accuracy_pct here is directly comparable to what the SKU
# drawer's "Prediction accuracy" card shows.
TRAIN_DAYS = 540
HOLDOUT_DAYS = 30

# Below this actual-total, accuracy_pct is dominated by rounding — a
# SKU with 5 units in a 30-day window and a 4-unit prediction reports
# 80% but tells you nothing. The FE (and the local benchmark harness)
# filters `low_volume=True` rows out of median calculations so the
# fleet-level number reflects SKUs the model was actually asked to
# forecast, not noise.
LOW_VOLUME_ACTUAL_THRESHOLD = 30


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
        "low_volume": actual_total < LOW_VOLUME_ACTUAL_THRESHOLD,
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
    always invoke it so the 3-way comparison has a baseline row.

    Uses the imputed series (stockouts filled with trailing DOW mean,
    subject to the decline guard) so the numbers match what the
    production picker sees for naive.
    """
    train_rows = [r for r in sku_rows if _row_day_naive(r) < cutoff.replace(tzinfo=None)]
    series = _build_series_imputed(train_rows, cutoff)
    result = _naive_forecast(series, horizon=HOLDOUT_DAYS, today=cutoff)
    _apply_recovery_bump(result, train_rows, cutoff)
    return _score_forecast(result, holdout_by_day)


def _lgbm_backtest(
    all_user_rows_by_sku: dict[str, list[dict]],
    target_sku: str,
    cutoff: datetime,
    holdout_by_day: dict[str, int],
) -> dict:
    """Train the user-global LightGBM on every SKU's data up to `cutoff`,
    then score its target-SKU forecast on the same holdout dict.

    Uses imputed series for the LGBM panel so the compare-endpoint
    numbers match the nightly-picker output; without imputation the
    LGBM lag features across stockout gaps go NaN and the model
    diverges from what production actually runs.
    """
    train_rows_by_sku: dict[str, list[dict]] = {}
    for sku, rows in all_user_rows_by_sku.items():
        train_rows_by_sku[sku] = [
            r for r in rows if _row_day_naive(r) < cutoff.replace(tzinfo=None)
        ]
    series_by_sku: dict[str, pd.DataFrame] = {}
    for sku, train_rows in train_rows_by_sku.items():
        s = _build_series_imputed(train_rows, cutoff)
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
    _apply_recovery_bump(result, train_rows_by_sku.get(target_sku, []), cutoff)
    return _score_forecast(result, holdout_by_day)


def _xgb_backtest(
    all_user_rows_by_sku: dict[str, list[dict]],
    target_sku: str,
    cutoff: datetime,
    holdout_by_day: dict[str, int],
) -> dict:
    """Sibling of _lgbm_backtest — trains a user-global XGBoost panel on
    the same imputed features and scores the target SKU's forecast.

    Kept in this routes module (rather than a new xgb/routes.py) so
    the compare endpoint stays in one place and doesn't need a second
    mount point.
    """
    try:
        from forecasting.xgb.model import (
            forecast_sku as xgb_forecast_sku,
            train_user_global as xgb_train,
        )
    except ImportError:
        return {"method": "xgb_missing", "days": [], "metrics": None}

    train_rows_by_sku: dict[str, list[dict]] = {}
    for sku, rows in all_user_rows_by_sku.items():
        train_rows_by_sku[sku] = [
            r for r in rows if _row_day_naive(r) < cutoff.replace(tzinfo=None)
        ]
    series_by_sku: dict[str, pd.DataFrame] = {}
    for sku, train_rows in train_rows_by_sku.items():
        s = _build_series_imputed(train_rows, cutoff)
        if not s.empty:
            series_by_sku[sku] = s
    if not series_by_sku:
        return {"method": "xgb_empty", "days": [], "metrics": None}

    train_end_ts = pd.Timestamp(cutoff.date()) - pd.Timedelta(days=1)
    fc = xgb_train(series_by_sku, train_end=train_end_ts)
    if fc is None:
        return {"method": "xgb_skipped", "days": [], "metrics": None}

    target_series = series_by_sku.get(target_sku)
    if target_series is None or target_series.empty:
        return {"method": "xgb_no_history", "days": [], "metrics": None}

    result = xgb_forecast_sku(fc, target_series, target_sku,
                              horizon=HOLDOUT_DAYS, today=cutoff)
    _apply_recovery_bump(result, train_rows_by_sku.get(target_sku, []), cutoff)
    return _score_forecast(result, holdout_by_day)


def _row_day_naive(r: dict) -> datetime | None:
    """Strip tz so date comparisons with `cutoff_naive` work regardless of
    whether Mongo returned naive or aware datetimes."""
    d = r.get("date")
    if not isinstance(d, datetime):
        return None
    return d.replace(tzinfo=None) if d.tzinfo is not None else d


def _ensemble_backtest(
    base_results: list[dict], holdout_by_day: dict[str, int],
) -> dict:
    """Blend the three base backtests: per-day mean of positive p50s.

    Missing days (a base backtest that didn't produce this date) are
    excluded rather than treated as zero — a zero prediction usually
    means the model degenerated and pulling it into the mean drags the
    ensemble down. p90 is the max across surviving bases (a conservative
    upper bound, not a mean, so coverage stays honest)."""
    by_date_p50: dict[str, list[float]] = {}
    by_date_p90: dict[str, list[float]] = {}
    all_dates: set[str] = set()
    for res in base_results:
        for d in res.get("days") or []:
            all_dates.add(d["date"])
            if d["p50"] > 0:
                by_date_p50.setdefault(d["date"], []).append(d["p50"])
                by_date_p90.setdefault(d["date"], []).append(d["p90"])
    if not all_dates:
        return {"method": "ensemble", "days": [], "metrics": None}
    ens_days = []
    for date in sorted(all_dates):
        p50s = by_date_p50.get(date, [])
        p90s = by_date_p90.get(date, [])
        p50 = sum(p50s) / len(p50s) if p50s else 0.0
        p90 = max(p90s) if p90s else 0.0
        ens_days.append({
            "date": date,
            "actual": int(holdout_by_day.get(date, 0)),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
        })
    return {
        "method": "ensemble",
        "days": ens_days,
        "metrics": _compute_metrics(ens_days),
        "drivers": None,
    }


async def _cached_bt_result(user: dict, sku: str, model_key: str) -> dict:
    """Pull a single model's cached backtest slice from forecast_cache.

    The nightly picker writes bt.all = {prophet: {...}, deepar: {...},
    ...} onto every cache row. For heavy models (deepar, tft) we prefer
    reading that slice over retraining inline — training a deep model
    per drawer open would cost the user minutes of latency for the same
    metrics the nightly job already computed against the identical
    30-day holdout.

    Returns the shape _score_forecast returns ({method, days, metrics,
    drivers}) so the caller and the FE treat it identically to a live
    backtest. Missing cache row / missing model key -> zeroed placeholder
    so the FE just doesn't render the row (the filter at index.html:5610
    drops entries with no metrics).
    """
    try:
        user_id = ObjectId(str(user.get("_id")))
    except Exception:
        return {"method": f"{model_key}_no_user", "days": [], "metrics": None}
    row = await _forecast_cache().find_one(
        {"userId": user_id, "sku": sku},
        {"backtest": 1},
    )
    if not row:
        return {"method": f"{model_key}_no_cache", "days": [], "metrics": None}
    bt = (row.get("backtest") or {}).get("all") or {}
    slice_ = bt.get(model_key)
    if not slice_ or not slice_.get("metrics"):
        return {"method": f"{model_key}_missing", "days": [], "metrics": None}
    return {
        "method": slice_.get("method") or model_key,
        "days": slice_.get("days") or [],
        "metrics": slice_.get("metrics"),
        "drivers": None,
    }


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
    try:
        xgb_result = _xgb_backtest(rows_by_sku, sku, cutoff, holdout_by_day)
    except Exception as e:
        log.exception("xgb backtest failed for sku=%s", sku)
        xgb_result = {
            "method": "xgb_error",
            "days": [],
            "metrics": None,
            "error": str(e),
        }

    # DeepAR + TFT read straight from forecast_cache.bt.all rather than
    # being trained inline — deep-model training is 1-5 min per model
    # per user, which would make every drawer open 5-10 min. The nightly
    # picker (when DEEPTS_ENABLED=1) already trained them against the
    # exact same 30-day holdout and cached the metrics in bt.all, so
    # we can surface those results in the compare panel at zero
    # additional API cost.
    deepar_result = await _cached_bt_result(user, sku, "deepar")
    tft_result = await _cached_bt_result(user, sku, "tft")

    # Ensemble candidate — per-day mean of positive p50s across every
    # base backtest. Matches the production picker's ensemble so the
    # drawer's Model comparison panel shows the same competitive row
    # the picker actually chose from. DeepAR/TFT can join the ensemble
    # too when their cached backtest days are available.
    ensemble_result = _ensemble_backtest(
        [prophet_result, naive_result, lgbm_result, xgb_result,
         deepar_result, tft_result],
        holdout_by_day,
    )

    return {
        "sku": sku,
        "train_start": since.date().isoformat(),
        "train_end": (cutoff - timedelta(days=1)).date().isoformat(),
        "holdout_start": cutoff.date().isoformat(),
        "holdout_end": (now_utc - timedelta(days=1)).date().isoformat(),
        "n_user_skus": len(rows_by_sku),
        "prophet": prophet_result,
        "lgbm": lgbm_result,
        "xgb": xgb_result,
        "deepar": deepar_result,
        "tft": tft_result,
        "naive": naive_result,
        "ensemble": ensemble_result,
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
