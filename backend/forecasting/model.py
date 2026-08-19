"""
Per-SKU demand forecasting.

Strategy:

- **Prophet** for SKUs with ≥ 60 days of non-stockout history. Daily grain,
  US holidays + custom Prime Day windows. If ad_spend is non-zero on ≥ 10%
  of training days, it's added as a regressor so the model can attribute
  some of the lift to PPC.
- **Naive fallback** for sparser SKUs: 28-day trimmed mean × day-of-week
  multiplier. p90 = p50 * 1.5 as a rough upper bound.

Prophet is CPU-bound, so each fit runs in a thread executor — we can fan
out across SKUs without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from bson import ObjectId

from database import (
    _forecast_cache,
    active_inbound_shipments_for_user,
    all_product_settings_for_user,
    get_forecast_settings_for_user,
    get_sales_daily_for_user,
    latest_inventory_for_user,
    upsert_forecast_cache,
)
from forecasting.reorder import compute_reorder

log = logging.getLogger("forecasting.model")

MIN_HISTORY_DAYS = 60
DEFAULT_HORIZON = 90

# Lookback windows shown in the Actions modal's Forecast tab.
VELOCITY_WINDOWS = (3, 7, 30, 60, 180)


def apply_returns_to_daily_rows(
    rows: list[dict], returns_by_day: dict[str, int] | None,
) -> list[dict]:
    """Return a new rows list with per-day returns subtracted from
    units_ordered (floored at 0). Pure function — safe to unit-test
    without any Mongo mocking.

    `returns_by_day` is keyed by ISO date string (YYYY-MM-DD) to match
    aurora_data.returned_units_by_sku_daily's output. Rows whose date
    field is not a datetime are copied through untouched. Original
    `rows` list is not mutated (the Restock endpoint reuses it for the
    gross-velocity pass)."""
    if not returns_by_day:
        return list(rows or [])
    out: list[dict] = []
    for r in rows or []:
        d = r.get("date")
        if not isinstance(d, datetime):
            out.append(r)
            continue
        day_key = d.date().isoformat()
        returned = int(returns_by_day.get(day_key) or 0)
        if returned <= 0:
            out.append(r)
            continue
        original = int(r.get("units_ordered") or 0)
        net = max(0, original - returned)
        new_row = dict(r)
        new_row["units_ordered"] = net
        out.append(new_row)
    return out


def compute_velocity_windows(
    rows: list[dict],
    today: datetime | None = None,
    windows: tuple[int, ...] | None = None,
) -> list[dict]:
    """Per-window sales velocity table (mirrors SellerBoard's Forecast tab).

    For each lookback window (3/7/30/60/180 days ending yesterday):
      - `units_sold`   = sum of units_ordered in the window
      - `days_in_stock` = count of days where NOT stockout_corrected
      - `velocity`     = units_sold / days_in_stock (0 when no stock days)

    Out-of-stock days are ignored in the denominator so a temporary
    stockout doesn't drag the velocity number down.
    """
    if today is None:
        today = datetime.now(timezone.utc)
    end = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=1)

    normalised: list[tuple[datetime, int, bool]] = []
    for r in rows or []:
        d = r.get("date")
        if isinstance(d, datetime):
            d = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        else:
            continue
        units = int(r.get("units_ordered") or 0)
        oos = bool(r.get("stockout_corrected", False))
        normalised.append((d, units, oos))

    out = []
    for w in (windows or VELOCITY_WINDOWS):
        start = end - timedelta(days=w - 1)
        units_sold = 0
        days_in_stock = 0
        seen: set[datetime] = set()
        for d, units, oos in normalised:
            if start <= d <= end:
                units_sold += units
                if not oos:
                    seen.add(d)
        days_in_stock = len(seen)
        velocity = (units_sold / days_in_stock) if days_in_stock > 0 else 0.0
        out.append({
            "period_days": w,
            "days_in_stock": days_in_stock,
            "units_sold": units_sold,
            "velocity": round(velocity, 2),
        })
    return out


def weighted_velocity(windows: list[dict], weights: dict | None) -> float | None:
    """Weighted average across the lookback windows.

    `weights` is a `{d3, d7, d30, d60, d180}` dict (any missing key = 0).
    Returns None when no positive weights are provided — signals the caller
    to skip the blend.
    """
    if not weights:
        return None
    key_map = {3: "d3", 7: "d7", 30: "d30", 60: "d60", 180: "d180"}
    num = 0.0
    denom = 0.0
    for w in windows:
        wt = float(weights.get(key_map.get(w["period_days"], ""), 0) or 0)
        if wt <= 0:
            continue
        num += wt * float(w["velocity"])
        denom += wt
    if denom <= 0:
        return None
    return round(num / denom, 4)


def _is_real_sku(sku: str) -> bool:
    """Drop Amazon-generated promo / giveaway SKUs from the forecast.

    Amazon mints SKUs like `amzn.gr.NQ-...` for Vine reviewer copies,
    one-off giveaways, and other internal events. They aren't real
    catalog inventory the seller would ever place a PO for — including
    them just clutters the restock dashboard.
    """
    s = (sku or "").strip().lower()
    if not s:
        return False
    if s.startswith("amzn.gr."):
        return False
    return True

# Confirmed Prime Day-style events. Extend yearly. lower/upper window padding
# lets Prophet attribute a few surrounding days of lift to the event.
PRIME_EVENTS = pd.DataFrame([
    # Summer Prime Day
    {"holiday": "prime_day", "ds": "2024-07-16", "lower_window": -1, "upper_window": 1},
    {"holiday": "prime_day", "ds": "2025-07-08", "lower_window": -1, "upper_window": 3},
    {"holiday": "prime_day", "ds": "2026-07-14", "lower_window": -1, "upper_window": 1},
    # Fall "Prime Big Deal Days"
    {"holiday": "prime_fall", "ds": "2024-10-08", "lower_window": 0, "upper_window": 1},
    {"holiday": "prime_fall", "ds": "2025-10-07", "lower_window": 0, "upper_window": 1},
])
PRIME_EVENTS["ds"] = pd.to_datetime(PRIME_EVENTS["ds"])


# ── Series prep ────────────────────────────────────────────────────────────


def _build_series(rows: list[dict], today: datetime) -> pd.DataFrame:
    """Build a dense daily series from sparse salesDaily rows.

    - Rows are left-joined onto a daily index from first observed date to
      yesterday; missing days are filled with 0 units (legitimate no-sale).
    - Rows flagged stockout_corrected are dropped — Prophet sees them as
      simply missing.
    """
    if not rows:
        return pd.DataFrame(columns=["ds", "y", "ad_spend"])
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    # Traffic-only and ads-only rows can land in salesDaily without
    # units_ordered (the upsert merges fields). Treat the missing column
    # as zero so those rows densify cleanly without crashing the fit.
    if "units_ordered" not in df.columns:
        df["units_ordered"] = 0
    if "ad_spend" not in df.columns:
        df["ad_spend"] = 0
    if "stockout_corrected" not in df.columns:
        df["stockout_corrected"] = False
    df["y"] = df["units_ordered"].fillna(0).astype(float)
    df["ad_spend"] = df["ad_spend"].fillna(0).astype(float)
    df = df[["ds", "y", "ad_spend", "stockout_corrected"]]

    start = df["ds"].min()
    end = pd.Timestamp(today.date()) - pd.Timedelta(days=1)
    if pd.isna(start) or start > end:
        return pd.DataFrame(columns=["ds", "y", "ad_spend"])

    full_idx = pd.DataFrame({"ds": pd.date_range(start, end, freq="D")})
    merged = full_idx.merge(df, on="ds", how="left")
    merged["y"] = merged["y"].fillna(0.0)
    merged["ad_spend"] = merged["ad_spend"].fillna(0.0)
    merged["stockout_corrected"] = merged["stockout_corrected"].fillna(False).astype(bool)
    merged = merged[~merged["stockout_corrected"]].copy()
    return merged[["ds", "y", "ad_spend"]]


# ── Naive fallback ─────────────────────────────────────────────────────────


def _naive_forecast(series: pd.DataFrame, horizon: int, today: datetime) -> dict:
    if series.empty:
        # Brand new SKU with no usable history — return zeros so the
        # restock dashboard can still render a row.
        return {
            "method": "empty",
            "forecast": [
                {"date": (today + timedelta(days=i + 1)).isoformat(),
                 "p50": 0.0, "p90": 0.0}
                for i in range(horizon)
            ],
            "drivers": {"recent_avg": 0.0, "recent_std": 0.0,
                        "growth_rate": 0.0, "ad_uplift": 0.0},
        }

    recent = series.tail(28)["y"].to_numpy()
    if len(recent) >= 5:
        lo, hi = np.percentile(recent, [10, 90])
        trimmed = recent[(recent >= lo) & (recent <= hi)]
        base = float(trimmed.mean()) if trimmed.size else float(recent.mean())
    else:
        base = float(recent.mean()) if recent.size else 0.0

    # Day-of-week multiplier from the last 8 weeks.
    last8 = series.tail(56).copy()
    overall = last8["y"].mean() if not last8.empty else base
    dow_mult: dict[int, float] = {}
    if overall and not last8.empty:
        for dow, group in last8.groupby(last8["ds"].dt.dayofweek):
            dow_mult[int(dow)] = float(group["y"].mean() / overall) if overall else 1.0

    out = []
    for i in range(horizon):
        d = pd.Timestamp(today.date()) + pd.Timedelta(days=i + 1)
        mult = dow_mult.get(int(d.dayofweek), 1.0)
        p50 = max(0.0, base * mult)
        out.append({"date": d.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                    "p50": round(p50, 2),
                    "p90": round(p50 * 1.5, 2)})
    recent_std = float(np.std(series["y"].tail(56).to_numpy(), ddof=0)) if len(series) >= 14 else 0.0
    return {
        "method": "naive",
        "forecast": out,
        "drivers": {
            "recent_avg": round(base, 2),
            "recent_std": round(recent_std, 2),
            "growth_rate": 0.0,
            "ad_uplift": 0.0,
        },
    }


# ── Prophet ────────────────────────────────────────────────────────────────


def _prophet_forecast(series: pd.DataFrame, horizon: int, today: datetime) -> dict:
    # Heavy import — keep it inside the function so the module is cheap to
    # import even when forecasting isn't being used in this process.
    from prophet import Prophet

    use_ad_regressor = (series["ad_spend"] > 0).mean() >= 0.10

    model = Prophet(
        interval_width=0.80,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        holidays=PRIME_EVENTS,
        # Default 0.05 is too conservative for this catalog — Prophet
        # regresses to the historical mean and misses recent acceleration
        # in demand. 0.5 lets the trend chase the last 30-60 days harder
        # so July's uptick is actually captured.
        changepoint_prior_scale=0.5,
    )
    model.add_country_holidays(country_name="US")
    if use_ad_regressor:
        model.add_regressor("ad_spend")

    fit_df = series.rename(columns={"ds": "ds", "y": "y"})
    model.fit(fit_df)

    future = model.make_future_dataframe(periods=horizon, freq="D",
                                         include_history=False)
    if use_ad_regressor:
        # Hold future ad_spend at the trailing 14-day average — caller can
        # re-forecast with a scenario value later.
        future["ad_spend"] = float(series["ad_spend"].tail(14).mean())
    fcst = model.predict(future)

    recent_avg = float(series["y"].tail(28).mean())
    recent_std = float(series["y"].tail(56).std(ddof=0)) if len(series) >= 14 else 0.0
    older_avg = float(series["y"].iloc[-56:-28].mean()) if len(series) >= 56 else recent_avg
    growth = ((recent_avg - older_avg) / older_avg) if older_avg > 0 else 0.0

    out = []
    for _, r in fcst.iterrows():
        d = r["ds"].to_pydatetime().replace(tzinfo=timezone.utc)
        out.append({
            "date": d.isoformat(),
            "p50": round(max(0.0, float(r["yhat"])), 2),
            "p90": round(max(0.0, float(r["yhat_upper"])), 2),
        })

    return {
        "method": "prophet" + ("+ads" if use_ad_regressor else ""),
        "forecast": out,
        "drivers": {
            "recent_avg": round(recent_avg, 2),
            "recent_std": round(recent_std, 2),
            "growth_rate": round(growth, 3),
            "ad_uplift": round(
                float(model.params.get("beta", [[0]])[0][-1]) if use_ad_regressor else 0.0, 4
            ),
        },
    }


# ── Public API ─────────────────────────────────────────────────────────────


def _forecast_one(rows: list[dict], horizon: int, today: datetime) -> dict:
    """Single-model picker: Prophet for SKUs with enough history, naive
    otherwise. Kept as the primitive that the dev-only /compare endpoint
    treats as the "Prophet path" — the multi-model picker below wraps
    this plus LGBM plus naive with backtest-based selection."""
    series = _build_series(rows, today)
    if len(series) < MIN_HISTORY_DAYS:
        return _naive_forecast(series, horizon, today)
    try:
        return _prophet_forecast(series, horizon, today)
    except Exception as e:
        log.warning("prophet failed (%s), falling back to naive", e)
        return _naive_forecast(series, horizon, today)


# ── Multi-model backtest picker ───────────────────────────────────────────
#
# The cache job runs a 30-day holdout backtest across Prophet + LightGBM +
# Naive, picks the highest-accuracy_pct winner per SKU, and refits it on
# full history. That winner's forecast is what lands in `forecast_cache`
# and drives the restock tab's 30-day forecast, Suggested qty, and (for
# brand-new SKUs with no live velocity) the stockout / ship-by dates.
#
# Guardrail: if the winner still scores accuracy_pct < 10 (e.g. a
# pathological Prophet fit with a runaway trend), force naive so the
# recommendation stays sane. Pathological forecasts inflate Suggested qty
# by orders of magnitude — a bad naive is dramatically less harmful than
# a bad Prophet.

BACKTEST_HOLDOUT_DAYS = 30
BACKTEST_ACCURACY_FLOOR_PCT = 10.0


def _row_day_naive(r: dict) -> datetime | None:
    d = r.get("date")
    if not isinstance(d, datetime):
        return None
    return d.replace(tzinfo=None) if d.tzinfo is not None else d


def _split_train_holdout(
    rows: list[dict], cutoff: datetime,
) -> tuple[list[dict], dict[str, int]]:
    """Rows before cutoff → train list. Rows on/after cutoff → holdout dict
    keyed by ISO date, values summed across duplicates."""
    cutoff_naive = cutoff.replace(tzinfo=None)
    train: list[dict] = []
    holdout: dict[str, int] = {}
    for r in rows or []:
        d = _row_day_naive(r)
        if d is None:
            continue
        if d < cutoff_naive:
            train.append(r)
        else:
            key = d.date().isoformat()
            holdout[key] = holdout.get(key, 0) + int(r.get("units_ordered") or 0)
    return train, holdout


def _score_forecast_days(
    forecast: list[dict], holdout_by_day: dict[str, int],
) -> tuple[list[dict], dict | None]:
    """Convert a forecast list + holdout dict into (days, metrics). Metric
    formulas match main.py:1043-1082 and lgbm/routes.py:43-82 exactly so
    accuracy_pct is comparable across the drawer, restock, and compare
    endpoint."""
    bt_days: list[dict] = []
    for r in forecast or []:
        d_key = r["date"][:10]
        bt_days.append({
            "date": d_key,
            "actual": int(holdout_by_day.get(d_key, 0)),
            "p50": round(float(r.get("p50", 0)), 2),
            "p90": round(float(r.get("p90", 0)), 2),
        })
    n = len(bt_days)
    if n == 0:
        return bt_days, None
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
    metrics = {
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
    return bt_days, metrics


def _try_lgbm_train(
    series_by_sku: dict, train_end: object,
):
    """Wrap LGBM training so a missing dep / small-catalog user doesn't
    crash the whole cache job. Returns (forecaster_or_None, lgbm_module_or_None).
    The module is returned so the caller can `lgbm_module.forecast_sku(...)`
    without a second import."""
    try:
        from forecasting.lgbm import model as lgbm_module
    except ImportError:
        log.info("lightgbm not installed — multi-model picker will use "
                 "Prophet + naive only")
        return None, None
    try:
        forecaster = lgbm_module.train_user_global(series_by_sku, train_end=train_end)
    except Exception as e:
        # RuntimeError from lgbm_module when lightgbm binary is missing,
        # or any downstream fit issue. Fall back to Prophet+Naive picker.
        log.warning("lgbm train_user_global failed (%s) — skipping LGBM "
                    "in this picker run", e)
        return None, None
    return forecaster, lgbm_module


def _pick_winner(candidates: dict[str, dict]) -> str | None:
    """Return the key of the candidate with the highest metrics.accuracy_pct.
    Ties broken by lower MAE. Returns None when no candidate scored."""
    scored = []
    for key, cand in candidates.items():
        m = cand.get("backtest_metrics") or {}
        acc = m.get("accuracy_pct")
        if acc is None:
            continue
        scored.append((acc, -(m.get("mae") or 0), key))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def _multimodel_forecast(
    sku: str,
    rows: list[dict],
    all_train_series_by_sku: dict,
    lgbm_state,
    lgbm_module,
    horizon: int,
    today: datetime,
    cutoff: datetime,
) -> dict:
    """Bake off Prophet + LGBM + Naive on the same 30-day holdout, pick
    the winner, refit it on FULL history, return the winner's forecast +
    the full backtest table so the drawer can show all three side-by-side
    without a live re-fit."""
    train_rows, holdout_by_day = _split_train_holdout(rows, cutoff)
    train_series = _build_series(train_rows, cutoff)
    full_series = _build_series(rows, today)

    candidates: dict[str, dict] = {}

    # ── Prophet backtest ──────────────────────────────────────
    prophet_bt_days: list[dict] = []
    prophet_metrics: dict | None = None
    if len(train_series) >= MIN_HISTORY_DAYS:
        try:
            proph_bt_result = _prophet_forecast(
                train_series, horizon=BACKTEST_HOLDOUT_DAYS, today=cutoff,
            )
            prophet_bt_days, prophet_metrics = _score_forecast_days(
                proph_bt_result.get("forecast") or [], holdout_by_day,
            )
            candidates["prophet"] = {
                "method_label": proph_bt_result.get("method") or "prophet",
                "backtest_days": prophet_bt_days,
                "backtest_metrics": prophet_metrics,
            }
        except Exception as e:
            log.warning("prophet backtest failed for sku=%s: %s", sku, e)

    # ── Naive backtest (always runnable, useful floor) ────────
    try:
        naive_bt_result = _naive_forecast(
            train_series, horizon=BACKTEST_HOLDOUT_DAYS, today=cutoff,
        )
        naive_bt_days, naive_metrics = _score_forecast_days(
            naive_bt_result.get("forecast") or [], holdout_by_day,
        )
        candidates["naive"] = {
            "method_label": "naive",
            "backtest_days": naive_bt_days,
            "backtest_metrics": naive_metrics,
        }
    except Exception as e:
        log.warning("naive backtest failed for sku=%s: %s", sku, e)

    # ── LGBM backtest (only when the user-global fit succeeded) ──
    if lgbm_state is not None and lgbm_module is not None:
        target_series = all_train_series_by_sku.get(sku)
        if target_series is not None and not target_series.empty:
            try:
                lgbm_bt_result = lgbm_module.forecast_sku(
                    lgbm_state, target_series, sku,
                    horizon=BACKTEST_HOLDOUT_DAYS, today=cutoff,
                )
                lgbm_bt_days, lgbm_metrics = _score_forecast_days(
                    lgbm_bt_result.get("forecast") or [], holdout_by_day,
                )
                candidates["lgbm"] = {
                    "method_label": "lgbm",
                    "backtest_days": lgbm_bt_days,
                    "backtest_metrics": lgbm_metrics,
                }
            except Exception as e:
                log.warning("lgbm backtest failed for sku=%s: %s", sku, e)

    # ── Pick winner and refit on full history ─────────────────
    winner = _pick_winner(candidates)
    winner_metrics = (candidates.get(winner) or {}).get("backtest_metrics") if winner else None
    winner_acc = (winner_metrics or {}).get("accuracy_pct")
    forced_fallback_reason: str | None = None
    # Guardrail: pathological winner (accuracy < 10%) → prefer naive so the
    # reorder math doesn't chase a broken forecast. Only applies when naive
    # is a viable candidate (i.e. produced its own backtest).
    if winner and winner != "naive" and winner_acc is not None and winner_acc < BACKTEST_ACCURACY_FLOOR_PCT:
        if "naive" in candidates:
            forced_fallback_reason = (
                f"forced_naive: {winner} accuracy_pct={winner_acc} "
                f"< floor {BACKTEST_ACCURACY_FLOOR_PCT}"
            )
            winner = "naive"

    # Refit winner on FULL history for the actual production forecast.
    if winner == "prophet" and len(full_series) >= MIN_HISTORY_DAYS:
        try:
            fwd = _prophet_forecast(full_series, horizon, today)
        except Exception as e:
            log.warning("prophet full-refit failed for sku=%s: %s — "
                        "falling back to naive", sku, e)
            winner = "naive"
            fwd = _naive_forecast(full_series, horizon, today)
    elif winner == "lgbm" and lgbm_state is not None and lgbm_module is not None:
        # LGBM was trained on data up to cutoff (t-30d). For the forward
        # forecast we ideally want a model trained on data up to today.
        # Pragmatically: re-use the same LGBM state to keep the picker
        # cheap. The 30-day gap in LGBM training data usually costs
        # ~1-2 accuracy points vs a fresh fit; acceptable trade-off for
        # avoiding a second N-SKU training pass per user.
        target_series = _build_series(rows, today)
        try:
            fwd = lgbm_module.forecast_sku(
                lgbm_state, target_series, sku, horizon, today,
            )
        except Exception as e:
            log.warning("lgbm forward-forecast failed for sku=%s: %s — "
                        "falling back to naive", sku, e)
            winner = "naive"
            fwd = _naive_forecast(full_series, horizon, today)
    else:
        winner = "naive"
        fwd = _naive_forecast(full_series, horizon, today)

    # Attach picker metadata so the drawer can render "★ best of N" and
    # the FE tooltip explains WHY this model was chosen.
    fwd["method"] = fwd.get("method") or winner
    fwd["picker"] = {
        "winner": winner,
        "candidates": {
            k: (v.get("backtest_metrics") or {}).get("accuracy_pct")
            for k, v in candidates.items()
        },
        "forced_fallback_reason": forced_fallback_reason,
    }
    # Full backtest table for FE. Drawer reads bt.method + bt.metrics for
    # the headline card; the compare panel (dev-only, gated behind
    # LGBM_BENCHMARK) reads bt.all for the side-by-side.
    winner_days = (candidates.get(winner) or {}).get("backtest_days") or []
    fwd["backtest"] = {
        "method": winner,
        "train_start": (
            train_rows[0].get("date").date().isoformat()
            if train_rows and isinstance(train_rows[0].get("date"), datetime)
            else None
        ),
        "train_end": (cutoff - timedelta(days=1)).date().isoformat(),
        "holdout_start": cutoff.date().isoformat(),
        "holdout_end": (today - timedelta(days=1)).date().isoformat(),
        "days": winner_days,
        "metrics": (candidates.get(winner) or {}).get("backtest_metrics"),
        "all": {
            k: {
                "method": v.get("method_label"),
                "metrics": v.get("backtest_metrics"),
                "days": v.get("backtest_days"),
            }
            for k, v in candidates.items()
        },
    }
    return fwd


async def forecast_sku_for_user(
    user_id: ObjectId,
    sku: str,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """Build a forecast for a single SKU. Used by the agent tool and the
    nightly refresh job."""
    today = datetime.now(timezone.utc)
    # Trailing 90 days — clears MIN_HISTORY_DAYS=60 so Prophet actually
    # runs (13 weekly cycles is enough for weekly seasonality), but short
    # enough that old sluggish months don't drag the trend fit down when
    # recent demand has accelerated. SKUs with <60 days fall through to
    # `_naive_forecast`.
    since = today - timedelta(days=90)
    rows = await get_sales_daily_for_user(user_id, sku=sku, since=since)
    result = await asyncio.to_thread(_forecast_one, rows, horizon, today)
    result["sku"] = sku
    result["horizon_days"] = horizon
    return result


async def refresh_forecasts_for_user(
    user_id: ObjectId,
    skus: Iterable[str] | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """Refresh the cache for every SKU with history (or just `skus` if
    provided). Called by the nightly job after ingest. Also computes the
    reorder fields so the dashboard can render from a single read."""
    today = datetime.now(timezone.utc)
    # Trailing 90 days — clears MIN_HISTORY_DAYS=60 so Prophet actually
    # runs (13 weekly cycles is enough for weekly seasonality), but short
    # enough that old sluggish months don't drag the trend fit down when
    # recent demand has accelerated. SKUs with <60 days fall through to
    # `_naive_forecast`.
    since = today - timedelta(days=90)

    if skus is None:
        # Pull the full history once, then group by SKU. Cheaper than 100s
        # of round-trips for a 100-SKU catalog.
        all_rows = await get_sales_daily_for_user(user_id, sku=None, since=since)
        by_sku: dict[str, list[dict]] = {}
        for r in all_rows:
            if not _is_real_sku(r["sku"]):
                continue
            by_sku.setdefault(r["sku"], []).append(r)
        # Drop any cache entries for SKUs we've now decided to exclude
        # (e.g. amzn.gr.* from earlier runs) so they vanish from the UI.
        await _forecast_cache().delete_many({
            "userId": user_id,
            "sku": {"$regex": "^amzn\\.gr\\."},
        })
    else:
        by_sku = {}
        for sku in skus:
            by_sku[sku] = await get_sales_daily_for_user(user_id, sku=sku, since=since)

    inv_map = await latest_inventory_for_user(user_id)
    settings = await get_forecast_settings_for_user(user_id)

    # ── Pre-train LGBM once for the whole user ────────────────────────
    # The picker needs each candidate scored against the same 30-day
    # holdout. LGBM is user-global (trained on every SKU's data at once),
    # so training it up-front is dramatically cheaper than fitting per-
    # SKU inside the loop. Trained on data BEFORE cutoff so its backtest
    # is a fair holdout evaluation, not a look-ahead.
    cutoff = (today - timedelta(days=BACKTEST_HOLDOUT_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    cutoff_naive = cutoff.replace(tzinfo=None)
    train_series_by_sku: dict = {}
    for sku_key, sku_rows in by_sku.items():
        train_rows_for_lgbm = [
            r for r in sku_rows
            if (d := _row_day_naive(r)) is not None and d < cutoff_naive
        ]
        s = _build_series(train_rows_for_lgbm, cutoff)
        if not s.empty:
            train_series_by_sku[sku_key] = s
    lgbm_state, lgbm_module = await asyncio.to_thread(
        _try_lgbm_train, train_series_by_sku,
        pd.Timestamp(cutoff.date()) - pd.Timedelta(days=1),
    )
    # Pulled once; each SKU's shipments feed the stockout timeline sim.
    shipments_by_sku = await active_inbound_shipments_for_user(user_id)
    # Per-SKU overrides from the Actions modal (lead-time, target cover,
    # velocity weights). Empty dict if the user hasn't opened the modal yet.
    product_settings_by_sku = await all_product_settings_for_user(user_id)

    written = 0
    methods: dict[str, int] = {}
    for sku, rows in by_sku.items():
        try:
            result = await asyncio.to_thread(
                _multimodel_forecast,
                sku, rows, train_series_by_sku,
                lgbm_state, lgbm_module,
                horizon, today, cutoff,
            )
        except Exception as e:
            log.exception("forecast failed for sku=%s: %s", sku, e)
            continue

        ps = product_settings_by_sku.get(sku) or {}

        # ── Weighted-velocity blend ─────────────────────────────────
        # If the seller has assigned weights across the 3/7/30/60/180
        # lookback windows, rescale Prophet's p50/p90 so the *level*
        # matches their opinion while preserving seasonality/DOW shape.
        windows = compute_velocity_windows(rows, today)
        w_vel = weighted_velocity(windows, ps.get("velocity_weights"))
        blend_scale: float | None = None
        if w_vel is not None:
            model_recent = float((result.get("drivers") or {}).get("recent_avg") or 0.0)
            if model_recent > 0:
                blend_scale = w_vel / model_recent
                for r in result.get("forecast") or []:
                    r["p50"] = round(float(r.get("p50", 0)) * blend_scale, 2)
                    r["p90"] = round(float(r.get("p90", 0)) * blend_scale, 2)
            elif w_vel > 0:
                # Model saw no recent demand but the user says X/day —
                # emit a flat forecast at that level so the reorder math
                # has something to work with.
                for r in result.get("forecast") or []:
                    r["p50"] = round(w_vel, 2)
                    r["p90"] = round(w_vel * 1.5, 2)

        result["sku"] = sku
        result["asin"] = (inv_map.get(sku) or {}).get("asin")
        result["horizon_days"] = horizon
        result["velocity_windows"] = windows
        result["weighted_velocity"] = w_vel
        if blend_scale is not None:
            result.setdefault("drivers", {})["blend_scale"] = round(blend_scale, 3)
        result["reorder"] = compute_reorder(
            result["forecast"], inv_map.get(sku), result["drivers"], settings,
            shipments=shipments_by_sku.get(sku, []),
            today=today,
            product_settings=ps,
        )
        await upsert_forecast_cache(user_id, sku, result)
        methods[result["method"]] = methods.get(result["method"], 0) + 1
        written += 1
    return {"skus": written, "methods": methods}
