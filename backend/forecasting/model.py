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
    get_forecast_settings_for_user,
    get_sales_daily_for_user,
    latest_inventory_for_user,
    upsert_forecast_cache,
)
from forecasting.reorder import compute_reorder

log = logging.getLogger("forecasting.model")

MIN_HISTORY_DAYS = 60
DEFAULT_HORIZON = 90


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
    series = _build_series(rows, today)
    if len(series) < MIN_HISTORY_DAYS:
        return _naive_forecast(series, horizon, today)
    try:
        return _prophet_forecast(series, horizon, today)
    except Exception as e:
        log.warning("prophet failed (%s), falling back to naive", e)
        return _naive_forecast(series, horizon, today)


async def forecast_sku_for_user(
    user_id: ObjectId,
    sku: str,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """Build a forecast for a single SKU. Used by the agent tool and the
    nightly refresh job."""
    today = datetime.now(timezone.utc)
    since = today - timedelta(days=540)
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
    since = today - timedelta(days=540)

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
    # Pulled once; each SKU's shipments feed the stockout timeline sim.
    shipments_by_sku = await active_inbound_shipments_for_user(user_id)

    written = 0
    methods: dict[str, int] = {}
    for sku, rows in by_sku.items():
        try:
            result = await asyncio.to_thread(_forecast_one, rows, horizon, today)
        except Exception as e:
            log.exception("forecast failed for sku=%s: %s", sku, e)
            continue
        result["sku"] = sku
        result["horizon_days"] = horizon
        result["reorder"] = compute_reorder(
            result["forecast"], inv_map.get(sku), result["drivers"], settings,
            shipments=shipments_by_sku.get(sku, []),
            today=today,
        )
        await upsert_forecast_cache(user_id, sku, result)
        methods[result["method"]] = methods.get(result["method"], 0) + 1
        written += 1
    return {"skus": written, "methods": methods}
