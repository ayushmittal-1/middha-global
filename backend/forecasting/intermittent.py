"""Croston + TSB — classical methods for intermittent-demand SKUs.

Both are lightweight exponential-smoothing methods that specifically
target the pattern our neural / tree models struggle with: sparse
sales where most days are zero and demand appears in bursts.

- **Croston (1972):** splits history into (size of nonzero orders,
  interval between nonzero orders). Both series get exponentially
  smoothed; the forecast is size/interval — i.e. average expected
  demand per period. Works well for stable intermittent demand.

- **TSB (Teunter-Syntetos-Babai, 2011):** improvement on Croston that
  ALSO updates the demand-probability during zero-demand stretches.
  This is what fixes the "obsolescence" problem — a dying SKU's
  forecast decays toward zero instead of getting stuck at the last
  observed size/interval ratio. Recommended over Croston when SKUs
  might genuinely go dormant.

- **SBA (Syntetos-Boylan Approximation, 2005):** Croston with a
  bias-correction multiplier `(1 - α/2)`. Croston's original method
  is known to over-estimate demand systematically; SBA fixes it with
  literally one scalar. Almost always ≥ Croston on real data.

- **ADIDA (Aggregate-Disaggregate Intermittent Demand Approach,
  Nikolopoulos et al. 2011):** aggregate the daily series to weekly
  (sum), run simple exponential smoothing at the weekly level, then
  disaggregate the weekly forecast uniformly back to daily. The
  aggregation step removes daily noise that makes the accuracy metric
  unstable for very low-volume SKUs (a 1-unit swing on 3 monthly
  units = 33% error). Best fit for SKUs below ~0.5/day.

Both output a flat forecast (day-of-week seasonality doesn't apply to
truly intermittent series — there's not enough signal per DOW to
estimate a multiplier). The picker will beat these methods with
Prophet/DeepAR whenever the series has real structure — they only win
when the target series is genuinely sparse enough that fancier models
overfit to phantom patterns.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

# Standard smoothing constants from the literature. Croston's original
# paper used 0.1; TSB authors recommend the same. Users don't tune these.
_ALPHA = 0.1  # smoothing for demand SIZE
_BETA = 0.1   # smoothing for demand PROBABILITY (TSB only)


def _fit_croston(y: list[float]) -> tuple[float, float]:
    """Return (size_estimate, interval_estimate) after processing y.
    y is a list of daily demand values (zeros allowed)."""
    z = None      # smoothed size of nonzero orders
    p = None      # smoothed interval between nonzero orders
    interval = 0  # days since last nonzero demand
    for val in y:
        interval += 1
        if val > 0:
            if z is None:
                z = float(val)
                p = float(interval)
            else:
                z = _ALPHA * float(val) + (1 - _ALPHA) * z
                p = _ALPHA * float(interval) + (1 - _ALPHA) * p
            interval = 0
    if z is None or p is None or p <= 0:
        return 0.0, 1.0
    return z, p


def _fit_tsb(y: list[float]) -> tuple[float, float]:
    """Return (size_estimate, prob_of_demand_estimate) after processing y."""
    z = None       # smoothed size of nonzero orders
    pi = None      # smoothed probability that any day has demand
    for val in y:
        indicator = 1.0 if val > 0 else 0.0
        if val > 0:
            if z is None:
                z = float(val)
            else:
                z = _ALPHA * float(val) + (1 - _ALPHA) * z
        # PROBABILITY updates EVERY period — this is the TSB fix that
        # lets the forecast decay when demand goes silent.
        if pi is None:
            pi = indicator
        else:
            pi = _BETA * indicator + (1 - _BETA) * pi
    if z is None:
        z = 0.0
    if pi is None:
        pi = 0.0
    return z, pi


def _forecast_flat(rate: float, horizon: int, today: datetime,
                    method_label: str) -> dict:
    """Emit `horizon` days of flat forecast at `rate` units/day.
    p90 = rate * 1.75 to reflect the burstiness of intermittent series."""
    out = []
    for i in range(horizon):
        d = pd.Timestamp(today.date()) + pd.Timedelta(days=i + 1)
        p50 = max(0.0, rate)
        out.append({
            "date": d.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
            "p50": round(p50, 2),
            "p90": round(p50 * 1.75, 2),
        })
    return {
        "method": method_label,
        "forecast": out,
        "drivers": {
            "recent_avg": round(rate, 3),
            "recent_std": 0.0,
            "growth_rate": 0.0,
            "ad_uplift": 0.0,
        },
    }


def croston_forecast(series: pd.DataFrame, horizon: int, today: datetime) -> dict:
    """Croston's method — flat forecast at size/interval rate."""
    if series is None or series.empty:
        return _forecast_flat(0.0, horizon, today, "croston")
    y = [float(v) for v in series["y"].to_numpy()]
    size, interval = _fit_croston(y)
    rate = size / interval if interval > 0 else 0.0
    return _forecast_flat(rate, horizon, today, "croston")


def tsb_forecast(series: pd.DataFrame, horizon: int, today: datetime) -> dict:
    """TSB method — flat forecast at prob_of_demand × size rate.
    Preferred over Croston when SKUs might go dormant."""
    if series is None or series.empty:
        return _forecast_flat(0.0, horizon, today, "tsb")
    y = [float(v) for v in series["y"].to_numpy()]
    size, prob = _fit_tsb(y)
    rate = prob * size
    return _forecast_flat(rate, horizon, today, "tsb")


def sba_forecast(series: pd.DataFrame, horizon: int, today: datetime) -> dict:
    """SBA (Syntetos-Boylan Approximation) — Croston with the
    bias-correction multiplier (1 - alpha/2). One-scalar fix for
    Croston's known positive bias. Almost always ≥ Croston."""
    if series is None or series.empty:
        return _forecast_flat(0.0, horizon, today, "sba")
    y = [float(v) for v in series["y"].to_numpy()]
    size, interval = _fit_croston(y)
    rate = (size / interval if interval > 0 else 0.0) * (1 - _ALPHA / 2)
    return _forecast_flat(rate, horizon, today, "sba")


def _ses(y: list[float], alpha: float = 0.3) -> float:
    """Simple exponential smoothing on a series — returns the final
    smoothed level (which is also the one-step-ahead forecast)."""
    if not y:
        return 0.0
    level = float(y[0])
    for v in y[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return level


def adida_forecast(series: pd.DataFrame, horizon: int, today: datetime) -> dict:
    """ADIDA — aggregate daily → weekly, forecast weekly via SES,
    disaggregate the weekly forecast uniformly back to daily
    (weekly_forecast / 7 per day).

    Aggregation removes daily noise that makes forecasting sub-1/day
    SKUs futile at daily resolution — a 1-unit swing on 3 monthly
    units is 33% error, but at weekly aggregation that same swing is
    a fraction of a week's demand and washes out."""
    if series is None or series.empty:
        return _forecast_flat(0.0, horizon, today, "adida")
    df = series.copy()
    if "ds" not in df.columns or "y" not in df.columns:
        return _forecast_flat(0.0, horizon, today, "adida")
    df["ds"] = pd.to_datetime(df["ds"])
    # Group by ISO week — .to_period('W') coerces every ds to its week's
    # Monday. .sum aligns; missing weeks are backfilled to 0 so the SES
    # sees the gaps (otherwise ADIDA would ignore silence — a fatal
    # obsolescence blindspot).
    weekly = df.set_index("ds")["y"].resample("W").sum().fillna(0)
    y_weekly = [float(v) for v in weekly.to_numpy()]
    if not y_weekly:
        return _forecast_flat(0.0, horizon, today, "adida")
    weekly_forecast = _ses(y_weekly)
    daily_rate = weekly_forecast / 7.0
    return _forecast_flat(daily_rate, horizon, today, "adida")
