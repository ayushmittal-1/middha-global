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
