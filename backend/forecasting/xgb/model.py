"""XGBoost user-global forecaster.

Mirrors forecasting.lgbm.model but uses xgboost instead of lightgbm.
Shares the feature engineering in forecasting.lgbm.features so both
tree-based models see the same lag/roll/DOW/categorical features and
their comparison in the picker is apples-to-apples.

Two objectives per model bundle:
  p50 — tweedie regression (matches LGBM's zero-inflated demand loss)
  p90 — quantile regression at α=0.9 for the upper interval

Unlike LightGBM's native categorical handling, XGBoost 3.x's
`enable_categorical` path crashes on single-row inference with our
feature shape (recode segfault in libxgboost). We keep `sku_cat` as a
plain integer feature instead — XGBoost splits it numerically, which
is slightly less powerful than true categorical splits but reliable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from forecasting.lgbm.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    build_inference_row,
    build_panel_features,
)

log = logging.getLogger("forecasting.xgb.model")


@dataclass
class XgbForecaster:
    """Trained bundle: p50 + p90 model + the SKU→categorical map so
    inference can rebuild the same features."""
    p50: Any
    p90: Any
    sku_to_cat: dict[str, int]
    train_end: pd.Timestamp
    n_train_rows: int


def train_user_global(
    series_by_sku: dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
) -> XgbForecaster | None:
    """Fit p50 (tweedie) + p90 (quantile 0.9) XGBoost models across
    every SKU's series, truncated to rows on or before `train_end`.

    Returns None on missing dependency, empty panel, or fewer than 50
    usable training rows — caller falls back to the LGBM/Prophet/naive
    results.
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError(
            "xgboost not installed. Run `pip install xgboost` or add it "
            "to requirements-dev.txt to enable the XGBoost picker candidate."
        ) from e

    panel, sku_to_cat = build_panel_features(series_by_sku)
    if panel.empty:
        return None

    train = panel[panel["ds"] <= train_end].copy()
    train = train.dropna(subset=["lag_1"]).reset_index(drop=True)
    if len(train) < 50:
        log.info("xgb skipped: only %d usable training rows", len(train))
        return None

    X = train[FEATURE_COLS]
    y = train["y"].astype("float32")

    common_params = dict(
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=400,
        tree_method="hist",
        verbosity=0,
    )

    # p50 — tweedie for the same zero-inflated demand shape LGBM handles.
    p50 = xgb.XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        **common_params,
    )
    p50.fit(X, y)

    # p90 — pinball loss at α=0.9. Available on xgboost ≥ 1.7.
    p90 = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.9,
        **common_params,
    )
    p90.fit(X, y)

    return XgbForecaster(
        p50=p50, p90=p90, sku_to_cat=sku_to_cat,
        train_end=train_end, n_train_rows=len(train),
    )


def forecast_sku(
    fc: XgbForecaster,
    series: pd.DataFrame,
    sku: str,
    horizon: int,
    today: datetime,
) -> dict:
    """Recursive multi-step forecast for one SKU using the fitted bundle.

    Same return schema as `_prophet_forecast` / lgbm.forecast_sku so
    the picker's metric code treats every model identically.
    """
    if series.empty:
        return {
            "method": "xgb_empty",
            "forecast": [
                {"date": (pd.Timestamp(today.date()) + pd.Timedelta(days=i + 1))
                    .to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                 "p50": 0.0, "p90": 0.0}
                for i in range(horizon)
            ],
            "drivers": {"recent_avg": 0.0, "recent_std": 0.0,
                        "growth_rate": 0.0, "ad_uplift": 0.0},
        }

    sku_cat = fc.sku_to_cat.get(sku)
    if sku_cat is None:
        sku_cat = -1

    history = series[["ds", "y"]].copy().reset_index(drop=True)
    history["ds"] = pd.to_datetime(history["ds"])
    ad_hint = float(series["ad_spend"].tail(14).mean()) if "ad_spend" in series.columns else 0.0

    out = []
    for i in range(horizon):
        target_date = pd.Timestamp(today.date()) + pd.Timedelta(days=i + 1)
        row = build_inference_row(history, target_date, sku_cat, ad_hint)
        X_row = row[FEATURE_COLS]
        p50 = float(max(0.0, fc.p50.predict(X_row)[0]))
        p90 = float(max(p50, fc.p90.predict(X_row)[0]))
        out.append({
            "date": target_date.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
        })
        history = pd.concat([
            history,
            pd.DataFrame([{"ds": target_date, "y": p50}]),
        ], ignore_index=True)

    recent_avg = float(series["y"].tail(28).mean())
    recent_std = (
        float(series["y"].tail(56).std(ddof=0)) if len(series) >= 14 else 0.0
    )
    older_avg = (
        float(series["y"].iloc[-56:-28].mean()) if len(series) >= 56 else recent_avg
    )
    growth = ((recent_avg - older_avg) / older_avg) if older_avg > 0 else 0.0

    return {
        "method": "xgb",
        "forecast": out,
        "drivers": {
            "recent_avg": round(recent_avg, 2),
            "recent_std": round(recent_std, 2),
            "growth_rate": round(growth, 3),
            "ad_uplift": 0.0,
            "n_train_rows": fc.n_train_rows,
            "n_train_skus": len(fc.sku_to_cat),
        },
    }
