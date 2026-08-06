"""LightGBM training + recursive multi-step prediction.

Trains one model per user across ALL of that user's SKUs (a "user-global"
model), so a SKU with 30 days of history borrows patterns from sibling
SKUs. Same-user only — no cross-tenant data pooling.

Returns forecasts in the exact dict schema of `_prophet_forecast` so
compare-endpoint metric code can treat both models identically.
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

log = logging.getLogger("forecasting.lgbm.model")


@dataclass
class LgbmForecaster:
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
) -> LgbmForecaster | None:
    """Fit p50 (tweedie) + p90 (quantile 0.9) models on all of the user's
    SKU series, truncated to rows on or before `train_end`.

    Returns None if there isn't enough data to fit — caller should fall
    back to the Prophet/naive result.
    """
    # Deferred import — LightGBM is a heavy binary wheel and lives in
    # requirements-dev.txt only. Fail loudly with a clear message if the
    # benchmark endpoint is enabled without the dep installed.
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError(
            "lightgbm not installed. Run `pip install -r requirements-dev.txt` "
            "to enable the /forecasting/sku/{sku}/compare endpoint."
        ) from e

    panel, sku_to_cat = build_panel_features(series_by_sku)
    if panel.empty:
        return None

    train = panel[panel["ds"] <= train_end].copy()
    # Drop rows where the shortest lag is still NaN — LightGBM can handle
    # NaN natively but very-early rows are near-useless as training data.
    train = train.dropna(subset=["lag_1"]).reset_index(drop=True)
    if len(train) < 50:
        log.info("lgbm skipped: only %d usable training rows", len(train))
        return None

    X = train[FEATURE_COLS]
    y = train["y"].astype("float32")

    common_params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # p50 — tweedie handles the zero-inflated, right-skewed distribution
    # of daily SKU demand better than plain L2 regression.
    p50 = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=400,
        **common_params,
    )
    p50.fit(X, y, categorical_feature=CATEGORICAL_COLS)

    # p90 — quantile regression at α=0.9. Same features, different loss.
    p90 = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.9,
        n_estimators=400,
        **common_params,
    )
    p90.fit(X, y, categorical_feature=CATEGORICAL_COLS)

    return LgbmForecaster(
        p50=p50, p90=p90, sku_to_cat=sku_to_cat,
        train_end=train_end, n_train_rows=len(train),
    )


def forecast_sku(
    fc: LgbmForecaster,
    series: pd.DataFrame,
    sku: str,
    horizon: int,
    today: datetime,
) -> dict:
    """Recursive multi-step forecast for one SKU using the fitted bundle.

    Returns the same dict schema as `_prophet_forecast` so the compare
    endpoint's metric code can process both without branching.
    """
    if series.empty:
        return {
            "method": "lgbm_empty",
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
        # SKU wasn't in the training panel — assign a fresh code so the
        # categorical branches fall through to whatever the tree learned
        # for "unseen SKU" splits.
        sku_cat = -1

    # Start from the SKU's known history; append each prediction so
    # tomorrow's lag_1 sees today's yhat.
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
        # Feed the p50 back as tomorrow's "y" so lag/rolling features
        # advance. Using the median (not p90) keeps the recursive path
        # unbiased on average.
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
        "method": "lgbm",
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
