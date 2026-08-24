"""DeepAR + TFT via gluonts, wrapped in the same interface as the
LightGBM path so `_multimodel_forecast` can drop them in as candidates.

Both are user-global (one model fit across every SKU) — the deep
network learns cross-SKU embeddings so a low-history SKU borrows from
siblings, same rationale as the LGBM user-global panel.

Runtime notes:
  - Torch device auto-selected: MPS on Apple Silicon → CUDA on Linux
    with NVIDIA → CPU otherwise. Log line at fit start tells you which.
  - Default `max_epochs=8`. On MPS with 45 SKUs × 500d it's ~1-3 min
    per model. CUDA T4 is ~30-90s. CPU-only is 10-20 min — feasible
    for a nightly job but pull the epochs down (or the batch count) if
    the seller catalog grows past ~500 SKUs.
  - Predictor is pickled per call; caching the trained bundle across
    runs is the next optimization (write to `_forecast_state` collection
    or disk under `AURORA_MODEL_CACHE_DIR`).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("forecasting.deepts.model")

DEFAULT_MAX_EPOCHS = 3
DEFAULT_NUM_BATCHES_PER_EPOCH = 25
DEFAULT_PREDICTION_LENGTH = 30
DEFAULT_CONTEXT_LENGTH = 60
DEFAULT_BATCH_SIZE = 32


@dataclass
class DeepTSForecaster:
    """Trained predictor + metadata. `kind` identifies which architecture
    so the caller can label forecasts and route quantile extraction."""
    predictor: Any
    kind: str  # "deepar" | "tft"
    sku_list: list[str]
    freq: str
    prediction_length: int
    context_length: int
    n_train_series: int
    train_end: pd.Timestamp


def _pick_device() -> str:
    """Auto-select the best torch device without importing torch until
    called (module import stays cheap).

    MPS is disabled by default: gluonts + lightning + MPS deadlocks
    silently on multi-series training (sleeps forever at 0% CPU). CUDA
    when available. Opt into MPS with DEEPTS_ALLOW_MPS=1 for testing
    once the upstream issue is fixed.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if (
        os.getenv("DEEPTS_ALLOW_MPS", "").lower() in ("1", "true", "yes")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _series_to_gluonts_dataset(series_by_sku: dict[str, pd.DataFrame],
                                freq: str = "D"):
    """Convert our {sku: dataframe} dict to a gluonts ListDataset.

    Each SKU becomes one time series entry with `target` = the y column
    and `start` = the first ds. We include `item_id` = sku so the
    predictor can distinguish them at inference time and the categorical
    embedding gets a stable key per SKU.
    """
    from gluonts.dataset.common import ListDataset

    entries = []
    for sku, df in series_by_sku.items():
        if df.empty:
            continue
        # gluonts wants the target as a 1-D array; DOW/seasonality is
        # learned by the network from the `start` + freq.
        target = df["y"].to_numpy(dtype=np.float32)
        if len(target) < 2:
            continue
        entries.append({
            "start": pd.Period(df["ds"].iloc[0], freq=freq),
            "target": target,
            "item_id": sku,
        })
    if not entries:
        return None
    return ListDataset(entries, freq=freq)


def _fit_estimator(
    estimator, dataset, kind: str,
) -> Any:
    """Fit and log wall-clock. Called after picking the device so
    Lightning can put the model on it."""
    import time
    t0 = time.time()
    log.info("deepts fit start: kind=%s device=%s series=%d",
             kind, _pick_device(), len(dataset))
    predictor = estimator.train(dataset)
    log.info("deepts fit done: kind=%s elapsed=%.1fs", kind, time.time() - t0)
    return predictor


def train_deepar(
    series_by_sku: dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    prediction_length: int = DEFAULT_PREDICTION_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
) -> DeepTSForecaster | None:
    """Train Amazon's DeepAR (autoregressive RNN) on the user's panel."""
    try:
        from gluonts.torch import DeepAREstimator
    except ImportError as e:
        raise RuntimeError(
            "gluonts[torch] not installed. Run "
            "`pip install -r requirements-dev.txt` to enable DeepAR/TFT."
        ) from e

    # Truncate every series to train_end so we're never leaking future
    # holdout data — same discipline the LGBM path uses.
    truncated: dict[str, pd.DataFrame] = {}
    for sku, df in series_by_sku.items():
        cut = df[df["ds"] <= train_end]
        if len(cut) >= context_length + 1:
            truncated[sku] = cut
    if not truncated:
        log.info("deepar skipped: no series with ≥ %d rows", context_length + 1)
        return None

    dataset = _series_to_gluonts_dataset(truncated)
    if dataset is None:
        return None

    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq="D",
        num_layers=2,
        hidden_size=40,
        batch_size=DEFAULT_BATCH_SIZE,
        num_batches_per_epoch=DEFAULT_NUM_BATCHES_PER_EPOCH,
        trainer_kwargs={
            "max_epochs": max_epochs,
            "accelerator": _pick_device(),
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "callbacks": [],
        },
    )
    try:
        predictor = _fit_estimator(estimator, dataset, "deepar")
    except Exception as e:
        log.warning("deepar train failed: %s", e)
        return None

    return DeepTSForecaster(
        predictor=predictor,
        kind="deepar",
        sku_list=list(truncated.keys()),
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length,
        n_train_series=len(truncated),
        train_end=train_end,
    )


def train_tft(
    series_by_sku: dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    prediction_length: int = DEFAULT_PREDICTION_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
) -> DeepTSForecaster | None:
    """Train Temporal Fusion Transformer on the user's panel."""
    try:
        from gluonts.torch import TemporalFusionTransformerEstimator
    except ImportError as e:
        raise RuntimeError(
            "gluonts[torch] not installed. Run "
            "`pip install -r requirements-dev.txt` to enable DeepAR/TFT."
        ) from e

    truncated: dict[str, pd.DataFrame] = {}
    for sku, df in series_by_sku.items():
        cut = df[df["ds"] <= train_end]
        if len(cut) >= context_length + 1:
            truncated[sku] = cut
    if not truncated:
        log.info("tft skipped: no series with ≥ %d rows", context_length + 1)
        return None

    dataset = _series_to_gluonts_dataset(truncated)
    if dataset is None:
        return None

    estimator = TemporalFusionTransformerEstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq="D",
        hidden_dim=32,
        num_heads=4,
        batch_size=DEFAULT_BATCH_SIZE,
        num_batches_per_epoch=DEFAULT_NUM_BATCHES_PER_EPOCH,
        trainer_kwargs={
            "max_epochs": max_epochs,
            "accelerator": _pick_device(),
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "callbacks": [],
        },
    )
    try:
        predictor = _fit_estimator(estimator, dataset, "tft")
    except Exception as e:
        log.warning("tft train failed: %s", e)
        return None

    return DeepTSForecaster(
        predictor=predictor,
        kind="tft",
        sku_list=list(truncated.keys()),
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length,
        n_train_series=len(truncated),
        train_end=train_end,
    )


def forecast_sku(
    fc: DeepTSForecaster,
    series: pd.DataFrame,
    sku: str,
    horizon: int,
    today: datetime,
) -> dict:
    """Predict `horizon` days for one SKU using the fitted deep model.

    Uses gluonts's probabilistic output to extract p50 (median) and p90
    quantiles per day. Returns the exact dict shape `_prophet_forecast`
    returns so the picker's metric code doesn't need to branch.
    """
    from gluonts.dataset.common import ListDataset

    if series.empty:
        return {
            "method": f"{fc.kind}_empty",
            "forecast": [
                {"date": (pd.Timestamp(today.date()) + pd.Timedelta(days=i + 1))
                    .to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                 "p50": 0.0, "p90": 0.0}
                for i in range(horizon)
            ],
            "drivers": {"recent_avg": 0.0, "recent_std": 0.0,
                        "growth_rate": 0.0, "ad_uplift": 0.0},
        }

    # gluonts expects the input series to END at (today - 1d). Predictor
    # then rolls forward `prediction_length` days from there. If the
    # requested horizon is longer than what the predictor was trained
    # for we still get prediction_length back — caller pads.
    target = series["y"].to_numpy(dtype=np.float32)
    dataset = ListDataset([{
        "start": pd.Period(series["ds"].iloc[0], freq=fc.freq),
        "target": target,
        "item_id": sku,
    }], freq=fc.freq)

    try:
        forecasts = list(fc.predictor.predict(dataset))
    except Exception as e:
        log.warning("%s predict failed sku=%s: %s", fc.kind, sku, e)
        return {
            "method": f"{fc.kind}_error",
            "forecast": [],
            "drivers": {"error": str(e)},
        }

    if not forecasts:
        return {"method": f"{fc.kind}_empty", "forecast": [], "drivers": {}}

    fcst = forecasts[0]
    p50_series = fcst.quantile(0.5)
    p90_series = fcst.quantile(0.9)

    out: list[dict] = []
    steps = min(len(p50_series), horizon)
    for i in range(steps):
        d = pd.Timestamp(today.date()) + pd.Timedelta(days=i + 1)
        out.append({
            "date": d.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
            "p50": round(float(max(0.0, p50_series[i])), 2),
            "p90": round(float(max(0.0, p90_series[i])), 2),
        })
    # Pad if horizon > prediction_length: repeat the last predicted day.
    while len(out) < horizon:
        last = out[-1] if out else {"p50": 0.0, "p90": 0.0}
        d = pd.Timestamp(today.date()) + pd.Timedelta(days=len(out) + 1)
        out.append({
            "date": d.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
            "p50": last["p50"],
            "p90": last["p90"],
        })

    recent_avg = float(series["y"].tail(28).mean())
    recent_std = (
        float(series["y"].tail(56).std(ddof=0)) if len(series) >= 14 else 0.0
    )
    older_avg = (
        float(series["y"].iloc[-56:-28].mean()) if len(series) >= 56 else recent_avg
    )
    growth = ((recent_avg - older_avg) / older_avg) if older_avg > 0 else 0.0

    return {
        "method": fc.kind,
        "forecast": out,
        "drivers": {
            "recent_avg": round(recent_avg, 2),
            "recent_std": round(recent_std, 2),
            "growth_rate": round(growth, 3),
            "ad_uplift": 0.0,
            "n_train_series": fc.n_train_series,
        },
    }


def _bench_load_ok() -> bool:
    """Cheap import smoke test — used by /bench and CI to fail fast when
    torch/gluonts wheels are missing on the deploy target."""
    try:
        import torch  # noqa: F401
        import gluonts  # noqa: F401
        from gluonts.torch import DeepAREstimator  # noqa: F401
        from gluonts.torch import TemporalFusionTransformerEstimator  # noqa: F401
    except Exception as e:
        log.warning("deepts unavailable: %s", e)
        return False
    return True
