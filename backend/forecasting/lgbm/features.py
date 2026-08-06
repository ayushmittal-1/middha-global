"""Feature engineering for the LightGBM benchmark.

Consumes the same (ds, y, ad_spend) frame that `_build_series` in
forecasting.model produces, and emits a feature matrix indexed by
(sku, ds). Deliberately mirrors the standard M5-competition recipe:
short + medium lags, rolling stats, calendar/holiday flags, categoricals.
"""

from __future__ import annotations

import pandas as pd

# Same Prime Day windows used by Prophet — kept in sync manually so the
# two models see the exact same event definition.
PRIME_DAYS = {
    pd.Timestamp("2024-07-16"),
    pd.Timestamp("2025-07-08"),
    pd.Timestamp("2026-07-14"),
}
PRIME_FALL_DAYS = {
    pd.Timestamp("2024-10-08"),
    pd.Timestamp("2025-10-07"),
}

# US federal holidays that materially move retail demand. Hardcoded so we
# don't pull in the `holidays` package just for the benchmark.
US_HOLIDAYS = {
    pd.Timestamp(d) for d in [
        # 2024
        "2024-01-01", "2024-01-15", "2024-05-27", "2024-07-04",
        "2024-09-02", "2024-11-11", "2024-11-28", "2024-12-25",
        # 2025
        "2025-01-01", "2025-01-20", "2025-05-26", "2025-07-04",
        "2025-09-01", "2025-11-11", "2025-11-27", "2025-12-25",
        # 2026
        "2026-01-01", "2026-01-19", "2026-05-25", "2026-07-04",
        "2026-09-07", "2026-11-11", "2026-11-26", "2026-12-25",
    ]
}

LAG_DAYS = (1, 2, 3, 7, 14, 28)
ROLL_WINDOWS = (7, 14, 28)

FEATURE_COLS = [
    *(f"lag_{d}" for d in LAG_DAYS),
    *(f"roll_mean_{w}" for w in ROLL_WINDOWS),
    *(f"roll_std_{w}" for w in ROLL_WINDOWS),
    "dow", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_us_holiday", "is_prime_day", "is_prime_fall",
    "days_to_prime_day",
    "ad_spend",
    "sku_cat",
]

CATEGORICAL_COLS = ["sku_cat"]


def _calendar_features(ds: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=ds.index)
    out["dow"] = ds.dt.dayofweek.astype("int16")
    out["day_of_month"] = ds.dt.day.astype("int16")
    out["month"] = ds.dt.month.astype("int16")
    out["week_of_year"] = ds.dt.isocalendar().week.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_us_holiday"] = ds.isin(US_HOLIDAYS).astype("int8")
    out["is_prime_day"] = ds.isin(PRIME_DAYS).astype("int8")
    out["is_prime_fall"] = ds.isin(PRIME_FALL_DAYS).astype("int8")
    # Continuous "closeness to next Prime Day" — lets the model learn a
    # ramp-up curve instead of a single-day spike.
    all_prime = sorted(PRIME_DAYS)
    def _days_to_next(d):
        for p in all_prime:
            if p >= d:
                return min((p - d).days, 365)
        return 365
    out["days_to_prime_day"] = ds.map(_days_to_next).astype("int16")
    return out


def build_panel_features(
    series_by_sku: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a panel feature matrix across many SKUs for one user.

    Input: {sku: DataFrame(ds, y, ad_spend)} — one entry per SKU, already
    densified by `_build_series`.

    Returns (features_df, sku_to_cat) where features_df has columns
    FEATURE_COLS + "y" + "ds" + "sku" (target + keys retained for
    train/test splitting). sku_to_cat maps sku string → int code so the
    same encoding can be reused at inference time.
    """
    frames = []
    for sku, s in series_by_sku.items():
        if s.empty:
            continue
        df = s.copy()
        df["sku"] = sku
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLS + ["y", "ds", "sku"]), {}

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["sku", "ds"]).reset_index(drop=True)

    # Lags + rolling stats computed per-SKU so they don't bleed across
    # series. groupby.transform keeps the row alignment simple.
    grp = panel.groupby("sku", sort=False)["y"]
    for d in LAG_DAYS:
        panel[f"lag_{d}"] = grp.shift(d)
    for w in ROLL_WINDOWS:
        # Shift by 1 so today's row can't see today's y — leakage guard.
        shifted = grp.shift(1)
        panel[f"roll_mean_{w}"] = (
            shifted.groupby(panel["sku"], sort=False)
            .rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        panel[f"roll_std_{w}"] = (
            shifted.groupby(panel["sku"], sort=False)
            .rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        )

    cal = _calendar_features(panel["ds"])
    panel = pd.concat([panel, cal], axis=1)

    # SKU categorical — LightGBM handles native categoricals as int codes.
    sku_to_cat = {s: i for i, s in enumerate(sorted(series_by_sku.keys()))}
    panel["sku_cat"] = panel["sku"].map(sku_to_cat).astype("int32")

    if "ad_spend" not in panel.columns:
        panel["ad_spend"] = 0.0
    panel["ad_spend"] = panel["ad_spend"].fillna(0.0).astype("float32")

    return panel, sku_to_cat


def build_inference_row(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    sku_cat: int,
    ad_spend_hint: float = 0.0,
) -> pd.DataFrame:
    """Build a single feature row for `target_date` given a history frame
    of (ds, y) values for that SKU. Used by the recursive predict loop —
    each predicted day gets appended to `history` before the next call.
    """
    y = history["y"].to_numpy()
    row = {"ds": target_date}

    for d in LAG_DAYS:
        row[f"lag_{d}"] = float(y[-d]) if len(y) >= d else 0.0
    for w in ROLL_WINDOWS:
        tail = y[-w:] if len(y) >= 1 else y
        if len(tail) == 0:
            row[f"roll_mean_{w}"] = 0.0
            row[f"roll_std_{w}"] = 0.0
        else:
            row[f"roll_mean_{w}"] = float(tail.mean())
            row[f"roll_std_{w}"] = float(tail.std(ddof=0)) if len(tail) >= 2 else 0.0

    cal = _calendar_features(pd.Series([target_date]))
    for c in cal.columns:
        row[c] = int(cal.iloc[0][c])

    row["ad_spend"] = float(ad_spend_hint)
    row["sku_cat"] = int(sku_cat)

    return pd.DataFrame([row])
