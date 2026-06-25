# Inventory Forecasting Module — Plan

## Scope

Add AI-driven inventory forecasting and restock recommendations for Amazon
sellers. Target scale: ~100s of SKUs per user. Forecasts surface both as
agent tool calls and as a dashboard tab in the existing frontend.

## Models (locked)

| Layer | Choice | Why |
| --- | --- | --- |
| Primary forecaster | **Prophet** (per SKU, daily) | Handles trend + weekly + yearly seasonality + holidays out of the box. `add_regressor("ad_spend")` bakes in PPC uplift. Native p50/p90 intervals drive safety stock. ~1–2 s per SKU → fits comfortably in a nightly job at this scale. |
| Cold/sparse fallback | **28-day trimmed mean × day-of-week multiplier** | Any SKU with < 60 days of stockout-corrected history. Prophet on tiny data is worse than a sensible average. |
| Business-rule overlay | Pure Python | Safety-stock floor, overstock ceiling, lead-time + MOQ rounding. This is what makes output actionable. |

**Explicitly not using:** SARIMA/ARIMA (Prophet strictly better here),
LightGBM/XGBoost as a separate stage (revisit only if backtest MAPE > 25%
on top-revenue SKUs), LSTM/TFT/DeepAR (pay off at 10k+ SKUs), Google
Trends / weather / social (diminishing returns), Keepa-class competitor
pricing (real money — defer until users ask).

**One knob to revisit:** if Prophet MAPE is bad after 4–6 weeks of real
data, add **Prophet residuals → LightGBM** with ad/price features. That
is a 1-day add, not a rewrite.

## Phase 1 — Data spine

### New Mongo collections

`salesDaily` — one doc per (user, sku, date):

```
{ userId, sku, asin, date,
  units_ordered, ordered_revenue,
  sessions, page_views, buy_box_pct,
  units_available,
  ad_spend, ad_impressions, ad_clicks,
  stockout_corrected: bool }
```

`inventorySnapshot` — daily snapshot per (user, sku, date):

```
{ userId, sku, date,
  fulfillable, inbound_working, inbound_shipped,
  reserved, unfulfillable }
```

`forecastCache` — latest forecast per (user, sku):

```
{ userId, sku, generated_at, horizon_days,
  forecast: [ { date, p50, p90 } ],
  drivers: { recent_avg, growth_rate, ad_uplift },
  reorder: { reorder_point, safety_stock, days_of_cover,
             reorder_by_date, recommended_po_qty } }
```

### Ingest job (`backend/forecasting/ingest.py`)

- `GET_SALES_AND_TRAFFIC_REPORT` (daily granularity, by ASIN) → daily sales
- `GET_FBA_INVENTORY_PLANNING_DATA` → inventory snapshot + on-hand
- Ads spend joined by ASIN → SKU (from existing `amazon_ads.py`)
- **Stockout correction:** mask zero-sales days where `units_available == 0`.
  Those rows get `stockout_corrected=true` and are excluded from Prophet
  training.
- Modes: **backfill 18 months** on first run per user; nightly incremental
  for yesterday after that.

### Scheduler

APScheduler in the FastAPI lifespan. One nightly job iterates over every
user with a saved SP-API refresh token and runs ingest + forecast refresh.
Plus a manual `POST /forecasting/ingest` endpoint for dev / on-demand
backfill.

## Phase 2 — Forecast + reorder math

`backend/forecasting/model.py` — per-SKU Prophet fit on stockout-corrected
units, `ad_spend` as regressor, US holidays + custom Prime Day. SKUs
without enough history go to the trimmed-mean fallback.

`backend/forecasting/reorder.py` — pure business math:

- `safety_stock = z(service_level) * std(daily_demand) * sqrt(lead_time)`
- `reorder_point = avg_daily_demand * lead_time + safety_stock`
- `days_of_cover = current_fulfillable / forecast_p50_daily`
- `recommended_po_qty = max(0, 90d_target_cover − inbound − on_hand)` rounded to MOQ
- Lead-time and MOQ defaults configurable **per user** (option a). Per-SKU
  override fields can be added to the existing COGS schema later — schema
  is forward-compatible.

Forecast cache refreshed by the same nightly job after ingest.

## Phase 3 — Surfaces

### Agent tools (registered in `agent.py`)

- `forecast_sku(sku, horizon_days=30)` — p50/p90 demand + drivers
- `restock_recommendations(top_n=20)` — ranked by stockout risk
- `days_until_stockout(sku=None)` — quick triage

### Frontend tab (`frontend/index.html`)

New "Restock" tab with table:

| SKU | On hand | Inbound | 30-day forecast | Days of cover | Reorder by | Suggested PO |
| --- | --- | --- | --- | --- | --- | --- |

- Color-coded urgency: red < 21 days cover, amber 21–45, green > 45
- Row click → 90-day forecast chart + history (Chart.js)

## Deliberately deferred

- XGBoost residual stage
- LSTM / TFT / DeepAR
- Google Trends / weather / social signals
- Competitor pricing feeds (Keepa, etc.)
- Per-SKU lead-time / MOQ overrides (defaults only for v1)

## Build order

1. `FORECASTING_PLAN.md` ← this file
2. Deps: prophet, apscheduler, pandas, numpy
3. New collections + helpers in `database.py`
4. `forecasting/ingest.py` + backfill mode
5. APScheduler in `main.py` lifespan + manual trigger endpoint
6. `forecasting/model.py` (Prophet + fallback)
7. `forecasting/reorder.py` (business math)
8. Forecast refresh job (chained to ingest)
9. Agent tools
10. Frontend Restock tab
