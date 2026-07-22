from ast import List
import csv
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from agent import stream_response
from meta_ads import shutdown_browser
from database import (
    init_db,
    create_session,
    list_sessions,
    get_messages,
    delete_session,
    upsert_cogs,
    get_cogs,
    delete_cogs,
    get_forecast_settings,
    update_forecast_settings,
    get_forecast_cache,
    get_sales_daily,
    active_inbound_shipments_for_user,
    latest_inventory_for_user,
    get_product_settings,
    upsert_product_settings,
    all_product_settings_for_user,
    list_purchase_orders,
    upsert_purchase_order,
    delete_purchase_order,
    open_ordered_qty_by_sku,
)
from forecasting.ingest import (
    backfill_user,
    ensure_sales_history,
    ingest_user_incremental,
    run_nightly_ingest,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta, timezone

import amazon_sp
import aurora_data
import data_resolver
from aurora_data import aurora_db_enabled
from agent import compute_profitability_data
from campaigns import analyze_performance_data
from amazon_ads import (
    exchange_auth_code,
    fetch_suggested_keywords,
    get_profiles,
    save_refresh_token,
    save_profile_id,
)
from auth import (
    protect,
    authenticate_ws,
    authenticate_credentials,
    current_user,
    generate_token,
)
from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB. Campaigns are fetched lazily per-user on first tool call.
    await init_db()
    print("Database initialized.")

    # Nightly forecasting ingest — 03:00 UTC daily. Skipped if the
    # FORECASTING_SCHEDULER env var is set to "0" (useful in dev so the
    # job doesn't fire while iterating on the code).
    scheduler: AsyncIOScheduler | None = None
    if os.getenv("FORECASTING_SCHEDULER", "1") != "0":
        scheduler = AsyncIOScheduler(timezone="UTC")
        scheduler.add_job(
            run_nightly_ingest,
            trigger=CronTrigger(hour=3, minute=0),
            id="forecasting_nightly_ingest",
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        print("Forecasting scheduler started (nightly 03:00 UTC).")

    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        # Shutdown: close shared Playwright browser
        await shutdown_browser()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/auth/login")
async def login(body: LoginRequest):
    """Local mirror of Aurora's POST /api/auth/login.

    Verifies the password against the shared Mongo `users` collection and
    issues a JWT signed with the same JWT_SECRET — so the resulting token
    works against both this backend and Aurora's. Lets us avoid a cross-
    origin call from the chatbot frontend to Aurora.
    """
    user = await authenticate_credentials(body.email, body.password)
    token = generate_token(str(user["_id"]))
    # Strip Mongo ObjectId so the response is JSON-serializable.
    user["_id"] = str(user["_id"])
    if user.get("sellerApplicationId") is not None:
        user["sellerApplicationId"] = str(user["sellerApplicationId"])
    return {**user, "token": token}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    user, auth_error = await authenticate_ws(websocket)
    if not user:
        print(f"[ws_chat] auth failed: {auth_error}")
        await websocket.send_json({"type": "error", "content": auth_error or "Not authorized"})
        await websocket.close(code=4401)
        return
    try:
        while True:
            raw = await websocket.receive_text()
            # Re-set the ContextVar inside the receive loop so each message
            # runs with the right user even after asyncio scheduling boundaries.
            current_user.set(user)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON"})
                continue

            message = data.get("message", "").strip()
            session_id = data.get("session_id", "default")

            if not message:
                await websocket.send_json({"type": "error", "content": "Message is required"})
                continue

            async for event in stream_response(message, session_id=session_id):
                await websocket.send_json(event)

            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass


@app.post("/chat")
async def chat(request: Request, user: dict = Depends(protect)):
    """Legacy SSE endpoint (kept for curl / debug use)."""
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", "default")

    if not message.strip():
        return {"error": "Message is required"}

    async def event_stream():
        # ContextVar is request-scoped — re-set inside the generator so the
        # streaming task sees the authenticated user.
        current_user.set(user)
        async for event in stream_response(message, session_id=session_id):
            if event["type"] == "token":
                yield f"data: {json.dumps({'content': event['content']})}\n\n"
            elif event["type"] == "error":
                yield f"data: {json.dumps({'error': event['content']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session endpoints ──────────────────────────────────────────────────────

@app.post("/sessions")
async def create_new_session(user: dict = Depends(protect)):
    session_id = str(uuid.uuid4())
    await create_session(session_id)
    return {"session_id": session_id}


@app.get("/sessions")
async def get_sessions(user: dict = Depends(protect)):
    sessions = await list_sessions()
    return sessions


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user: dict = Depends(protect)):
    messages = await get_messages(session_id)
    return [m for m in messages if m.get("role") in ("user", "assistant")]


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, user: dict = Depends(protect)):
    await delete_session(session_id)
    return {"ok": True}


# ── COGS endpoints ─────────────────────────────────────────────────────────

@app.post("/cogs/upload")
async def upload_cogs(request: Request, user: dict = Depends(protect)):
    """Accept a CSV body (text/csv or text/plain) with columns sku, unit_cost,
    and optionally inbound_shipping_per_unit. Upserts rows into the cogs table.
    """
    raw = (await request.body()).decode("utf-8", errors="replace")
    if not raw.strip():
        return {"error": "empty body"}

    reader = csv.DictReader(io.StringIO(raw))
    rows: list[dict] = []
    skipped: list[str] = []
    for row in reader:
        if not row.get("sku"):
            continue
        try:
            cost = float((row.get("unit_cost") or "").strip() or 0)
        except ValueError:
            skipped.append(f"{row.get('sku')}: bad unit_cost {row.get('unit_cost')!r}")
            continue
        if cost <= 0:
            skipped.append(f"{row.get('sku')}: missing unit_cost")
            continue
        rows.append(row)

    written = await upsert_cogs(rows)
    return {"saved": written, "skipped": len(skipped), "skipped_details": skipped[:20]}


@app.get("/cogs")
async def list_cogs(user: dict = Depends(protect)):
    rows = await get_cogs()
    return {"count": len(rows), "rows": rows}


@app.put("/cogs/{sku}")
async def upsert_cogs_single(
    sku: str, request: Request, user: dict = Depends(protect),
):
    """Insert or update a single COGS row. Body: {unit_cost,
    inbound_shipping_per_unit?}. Used by the inline COGS editor in the
    FE — one PUT per row edit, keyed by SKU."""
    body = await request.json()
    written = await upsert_cogs([{
        "sku": sku,
        "unit_cost": body.get("unit_cost"),
        "inbound_shipping_per_unit": body.get("inbound_shipping_per_unit", 0),
    }])
    if not written:
        return {"error": "invalid unit_cost (must be a positive number)"}
    return {"saved": 1, "sku": sku}


@app.delete("/cogs/{sku}")
async def delete_cogs_endpoint(sku: str, user: dict = Depends(protect)):
    removed = await delete_cogs(sku)
    return {"removed": removed, "sku": sku}


# ── Forecasting endpoints ──────────────────────────────────────────────────


@app.post("/forecasting/ingest")
async def forecasting_ingest(
    user: dict = Depends(protect),
    backfill: bool = False,
    days_back: int = 540,
):
    """Manually trigger an ingest for the authenticated user. By default
    runs the incremental flow (yesterday's orders + today's inventory).
    Pass `?backfill=true` to do the full 18-month pull instead.
    """
    if not user.get("amazonRefreshToken"):
        return {"error": "SP-API not connected. Authorize at /amazon/sp-login first."}
    current_user.set(user)
    if backfill:
        result = await backfill_user(user, days_back=days_back)
        return {"mode": "backfill", "days_back": days_back, **result}
    result = await ingest_user_incremental(user)
    return {"mode": "incremental", **result}


@app.post("/forecasting/refresh")
async def forecasting_refresh(user: dict = Depends(protect)):
    """Ingest sales history when missing, then refit forecasts + reorder math.

    Reads Aurora Mongo orders first (SP-API fallback only when DB has no
    history). No redundant Amazon calls when salesDaily is already populated.
    """
    from bson import ObjectId as _OID
    from forecasting.model import refresh_forecasts_for_user
    current_user.set(user)
    ingest = await ensure_sales_history(user)
    refresh = await refresh_forecasts_for_user(_OID(str(user["_id"])))
    out = {**refresh, "ingest": ingest}
    if refresh.get("skus", 0) == 0:
        out["hint"] = (
            "No SKU sales history found. Sync orders in Aurora first, or "
            "ensure Amazon SP-API is connected so ingest can pull reports."
        )
    return out


@app.get("/forecasting/settings")
async def get_forecasting_settings_endpoint(user: dict = Depends(protect)):
    return await get_forecast_settings()


@app.put("/forecasting/settings")
async def update_forecasting_settings_endpoint(
    request: Request, user: dict = Depends(protect)
):
    body = await request.json()
    return await update_forecast_settings(body or {})


@app.get("/forecasting/restock")
async def forecasting_restock(user: dict = Depends(protect)):
    """The Restock dashboard's data source. One row per SKU with the
    forecast headline, reorder math, and method used."""
    from bson import ObjectId as _OID
    user_id = _OID(str(user["_id"]))

    cached = await get_forecast_cache()

    # Side joins for the derived columns.
    cogs_rows = await get_cogs()
    cogs_by_sku: dict[str, dict] = {r["sku"]: r for r in cogs_rows}
    settings_by_sku = await all_product_settings_for_user(user_id)
    ordered_by_sku = await open_ordered_qty_by_sku(user_id)
    # Amazon's own historical-days-of-supply and recommended-ship-in-quantity
    # come from the FBA Inventory Planning report. Read-only from cache here —
    # the report takes 30-120s to generate, so we let /profitability warm it
    # and just consume the fresh copy when it's available.
    # Prefer Aurora's fbaagedinventoryfees snapshot when populated — it
    # carries the same fields (monthly_fee, historical_days_of_supply) as
    # our legacy cache. Falls back to the SP-API-driven cache otherwise.
    import aurora_data
    aged_supplements: dict[str, dict] = {}
    try:
        aurora_aged = await aurora_data.fba_aged_inventory_by_sku(user)
    except Exception:
        aurora_aged = None
    if aurora_aged:
        aged_supplements = aurora_aged
    else:
        from database import get_aged_inventory_cache
        aged_cache = await get_aged_inventory_cache(max_age_hours=24)
        aged_supplements = (aged_cache or {}).get("per_sku", {})
    # Fresh read — buyability can flip Active/Inactive between forecast
    # refreshes, and we want the UI to reflect that instantly rather than
    # waiting for the next cache rebuild.
    inv_map = await latest_inventory_for_user(user_id)

    # Live Aurora-sourced velocities per SKU for the last 7/30/90 days,
    # so the restock table's Orders/day column reflects real trailing
    # demand rather than the forecast_cache's frozen number. One 90-day
    # Mongo aggregation covers every SKU — the per-window slicing happens
    # in-memory in `compute_velocity_windows`.
    from forecasting.model import compute_velocity_windows, weighted_velocity
    from database import DEFAULT_PRODUCT_SETTINGS
    since_180d = datetime.now(timezone.utc) - timedelta(days=180)
    sales_rows = await get_sales_daily(sku=None, since=since_180d)
    sales_by_sku: dict[str, list[dict]] = {}
    for r in sales_rows:
        sales_by_sku.setdefault(r.get("sku") or "", []).append(r)
    now_utc = datetime.now(timezone.utc)
    default_weights = DEFAULT_PRODUCT_SETTINGS["velocity_weights"]
    weighted_by_sku: dict[str, float] = {}
    orders_7d_by_sku: dict[str, int] = {}
    orders_30d_by_sku: dict[str, int] = {}
    orders_60d_by_sku: dict[str, int] = {}
    stockout_days_by_sku: dict[str, int] = {}
    # Mongo returns naive UTC datetimes — strip the tz on the cutoff so
    # `date >= cutoff_90d` doesn't blow up on the naive/aware mismatch.
    cutoff_90d_naive = (now_utc - timedelta(days=90)).replace(tzinfo=None)
    for sku_key, sku_rows in sales_by_sku.items():
        windows = compute_velocity_windows(
            sku_rows, now_utc, windows=(3, 7, 30, 60, 180),
        )
        wv = weighted_velocity(windows, default_weights)
        weighted_by_sku[sku_key] = float(wv) if wv is not None else 0.0
        for w in windows:
            pd_val = int(w.get("period_days") or 0)
            units = int(w.get("units_sold") or 0)
            if pd_val == 7:
                orders_7d_by_sku[sku_key] = units
            elif pd_val == 30:
                orders_30d_by_sku[sku_key] = units
            elif pd_val == 60:
                orders_60d_by_sku[sku_key] = units
        # Live count of stockout-corrected days in the trailing 90d —
        # get_sales_daily runs `_flag_stockout_runs` on read, so any row
        # inside the window with the flag set is a missed-sales day.
        count = 0
        for r in sku_rows:
            if not r.get("stockout_corrected"):
                continue
            d = r.get("date")
            if not isinstance(d, datetime):
                continue
            d_naive = d.replace(tzinfo=None) if d.tzinfo is not None else d
            if d_naive >= cutoff_90d_naive:
                count += 1
        stockout_days_by_sku[sku_key] = count

    today = datetime.now(timezone.utc).date()

    rows = []
    for c in cached:
        sku = c["sku"]
        reorder = c.get("reorder") or {}
        forecast = c.get("forecast") or []
        next30 = sum(float(r.get("p50", 0)) for r in forecast[:30])

        # Prefer the fresh Aurora snapshot for the 5 SP-API-sourced counts;
        # fall back to the forecast cache when the SKU isn't in `inv_map`
        # (e.g. delisted from Aurora but still in forecast_cache). `inbound`
        # comes from the shipments collection and stays on `reorder`.
        inv_row = inv_map.get(sku) or {}
        # `available` = Amazon's fulfillable (currently sellable) quantity.
        # Reorder/days-of-cover math below wants this — it explicitly adds
        # `reserved` back on top to reconstruct total forward-looking stock.
        available = int(
            inv_row.get("fulfillable", reorder.get("available", reorder.get("on_hand", 0))) or 0
        )
        reserved = int(inv_row.get("reserved", reorder.get("reserved", 0)) or 0)
        sent_to_fba = int(inv_row.get("inbound_shipped", reorder.get("sent_to_fba", 0)) or 0)
        inbound_working = int(inv_row.get("inbound_working", reorder.get("inbound_working", 0)) or 0)
        unfulfillable = int(inv_row.get("unfulfillable", reorder.get("unfulfillable", 0)) or 0)
        # Seller Central "On-hand (FBA)" = Available + FC Transfer
        # (pendingTransshipment). See latest_inventory_for_user.
        fc_transfer = int(inv_row.get("fc_transfer") or 0)
        on_hand = int(
            inv_row.get("on_hand")
            or (available + fc_transfer)
            or reorder.get("on_hand", 0)
            or 0
        )

        # Stock value = (available + reserved + inbound + sent_to_fba) × unit landed cost.
        cogs = cogs_by_sku.get(sku) or {}
        unit_cost = float(cogs.get("unit_cost") or 0)
        unit_ship = float(cogs.get("inbound_shipping_per_unit") or 0)
        landed_cost = unit_cost + unit_ship
        stock_units = available + reserved + sent_to_fba + inbound_working
        stock_value = round(stock_units * landed_cost, 2) if landed_cost > 0 else 0.0

        # Missed profit estimate — count stockout-corrected days in the
        # trailing 90-day history (live from Aurora sales, not the frozen
        # forecast_cache drivers) and value them at weighted velocity ×
        # 25% margin. The 25% is a coarse proxy until per-SKU sale price
        # is threaded through this endpoint.
        drivers = c.get("drivers") or {}
        stockout_days = stockout_days_by_sku.get(sku, 0) or int(drivers.get("stockout_days_90d") or 0)
        velocity = weighted_by_sku.get(sku, 0.0) or float(reorder.get("avg_daily_demand") or 0)
        missed_units = round(stockout_days * velocity, 1)
        missed_profit_est = round(missed_units * landed_cost * 0.25, 2) if landed_cost > 0 else 0.0

        # Override the forecast-cache reorder numbers with values derived
        # from the weighted velocity so the whole restock row tells one
        # consistent story: Orders/day, Days of cover, Stockout on, and
        # both Ship-by dates all trust the same demand signal the seller
        # sees in the modal. When weighted velocity is 0 (brand new SKU
        # with no trailing sales), fall through to the Prophet-driven
        # cached values so we don't clobber a legitimate horizon-based
        # forecast with zeros.
        wv = weighted_by_sku.get(sku, 0.0)
        stock_forward = available + reserved + sent_to_fba + inbound_working
        days_of_cover_val = reorder.get("days_of_cover")
        stockout_date_iso = reorder.get("stockout_date")
        reorder_by_date_air_iso = reorder.get("reorder_by_date_air")
        reorder_by_date_ocean_iso = reorder.get("reorder_by_date_ocean")
        if wv > 0:
            days_of_cover_val = round(stock_forward / wv, 1)
            stockout_date_obj = today + timedelta(days=int(stock_forward / wv))
            stockout_date_iso = stockout_date_obj.isoformat()
            air_transit = int(reorder.get("air_transit_days") or 10)
            ocean_transit = int(reorder.get("ocean_transit_days") or 45)
            # Clamp ship-by dates to today when they'd be in the past —
            # a past date is confusing UI; "ship NOW" is the message.
            reorder_by_date_air_iso = max(
                today, stockout_date_obj - timedelta(days=air_transit),
            ).isoformat()
            reorder_by_date_ocean_iso = max(
                today, stockout_date_obj - timedelta(days=ocean_transit),
            ).isoformat()

        # Days until next-order deadline (based on air ship-by date).
        days_until_next_order = None
        if reorder_by_date_air_iso:
            try:
                d = datetime.fromisoformat(reorder_by_date_air_iso).date()
                days_until_next_order = (d - today).days
            except ValueError:
                pass

        aged_sup = aged_supplements.get(sku) or {}
        historical_dos = aged_sup.get("historical_days_of_supply")
        amazon_rec_qty = aged_sup.get("recommended_ship_in_quantity")
        amazon_rec_date = aged_sup.get("recommended_ship_in_date")

        ps = settings_by_sku.get(sku) or {}
        is_buyable = bool(inv_row.get("is_buyable", True))
        # For non-buyable SKUs, zero the reorder recommendation regardless
        # of what the forecast says — no point telling the seller to ship
        # 500 units of an inactive listing.
        recommended_po_qty = reorder.get("recommended_po_qty", 0) if is_buyable else 0

        rows.append({
            "sku": sku,
            "asin": c.get("asin"),
            "fnsku": inv_row.get("fnsku"),
            "method": c.get("method"),
            "is_buyable": is_buyable,
            "status": inv_row.get("status"),
            "listing_status": inv_row.get("listing_status"),
            "generated_at": c.get("generated_at").isoformat() if c.get("generated_at") else None,
            "on_hand": on_hand,
            "available": available,
            "reserved": reserved,
            "sent_to_fba": sent_to_fba,
            "inbound_working": inbound_working,
            "unfulfillable": unfulfillable,
            "inbound": reorder.get("inbound", 0),
            "ordered": int(ordered_by_sku.get(sku) or 0),
            "avg_daily_demand": reorder.get("avg_daily_demand", 0),
            "weighted_velocity": round(weighted_by_sku.get(sku, 0.0), 2),
            "orders_7d": int(orders_7d_by_sku.get(sku, 0)),
            "orders_30d": int(orders_30d_by_sku.get(sku, 0)),
            "orders_60d": int(orders_60d_by_sku.get(sku, 0)),
            "next_30_day_forecast": round(next30, 1),
            "days_of_cover": days_of_cover_val,
            "stockout_date": stockout_date_iso,
            "reorder_by_date": reorder.get("reorder_by_date"),
            "reorder_by_date_air": reorder_by_date_air_iso,
            "reorder_by_date_ocean": reorder_by_date_ocean_iso,
            "reorder_by_date_sea": reorder_by_date_ocean_iso,  # legacy alias
            "days_until_next_order": days_until_next_order,
            "air_transit_days": reorder.get("air_transit_days"),
            "ocean_transit_days": reorder.get("ocean_transit_days"),
            "inbound_shipments_count": reorder.get("inbound_shipments_count", 0),
            "next_shipment_eta": reorder.get("next_shipment_eta"),
            "next_shipment_qty": reorder.get("next_shipment_qty"),
            "recommended_po_qty": recommended_po_qty,
            "amazon_recommended_ship_qty": amazon_rec_qty,
            "amazon_recommended_ship_date": amazon_rec_date,
            "historical_days_of_supply": historical_dos,
            "unit_cost": unit_cost,
            "landed_cost": round(landed_cost, 4),
            "stock_value": stock_value,
            "missed_profit_est": missed_profit_est,
            "comment": ps.get("comment") or "",
            "drivers": drivers,
        })
    # Sort: stockouts first (None or low days_of_cover), then ascending.
    def _sort_key(r):
        d = r.get("days_of_cover")
        return (1, 0) if d is None else (0, d)
    rows.sort(key=_sort_key)
    return {"count": len(rows), "rows": rows}


@app.get("/forecasting/sku/{sku}")
async def forecasting_sku_detail(sku: str, user: dict = Depends(protect)):
    """Full forecast (90-day p50/p90) plus last 90 days of actuals for one
    SKU — drives the detail chart. Stockout days are flagged so the FE can
    render them differently (the model trained without them)."""
    from datetime import datetime, timezone, timedelta as _td
    from forecasting.model import (
        _forecast_one, compute_velocity_windows, weighted_velocity,
    )
    from database import DEFAULT_PRODUCT_SETTINGS
    from bson import ObjectId as _OID

    cached = await get_forecast_cache(skus=[sku])
    if not cached:
        return {"error": f"No forecast for {sku}. Run /forecasting/refresh first."}
    c = cached[0]

    # Chart data — always 90 days of actuals for context.
    since = datetime.now(timezone.utc) - _td(days=90)
    raw_history = await get_sales_daily(sku=sku, since=since)

    # Live model refresh: recompute the forecast on the current
    # 30-day training window so the chart's p50/p90 reflects the model
    # we're actually using. Also recompute the reorder-side numbers
    # (days_of_cover, stockout_date, ship-by dates) off the same
    # weighted velocity that drives the restock table's Orders/day,
    # so this modal and the row underneath tell one consistent story.
    now_utc = datetime.now(timezone.utc)
    train_since = now_utc - _td(days=30)
    train_rows = await get_sales_daily(sku=sku, since=train_since)
    try:
        fresh = _forecast_one(train_rows, horizon=90, today=now_utc)
        live_forecast = fresh.get("forecast") or c.get("forecast")
        live_method = fresh.get("method") or c.get("method")
        live_drivers = fresh.get("drivers") or c.get("drivers")
    except Exception:
        live_forecast = c.get("forecast")
        live_method = c.get("method")
        live_drivers = c.get("drivers")

    # Weighted velocity from the last 180 days of Aurora sales — same
    # blend the restock endpoint uses.
    wv_rows = await get_sales_daily(sku=sku, since=now_utc - _td(days=180))
    windows = compute_velocity_windows(
        wv_rows, now_utc, windows=(3, 7, 30, 60, 180),
    )
    wv_raw = weighted_velocity(
        windows, DEFAULT_PRODUCT_SETTINGS["velocity_weights"],
    )
    wv = float(wv_raw) if wv_raw is not None else 0.0

    # Override cached reorder numbers with the weighted-velocity view
    # exactly like /forecasting/restock does.
    reorder = dict(c.get("reorder") or {})
    inv_map = await latest_inventory_for_user(_OID(str(user["_id"])))
    inv_row = inv_map.get(sku) or {}
    available = int(
        inv_row.get("fulfillable", reorder.get("available", reorder.get("on_hand", 0))) or 0
    )
    reserved = int(inv_row.get("reserved", reorder.get("reserved", 0)) or 0)
    sent_to_fba = int(inv_row.get("inbound_shipped", reorder.get("sent_to_fba", 0)) or 0)
    inbound_working = int(inv_row.get("inbound_working", reorder.get("inbound_working", 0)) or 0)
    unfulfillable = int(inv_row.get("unfulfillable", reorder.get("unfulfillable", 0)) or 0)
    # Seller Central "On-hand (FBA)" = Available + FC Transfer.
    fc_transfer = int(inv_row.get("fc_transfer") or 0)
    on_hand = int(
        inv_row.get("on_hand")
        or (available + fc_transfer)
        or reorder.get("on_hand", 0)
        or 0
    )
    stock_forward = available + reserved + sent_to_fba + inbound_working
    today_date = now_utc.date()
    if wv > 0 and stock_forward > 0:
        reorder["days_of_cover"] = round(stock_forward / wv, 1)
        stockout_dt = today_date + _td(days=int(stock_forward / wv))
        reorder["stockout_date"] = stockout_dt.isoformat()
        air_transit = int(reorder.get("air_transit_days") or 10)
        ocean_transit = int(reorder.get("ocean_transit_days") or 45)
        reorder["reorder_by_date_air"] = max(
            today_date, stockout_dt - _td(days=air_transit),
        ).isoformat()
        reorder["reorder_by_date_ocean"] = max(
            today_date, stockout_dt - _td(days=ocean_transit),
        ).isoformat()
    reorder["on_hand"] = on_hand
    reorder["available"] = available
    reorder["reserved"] = reserved
    reorder["sent_to_fba"] = sent_to_fba
    reorder["inbound_working"] = inbound_working
    reorder["avg_daily_demand"] = round(wv, 2) if wv > 0 else reorder.get("avg_daily_demand", 0)

    # Densify the array on the frontend; here we just emit the (date,
    # units, stockout_corrected) tuples for whichever days we have rows.
    history = [
        {
            "date": r["date"].isoformat() if hasattr(r.get("date"), "isoformat") else r.get("date"),
            "units": int(r.get("units_ordered") or 0),
            "stockout_corrected": bool(r.get("stockout_corrected", False)),
        }
        for r in raw_history
    ]

    shipments_by_sku = await active_inbound_shipments_for_user(
        _OID(str(user["_id"])),
    )
    shipments = [
        {
            "shipment_id": s.get("shipment_id"),
            "name": s.get("name"),
            "status": s.get("status"),
            "display_status": s.get("display_status"),
            "eta": s["eta"].date().isoformat() if s.get("eta") else None,
            "qty_outstanding": s.get("qty_outstanding"),
            "carrier_name": s.get("carrier_name"),
            "mode": s.get("mode"),
        }
        for s in shipments_by_sku.get(sku, [])
    ]

    # ── Backtest / holdout validation ────────────────────────────────────
    # Train on days [-60, -31], predict days [-30, -1], compare against
    # actual. Gives the seller a concrete accuracy read on the model
    # they're seeing — not a black-box "trust me" number.
    backtest: dict | None = None
    try:
        bt_since = now_utc - _td(days=60)
        bt_rows = await get_sales_daily(sku=sku, since=bt_since)
        cutoff = (now_utc - _td(days=30)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        cutoff_naive = cutoff.replace(tzinfo=None)

        def _row_day(r):
            d = r.get("date")
            if not isinstance(d, datetime):
                return None
            return d.replace(tzinfo=None) if d.tzinfo is not None else d

        train_rows_bt: list[dict] = []
        holdout_by_day: dict[str, int] = {}
        for r in bt_rows:
            d = _row_day(r)
            if d is None:
                continue
            if d < cutoff_naive:
                train_rows_bt.append(r)
            else:
                key = d.date().isoformat()
                holdout_by_day[key] = holdout_by_day.get(key, 0) + int(
                    r.get("units_ordered") or 0,
                )

        bt_result = _forecast_one(train_rows_bt, horizon=30, today=cutoff)
        bt_days: list[dict] = []
        for r in bt_result.get("forecast") or []:
            d_key = r["date"][:10]
            actual = int(holdout_by_day.get(d_key, 0))
            p50 = float(r.get("p50", 0))
            p90 = float(r.get("p90", 0))
            bt_days.append({
                "date": d_key,
                "actual": actual,
                "p50": round(p50, 2),
                "p90": round(p90, 2),
            })

        metrics: dict | None = None
        n = len(bt_days)
        if n > 0:
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
            # Volume accuracy — how close is the total predicted volume to
            # the total actual? Forgives offsetting daily misses (some days
            # over, some under) so the metric matches the seller's intuition
            # for "did we order the right amount". WAPE-style daily-error
            # accuracy stays available in `accuracy_daily_pct` for anyone
            # who cares about day-level jitter (stockout-timing analyses).
            volume_error = abs(pred_total - actual_total)
            accuracy_pct = (
                round(max(0.0, (1 - volume_error / actual_total) * 100), 1)
                if actual_total > 0 else None
            )
            wape = (
                sum(abs_errors) / actual_total if actual_total > 0 else None
            )
            accuracy_daily_pct = (
                round(max(0.0, (1 - wape) * 100), 1) if wape is not None else None
            )
            metrics = {
                "mae": round(sum(abs_errors) / n, 2),
                "rmse": round((sum(squared) / n) ** 0.5, 2),
                "bias": round(sum(errors) / n, 2),
                "mape_pct": round(
                    (sum(pct_errors) / len(pct_errors)) * 100, 1,
                ) if pct_errors else None,
                "accuracy_pct": accuracy_pct,
                "accuracy_daily_pct": accuracy_daily_pct,
                "actual_total": actual_total,
                "predicted_total": round(pred_total, 1),
                "coverage_pct": round((coverage_hits / n) * 100, 1),
                "days_evaluated": n,
            }

        train_start = train_rows_bt[0].get("date") if train_rows_bt else None
        backtest = {
            "train_start": (
                train_start.date().isoformat()
                if isinstance(train_start, datetime) else None
            ),
            "train_end": (cutoff - _td(days=1)).date().isoformat(),
            "holdout_start": cutoff.date().isoformat(),
            "holdout_end": (now_utc - _td(days=1)).date().isoformat(),
            "method": bt_result.get("method"),
            "days": bt_days,
            "metrics": metrics,
        }
    except Exception:
        backtest = None

    return {
        "sku": sku,
        "asin": c.get("asin"),
        "method": live_method,
        "generated_at": c.get("generated_at").isoformat() if c.get("generated_at") else None,
        "horizon_days": c.get("horizon_days"),
        "drivers": live_drivers,
        "reorder": reorder,
        "forecast": live_forecast,
        "history": history,
        "inbound_shipments": shipments,
        "backtest": backtest,
    }


# ── Per-SKU product settings (Actions modal) ──────────────────────────────

@app.get("/product-settings/{sku}")
async def get_product_settings_endpoint(sku: str, user: dict = Depends(protect)):
    """Full settings for one SKU + the computed velocity-windows table the
    Forecast tab renders."""
    from forecasting.model import compute_velocity_windows
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    settings = await get_product_settings(sku)
    since = _dt.now(_tz.utc) - _td(days=200)
    rows = await get_sales_daily(sku=sku, since=since)
    windows = compute_velocity_windows(rows, _dt.now(_tz.utc))
    return {"sku": sku, "settings": settings, "velocity_windows": windows}


@app.put("/product-settings/{sku}")
async def put_product_settings_endpoint(
    sku: str, patch: dict, user: dict = Depends(protect),
):
    settings = await upsert_product_settings(sku, patch)
    return {"sku": sku, "settings": settings}


# ── Purchase orders (drives the "Ordered" column) ────────────────────────

@app.get("/purchase-orders")
async def list_purchase_orders_endpoint(
    status: str | None = None, user: dict = Depends(protect),
):
    return {"rows": await list_purchase_orders(status=status)}


@app.post("/purchase-orders")
async def create_purchase_order_endpoint(
    body: dict, user: dict = Depends(protect),
):
    return await upsert_purchase_order(body)


@app.patch("/purchase-orders/{po_id}")
async def update_purchase_order_endpoint(
    po_id: str, body: dict, user: dict = Depends(protect),
):
    body["poId"] = po_id
    return await upsert_purchase_order(body)


@app.delete("/purchase-orders/{po_id}")
async def delete_purchase_order_endpoint(
    po_id: str, user: dict = Depends(protect),
):
    n = await delete_purchase_order(po_id)
    return {"deleted": n}


# ── Amazon SP-API data endpoints (for FE tables) ───────────────────────────

ORDERS_SOURCE = os.getenv("AURORA_DATA_SOURCE") or os.getenv("AURORA_ORDERS_SOURCE", "db")
ORDERS_SOURCE = ORDERS_SOURCE.strip().lower()


@app.get("/amazon/marketplaces")
async def amazon_marketplaces(user: dict = Depends(protect)):
    """Marketplaces the user is registered in. Feeds the Orders tab selector."""
    return {"marketplaces": data_resolver.list_marketplaces_resolved(user)}


@app.get("/amazon/orders")
async def amazon_orders(
    days_back: int = 30,
    start: str | None = None,
    end: str | None = None,
    status: str | None = None,
    marketplace: str | None = None,
    buyer_email: str | None = None,
    user: dict = Depends(protect),
):
    """List orders in the requested window across all pages (no FE pagination).

    Filters: date range (days_back OR explicit start/end ISO-8601), comma-
    separated OrderStatuses, single marketplace (id / short code / country
    name; default = all the user has), and optional buyer email substring
    match applied after fetch."""
    if start:
        created_after = start
    else:
        created_after = (
            datetime.now(timezone.utc) - timedelta(days=max(1, days_back))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    created_before = end
    statuses = [s.strip() for s in status.split(",") if s.strip()] if status else None

    if ORDERS_SOURCE == "db":
        return await data_resolver.list_orders_resolved(
            user,
            created_after=created_after,
            created_before=created_before,
            statuses=statuses,
            marketplace=marketplace,
            buyer_email=buyer_email,
            paginate=True,
        )

    try:
        data = await amazon_sp.get_orders(
            created_after=created_after,
            created_before=created_before,
            statuses=statuses,
            max_results=100,
            marketplace=marketplace,
            paginate=True,
        )
    except Exception as e:
        # Return a JSON error the FE can render, not FastAPI's bare 500.
        # 429s are especially common on multi-marketplace sellers because
        # Orders API is 1 req/min sustained and burst is shared LWA-wide.
        msg = str(e)
        return {
            "error": f"Failed to fetch orders: {msg}",
            "error_kind": "rate_limited" if "429" in msg else "orders_fetch_failed",
            "count": 0, "orders": [],
            "created_after": created_after,
            "created_before": created_before,
        }
    orders = (data.get("payload") or {}).get("Orders") or []
    if buyer_email:
        needle = buyer_email.lower()
        orders = [
            o for o in orders
            if needle in (o.get("BuyerInfo", {}).get("BuyerEmail") or "").lower()
        ]
    out = {
        "count": len(orders),
        "created_after": created_after,
        "created_before": created_before,
        "orders": orders,
    }
    if data.get("_partial"):
        out["partial_warning"] = data["_partial"]
    return out


@app.get("/amazon/orders/{order_id}/items")
async def amazon_order_items(order_id: str, user: dict = Depends(protect)):
    """Line items for a single order — used to expand a row in the FE table."""
    if ORDERS_SOURCE == "db":
        return await data_resolver.get_order_items_resolved(user, order_id)
    data = await amazon_sp.get_order_items(order_id)
    return {"items": (data.get("payload") or {}).get("OrderItems") or []}


@app.get("/campaigns/performance")
async def campaigns_performance(user: dict = Depends(protect)):
    """Structured campaign performance — feeds the Campaigns tab."""
    return await analyze_performance_data(full=True)


@app.get("/profitability")
async def profitability(
    days_back: int = 7,
    start: str | None = None,
    end: str | None = None,
    timeZone: str | None = None,
    user: dict = Depends(protect),
):
    """Per-SKU profitability for the requested window. Walks SP-API
    NextToken so the FE sees the whole window.

    Accepts either `?start=YYYY-MM-DD&end=YYYY-MM-DD` (preferred, matches
    the FE date pickers) or `?days_back=N` (legacy, still used by the LLM
    tool). If both are given, start/end wins.

    Day boundaries use the seller's marketplace timezone (same as Aurora Orders)."""
    return await compute_profitability_data(
        days_back=days_back, start=start, end=end, paginate=True,
        time_zone=timeZone,
    )


@app.get("/profitability/sku/{sku}/prices")
async def profitability_sku_prices(
    sku: str,
    start: str | None = None,
    end: str | None = None,
    user: dict = Depends(protect),
):
    """Per-line-item price breakdown for one SKU over a date range.

    Powers the "click Avg Price → see every unit's price" drill-down.
    Same date-range parsing rules as /profitability so the modal shows
    the same universe of orders as the row above it.
    """
    from bson import ObjectId as _OID
    from database import _db
    from marketplace_timezone import resolve_dashboard_timezone, parse_date_range_for_query

    user_id = _OID(str(user["_id"]))
    mp_tz = resolve_dashboard_timezone(user, None)
    now = datetime.now(timezone.utc)

    if start and end:
        start_dt, end_dt = parse_date_range_for_query(start, end, mp_tz)
    else:
        start_dt = now - timedelta(days=7)
        end_dt = now
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    pipeline = [
        {"$match": {
            "sellerId": user_id,
            "purchaseDate": {"$gte": start_dt, "$lte": end_dt},
        }},
        {"$unwind": "$orderItems"},
        {"$match": {"orderItems.sellerSku": sku}},
        {"$project": {
            "_id": 0,
            "amazon_order_id": "$amazonOrderId",
            "order_status": "$orderStatus",
            "purchase_date": "$purchaseDate",
            "qty": {"$ifNull": ["$orderItems.quantityOrdered", 0]},
            "item_price_line": {"$ifNull": ["$orderItems.itemPrice.amount", 0]},
            "item_subtotal_line": {"$ifNull": ["$orderItems.itemSubtotal.amount", 0]},
            "promotion_discount": {"$ifNull": ["$orderItems.promotionDiscount.amount", 0]},
            "currency": {"$ifNull": ["$orderItems.itemPrice.currencyCode", "USD"]},
            "asin": "$orderItems.asin",
            "city": "$shippingAddress.city",
            "state": "$shippingAddress.stateOrRegion",
        }},
        {"$sort": {"purchase_date": -1}},
    ]
    docs = await _db().orders.aggregate(pipeline).to_list(length=None)

    rows = []
    for d in docs:
        qty = int(d.get("qty") or 0)
        subtotal = float(d.get("item_subtotal_line") or 0)
        item_price = float(d.get("item_price_line") or 0)
        promo = float(d.get("promotion_discount") or 0)
        gross_line = subtotal if subtotal > 0 else item_price
        net_line = max(0.0, gross_line - promo)
        unit_price = net_line / qty if qty > 0 else 0.0
        rows.append({
            "amazon_order_id": d.get("amazon_order_id"),
            "order_status": d.get("order_status"),
            "purchase_date": d["purchase_date"].isoformat() if d.get("purchase_date") else None,
            "qty": qty,
            "gross_line": round(gross_line, 2),
            "promo_discount": round(promo, 2),
            "net_line": round(net_line, 2),
            "unit_price": round(unit_price, 2),
            "currency": d.get("currency") or "USD",
            "asin": d.get("asin"),
            "ship_to": ", ".join(
                x for x in [d.get("city"), d.get("state")] if x
            ) or None,
        })

    total_units = sum(r["qty"] for r in rows)
    total_net_revenue = round(sum(r["net_line"] for r in rows), 2)
    prices = [r["unit_price"] for r in rows if r["qty"] > 0]
    return {
        "sku": sku,
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "line_count": len(rows),
        "total_units": total_units,
        "total_net_revenue": total_net_revenue,
        "avg_price": round(total_net_revenue / total_units, 2) if total_units else 0.0,
        "min_price": round(min(prices), 2) if prices else 0.0,
        "max_price": round(max(prices), 2) if prices else 0.0,
        "lines": rows,
    }


# ── Amazon OAuth endpoints ─────────────────────────────────────────────────

AMAZON_AUTH_URL = "https://www.amazon.com/ap/oa"
DEFAULT_REDIRECT_URI = "https://d24d-2409-4085-e8c-b458-f90e-780c-de78-1437.ngrok-free.app/amazon/sp-callback"


@app.get("/amazon/login")
async def amazon_login(redirect_uri: str | None = None):
    """Redirect user to Amazon's OAuth consent page."""
    ru = redirect_uri or DEFAULT_REDIRECT_URI
    params = urlencode({
        "client_id": os.getenv("AMAZON_LWA_CLIENT_ID", ""),
        "scope": "advertising::campaign_management",
        "response_type": "code",
        "redirect_uri": ru,
    })
    print(f"[amazon_login] Redirecting to Amazon OAuth consent page with params: {params}")
    return RedirectResponse(f"{AMAZON_AUTH_URL}?{params}")


@app.get("/callback")
@app.get("/api/auth/amazon/ads-callback")
async def oauth_callback(
    request: Request,
    user: dict = Depends(protect),
    code: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle Amazon Ads OAuth callback — exchange code for refresh token and
    persist on the authenticated user's Mongo doc."""
    if error or not code:
        detail = error_description or error or "No authorization code was provided."
        print(f"[oauth_callback] no code. error={error!r} description={error_description!r}")
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{detail}</p>"
            "<p>Start the flow again from <code>/amazon/login</code>.</p>",
            status_code=400,
        )

    redirect_uri = f"https://{request.url.netloc}{request.url.path}"
    try:
        tokens = await exchange_auth_code(code, redirect_uri)
    except Exception as e:
        return HTMLResponse(f"<h2>OAuth Error</h2><pre>{e}</pre>", status_code=400)

    refresh_token = tokens.get("refresh_token", "")
    if refresh_token:
        await save_refresh_token(refresh_token, scope="ads")

    return HTMLResponse(
        "<h2>Authorization successful!</h2>"
        f"<p>Refresh token has been saved on your account ({user.get('email')}).</p>"
        f"<pre>access_token (truncated): {tokens.get('access_token', '')[:20]}...</pre>"
    )


# ── SP-API OAuth ──────────────────────────────────────────────────────────

SP_API_APP_ID = "amzn1.sp.solution.30028133-1d34-4996-b191-fb3ff4ce57f2"
SP_API_AUTH_URL = "https://sellercentral.amazon.com/apps/authorize/consent"


@app.get("/amazon/sp-login")
async def sp_api_login(
    request: Request,
    redirect_uri: str | None = None,
    token: str | None = None,
    authorization: str | None = None,
):
    """Redirect user to Seller Central to authorize the SP-API app.

    The caller's JWT is embedded into OAuth `state` so `/amazon/sp-callback`
    can identify the user after Amazon's 302 redirect — browsers can't
    attach an Authorization header to a redirect, so we can't use the
    normal `protect` dependency on the callback. The token is accepted
    either as `?token=<jwt>` (works from a plain browser link) or as an
    `Authorization: Bearer <jwt>` header (works from XHR)."""
    # Local import — avoids exposing internal helpers in the module namespace.
    from auth import _verify_token, _load_user

    jwt_token = token
    if not jwt_token:
        auth_header = authorization or request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            jwt_token = auth_header.split(" ", 1)[1]
    if not jwt_token:
        return HTMLResponse(
            "<h2>SP-API OAuth Error</h2><p>Missing auth — pass your JWT as "
            "<code>?token=&lt;jwt&gt;</code> or in an <code>Authorization: "
            "Bearer</code> header.</p>",
            status_code=401,
        )
    try:
        _load_user_id = _verify_token(jwt_token).get("id")
        await _load_user(_load_user_id)  # ensure the user still exists
    except Exception as e:
        return HTMLResponse(
            f"<h2>SP-API OAuth Error</h2><p>Auth failed: {e}</p>",
            status_code=401,
        )

    ru = redirect_uri or DEFAULT_REDIRECT_URI
    params = urlencode({
        "application_id": SP_API_APP_ID,
        "redirect_uri": ru,
        "state": jwt_token,
    })
    print(f"[sp_login] Redirecting to Seller Central consent: {SP_API_AUTH_URL}?{params}")
    return RedirectResponse(f"{SP_API_AUTH_URL}?{params}")


@app.get("/amazon/sp-callback")
async def sp_api_callback(
    request: Request,
    spapi_oauth_code: str | None = None,
    state: str | None = None,
    selling_partner_id: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle SP-API OAuth callback — exchange code for refresh token and
    persist it as the authenticated user's amazonRefreshToken.

    Authenticates via the JWT embedded in `state` (set by /amazon/sp-login),
    since Amazon's redirect strips any Authorization header."""
    from auth import _verify_token, _load_user

    if error or not spapi_oauth_code:
        detail = error_description or error or "No authorization code was provided."
        print(f"[sp_callback] no code. error={error!r} description={error_description!r}")
        return HTMLResponse(
            f"<h2>SP-API OAuth Error</h2><p>{detail}</p>"
            "<p>Start the flow again from <code>/amazon/sp-login</code>.</p>",
            status_code=400,
        )

    if not state:
        return HTMLResponse(
            "<h2>SP-API OAuth Error</h2><p>Missing state — start the flow at "
            "<code>/amazon/sp-login</code>.</p>",
            status_code=400,
        )

    try:
        decoded = _verify_token(state)
        user = await _load_user(decoded.get("id"))
    except Exception as e:
        return HTMLResponse(
            f"<h2>SP-API OAuth Error</h2><p>Auth failed: {e}</p>",
            status_code=401,
        )

    user["_token"] = state
    current_user.set(user)

    print(f"[sp_callback] code={spapi_oauth_code[:8]}... seller_id={selling_partner_id}")

    redirect_uri = f"https://{request.url.netloc}{request.url.path}"
    try:
        tokens = await exchange_auth_code(spapi_oauth_code, redirect_uri)
    except Exception as e:
        return HTMLResponse(f"<h2>SP-API OAuth Error</h2><pre>{e}</pre>", status_code=400)

    refresh_token = tokens.get("refresh_token", "")
    if refresh_token:
        await save_refresh_token(refresh_token, scope="sp")
        print(f"[sp_callback] SP-API refresh token saved for user {user.get('email')}")

    return HTMLResponse(
        "<h2>SP-API Authorization successful!</h2>"
        f"<p>Refresh token has been saved on your account ({user.get('email')}).</p>"
        f"<pre>selling_partner_id: {selling_partner_id}</pre>"
        f"<pre>access_token (truncated): {tokens.get('access_token', '')[:20]}...</pre>"
    )


@app.get("/amazon/profiles")
async def list_profiles(user: dict = Depends(protect)):
    """List Amazon Advertising profiles for the authenticated account."""
    try:
        profiles = await get_profiles()
    except Exception as e:
        return {"error": str(e)}
    return profiles


@app.post("/amazon/profiles/{profile_id}/select")
async def select_profile(profile_id: str, user: dict = Depends(protect)):
    """Save the chosen profile ID onto the user's amazonAdsProfileIds."""
    await save_profile_id(profile_id)
    return {"ok": True, "profile_id": profile_id}


# ==========================================
# NEW ROUTE: AMAZON BRAND ANALYTICS
# ==========================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

class BrandAnalyticsRequest(BaseModel):
    keywords: List[str]
    start_date: str
    end_date: str
    period: str = "WEEK"
    marketplace: str | None = None

@app.post("/brand-analytics/keyword-matches")
async def get_keyword_matches(request: BrandAnalyticsRequest, user: dict = Depends(protect)):
    """
    Checks exact, phrase, and broad matches for target keywords using Amazon Brand Analytics.
    """
    try:
        results = await amazon_sp.process_brand_analytics_keywords(
            keywords=request.keywords,
            start_date=request.start_date,
            end_date=request.end_date,
            period=request.period,
            marketplace=request.marketplace
        )

        return {
            "status": "success",
            "data": results
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving brand analytics: {str(e)}"
        )


# ==========================================
# NEW ROUTE: AMAZON ADS SUGGESTED KEYWORDS
# ==========================================

class SuggestedKeywordsRequest(BaseModel):
    asin: str
    max_suggestions: int = 100


@app.post("/ads/suggested-keywords")
async def get_ads_suggested_keywords(
    request: SuggestedKeywordsRequest,
    user: dict = Depends(protect),
):
    """Amazon-recommended keywords for a given ASIN via the Sponsored Products API."""
    try:
        results = await fetch_suggested_keywords(
            asin=request.asin,
            max_suggestions=request.max_suggestions,
        )
        return {"status": "success", "data": results}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching suggested keywords: {str(e)}",
        )


# ==========================================
# NEW ROUTE: KEYWORD MATRIX JOURNEY
# ==========================================
# Async multi-step flow: source keywords from 3 paths (Amazon ASIN, Meta,
# Amazon Searchbar), enrich with Brand Analytics + CPC, score, and lay out a
# 3x3 (Top/Medium/Low × source) matrix. The endpoint kicks off a background
# job; the client polls the GET endpoint to render the stepper UI.
import keyword_matrix


class KeywordMatrixStartRequest(BaseModel):
    asins: List[str]
    ad_group_id: str | None = None
    campaign_id: str | None = None


@app.post("/keyword-matrix/start")
async def start_keyword_matrix(
    request: KeywordMatrixStartRequest,
    user: dict = Depends(protect),
):
    try:
        job_id = keyword_matrix.start_job(
            asins=request.asins,
            ad_group_id=request.ad_group_id,
            campaign_id=request.campaign_id,
        )
        return {"status": "started", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/keyword-matrix/{job_id}")
async def get_keyword_matrix(job_id: str, user: dict = Depends(protect)):
    job = keyword_matrix.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
