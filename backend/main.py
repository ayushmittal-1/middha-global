import csv
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

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
    get_forecast_settings,
    update_forecast_settings,
    get_forecast_cache,
)
from forecasting.ingest import (
    backfill_user,
    ingest_user_incremental,
    run_nightly_ingest,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta, timezone

import amazon_sp
from agent import compute_profitability_data
from campaigns import analyze_performance_data
from amazon_ads import (
    exchange_auth_code,
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
    """Refit forecasts + reorder math for every SKU with history. Reads
    from salesDaily/inventorySnapshot (no SP-API calls)."""
    from bson import ObjectId as _OID
    from forecasting.model import refresh_forecasts_for_user
    current_user.set(user)
    return await refresh_forecasts_for_user(_OID(str(user["_id"])))


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
    cached = await get_forecast_cache()
    rows = []
    for c in cached:
        reorder = c.get("reorder") or {}
        forecast = c.get("forecast") or []
        next30 = sum(float(r.get("p50", 0)) for r in forecast[:30])
        rows.append({
            "sku": c["sku"],
            "method": c.get("method"),
            "generated_at": c.get("generated_at").isoformat() if c.get("generated_at") else None,
            "on_hand": reorder.get("on_hand", 0),
            "inbound": reorder.get("inbound", 0),
            "next_30_day_forecast": round(next30, 1),
            "days_of_cover": reorder.get("days_of_cover"),
            "reorder_by_date": reorder.get("reorder_by_date"),
            "recommended_po_qty": reorder.get("recommended_po_qty", 0),
            "drivers": c.get("drivers"),
        })
    # Sort: stockouts first (None or low days_of_cover), then ascending.
    def _sort_key(r):
        d = r.get("days_of_cover")
        return (1, 0) if d is None else (0, d)
    rows.sort(key=_sort_key)
    return {"count": len(rows), "rows": rows}


@app.get("/forecasting/sku/{sku}")
async def forecasting_sku_detail(sku: str, user: dict = Depends(protect)):
    """Full forecast (90-day p50/p90) for one SKU — drives the detail chart."""
    cached = await get_forecast_cache(skus=[sku])
    if not cached:
        return {"error": f"No forecast for {sku}. Run /forecasting/refresh first."}
    c = cached[0]
    return {
        "sku": sku,
        "method": c.get("method"),
        "generated_at": c.get("generated_at").isoformat() if c.get("generated_at") else None,
        "horizon_days": c.get("horizon_days"),
        "drivers": c.get("drivers"),
        "reorder": c.get("reorder"),
        "forecast": c.get("forecast"),
    }


# ── Amazon SP-API data endpoints (for FE tables) ───────────────────────────


@app.get("/amazon/marketplaces")
async def amazon_marketplaces(user: dict = Depends(protect)):
    """Marketplaces the user is registered in. Feeds the Orders tab selector."""
    return {"marketplaces": amazon_sp.list_marketplaces()}


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

    data = await amazon_sp.get_orders(
        created_after=created_after,
        created_before=created_before,
        statuses=statuses,
        max_results=100,
        marketplace=marketplace,
        paginate=True,
    )
    orders = (data.get("payload") or {}).get("Orders") or []
    if buyer_email:
        needle = buyer_email.lower()
        orders = [
            o for o in orders
            if needle in (o.get("BuyerInfo", {}).get("BuyerEmail") or "").lower()
        ]
    return {
        "count": len(orders),
        "created_after": created_after,
        "created_before": created_before,
        "orders": orders,
    }


@app.get("/amazon/orders/{order_id}/items")
async def amazon_order_items(order_id: str, user: dict = Depends(protect)):
    """Line items for a single order — used to expand a row in the FE table."""
    data = await amazon_sp.get_order_items(order_id)
    return {"items": (data.get("payload") or {}).get("OrderItems") or []}


@app.get("/campaigns/performance")
async def campaigns_performance(user: dict = Depends(protect)):
    """Structured campaign performance — feeds the Campaigns tab."""
    return await analyze_performance_data(full=True)


@app.get("/profitability")
async def profitability(days_back: int = 7, user: dict = Depends(protect)):
    """Per-SKU profitability for the requested window. Walks SP-API
    NextToken so the FE sees the whole window."""
    return await compute_profitability_data(days_back=days_back, paginate=True)


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
async def sp_api_login(redirect_uri: str | None = None):
    """Redirect user to Seller Central to authorize the SP-API app."""
    ru = redirect_uri or DEFAULT_REDIRECT_URI
    params = urlencode({
        "application_id": SP_API_APP_ID,
        "redirect_uri": ru,
        "state": "sp_api_auth",
    })
    print(f"[sp_login] Redirecting to Seller Central consent: {SP_API_AUTH_URL}?{params}")
    return RedirectResponse(f"{SP_API_AUTH_URL}?{params}")


@app.get("/amazon/sp-callback")
async def sp_api_callback(
    request: Request,
    user: dict = Depends(protect),
    spapi_oauth_code: str | None = None,
    state: str | None = None,
    selling_partner_id: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle SP-API OAuth callback — exchange code for refresh token and
    persist it as the authenticated user's amazonRefreshToken."""
    if error or not spapi_oauth_code:
        detail = error_description or error or "No authorization code was provided."
        print(f"[sp_callback] no code. error={error!r} description={error_description!r}")
        return HTMLResponse(
            f"<h2>SP-API OAuth Error</h2><p>{detail}</p>"
            "<p>Start the flow again from <code>/amazon/sp-login</code>.</p>",
            status_code=400,
        )

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


# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
