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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from campaigns import fetch_all_campaigns
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
)
from amazon_ads import (
    exchange_auth_code,
    get_profiles,
    save_refresh_token,
    save_profile_id,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB and fetch campaigns
    await init_db()
    print("Database initialized.")
    print("Fetching campaigns from Aurora API...")
    try:
        await fetch_all_campaigns()
        print("Campaigns loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to fetch campaigns: {e}")
    yield
    # Shutdown: close shared Playwright browser
    await shutdown_browser()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
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
async def chat(request: Request):
    """Legacy SSE endpoint (kept for curl / debug use)."""
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", "default")

    if not message.strip():
        return {"error": "Message is required"}

    async def event_stream():
        async for event in stream_response(message, session_id=session_id):
            if event["type"] == "token":
                yield f"data: {json.dumps({'content': event['content']})}\n\n"
            elif event["type"] == "error":
                yield f"data: {json.dumps({'error': event['content']})}\n\n"
            # tool_start / tool_result are silently skipped in SSE mode
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session endpoints ──────────────────────────────────────────────────────

@app.post("/sessions")
async def create_new_session():
    session_id = str(uuid.uuid4())
    await create_session(session_id)
    return {"session_id": session_id}


@app.get("/sessions")
async def get_sessions():
    sessions = await list_sessions()
    return sessions


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    messages = await get_messages(session_id)
    # Filter to only user/assistant messages for the frontend
    return [m for m in messages if m.get("role") in ("user", "assistant")]


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    await delete_session(session_id)
    return {"ok": True}


# ── COGS endpoints ─────────────────────────────────────────────────────────

@app.post("/cogs/upload")
async def upload_cogs(request: Request):
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
async def list_cogs():
    rows = await get_cogs()
    return {"count": len(rows), "rows": rows}


# ── Amazon OAuth endpoints ─────────────────────────────────────────────────

AMAZON_AUTH_URL = "https://www.amazon.com/ap/oa"
DEFAULT_REDIRECT_URI = "https://d24d-2409-4085-e8c-b458-f90e-780c-de78-1437.ngrok-free.app/amazon/sp-callback"


@app.get("/amazon/login")
async def amazon_login(redirect_uri: str | None = None):
    """Redirect user to Amazon's OAuth consent page."""
    ru = redirect_uri or DEFAULT_REDIRECT_URI
    params = urlencode({
        "client_id": os.getenv("AMAZON_LWA_CLIENT_ID", ""),
        # advertising::campaign_management is required to create/manage campaigns
        # via the Amazon Ads API. profile:user_id alone returns 401/403 on Ads calls.
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
    code: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle Amazon OAuth callback — exchange code for refresh token.

    Registered under both our own /callback and Aurora's
    /api/auth/amazon/ads-callback path so either Allowed Return URL works.
    """
    # Amazon redirects here with ?error=... when consent fails or is declined.
    if error or not code:
        detail = error_description or error or "No authorization code was provided."
        print(f"[oauth_callback] no code. error={error!r} description={error_description!r}")
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{detail}</p>"
            "<p>Start the flow again from <code>/amazon/login</code>.</p>",
            status_code=400,
        )

    # Reconstruct the exact redirect URI used during authorization. Force https
    # because behind ngrok the request scheme is often http, and Amazon requires
    # the redirect_uri in the token exchange to match the one used at consent.
    redirect_uri = f"https://{request.url.netloc}{request.url.path}"
    try:
        tokens = await exchange_auth_code(code, redirect_uri)
    except Exception as e:
        return HTMLResponse(f"<h2>OAuth Error</h2><pre>{e}</pre>", status_code=400)

    refresh_token = tokens.get("refresh_token", "")
    if refresh_token:
        save_refresh_token(refresh_token)

    return HTMLResponse(
        "<h2>Authorization successful!</h2>"
        "<p>Refresh token has been saved. You can close this window.</p>"
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
    spapi_oauth_code: str | None = None,
    state: str | None = None,
    selling_partner_id: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle SP-API OAuth callback — exchange code for refresh token."""
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
        save_refresh_token(refresh_token)
        print(f"[sp_callback] SP-API refresh token saved ({len(refresh_token)} chars)")

    return HTMLResponse(
        "<h2>SP-API Authorization successful!</h2>"
        "<p>Refresh token has been saved. You can close this window.</p>"
        f"<pre>selling_partner_id: {selling_partner_id}</pre>"
        f"<pre>access_token (truncated): {tokens.get('access_token', '')[:20]}...</pre>"
    )


@app.get("/amazon/profiles")
async def list_profiles():
    """List Amazon Advertising profiles for the authenticated account."""
    try:
        profiles = await get_profiles()
    except Exception as e:
        return {"error": str(e)}
    return profiles


@app.post("/amazon/profiles/{profile_id}/select")
async def select_profile(profile_id: str):
    """Save the chosen profile ID to .env."""
    save_profile_id(profile_id)
    return {"ok": True, "profile_id": profile_id}


# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
