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
from database import init_db, create_session, list_sessions, get_messages, delete_session
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


# ── Amazon OAuth endpoints ─────────────────────────────────────────────────

AMAZON_AUTH_URL = "https://www.amazon.com/ap/oa"
DEFAULT_REDIRECT_URI = "https://powdered-tragedy-crafty.ngrok-free.dev/api/auth/amazon/ads-callback"


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
