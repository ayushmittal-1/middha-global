"""
Authentication — mirrors auroraBackend/src/middleware/auth.js.

A user logged in to Aurora can hit this backend with the same Bearer JWT:
we verify it against Aurora's JWT_SECRET and load the user from the same
MongoDB `users` collection. Per-user Amazon credentials are then pulled
from that user document instead of from .env.
"""

import os
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from bson import ObjectId
from fastapi import Depends, Header, HTTPException, WebSocket, status
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

MONGO_URI = os.getenv("MONGO_URI", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "")

_client: Optional[AsyncIOMotorClient] = None

# Threads the authenticated user through nested async calls (agent tools,
# Amazon SDK helpers) without changing every signature.
current_user: ContextVar[Optional[dict]] = ContextVar("current_user", default=None)


def _db() -> AsyncIOMotorDatabase:
    global _client
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI is not configured")
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    if MONGO_DB_NAME:
        return _client[MONGO_DB_NAME]
    return _client.get_default_database()


def _verify_token(token: str) -> dict:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET not configured")
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Not authorized, token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Not authorized, token failed")


async def _load_user(user_id: Optional[str]) -> dict:
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authorized, no user id in token")
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=401, detail="Not authorized, invalid user id")
    user = await _db().users.find_one({"_id": oid}, {"password": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Not authorized, user not found")
    return user


async def protect(authorization: Optional[str] = Header(default=None)) -> dict:
    """FastAPI dependency — equivalent to Aurora's `protect` middleware."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authorized, no token")
    token = authorization.split(" ", 1)[1]
    decoded = _verify_token(token)
    user = await _load_user(decoded.get("id"))
    # Stash the raw JWT so downstream code (e.g. calls to Aurora's REST API)
    # can reuse it without re-signing.
    user["_token"] = token
    current_user.set(user)
    return user


async def authenticate_ws(websocket: WebSocket) -> tuple[Optional[dict], Optional[str]]:
    """WebSocket auth — token comes from `?token=...` query param or the
    `Authorization: Bearer ...` header. Returns (user, error_message). On
    success, user is set and error_message is None; on failure, user is None
    and error_message describes which check failed."""
    token = websocket.query_params.get("token")
    if not token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
    if not token:
        return None, "No token provided. Pass ?token=<jwt> in the WebSocket URL."
    try:
        decoded = _verify_token(token)
    except HTTPException as e:
        return None, f"Token verification failed: {e.detail}"
    try:
        user = await _load_user(decoded.get("id"))
    except HTTPException as e:
        return None, f"User lookup failed: {e.detail}"
    user["_token"] = token
    current_user.set(user)
    return user, None


def authorize(*roles: str):
    """Equivalent of Aurora's `authorize(...roles)` — role-gating dependency."""
    async def _check(user: dict = Depends(protect)) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User role {user.get('role')} is not authorized to access this route",
            )
        return user
    return _check


def generate_token(user_id: str) -> str:
    """Mint a JWT identical in shape to Aurora's `generateToken` —
    HS256, payload `{id: <user_id>}`, 30-day expiry."""
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET not configured")
    now = datetime.now(timezone.utc)
    payload = {
        "id": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=30)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def authenticate_credentials(email: str, password: str) -> dict:
    """Verify an email+password against the shared Mongo users collection.

    Returns the user document (with password stripped) on success. Raises
    401 on bad credentials, matching Aurora's `login` controller behavior.
    """
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    # Aurora's User schema lowercases emails on save; match that here.
    user = await _db().users.find_one({"email": email.lower().strip()})
    if not user or not user.get("password"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8"))
    except (ValueError, TypeError):
        ok = False
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user.pop("password", None)
    return user


def require_user() -> dict:
    """Look up the current authenticated user from the ContextVar.

    Used by helper modules (amazon_ads, amazon_sp) that don't take a user
    argument directly — the FastAPI request handler / WS handler must have
    already set the ContextVar via `protect` / `authenticate_ws`."""
    user = current_user.get()
    if user is None:
        raise RuntimeError(
            "No authenticated user in context. Did you forget to add "
            "Depends(protect) to the endpoint, or to authenticate the WebSocket?"
        )
    return user
