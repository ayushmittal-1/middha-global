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
from pymongo import ReturnDocument

from token_encryption import hydrate_user_tokens

MONGO_URI = os.getenv("MONGO_URI", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
# Aurora's mongoose connection uses the `test` database when the URI has no path.
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "test")

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
    return _client[MONGO_DB_NAME or "test"]


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
    return hydrate_user_tokens(user)


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
    """WebSocket auth. Token resolution order (audit H1):
      1. `Authorization: Bearer <jwt>` header  — preferred; not logged.
      2. `Sec-WebSocket-Protocol: bearer, <jwt>` subprotocol — preferred
         for browsers, since the JS WebSocket API can't set arbitrary
         headers but does support subprotocols.
      3. `?token=<jwt>` query string — DISCOURAGED. Query strings are
         written to server access logs, reverse-proxy logs, and browser
         history, so a leaked log file leaks a live session token. Kept
         for backward compatibility; emits a warning log per use so ops
         can watch clients migrate off it.

    Returns (user, error_message). On success user is set; on failure
    it's None and error_message describes which check failed."""
    import logging
    _log = logging.getLogger("auth.ws")

    token: Optional[str] = None
    token_source = ""
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        token_source = "header"
    if not token:
        # Sec-WebSocket-Protocol comes in as a comma-separated list; the
        # client sends ["bearer", <jwt>] as two protocols.
        proto_header = websocket.headers.get("sec-websocket-protocol", "")
        parts = [p.strip() for p in proto_header.split(",") if p.strip()]
        if len(parts) >= 2 and parts[0].lower() == "bearer":
            token = parts[1]
            token_source = "subprotocol"
    if not token:
        # Query-string fallback (audit H1: discouraged path).
        token = websocket.query_params.get("token")
        if token:
            token_source = "query"
            _log.warning(
                "WebSocket auth via query-string token — this ends up in "
                "access logs. Migrate the client to the Authorization "
                "header or Sec-WebSocket-Protocol subprotocol."
            )
    if not token:
        return None, (
            "No token provided. Send Authorization: Bearer <jwt> or the "
            "Sec-WebSocket-Protocol subprotocol (query-string fallback "
            "is deprecated)."
        )
    try:
        decoded = _verify_token(token)
    except HTTPException as e:
        return None, f"Token verification failed: {e.detail}"
    try:
        user = await _load_user(decoded.get("id"))
    except HTTPException as e:
        return None, f"User lookup failed: {e.detail}"
    user["_token"] = token
    user["_token_source"] = token_source
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


# ── Login lockout (audit M4) ─────────────────────────────────────────────
# Progressive backoff on repeated failed logins per account. Defense-in-
# depth on top of the IP-based rate limiter (audit C2): a rotating-IP
# attacker still trips the per-account counter, and a legitimate user
# fat-fingering their password five times gets a short lock, not a total
# ban.

# (failures_before_this_lock, lock_minutes) — applied in order.
_LOGIN_LOCKOUT_TIERS = (
    (5,  5),      # 5 failures  →  5-minute lock
    (10, 60),     # 10 failures → 1-hour lock
    (15, 24 * 60),  # 15 failures → 24-hour lock
)


def compute_login_lockout_seconds(
    failed_count: int, now: Optional[datetime] = None,
) -> int:
    """Given the current failure counter, return how many seconds the
    account should be locked out for after THIS failed attempt.

    Pure function so the tiering rules are unit-testable without any
    Mongo mocking. `failed_count` is the counter AFTER incrementing for
    the current failure (i.e. the 5th failure passes failed_count=5)."""
    duration_min = 0
    for threshold, minutes in _LOGIN_LOCKOUT_TIERS:
        if failed_count >= threshold:
            duration_min = minutes
    return duration_min * 60


def _is_locked_out(user: dict, now: datetime) -> Optional[datetime]:
    """Return the lockout expiry timestamp if the account is currently
    locked, else None. Handles legacy user docs that lack the field."""
    locked_until = user.get("loginLockedUntil")
    if not isinstance(locked_until, datetime):
        return None
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until if locked_until > now else None


async def authenticate_credentials(email: str, password: str) -> dict:
    """Verify an email+password against the shared Mongo users collection.

    Returns the user document (with password stripped) on success. Raises
    401 on bad credentials, matching Aurora's `login` controller behavior.
    Additionally (audit M4): locked-out accounts return 429 with the
    remaining lockout window instead of running bcrypt at all, and
    failure/success both persist counter state to the user document."""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    # Aurora's User schema lowercases emails on save; match that here.
    users = _db().users
    user = await users.find_one({"email": email.lower().strip()})
    if not user or not user.get("password"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    now = datetime.now(timezone.utc)
    locked_until = _is_locked_out(user, now)
    if locked_until is not None:
        retry_after = max(1, int((locked_until - now).total_seconds()))
        raise HTTPException(
            status_code=429,
            detail=(
                "Account temporarily locked due to repeated failed "
                f"logins. Try again in {retry_after}s."
            ),
            headers={"Retry-After": str(retry_after)},
        )
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8"))
    except (ValueError, TypeError):
        ok = False
    if not ok:
        # Round-4-audit followup on M4: the previous read-then-write
        # pattern (find_one → increment in Python → update_one) had a
        # race — two concurrent failed attempts both read count=4,
        # both wrote count=5, and the true count silently landed at 5
        # instead of 6. Under credential-stuffing that could keep an
        # attacker's parallel attempts under the lockout threshold
        # indefinitely. `find_one_and_update` with `$inc` is atomic at
        # the Mongo level — N concurrent failures produce distinct
        # sequential counter values.
        try:
            updated = await users.find_one_and_update(
                {"_id": user["_id"]},
                {
                    "$inc": {"failedLoginCount": 1},
                    "$set": {"lastFailedLoginAt": now},
                },
                projection={"failedLoginCount": 1},
                return_document=ReturnDocument.AFTER,
            )
            new_count = int((updated or {}).get("failedLoginCount") or 0)
            lockout_seconds = compute_login_lockout_seconds(new_count, now)
            if lockout_seconds:
                # Second write is fine — the counter is already atomic.
                # This just stamps the lockout expiry for this attempt.
                await users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"loginLockedUntil": now + timedelta(seconds=lockout_seconds)}},
                )
        except Exception:
            # Persistence failure must not turn into a 500 on the login
            # path — the user still gets a 401 for wrong password.
            pass
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Success — reset the lockout counter atomically.
    try:
        await users.update_one(
            {"_id": user["_id"]},
            {"$set": {"failedLoginCount": 0, "loginLockedUntil": None,
                      "lastSuccessfulLoginAt": now}},
        )
    except Exception:
        pass
    user.pop("password", None)
    return hydrate_user_tokens(user)


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
