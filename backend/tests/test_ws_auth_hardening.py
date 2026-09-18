"""WebSocket auth token resolution — audit H1 / H-03.

Tokens must not be read from `websocket.query_params` (access logs /
browser history). Header + Sec-WebSocket-Protocol are the only accepted
sources; query-string attempts are rejected."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from unittest.mock import patch

import jwt
import pytest

import auth


# `auth.JWT_SECRET` is read from the env at IMPORT time. In a full-suite
# run, another test file may have imported `auth` first without a
# secret set, freezing `auth.JWT_SECRET = ""`. Patch the module-level
# constant directly so these tests are independent of import order.
_TEST_JWT_SECRET = "test-secret-for-ws-tests-please-make-me-long-enough"


@pytest.fixture(autouse=True)
def _pin_jwt_secret(monkeypatch):
    monkeypatch.setattr(auth, "JWT_SECRET", _TEST_JWT_SECRET)


def _valid_jwt() -> str:
    return jwt.encode({"id": "507f1f77bcf86cd799439011"}, _TEST_JWT_SECRET, "HS256")


class _FakeWebSocket:
    """Minimal stand-in — real FastAPI WebSocket isn't easy to construct
    in a unit test, but authenticate_ws only touches headers and
    query_params."""

    def __init__(self, *, headers=None, query_params=None):
        class _MutableMapping(dict):
            def get(self, key, default=None):  # type: ignore[override]
                return dict.get(self, key.lower(), default)

        self.headers = _MutableMapping((k.lower(), v) for k, v in (headers or {}).items())
        self.query_params = query_params or {}


@pytest.mark.asyncio
async def test_authorization_header_is_preferred_over_query_string(caplog):
    token = _valid_jwt()
    ws = _FakeWebSocket(
        headers={"Authorization": f"Bearer {token}"},
        query_params={"token": "STALE-DO-NOT-USE"},
    )
    with patch("auth._load_user", return_value={"_id": "u", "email": "x@y"}):
        user, err = await auth.authenticate_ws(ws)
    assert err is None
    assert user is not None
    assert user["_token"] == token
    assert user["_token_source"] == "header"


@pytest.mark.asyncio
async def test_subprotocol_bearer_is_accepted():
    token = _valid_jwt()
    ws = _FakeWebSocket(
        headers={"sec-websocket-protocol": f"bearer, {token}"},
    )
    with patch("auth._load_user", return_value={"_id": "u", "email": "x@y"}):
        user, err = await auth.authenticate_ws(ws)
    assert err is None
    assert user["_token_source"] == "subprotocol"


@pytest.mark.asyncio
async def test_query_string_token_is_rejected(caplog):
    token = _valid_jwt()
    ws = _FakeWebSocket(query_params={"token": token})
    with caplog.at_level("WARNING", logger="auth.ws"):
        user, err = await auth.authenticate_ws(ws)
    assert user is None
    assert err is not None
    assert "query-string" in err.lower() or "Query-string" in err
    assert any("query-string" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_missing_token_returns_clear_error():
    ws = _FakeWebSocket()
    user, err = await auth.authenticate_ws(ws)
    assert user is None
    assert err is not None
    assert "Authorization" in err or "authorization" in err.lower()


@pytest.mark.asyncio
async def test_invalid_token_is_rejected():
    ws = _FakeWebSocket(headers={"Authorization": "Bearer not-a-real-jwt"})
    user, err = await auth.authenticate_ws(ws)
    assert user is None
    assert err is not None
    assert "verification failed" in err.lower()


def test_aurora_embed_audience_token_verifies():
    """Aurora signAiEmbedToken sets aud=aurora-ai-embed; must not 401."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "id": "507f1f77bcf86cd799439011",
            "type": "ai_embed",
            "iat": now,
            "exp": now + timedelta(minutes=15),
            "iss": "aurora-backend",
            "aud": "aurora-ai-embed",
        },
        _TEST_JWT_SECRET,
        algorithm="HS256",
    )
    decoded = auth._verify_token(token)
    assert decoded["id"] == "507f1f77bcf86cd799439011"
    assert decoded["aud"] == "aurora-ai-embed"


def test_wrong_audience_is_rejected():
    from datetime import datetime, timedelta, timezone
    from fastapi import HTTPException

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "id": "507f1f77bcf86cd799439011",
            "exp": now + timedelta(minutes=15),
            "aud": "not-a-valid-audience",
        },
        _TEST_JWT_SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        auth._verify_token(token)
    assert exc.value.status_code == 401
    assert "token failed" in exc.value.detail


def test_embed_token_wrong_issuer_is_rejected():
    from datetime import datetime, timedelta, timezone
    from fastapi import HTTPException

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "id": "507f1f77bcf86cd799439011",
            "type": "ai_embed",
            "exp": now + timedelta(minutes=15),
            "iss": "evil-issuer",
            "aud": "aurora-ai-embed",
        },
        _TEST_JWT_SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        auth._verify_token(token)
    assert exc.value.status_code == 401


def test_embed_token_missing_type_is_rejected():
    from datetime import datetime, timedelta, timezone
    from fastapi import HTTPException

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "id": "507f1f77bcf86cd799439011",
            "exp": now + timedelta(minutes=15),
            "iss": "aurora-backend",
            "aud": "aurora-ai-embed",
        },
        _TEST_JWT_SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        auth._verify_token(token)
    assert exc.value.status_code == 401
