"""WebSocket auth token resolution — audit H1.

Pre-fix, tokens were read from `websocket.query_params` first, which
means every WS connection ended up with the live JWT in server access
logs and browser history. Post-fix, header + subprotocol are preferred
and the query fallback logs a warning."""

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
async def test_query_string_still_works_as_deprecated_fallback(caplog):
    token = _valid_jwt()
    ws = _FakeWebSocket(query_params={"token": token})
    with caplog.at_level("WARNING", logger="auth.ws"):
        with patch("auth._load_user", return_value={"_id": "u", "email": "x@y"}):
            user, err = await auth.authenticate_ws(ws)
    assert err is None
    assert user["_token_source"] == "query"
    # Deprecation warning must be logged so ops can watch clients migrate off.
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
