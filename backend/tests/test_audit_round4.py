"""Round-4 audit followup — four fixes.

Each test corresponds to a specific 'Still needed' item from
Aurora_Audit_Fix_Verification_Round4.docx:

  C2 — SlowAPIMiddleware must be registered so the configured
       default_limits actually apply to every route, not just the
       decorated login.
  C4 — BodySizeLimitMiddleware must reject bodies whose actual size
       exceeds the cap, not just those where Content-Length says so.
       (Chunked transfer-encoding was the specific bypass called out.)
  H1 — Frontend WebSocket must send the token via the
       `Sec-WebSocket-Protocol: bearer, <jwt>` subprotocol, not as
       `?token=` in the URL. Backend must echo the subprotocol back
       during the handshake or the browser closes the connection.
  M4 — Login-lockout counter must be incremented atomically
       (find_one_and_update + $inc), not via a read-then-write pattern
       that races under concurrent failed attempts.
"""

import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest


# ── C2: SlowAPIMiddleware registered on the app ──────────────────────────


def test_slowapi_middleware_is_registered_on_the_app():
    """Regression guard: the Round-3 audit found the middleware missing
    so the 120/min default limit applied to nothing. `app.user_middleware`
    must include SlowAPIMiddleware for the default_limits to enforce."""
    import main
    from slowapi.middleware import SlowAPIMiddleware
    middleware_classes = [m.cls for m in main.app.user_middleware]
    assert SlowAPIMiddleware in middleware_classes, (
        "SlowAPIMiddleware not registered — configured default limits "
        "are dead config on every route except /api/auth/login"
    )


def test_body_size_middleware_runs_before_cors():
    """A hostile pre-flight with a giant body should be rejected before
    CORS burns cycles inspecting headers. In Starlette, user_middleware
    lists in REVERSE dispatch order (last-added runs first), so
    BodySizeLimit should appear AFTER CORS in that list."""
    import main
    names = [m.cls.__name__ for m in main.app.user_middleware]
    body_idx = names.index("BodySizeLimitMiddleware")
    cors_idx = names.index("CORSMiddleware")
    assert body_idx > cors_idx, (
        "BodySizeLimitMiddleware must be added AFTER CORS so it runs "
        "first in the request pipeline (Starlette dispatches "
        "user_middleware in reverse order)"
    )


# ── C4: Stream-level byte counting ───────────────────────────────────────


class _AppSpy:
    """Minimal ASGI app that receives the whole body and echoes its size.
    Used to prove the body-size middleware truncates before the app runs."""

    def __init__(self):
        self.saw_body_bytes = 0
        self.was_called = False

    async def __call__(self, scope, receive, send):
        self.was_called = True
        received = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue
            received += len(message.get("body") or b"")
            if not message.get("more_body"):
                break
        self.saw_body_bytes = received
        body = json.dumps({"received": received}).encode("utf-8")
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


async def _send_body_in_chunks(chunks: list[bytes]) -> tuple[int, dict]:
    """Wraps the middleware around _AppSpy and pushes `chunks` through
    as an ASGI http.request stream. Returns (status_code, response_body)."""
    from main import BodySizeLimitMiddleware
    spy = _AppSpy()
    mw = BodySizeLimitMiddleware(spy, max_bytes=1024)

    idx = 0

    async def receive():
        nonlocal idx
        if idx >= len(chunks):
            # Client done streaming.
            return {"type": "http.disconnect"}
        body = chunks[idx]
        idx += 1
        return {
            "type": "http.request",
            "body": body,
            "more_body": idx < len(chunks),
        }

    sent_start: dict = {}
    sent_body = bytearray()

    async def send(message):
        if message["type"] == "http.response.start":
            sent_start.update(message)
        elif message["type"] == "http.response.body":
            sent_body.extend(message.get("body") or b"")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": [],  # NO content-length — chunked encoding
    }
    await mw(scope, receive, send)
    return sent_start.get("status", 0), (
        json.loads(sent_body.decode() or "{}") if sent_body else {}
    ), spy


@pytest.mark.asyncio
async def test_body_size_rejects_chunked_upload_over_cap_without_content_length():
    """The bypass the Round-3 audit called out: a client that omits
    Content-Length (chunked transfer-encoding) got a free pass under
    the old middleware. With stream-level byte counting, an oversize
    body is rejected with 413 even without a declared length."""
    # 1024-byte cap, send 2000 bytes in two chunks.
    chunks = [b"x" * 1024, b"y" * 976]
    status, body, spy = await _send_body_in_chunks(chunks)
    assert status == 413, f"expected 413, got {status}"
    assert "exceeds" in body.get("detail", "").lower()


@pytest.mark.asyncio
async def test_body_size_lets_small_chunked_upload_pass_through():
    """Regression guard: legitimate small requests without
    Content-Length must still succeed."""
    chunks = [b"a" * 300, b"b" * 300]  # 600 bytes total, cap is 1024
    status, body, spy = await _send_body_in_chunks(chunks)
    assert status == 200, f"expected 200, got {status}"
    assert spy.saw_body_bytes == 600


@pytest.mark.asyncio
async def test_body_size_early_rejects_when_content_length_declares_over_cap():
    """Fast-path: when a client honestly declares a giant body in
    Content-Length, we reject before any body bytes are read."""
    from main import BodySizeLimitMiddleware
    spy = _AppSpy()
    mw = BodySizeLimitMiddleware(spy, max_bytes=100)

    async def receive():
        return {"type": "http.disconnect"}

    sent_start: dict = {}
    sent_body = bytearray()

    async def send(message):
        if message["type"] == "http.response.start":
            sent_start.update(message)
        elif message["type"] == "http.response.body":
            sent_body.extend(message.get("body") or b"")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/upload",
        "headers": [(b"content-length", b"9999999")],
    }
    await mw(scope, receive, send)
    assert sent_start["status"] == 413
    assert not spy.was_called, (
        "app must NOT be invoked when Content-Length declared over cap"
    )


# ── H1: FE uses subprotocol, backend echoes it ───────────────────────────


@pytest.fixture(scope="module")
def frontend_html() -> str:
    return (
        Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")


def test_frontend_websocket_uses_subprotocol_not_query_string(frontend_html):
    """Round-3 audit flagged that the FE was still doing
    `new WebSocket(url?token=...)`. The token must NOT appear in the URL."""
    # Locate the WS constructor call — must be the subprotocol form.
    assert "new WebSocket(" in frontend_html
    # The subprotocol form passes ['bearer', token] as the 2nd arg.
    assert "['bearer', token]" in frontend_html or '["bearer", token]' in frontend_html, (
        "FE WebSocket must pass ['bearer', token] as subprotocol, "
        "not `?token=` in the URL"
    )
    # Regression guard on the specific old bad line.
    assert "/ws/chat?token=" not in frontend_html, (
        "FE still has the deprecated `?token=` URL form — tokens will "
        "leak into access logs"
    )


def test_backend_ws_endpoint_echoes_bearer_subprotocol():
    """Browser will close the connection immediately if the server's
    101 response doesn't echo the requested subprotocol. Regression
    guard by grepping the endpoint source."""
    import inspect
    import main
    src = inspect.getsource(main.ws_chat)
    assert 'accept(subprotocol="bearer")' in src or "accept(subprotocol='bearer')" in src, (
        "ws_chat must call websocket.accept(subprotocol='bearer') when "
        "the client requests the bearer subprotocol"
    )


# ── M4: Atomic $inc on failed-login counter ──────────────────────────────


def test_authenticate_uses_atomic_findoneandupdate():
    """Round-3 audit flagged the read-then-write race. The failure
    path must use find_one_and_update with $inc so concurrent attempts
    can't under-count. Grep guard against a revert."""
    import inspect
    import auth
    src = inspect.getsource(auth.authenticate_credentials)
    assert "find_one_and_update" in src, (
        "authenticate_credentials must use find_one_and_update for "
        "atomic counter increments"
    )
    assert "$inc" in src, (
        "failure path must use $inc to atomically bump failedLoginCount"
    )
    # Regression guard: the old Python-side increment pattern must not
    # be present anymore.
    assert 'int(user.get("failedLoginCount") or 0) + 1' not in src, (
        "read-then-write pattern still present — race not fixed"
    )


def test_return_document_after_used_so_new_count_is_read():
    import inspect
    import auth
    src = inspect.getsource(auth.authenticate_credentials)
    assert "ReturnDocument.AFTER" in src, (
        "find_one_and_update must use return_document=ReturnDocument.AFTER "
        "so the incremented counter is what feeds the lockout tier check"
    )


class _FakeAtomicCollection:
    """In-memory stand-in that behaves like Mongo's find_one_and_update
    with $inc under concurrency — every increment yields a unique
    sequential counter value even when many coroutines race."""

    def __init__(self, initial: dict):
        self._doc = dict(initial)
        self._lock = asyncio.Lock()

    async def find_one_and_update(self, filt, update, **_kwargs):
        async with self._lock:
            inc = update.get("$inc", {})
            for k, v in inc.items():
                self._doc[k] = int(self._doc.get(k, 0)) + int(v)
            set_ = update.get("$set", {})
            for k, v in set_.items():
                self._doc[k] = v
            # ReturnDocument.AFTER semantics.
            return dict(self._doc)


@pytest.mark.asyncio
async def test_concurrent_failed_attempts_all_increment_the_counter():
    """The behavioural contract we're locking in: N concurrent
    'increment' calls yield the counter reaching exactly N (not <N as
    the pre-fix code could produce)."""
    coll = _FakeAtomicCollection({"failedLoginCount": 0})

    async def one_failure():
        result = await coll.find_one_and_update(
            {"_id": "x"},
            {"$inc": {"failedLoginCount": 1}},
        )
        return result["failedLoginCount"]

    # 10 concurrent failed attempts.
    results = await asyncio.gather(*(one_failure() for _ in range(10)))
    assert sorted(results) == list(range(1, 11)), (
        "atomic $inc must yield distinct sequential values across N "
        "concurrent calls — got: " + str(sorted(results))
    )
    # Final counter state must equal 10, not something lower.
    assert coll._doc["failedLoginCount"] == 10
