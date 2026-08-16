"""Body-size limit middleware — audit C4.

Pre-fix, /cogs/upload called `await request.body()` with no size limit,
so a hostile 10 GB body would be buffered into memory and take the
process down. The middleware rejects with HTTP 413 before the body is
read."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from main import BodySizeLimitMiddleware


def _make_app(max_bytes: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    async def echo(body: dict):
        return {"received_bytes": len(str(body))}

    return app


def test_request_under_cap_passes_through():
    client = TestClient(_make_app(max_bytes=1024))
    r = client.post("/echo", json={"hi": "there"})
    assert r.status_code == 200


def test_request_over_cap_is_rejected_with_413():
    client = TestClient(_make_app(max_bytes=100))
    # Content-Length header of 999 bytes should be rejected immediately.
    payload = "x" * 999
    r = client.post(
        "/echo",
        content=payload,
        headers={"content-type": "application/octet-stream"},
    )
    assert r.status_code == 413
    assert "exceeds" in r.json()["detail"]


def test_rejected_response_names_the_cap_size():
    client = TestClient(_make_app(max_bytes=100))
    r = client.post(
        "/echo",
        content="x" * 999,
        headers={"content-type": "application/octet-stream"},
    )
    assert "100 bytes" in r.json()["detail"]


def test_missing_content_length_is_not_rejected():
    """Streaming clients without Content-Length must fall through to
    the reverse-proxy cap — this middleware only enforces the length-
    known case (defense-in-depth)."""
    client = TestClient(_make_app(max_bytes=100))
    r = client.post("/echo", json={"hi": "there"})
    assert r.status_code == 200


def test_default_cap_is_50_mb():
    """The audit-followup Q&A picked 50 MB. Regression guard."""
    from main import _max_body_bytes
    assert _max_body_bytes() == 50 * 1024 * 1024
