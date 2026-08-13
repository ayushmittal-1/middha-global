"""Bounded-concurrency SP-API GetOrderItems fetch (review issue #3).

Was strictly serial with a 2.1s sleep per call — a 60-order window took
~60 s and easily blew past the function's 55 s critical deadline. Now
uses an asyncio.Semaphore(concurrency) so N calls run in parallel while
per-slot pacing keeps sustained rate within SP-API's quota. These tests
mock `amazon_sp.get_order_items` so nothing hits the network."""

import asyncio
import os
import time

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import agent


class _FakeSpApi:
    """Records concurrency + total call count for a fake get_order_items."""

    def __init__(self, per_call_delay: float = 0.05, error_ids: set | None = None):
        self.per_call_delay = per_call_delay
        self.error_ids = error_ids or set()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.calls: list[str] = []
        self._lock = asyncio.Lock()

    async def get_order_items(self, oid: str):
        async with self._lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.calls.append(oid)
        try:
            await asyncio.sleep(self.per_call_delay)
            if oid in self.error_ids:
                raise RuntimeError(f"boom-{oid}")
            return {"AmazonOrderId": oid, "OrderItems": []}
        finally:
            async with self._lock:
                self.in_flight -= 1


# Capture the real asyncio.sleep BEFORE anyone patches it, so the fast
# stand-in can yield without recursing into itself.
_real_sleep = asyncio.sleep


async def _fast_sleep(_seconds):
    # Just yield once so asyncio can schedule other coroutines — but
    # never actually sleep, so tests stay under a second.
    await _real_sleep(0)


@pytest.fixture
def fake_sp(monkeypatch):
    fake = _FakeSpApi()

    async def _proxy(oid):
        return await fake.get_order_items(oid)

    monkeypatch.setattr(agent.amazon_sp, "get_order_items", _proxy)
    # Neutralize the retry cool-down and per-slot pacing so tests stay
    # fast — we're asserting concurrency shape, not the pacing sleep.
    monkeypatch.setattr(agent.asyncio, "sleep", _fast_sleep)
    return fake


@pytest.mark.asyncio
async def test_all_order_ids_get_a_result_tuple(fake_sp):
    order_ids = [f"O-{i}" for i in range(20)]
    results = await agent._fetch_order_items_paced(order_ids, concurrency=5)
    assert len(results) == 20
    assert {oid for oid, _ in results} == set(order_ids)


@pytest.mark.asyncio
async def test_calls_run_in_parallel_up_to_concurrency_limit(fake_sp):
    order_ids = [f"O-{i}" for i in range(20)]
    await agent._fetch_order_items_paced(order_ids, concurrency=5)
    # Regression guard: the pre-fix serial version had peak_in_flight == 1.
    assert fake_sp.peak_in_flight >= 2, (
        "concurrent execution not observed — regression to serial fetch"
    )
    assert fake_sp.peak_in_flight <= 5, (
        f"concurrency limit violated: saw {fake_sp.peak_in_flight} in flight"
    )


@pytest.mark.asyncio
async def test_concurrency_of_one_matches_serial_behavior(fake_sp):
    order_ids = [f"O-{i}" for i in range(5)]
    await agent._fetch_order_items_paced(order_ids, concurrency=1)
    assert fake_sp.peak_in_flight == 1


@pytest.mark.asyncio
async def test_non_retryable_error_is_recorded_and_does_not_stop_others(monkeypatch):
    fake = _FakeSpApi(error_ids={"O-2"})

    async def _proxy(oid):
        return await fake.get_order_items(oid)

    monkeypatch.setattr(agent.amazon_sp, "get_order_items", _proxy)
    monkeypatch.setattr(agent.asyncio, "sleep", _fast_sleep)  # captured pre-patch above

    results = await agent._fetch_order_items_paced(
        [f"O-{i}" for i in range(5)], concurrency=3,
    )
    by_id = dict(results)
    assert "_error" in by_id["O-2"]
    # Other orders succeeded.
    for good in ("O-0", "O-1", "O-3", "O-4"):
        assert "OrderItems" in by_id[good]


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list(fake_sp):
    assert await agent._fetch_order_items_paced([], concurrency=5) == []
