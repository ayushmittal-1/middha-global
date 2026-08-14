"""Bounded browser concurrency for Meta Ads scraper — audit H6.

Pre-fix, the shared Chromium instance had no cap on how many contexts
could open against it simultaneously — an unbounded burst under load
could OOM the container. The semaphore caps concurrent scrapes."""

import asyncio
import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import meta_ads


@pytest.fixture(autouse=True)
def _reset_semaphore():
    meta_ads._browser_semaphore = None
    yield
    meta_ads._browser_semaphore = None


def test_default_concurrency_is_five(monkeypatch):
    monkeypatch.delenv("META_ADS_MAX_CONCURRENCY", raising=False)
    assert meta_ads._max_browser_concurrency() == 5


def test_env_override_is_respected(monkeypatch):
    monkeypatch.setenv("META_ADS_MAX_CONCURRENCY", "12")
    assert meta_ads._max_browser_concurrency() == 12


def test_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("META_ADS_MAX_CONCURRENCY", "not-a-number")
    assert meta_ads._max_browser_concurrency() == 5


def test_zero_or_negative_clamped_to_one(monkeypatch):
    monkeypatch.setenv("META_ADS_MAX_CONCURRENCY", "0")
    assert meta_ads._max_browser_concurrency() == 1
    monkeypatch.setenv("META_ADS_MAX_CONCURRENCY", "-3")
    assert meta_ads._max_browser_concurrency() == 1


@pytest.mark.asyncio
async def test_semaphore_bounds_concurrent_holders(monkeypatch):
    """Two coroutines can both hold the semaphore when the cap is 2;
    a third has to wait."""
    monkeypatch.setenv("META_ADS_MAX_CONCURRENCY", "2")
    meta_ads._browser_semaphore = None  # re-init at the new cap

    sem = meta_ads._get_browser_semaphore()
    in_flight = 0
    peak = 0
    release_last = asyncio.Event()

    async def hold():
        nonlocal in_flight, peak
        async with sem:
            in_flight += 1
            peak = max(peak, in_flight)
            await release_last.wait()
            in_flight -= 1

    a = asyncio.create_task(hold())
    b = asyncio.create_task(hold())
    # Give A and B a moment to enter the semaphore.
    await asyncio.sleep(0.02)
    assert in_flight == 2, "cap-2 semaphore should allow 2 concurrent holders"

    # Third caller must block until we release.
    async def try_acquire():
        async with sem:
            pass

    c = asyncio.create_task(try_acquire())
    await asyncio.sleep(0.02)
    assert not c.done(), "third caller must be waiting"

    release_last.set()
    await asyncio.gather(a, b, c)
    assert peak == 2, "peak concurrency must equal the cap, not exceed it"


@pytest.mark.asyncio
async def test_semaphore_is_lazily_instantiated_once():
    """The semaphore is created once and reused so concurrent callers
    contend on the same slot pool, not on independent semaphores."""
    s1 = meta_ads._get_browser_semaphore()
    s2 = meta_ads._get_browser_semaphore()
    assert s1 is s2
