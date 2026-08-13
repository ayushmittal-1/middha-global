"""Per-account SP-API report semaphore.

Guards against a regression of the review-doc issue where `_REPORT_SEM` was
a single module-global slot: one seller holding it would queue every other
seller's request platform-wide even though SP-API's /reports quota is
per-account.
"""

import asyncio
import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import agent
from auth import current_user


@pytest.fixture(autouse=True)
def _clear_semaphores():
    agent._REPORT_SEMS.clear()
    yield
    agent._REPORT_SEMS.clear()


def _as_user(uid: str) -> dict:
    return {"_id": uid, "email": f"{uid}@example.test"}


@pytest.mark.asyncio
async def test_same_user_reuses_same_semaphore():
    token = current_user.set(_as_user("user-a"))
    try:
        sem1 = await agent._get_report_sem()
        sem2 = await agent._get_report_sem()
        assert sem1 is sem2, "same user must get the same semaphore instance"
    finally:
        current_user.reset(token)


@pytest.mark.asyncio
async def test_different_users_get_distinct_semaphores():
    tok_a = current_user.set(_as_user("user-a"))
    sem_a = await agent._get_report_sem()
    current_user.reset(tok_a)

    tok_b = current_user.set(_as_user("user-b"))
    sem_b = await agent._get_report_sem()
    current_user.reset(tok_b)

    assert sem_a is not sem_b, "different users must get separate semaphores"


@pytest.mark.asyncio
async def test_one_user_holding_sem_does_not_block_another():
    """Regression test for review issue #2.

    User A holds their per-account slot for a fake 'slow report' call;
    User B must be able to acquire and release their own slot without
    waiting. Before the fix (single global Semaphore(1)), B would queue
    behind A."""
    b_acquired = asyncio.Event()

    async def slow_holder_for_user_a():
        tok = current_user.set(_as_user("user-a"))
        try:
            sem = await agent._get_report_sem()
            async with sem:
                await b_acquired.wait()
        finally:
            current_user.reset(tok)

    async def try_acquire_for_user_b():
        tok = current_user.set(_as_user("user-b"))
        try:
            sem = await agent._get_report_sem()
            async with sem:
                b_acquired.set()
        finally:
            current_user.reset(tok)

    # If the semaphore were still global, holder never releases -> B's
    # acquire blocks forever -> wait_for times out.
    await asyncio.wait_for(
        asyncio.gather(slow_holder_for_user_a(), try_acquire_for_user_b()),
        timeout=2.0,
    )


@pytest.mark.asyncio
async def test_same_user_still_serializes_calls():
    """Per-account fix must NOT weaken the intended per-account serialization.

    Two concurrent report calls for the SAME user should still run one after
    the other (the underlying reason the semaphore exists: SP-API's per-
    account quota)."""
    events: list[str] = []

    async def report_call(label: str):
        tok = current_user.set(_as_user("shared-user"))
        try:
            sem = await agent._get_report_sem()
            async with sem:
                events.append(f"{label}:start")
                await asyncio.sleep(0.05)
                events.append(f"{label}:end")
        finally:
            current_user.reset(tok)

    await asyncio.gather(report_call("A"), report_call("B"))

    # Must be strictly interleaved as start,end,start,end — never overlap.
    assert events[0].endswith(":start")
    assert events[1].endswith(":end")
    assert events[2].endswith(":start")
    assert events[3].endswith(":end")
    assert events[0].split(":")[0] == events[1].split(":")[0]
    assert events[2].split(":")[0] == events[3].split(":")[0]


@pytest.mark.asyncio
async def test_no_user_context_falls_back_to_shared_slot():
    """Called without a request context, the helper must still return a slot
    (used by the fallback path so nothing crashes if the ContextVar is unset)."""
    # No current_user.set — default is None.
    sem = await agent._get_report_sem()
    assert sem is not None
    assert "__anon__" in agent._REPORT_SEMS
