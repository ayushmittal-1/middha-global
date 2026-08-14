"""SP-API GetOrderItems pacing math (verification-review followup on #3).

The first fix parallelised the fetch but kept a fixed 2.1s per-slot
sleep, which meant `concurrency=5` produced ~2.4 req/s — roughly 5×
SP-API's documented ~0.5 req/s sustained quota, opening a real 429
regression risk under load.

`_pacing_sleep_for` now scales per-slot sleep with concurrency so the
effective sustained rate never exceeds the quota, regardless of what
`concurrency` is set to."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import agent


def test_serial_matches_original_2_1s_per_call():
    """concurrency=1 must reproduce the original ~2.1s per-call spacing
    that the pre-parallel serial implementation used — otherwise this
    change would silently slow the fallback path for small windows."""
    assert agent._pacing_sleep_for(concurrency=1, attempt=1) == pytest.approx(2.0)


@pytest.mark.parametrize("concurrency", [1, 3, 5, 10, 20])
def test_effective_rate_never_exceeds_quota(concurrency):
    """The whole point of this fix: `concurrency / sleep` must be <=
    the documented sustained quota, regardless of how many slots the
    caller opens up."""
    sleep = agent._pacing_sleep_for(concurrency=concurrency, attempt=1)
    effective_rate = concurrency / sleep
    assert effective_rate <= agent.SP_API_ORDER_ITEMS_SUSTAINED_REQ_PER_SEC + 1e-9, (
        f"concurrency={concurrency} yields {effective_rate:.3f} req/s, "
        f"which exceeds the {agent.SP_API_ORDER_ITEMS_SUSTAINED_REQ_PER_SEC} req/s "
        f"sustained quota — regression to the pre-followup pacing bug"
    )


def test_effective_rate_stays_at_quota_not_far_below():
    """Guard against the opposite regression — a fix that over-corrects
    into serial-equivalent throughput. concurrency=5 should still
    saturate the quota, not throttle to (say) 0.1 req/s."""
    sleep = agent._pacing_sleep_for(concurrency=5, attempt=1)
    effective_rate = 5 / sleep
    # Within 10% of the sustained quota.
    target = agent.SP_API_ORDER_ITEMS_SUSTAINED_REQ_PER_SEC
    assert effective_rate >= target * 0.9


def test_attempt_backoff_multiplies_sleep():
    """Retry attempt N should sleep proportionally longer — that's how
    the original serial path handled repeated 429s. Attempt 2 = 2×
    attempt 1 sleep, attempt 3 = 3×, etc."""
    s1 = agent._pacing_sleep_for(concurrency=5, attempt=1)
    s2 = agent._pacing_sleep_for(concurrency=5, attempt=2)
    s3 = agent._pacing_sleep_for(concurrency=5, attempt=3)
    assert s2 == pytest.approx(2 * s1)
    assert s3 == pytest.approx(3 * s1)


def test_concurrency_zero_or_negative_is_clamped_to_one():
    """Defensive — a bad `concurrency` arg shouldn't zero-divide the
    caller. Clamp to serial-equivalent instead."""
    assert agent._pacing_sleep_for(concurrency=0, attempt=1) > 0
    assert agent._pacing_sleep_for(concurrency=-3, attempt=1) > 0
