"""Per-account failed-login lockout — audit M4.

Defense-in-depth on top of the IP-based rate limiter (audit C2): a
rotating-IP attacker still trips a per-account counter, and a
legitimate user fat-fingering their password five times gets a short
lock, not a total ban."""

import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from auth import (
    _LOGIN_LOCKOUT_TIERS,
    _is_locked_out,
    compute_login_lockout_seconds,
)


def test_first_four_failures_do_not_trigger_lockout():
    for n in (1, 2, 3, 4):
        assert compute_login_lockout_seconds(n) == 0


def test_fifth_failure_triggers_five_minute_lock():
    assert compute_login_lockout_seconds(5) == 5 * 60


def test_tenth_failure_escalates_to_one_hour():
    assert compute_login_lockout_seconds(10) == 60 * 60


def test_fifteenth_failure_escalates_to_24_hours():
    assert compute_login_lockout_seconds(15) == 24 * 60 * 60


def test_lockout_stays_at_highest_tier_beyond_15():
    """A determined attacker who's already at 15+ failures shouldn't
    somehow lose their lock by continuing to try."""
    assert compute_login_lockout_seconds(20) == 24 * 60 * 60
    assert compute_login_lockout_seconds(100) == 24 * 60 * 60


def test_is_locked_out_returns_none_when_field_missing():
    assert _is_locked_out({}, datetime.now(timezone.utc)) is None
    assert _is_locked_out({"loginLockedUntil": None}, datetime.now(timezone.utc)) is None


def test_is_locked_out_returns_expiry_when_still_locked():
    now = datetime.now(timezone.utc)
    future = now + timedelta(minutes=3)
    result = _is_locked_out({"loginLockedUntil": future}, now)
    assert result == future


def test_is_locked_out_returns_none_after_expiry():
    now = datetime.now(timezone.utc)
    past = now - timedelta(minutes=1)
    assert _is_locked_out({"loginLockedUntil": past}, now) is None


def test_is_locked_out_normalises_naive_datetime_as_utc():
    """Legacy user documents may have naive datetimes. The helper must
    treat them as UTC, not raise or crash."""
    now = datetime.now(timezone.utc)
    naive_future = (now + timedelta(minutes=3)).replace(tzinfo=None)
    result = _is_locked_out({"loginLockedUntil": naive_future}, now)
    assert result is not None
    assert result.tzinfo is not None


def test_lockout_tiers_are_monotonically_increasing():
    """Regression guard on the tiering constant so a future edit that
    reorders it doesn't silently produce a weaker higher tier."""
    prev_threshold = -1
    prev_minutes = -1
    for threshold, minutes in _LOGIN_LOCKOUT_TIERS:
        assert threshold > prev_threshold, "thresholds must be strictly increasing"
        assert minutes > prev_minutes, "durations must be strictly increasing"
        prev_threshold, prev_minutes = threshold, minutes
