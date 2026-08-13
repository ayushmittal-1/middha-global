"""Penny-accurate money accumulation (review issue #7).

Naive `sum(floats)` accumulates binary-float drift that can produce
totals off by a cent from Seller Central's own figures. `sum_money`
quantizes each addend with Decimal + ROUND_HALF_UP, and
`reconcile_row_derived_totals` uses it to re-derive account-level totals
from per-SKU rows (extracted stage — review issue #8)."""

import os
from decimal import Decimal

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from aurora_data import (
    reconcile_row_derived_totals,
    ROW_SUM_MONEY_KEYS,
    sum_money,
)


# ── sum_money ─────────────────────────────────────────────────────────────


def test_classic_binary_drift_case_is_exact():
    """0.1 + 0.2 == 0.30000000000000004 in float, but must be 0.30 here."""
    assert sum_money([0.1, 0.2]) == 0.30
    assert sum_money([0.1, 0.2, 0.3]) == 0.60


def test_thousand_row_sum_matches_decimal_expectation():
    """1000 × 0.01 must be exactly 10.00 — float sum drifts by ~1e-13."""
    assert sum_money([0.01] * 1000) == 10.00


def test_penny_rounding_uses_round_half_up_not_banker():
    """Python's built-in round() uses banker's rounding (0.5 → nearest even).
    Money totals should use ROUND_HALF_UP to match Seller Central and
    finance-team expectations."""
    # 0.125 quantized to 2dp: banker's would give 0.12, we want 0.13.
    assert sum_money([Decimal("0.125")]) == 0.13
    # Round-trip check: 0.005 + 0.005 = 0.01 (each rounds to 0.01, then sums)
    assert sum_money([Decimal("0.005"), Decimal("0.005")]) == 0.02


def test_none_and_bogus_values_are_ignored():
    assert sum_money([1.0, None, "not-a-number", 2.0]) == 3.00
    assert sum_money([]) == 0.0
    assert sum_money([None, None]) == 0.0


def test_accepts_strings_ints_decimals_and_floats():
    values = [1, "2.50", 3.25, Decimal("0.75")]
    assert sum_money(values) == 7.50


# ── reconcile_row_derived_totals ──────────────────────────────────────────


def _row(**overrides) -> dict:
    base = {k: 0.0 for k in ROW_SUM_MONEY_KEYS}
    base.update(overrides)
    return base


def test_reconcile_empty_rows_returns_zeros_for_every_key():
    result = reconcile_row_derived_totals([])
    assert set(result.keys()) == set(ROW_SUM_MONEY_KEYS)
    assert all(v == 0.0 for v in result.values())


def test_reconcile_sums_each_key_across_rows():
    rows = [
        _row(revenue=10.50, referral_fee=1.58, fba_fee=3.00),
        _row(revenue=20.00, referral_fee=3.00, fba_fee=5.50),
        _row(revenue=15.25, referral_fee=2.29, fba_fee=4.10),
    ]
    result = reconcile_row_derived_totals(rows)
    assert result["revenue"] == 45.75
    assert result["referral_fee"] == 6.87
    assert result["fba_fee"] == 12.60


def test_reconcile_matches_penny_when_float_would_drift():
    """Reconciled totals must match penny-perfect the sum-of-Decimals,
    even where a float loop would produce a trailing …9999999 or
    …0000001 result."""
    rows = [_row(revenue=0.01) for _ in range(1000)]
    result = reconcile_row_derived_totals(rows)
    assert result["revenue"] == 10.00


def test_reconcile_missing_keys_treated_as_zero():
    """Partial rows (e.g. from an older serializer) shouldn't crash the
    reconciliation — missing keys count as 0.0."""
    rows = [{"revenue": 5.00}, {"revenue": 3.00, "referral_fee": 1.00}]
    result = reconcile_row_derived_totals(rows)
    assert result["revenue"] == 8.00
    assert result["referral_fee"] == 1.00
    assert result["fba_fee"] == 0.0


def test_row_sum_keys_are_the_documented_set():
    """Guard against silent drift between the constant and any inline
    duplicate — several downstream callers read the same list to know
    which keys are safe to re-derive."""
    assert "revenue" in ROW_SUM_MONEY_KEYS
    assert "cogs_total" in ROW_SUM_MONEY_KEYS
    # Blended/overwritten totals must NOT be in the row-sum list, or the
    # reconciliation would clobber their blended values.
    for blended in (
        "inbound_placement_fee", "storage_fee", "aged_inventory_fee",
        "removal_fee", "reimbursement",
    ):
        assert blended not in ROW_SUM_MONEY_KEYS
