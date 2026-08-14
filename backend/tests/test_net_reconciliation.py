"""Penny-consistent `net` reconciliation (verification-review followup on #7).

The first fix reconciled row-sum component totals via Decimal but left
`totals["net"]` on the float-accumulator path. That opened a subtle new
tie-out gap: components could be Decimal-accurate while net drifted by a
cent, so `net != revenue - fees - ...` at the penny level. This test
locks the invariant that reconciled net equals sum_money() of the
signed components."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from aurora_data import reconcile_net_total, sum_money


def _row(**kv) -> dict:
    return {"net": 0.0, **kv}


def test_reconcile_net_returns_zero_for_empty_rows():
    assert reconcile_net_total([]) == 0.0
    assert reconcile_net_total(None) == 0.0  # type: ignore[arg-type]


def test_reconcile_net_sums_row_net_via_decimal():
    """Classic binary-float case: 0.1 + 0.2 + 0.3 sums to 0.6 exactly
    under Decimal, not 0.6000000000000001 as float would."""
    rows = [_row(net=0.1), _row(net=0.2), _row(net=0.3)]
    assert reconcile_net_total(rows) == 0.60


def test_reconcile_net_matches_expected_when_summing_1000_rows():
    """1000 × $0.01 net rows should be exactly $10.00, not $9.99…"""
    rows = [_row(net=0.01) for _ in range(1000)]
    assert reconcile_net_total(rows) == 10.00


def test_reconcile_net_handles_negative_rows():
    """A SKU can genuinely be net-negative (fees + ad cost exceed
    revenue). Reconciliation must not silently drop those."""
    rows = [_row(net=5.00), _row(net=-3.50), _row(net=-1.25)]
    assert reconcile_net_total(rows) == 0.25


def test_reconcile_net_missing_key_is_zero():
    """A row missing the `net` key (e.g. from a partial serializer)
    must not crash — it just contributes 0."""
    rows = [{"revenue": 5.0}, _row(net=1.25)]
    assert reconcile_net_total(rows) == 1.25


def test_reconciled_net_equals_signed_component_sum():
    """The core tie-out invariant the followup exists to enforce:
    `net` derived via sum_money over signed components must equal
    reconcile_net_total when each row's `net` was computed as
    `revenue - amazon_fees - ... + reimbursement`."""
    rows = []
    for _ in range(500):
        revenue = 19.99
        amazon_fees = 4.15
        ad_cost = 1.10
        product_cost = 6.00
        row_net = round(revenue - amazon_fees - ad_cost - product_cost, 2)
        rows.append({
            "net": row_net,
            "revenue": revenue,
            "amazon_fees": amazon_fees,
            "ad_cost": ad_cost,
            "product_cost": product_cost,
        })
    net_from_rows = reconcile_net_total(rows)
    net_from_components = sum_money([
        sum_money(r["revenue"] for r in rows),
        -sum_money(r["amazon_fees"] for r in rows),
        -sum_money(r["ad_cost"] for r in rows),
        -sum_money(r["product_cost"] for r in rows),
    ])
    assert net_from_rows == net_from_components
