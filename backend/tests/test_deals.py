"""Deals tab — historical promotion aggregation + FE wiring.

The Deals tab has three sections:
  1. Historical promotions (from Aurora order docs — pure DB read)
  2. Active coupons (SP-API, best-effort)
  3. Prime Exclusive Discounts (SP-API, best-effort)

Plus an informational callout listing what Amazon doesn't expose via
SP-API (Lightning Deals, deal opportunities, suggested inventory).

These tests lock in the pure historical aggregator, the defensive
SP-API fetches, and the FE wiring."""

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

from deals import (
    UNSUPPORTED_DEAL_TYPES,
    _sp_api_unavailable,
    aggregate_historical_promotions,
    fetch_active_coupons,
    fetch_prime_exclusive_discounts,
)


# ── Historical aggregation (pure) ────────────────────────────────────────


def _order(oid: str, day: int, items: list[dict]) -> dict:
    return {
        "amazonOrderId": oid,
        "purchaseDate": datetime(2026, 8, day, tzinfo=timezone.utc),
        "orderItems": items,
    }


def _item(sku: str, qty: int, gross: float, discount: float, promo_ids: list[str]) -> dict:
    return {
        "sellerSku": sku,
        "quantityOrdered": qty,
        "itemSubtotal": {"amount": gross},
        "promotionDiscount": {"amount": discount},
        "promotionIds": promo_ids,
    }


def test_empty_input_returns_empty_shape():
    result = aggregate_historical_promotions([])
    assert result == {
        "promotions": [],
        "totals": {
            "unique_promotions": 0,
            "units_sold": 0,
            "gross_revenue": 0,
            "discount_amount": 0,
            "net_revenue": 0,
        },
    }


def test_none_input_is_safe():
    result = aggregate_historical_promotions(None)  # type: ignore[arg-type]
    assert result["promotions"] == []


def test_items_without_promotion_ids_are_ignored():
    """A regular sale (no promo) contributes nothing to the Deals view —
    only sales with an explicit promotionIds entry count."""
    docs = [_order("O1", 10, [_item("A", 2, 20.0, 0.0, [])])]
    result = aggregate_historical_promotions(docs)
    assert result["totals"]["unique_promotions"] == 0


def test_single_promotion_gets_aggregated():
    docs = [_order("O1", 10, [_item("SKU-1", 2, 20.0, 4.0, ["LDEAL-XYZ"])])]
    result = aggregate_historical_promotions(docs)
    assert len(result["promotions"]) == 1
    p = result["promotions"][0]
    assert p["promotion_id"] == "LDEAL-XYZ"
    assert p["units_sold"] == 2
    assert p["orders"] == 1
    assert p["gross_revenue"] == 20.0
    assert p["discount_amount"] == 4.0
    assert p["net_revenue"] == 16.0
    assert p["avg_discount_per_unit"] == 2.0
    assert p["skus"] == ["SKU-1"]


def test_multiple_orders_same_promotion_aggregate():
    docs = [
        _order("O1", 10, [_item("A", 2, 20.0, 4.0, ["DEAL-1"])]),
        _order("O2", 11, [_item("A", 3, 30.0, 6.0, ["DEAL-1"])]),
        _order("O3", 12, [_item("B", 1, 15.0, 3.0, ["DEAL-1"])]),
    ]
    result = aggregate_historical_promotions(docs)
    assert len(result["promotions"]) == 1
    p = result["promotions"][0]
    assert p["units_sold"] == 6
    assert p["orders"] == 3
    assert p["gross_revenue"] == 65.0
    assert p["discount_amount"] == 13.0


def test_first_and_last_seen_track_date_range():
    docs = [
        _order("O1", 15, [_item("A", 1, 10.0, 2.0, ["DEAL-1"])]),
        _order("O2", 5, [_item("A", 1, 10.0, 2.0, ["DEAL-1"])]),
        _order("O3", 20, [_item("A", 1, 10.0, 2.0, ["DEAL-1"])]),
    ]
    p = aggregate_historical_promotions(docs)["promotions"][0]
    assert p["first_seen"] == "2026-08-05"
    assert p["last_seen"] == "2026-08-20"


def test_multiple_promo_ids_on_one_item_credits_only_the_first():
    """When Amazon attaches multiple promotions to a single line
    (e.g. Lightning Deal + coupon stacked), attribution goes to the
    FIRST id only — otherwise the total across all promotions
    over-counts sales and discount."""
    docs = [
        _order("O1", 10, [_item("A", 1, 10.0, 3.0, ["PRIMARY-DEAL", "SNS-STACK"])])
    ]
    result = aggregate_historical_promotions(docs)
    ids = {p["promotion_id"] for p in result["promotions"]}
    assert ids == {"PRIMARY-DEAL"}
    assert "SNS-STACK" not in ids


def test_sku_count_reflects_distinct_skus_and_top_skus_capped_at_10():
    """Different SKUs sold under the same promo must all be tracked;
    display list is capped at 10 to keep the table row readable."""
    items = [
        _item(f"SKU-{i}", 1, 10.0, 1.0, ["DEAL-X"])
        for i in range(15)
    ]
    docs = [_order("O1", 10, items)]
    p = aggregate_historical_promotions(docs)["promotions"][0]
    assert p["sku_count"] == 15
    assert len(p["skus"]) == 10


def test_promotions_sorted_by_discount_desc():
    """A seller asking 'which deals cost me the most?' should see the
    top-cost deal first."""
    docs = [
        _order("O1", 10, [_item("A", 1, 100.0, 5.0, ["SMALL-DEAL"])]),
        _order("O2", 10, [_item("B", 1, 100.0, 50.0, ["BIG-DEAL"])]),
        _order("O3", 10, [_item("C", 1, 100.0, 20.0, ["MED-DEAL"])]),
    ]
    ids = [p["promotion_id"] for p in aggregate_historical_promotions(docs)["promotions"]]
    assert ids == ["BIG-DEAL", "MED-DEAL", "SMALL-DEAL"]


def test_totals_match_sum_of_promotion_rows():
    docs = [
        _order("O1", 10, [_item("A", 2, 20.0, 4.0, ["D1"])]),
        _order("O2", 11, [_item("B", 3, 30.0, 6.0, ["D2"])]),
    ]
    r = aggregate_historical_promotions(docs)
    assert r["totals"]["unique_promotions"] == 2
    assert r["totals"]["units_sold"] == 5
    assert r["totals"]["gross_revenue"] == 50.0
    assert r["totals"]["discount_amount"] == 10.0
    assert r["totals"]["net_revenue"] == 40.0


def test_falls_back_to_itemPrice_when_itemSubtotal_missing():
    """Aurora order sync uses itemSubtotal, but SP-API-shape orders
    can carry only itemPrice. The aggregator handles both."""
    docs = [_order("O1", 10, [{
        "sellerSku": "A",
        "quantityOrdered": 1,
        "itemPrice": {"amount": 25.0},
        "promotionDiscount": {"amount": 5.0},
        "promotionIds": ["FALLBACK-DEAL"],
    }])]
    p = aggregate_historical_promotions(docs)["promotions"][0]
    assert p["gross_revenue"] == 25.0
    assert p["discount_amount"] == 5.0


def test_negative_discount_amount_is_normalised_positive():
    """Amazon sometimes stores promotionDiscount as a signed negative
    (money OUT). The report treats discount as always-positive so the
    display line 'Discount −$X' can prefix its own sign."""
    docs = [_order("O1", 10, [_item("A", 1, 10.0, -3.0, ["DEAL"])])]
    p = aggregate_historical_promotions(docs)["promotions"][0]
    assert p["discount_amount"] == 3.0


def test_malformed_documents_are_skipped_not_crashed():
    """A partial / half-migrated order doc shouldn't take down the
    endpoint — the aggregator skips over anything it can't parse."""
    docs = [
        "not a dict",  # type: ignore[list-item]
        {},
        {"orderItems": "not a list"},
        {"orderItems": ["not a dict"]},
        _order("O1", 10, [_item("A", 1, 10.0, 2.0, ["REAL-DEAL"])]),
    ]
    p = aggregate_historical_promotions(docs)  # type: ignore[arg-type]
    assert p["totals"]["unique_promotions"] == 1


# ── Unsupported-types callout ────────────────────────────────────────────


def test_unsupported_list_names_the_expected_gaps():
    """Callout must explicitly mention the four things we can't
    provide via SP-API — Lightning Deals, 7-Day/Best, deal
    opportunities, and Amazon-suggested inventory."""
    names = [u["name"] for u in UNSUPPORTED_DEAL_TYPES]
    joined = "\n".join(names)
    assert "Lightning" in joined
    assert "7-Day" in joined or "Best" in joined
    assert "opportunit" in joined.lower()
    assert "suggested inventory" in joined.lower()


def test_every_unsupported_item_carries_a_reason():
    for u in UNSUPPORTED_DEAL_TYPES:
        assert u.get("reason"), f"missing reason for {u.get('name')}"
        assert len(u["reason"]) > 20, "reason should be a real explanation"


# ── Defensive SP-API fetches ─────────────────────────────────────────────


def test_sp_api_unavailable_shape():
    r = _sp_api_unavailable("some reason")
    assert r == {"unavailable": "some reason", "items": []}


@pytest.mark.asyncio
async def test_fetch_active_coupons_gracefully_handles_403(monkeypatch):
    import amazon_sp

    async def _boom(*args, **kwargs):
        raise RuntimeError("HTTP 403 AccessDenied")

    monkeypatch.setattr(amazon_sp, "_sp_request", _boom)
    result = await fetch_active_coupons({"_id": "u1"})
    assert "unavailable" in result
    assert "403" in result["unavailable"] or "authorization" in result["unavailable"]


@pytest.mark.asyncio
async def test_fetch_active_coupons_gracefully_handles_404(monkeypatch):
    import amazon_sp

    async def _boom(*args, **kwargs):
        raise RuntimeError("HTTP 404 NotFound")

    monkeypatch.setattr(amazon_sp, "_sp_request", _boom)
    result = await fetch_active_coupons({"_id": "u1"})
    assert "unavailable" in result
    assert "404" in result["unavailable"] or "not expose" in result["unavailable"].lower() or "marketplace" in result["unavailable"].lower()


@pytest.mark.asyncio
async def test_fetch_active_coupons_returns_items_on_success(monkeypatch):
    import amazon_sp

    fake_response = {"coupons": [{"couponId": "C1", "name": "Test"}]}

    async def _ok(method, path, **kwargs):
        assert method == "GET"
        assert "coupons" in path
        return fake_response

    monkeypatch.setattr(amazon_sp, "_sp_request", _ok)
    result = await fetch_active_coupons({"_id": "u1"})
    assert result.get("items") == fake_response["coupons"]


@pytest.mark.asyncio
async def test_fetch_ped_gracefully_handles_403(monkeypatch):
    import amazon_sp

    async def _boom(*args, **kwargs):
        raise RuntimeError("HTTP 403 Forbidden")

    monkeypatch.setattr(amazon_sp, "_sp_request", _boom)
    result = await fetch_prime_exclusive_discounts({"_id": "u1"})
    assert "unavailable" in result


# ── Frontend wiring ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def frontend_html() -> str:
    return (
        Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")


def test_deals_tab_button_exists_in_nav(frontend_html):
    assert 'data-tab="deals"' in frontend_html


def test_deals_view_body_exists(frontend_html):
    assert 'id="deals-view"' in frontend_html
    assert 'id="deals-historical-body"' in frontend_html
    assert 'id="deals-coupons-body"' in frontend_html
    assert 'id="deals-ped-body"' in frontend_html


def test_deals_tab_calls_loadDeals(frontend_html):
    assert "loadDeals" in frontend_html
    assert "/deals?days_back=" in frontend_html


def test_unsupported_callout_renders_from_response(frontend_html):
    """The FE must consume data.unsupported and render each item so
    users see WHY Lightning Deals aren't listed."""
    assert "data.unsupported" in frontend_html
    assert "deals-unsupported" in frontend_html


def test_sp_api_unavailable_gets_a_friendly_message(frontend_html):
    """When coupons/PED come back with `unavailable`, the FE shows the
    reason instead of an empty section (or worse, a JSON dump)."""
    assert ".unavailable" in frontend_html


def test_sns_zero_discount_caveat_is_documented_in_ui(frontend_html):
    """Live-Mongo probe surfaced that Subscribe & Save orders arrive
    with promotionDiscount=$0 because SNS discounts are accounted for
    in Finances, not on the order line. The UI must flag this so a
    user seeing '$0 discount, 129 units' on a SNS row doesn't file a
    bug ticket."""
    assert "Subscribe" in frontend_html and "Save" in frontend_html
    assert "$0" in frontend_html or "Finances" in frontend_html
