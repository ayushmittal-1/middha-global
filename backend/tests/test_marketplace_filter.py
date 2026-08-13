"""Marketplace-scoped profitability (review issue #1, full fix).

`GET /profitability?marketplaceId=…` filters loaded orders to a single
marketplace before aggregating, so multi-marketplace sellers get a
single-currency response instead of a silently-blended total. These
tests lock in the pure filter helper that both the DB and SP-API paths
use."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from aurora_data import filter_orders_by_marketplace


def test_none_marketplace_id_returns_all_orders():
    orders = [
        {"amazonOrderId": "A", "marketplaceId": "ATVPDKIKX0DER"},
        {"amazonOrderId": "B", "marketplaceId": "A1F83G8C2ARO7P"},
    ]
    assert filter_orders_by_marketplace(orders, None) == orders
    assert filter_orders_by_marketplace(orders, "") == orders
    assert filter_orders_by_marketplace(orders, "   ") == orders


def test_filters_aurora_db_shape_marketplace_id():
    orders = [
        {"amazonOrderId": "US-1", "marketplaceId": "ATVPDKIKX0DER"},
        {"amazonOrderId": "UK-1", "marketplaceId": "A1F83G8C2ARO7P"},
        {"amazonOrderId": "US-2", "marketplaceId": "ATVPDKIKX0DER"},
    ]
    kept = filter_orders_by_marketplace(orders, "ATVPDKIKX0DER")
    assert [o["amazonOrderId"] for o in kept] == ["US-1", "US-2"]


def test_filters_sp_api_shape_marketplace_id():
    """SP-API GetOrders returns `MarketplaceId` (PascalCase). The same
    helper must handle both shapes so the DB-empty fallback path works."""
    orders = [
        {"AmazonOrderId": "US-1", "MarketplaceId": "ATVPDKIKX0DER"},
        {"AmazonOrderId": "UK-1", "MarketplaceId": "A1F83G8C2ARO7P"},
    ]
    kept = filter_orders_by_marketplace(orders, "ATVPDKIKX0DER")
    assert len(kept) == 1
    assert kept[0]["AmazonOrderId"] == "US-1"


def test_orders_without_marketplace_are_dropped_when_filtering():
    """Defensive: an order with no marketplaceId field can't be proven to
    belong to the requested marketplace, so it's excluded rather than
    silently included."""
    orders = [
        {"amazonOrderId": "US-1", "marketplaceId": "ATVPDKIKX0DER"},
        {"amazonOrderId": "MYSTERY"},  # no marketplaceId
    ]
    kept = filter_orders_by_marketplace(orders, "ATVPDKIKX0DER")
    assert [o["amazonOrderId"] for o in kept] == ["US-1"]


def test_empty_and_none_input_are_safe():
    assert filter_orders_by_marketplace([], "ATVPDKIKX0DER") == []
    assert filter_orders_by_marketplace(None, "ATVPDKIKX0DER") == []  # type: ignore[arg-type]


def test_whitespace_in_target_id_is_stripped():
    orders = [{"amazonOrderId": "A", "marketplaceId": "ATVPDKIKX0DER"}]
    kept = filter_orders_by_marketplace(orders, "  ATVPDKIKX0DER  ")
    assert len(kept) == 1
