"""Ordered units must match Amazon All Orders quantity.

A 3-unit order counts as 3 (sum of quantityOrdered), never as 1 order.
Returns reduce returned_units / net_units by return quantity, not by
order count, and must not change the displayed Units (ordered) figure.
"""

from aurora_data import aggregate_sku_metrics_from_orders, line_item_quantity
from agent import _apply_returns_to_sku_data


def test_line_item_quantity_uses_ordered_not_one_per_order():
    assert line_item_quantity({"quantityOrdered": 3}) == 3
    assert line_item_quantity({"quantityOrdered": 0, "quantityShipped": 2}) == 2
    assert line_item_quantity({}) == 0


def test_multi_unit_order_counts_quantity_not_one():
    orders = [
        {
            "amazonOrderId": "111-multi",
            "orderStatus": "Shipped",
            "orderItems": [
                {
                    "sellerSku": "SKU-A",
                    "asin": "B00TEST001",
                    "quantityOrdered": 3,
                    "itemPrice": {"amount": 14.7, "currencyCode": "USD"},
                    "itemSubtotal": {"amount": 14.7, "currencyCode": "USD"},
                    "promotionDiscount": {"amount": 0, "currencyCode": "USD"},
                    "referralFee": {"amount": 2.0, "currencyCode": "USD"},
                    "fulfillmentFee": {"amount": 6.0, "currencyCode": "USD"},
                },
            ],
        }
    ]
    sku_data, _, orders_count = aggregate_sku_metrics_from_orders(orders)
    assert orders_count == 1
    assert sku_data["SKU-A"]["units"] == 3
    assert sku_data["SKU-A"]["revenue"] == 14.7


def test_zero_price_lines_excluded_from_units_and_revenue():
    """$0 / replacement lines must not dilute Units or avg price."""
    orders = [
        {
            "amazonOrderId": "111-1",
            "orderStatus": "Shipped",
            "orderItems": [
                {
                    "sellerSku": "SKU-A",
                    "asin": "B00TEST001",
                    "quantityOrdered": 3,
                    "itemPrice": {"amount": 0, "currencyCode": "USD"},
                    "itemSubtotal": {"amount": 0, "currencyCode": "USD"},
                    "promotionDiscount": {"amount": 0, "currencyCode": "USD"},
                    "referralFee": {"amount": 0, "currencyCode": "USD"},
                    "fulfillmentFee": {"amount": 0, "currencyCode": "USD"},
                },
                {
                    "sellerSku": "SKU-A",
                    "asin": "B00TEST001",
                    "quantityOrdered": 2,
                    "itemPrice": {"amount": 9.8, "currencyCode": "USD"},
                    "itemSubtotal": {"amount": 9.8, "currencyCode": "USD"},
                    "promotionDiscount": {"amount": 0, "currencyCode": "USD"},
                    "referralFee": {"amount": 1.0, "currencyCode": "USD"},
                    "fulfillmentFee": {"amount": 3.0, "currencyCode": "USD"},
                },
            ],
        }
    ]
    sku_data, na_rows, _ = aggregate_sku_metrics_from_orders(orders)
    assert sku_data["SKU-A"]["units"] == 2
    assert sku_data["SKU-A"]["revenue"] == 9.8
    assert sum(r["qty"] for r in na_rows) == 3


def test_returns_reduce_by_quantity_not_order_count_and_keep_units():
    sku_data = {
        "SKU-A": {
            "units": 10,
            "revenue": 49.0,
            "asin": "B00TEST001",
            "referral_total": 5.0,
            "fba_total": 15.0,
        }
    }
    # Legacy map (units only) still uses kept/ordered ratio.
    returns = {"SKU-A": {"returned_units": 3, "refunded_referral": 1.5, "asin": "B00TEST001"}}
    _apply_returns_to_sku_data(sku_data, returns)
    assert sku_data["SKU-A"]["units"] == 10  # still ordered qty
    assert sku_data["SKU-A"]["ordered_units"] == 10
    assert sku_data["SKU-A"]["returned_units"] == 3  # qty, not 1
    assert sku_data["SKU-A"]["net_units"] == 7
    assert abs(sku_data["SKU-A"]["revenue"] - 34.3) < 0.001


def test_returns_subtract_actual_refunded_line_not_sku_average():
    """Mixed prices: proportional kept/ordered drifts (Andexports B08P3CD3WR)."""
    sku_data = {
        "SKU-A": {
            "units": 35,
            "revenue": 399.68,
            "asin": "B08P3CD3WR",
            "referral_total": 59.90,
            "fba_total": 152.25,
        }
    }
    returns = {
        "SKU-A": {
            "returned_units": 1,
            "refunded_revenue": 11.76,
            "refunded_referral": 1.76,
            "refunded_fulfillment": 4.35,
            "asin": "B08P3CD3WR",
        }
    }
    _apply_returns_to_sku_data(sku_data, returns, {"SKU-A": 0.15})
    assert sku_data["SKU-A"]["units"] == 35
    assert sku_data["SKU-A"]["returned_units"] == 1
    assert sku_data["SKU-A"]["net_units"] == 34
    assert abs(sku_data["SKU-A"]["revenue"] - 387.92) < 0.001
    # Not the buggy proportional 399.68 * 34/35 = 388.26
    assert abs(sku_data["SKU-A"]["revenue"] - 388.26) > 0.01
    assert abs(sku_data["SKU-A"]["referral_total"] - 58.14) < 0.01
    assert abs(sku_data["SKU-A"]["fba_total"] - 147.90) < 0.01


def test_physical_return_without_refund_does_not_count():
    """SC Payment-complete / no refund must not count as returned."""
    from aurora_data import returned_qty_for_sku

    assert returned_qty_for_sku(physical=1, refunded=0) == 0
    assert returned_qty_for_sku(physical=3, refunded=1) == 3
    assert returned_qty_for_sku(physical=0, refunded=2) == 2
