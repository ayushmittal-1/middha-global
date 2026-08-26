"""Mixed-currency detection for profitability totals.

Source currencies are still detected so the API can report CAD/MXN/etc.
Conversion to USD happens in `currency_fx` / aggregation (see
test_usd_fx_profitability.py). Detection must not invent a blank code.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from aurora_data import detect_order_currencies
from agent import _sp_api_order_currencies


def _db_order(status: str, currencies_by_field: list[dict]) -> dict:
    """Build an Aurora-DB-shape order with the given per-line-item currency
    codes. currencies_by_field is a list where each dict maps
    money-field-name -> currencyCode."""
    return {
        "amazonOrderId": "A-1",
        "orderStatus": status,
        "orderItems": [
            {
                "sellerSku": f"SKU-{i}",
                "quantityOrdered": 1,
                **{
                    field: {"amount": 10.0, "currencyCode": code}
                    for field, code in fields.items()
                },
            }
            for i, fields in enumerate(currencies_by_field)
        ],
    }


# ── Aurora-DB shape ──────────────────────────────────────────────────────


def test_detect_single_currency_from_line_items():
    orders = [_db_order("Shipped", [{"itemSubtotal": "USD", "itemPrice": "USD"}])]
    assert detect_order_currencies(orders) == {"USD"}


def test_detect_multiple_currencies_across_orders():
    orders = [
        _db_order("Shipped", [{"itemSubtotal": "USD"}]),
        _db_order("Shipped", [{"itemSubtotal": "GBP"}]),
        _db_order("Shipped", [{"itemSubtotal": "EUR"}]),
    ]
    assert detect_order_currencies(orders) == {"USD", "GBP", "EUR"}


def test_detect_multiple_currencies_within_one_order():
    orders = [
        _db_order(
            "Shipped",
            [{"itemSubtotal": "USD"}, {"itemSubtotal": "CAD"}],
        )
    ]
    assert detect_order_currencies(orders) == {"USD", "CAD"}


def test_cancelled_orders_are_excluded_from_currency_scan():
    """Match the revenue path — cancelled orders don't contribute to
    totals, so they shouldn't drag a stale currency into the warning."""
    orders = [
        _db_order("Shipped", [{"itemSubtotal": "USD"}]),
        _db_order("Cancelled", [{"itemSubtotal": "GBP"}]),
    ]
    assert detect_order_currencies(orders) == {"USD"}


def test_blank_and_none_currency_codes_ignored():
    orders = [
        _db_order("Shipped", [{"itemSubtotal": "USD"}]),
        _db_order("Shipped", [{"itemSubtotal": ""}]),
    ]
    orders.append({
        "orderStatus": "Shipped",
        "orderItems": [{"itemSubtotal": {"amount": 5.0, "currencyCode": None}}],
    })
    assert detect_order_currencies(orders) == {"USD"}


def test_top_level_order_total_currency_is_seen():
    orders = [
        {
            "amazonOrderId": "A-1",
            "orderStatus": "Shipped",
            "orderTotal": {"amount": 30.0, "currencyCode": "jpy"},
            "orderItems": [],
        }
    ]
    # Codes normalise to upper-case.
    assert detect_order_currencies(orders) == {"JPY"}


def test_sales_channel_amazon_ca_counts_as_cad():
    orders = [
        {
            "amazonOrderId": "A-1",
            "orderStatus": "Shipped",
            "salesChannel": "Amazon.ca",
            "orderItems": [{"itemSubtotal": {"amount": 10.0}}],
        }
    ]
    assert detect_order_currencies(orders) == {"CAD"}


def test_empty_orders_returns_empty_set():
    assert detect_order_currencies([]) == set()
    assert detect_order_currencies(None) == set()  # type: ignore[arg-type]


# ── SP-API-shape sibling ─────────────────────────────────────────────────


def test_sp_api_currencies_from_order_total():
    orders = [
        {"AmazonOrderId": "A-1", "OrderTotal": {"Amount": "10", "CurrencyCode": "USD"}},
        {"AmazonOrderId": "A-2", "OrderTotal": {"Amount": "5", "CurrencyCode": "GBP"}},
    ]
    assert _sp_api_order_currencies(orders) == {"USD", "GBP"}


def test_sp_api_currencies_handles_missing_order_total():
    orders = [{"AmazonOrderId": "A-1"}, {"AmazonOrderId": "A-2", "OrderTotal": None}]
    assert _sp_api_order_currencies(orders) == set()
