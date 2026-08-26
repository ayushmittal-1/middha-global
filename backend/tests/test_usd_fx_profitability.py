"""USD conversion for mixed CAD/USD profitability (B09JZL4J8S January).

Amazon.com lines are USD; Amazon.ca lines are CAD. Summing the raw
floats treated C$17.98 as $17.98 and then took 15% referral of that
CAD face value. Convert to USD first, then compute referral / FBA.
"""

import os
from datetime import date, datetime, timezone

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from aurora_data import (
    aggregate_sku_metrics_from_orders,
    detect_order_currencies,
    filter_orders_by_marketplace,
    line_referral_fee,
    recompute_referral_totals,
    round_money,
)
from currency_fx import UsdFx, infer_line_currency, infer_marketplace_id


SKU = "Green Lines 50 Pack Eleet"
ASIN = "B09JZL4J8S"
JAN = datetime(2026, 1, 15, tzinfo=timezone.utc)
CAD_USD = 0.71922


def _fx(rate: float = CAD_USD) -> UsdFx:
    table = UsdFx(source="test")
    table.rates[("CAD", date(2026, 1, 15))] = rate
    return table


def _line(*, amount: float, currency: str, qty: int = 1, fba: float = 4.35, fba_ccy: str = "USD"):
    return {
        "sellerSku": SKU,
        "asin": ASIN,
        "quantityOrdered": qty,
        "itemPrice": {"amount": amount, "currencyCode": currency},
        "itemSubtotal": {"amount": amount, "currencyCode": currency},
        "promotionDiscount": {"amount": 0, "currencyCode": currency},
        "referralFee": {
            "amount": round_money(amount * 0.15),
            "currencyCode": "USD",
        },
        "fulfillmentFee": {"amount": fba, "currencyCode": fba_ccy},
    }


def _order(channel: str, amount: float, currency: str) -> dict:
    return {
        "amazonOrderId": f"{channel}-{amount}",
        "orderStatus": "Shipped",
        "salesChannel": channel,
        "purchaseDate": JAN,
        "orderTotal": {"amount": amount, "currencyCode": currency},
        "orderItems": [_line(amount=amount, currency=currency)],
    }


def _jan_asin_orders() -> list[dict]:
    """10 × $6.99 USD Amazon.com + 2 × C$17.98 Amazon.ca."""
    orders = [_order("Amazon.com", 6.99, "USD") for _ in range(10)]
    orders.extend(_order("Amazon.ca", 17.98, "CAD") for _ in range(2))
    return orders


def test_observation_raw_sum_is_wrong_for_january_asin():
    """The bug: CAD face values added to USD as if they were dollars."""
    sku_data, _, _ = aggregate_sku_metrics_from_orders(
        _jan_asin_orders(), fx=UsdFx(source="identity", fallback={"USD": 1.0}),
    )
    d = sku_data[SKU]
    assert d["units"] == 12
    # 69.90 USD + 35.96 CAD treated as $105.86
    assert round(d["revenue"], 2) == 105.86
    assert round(d["referral_total"], 2) == 15.90


def test_convert_cad_then_calculate_matches_fx():
    sku_data, _, _ = aggregate_sku_metrics_from_orders(
        _jan_asin_orders(), fx=_fx(),
    )
    d = sku_data[SKU]
    cad_unit_usd = round_money(17.98 * CAD_USD)
    assert cad_unit_usd == 12.93
    expected_rev = round_money(69.90 + 2 * cad_unit_usd)
    assert round(d["revenue"], 2) == expected_rev == 95.76

    expected_ref = round_money(
        10 * line_referral_fee(6.99, 0.15, 1)
        + 2 * line_referral_fee(cad_unit_usd, 0.15, 1)
    )
    assert round(d["referral_total"], 2) == expected_ref == 14.38
    # FBA stored as USD even on CA orders — leave it, don't treat 4.35 as CAD.
    assert round(d["fba_total"], 2) == round_money(4.35 * 12)


def test_recompute_referral_uses_usd_not_cad_face_value():
    orders = _jan_asin_orders()
    sku_data, _, _ = aggregate_sku_metrics_from_orders(orders, fx=_fx())
    recompute_referral_totals(orders, sku_data, {SKU: 0.15}, fx=_fx())
    assert round(sku_data[SKU]["referral_total"], 2) == 14.38


def test_detects_cad_and_usd_on_this_asin():
    assert detect_order_currencies(_jan_asin_orders()) == {"USD", "CAD"}


def test_sales_channel_infers_canada_marketplace_when_id_missing():
    """Aurora All-Orders sync leaves marketplaceId empty — only salesChannel."""
    orders = [
        {"amazonOrderId": "US", "salesChannel": "Amazon.com"},
        {"amazonOrderId": "CA", "salesChannel": "Amazon.ca"},
        {"amazonOrderId": "MX", "salesChannel": "Amazon.com.mx"},
    ]
    assert infer_marketplace_id(orders[0]) == "ATVPDKIKX0DER"
    assert infer_marketplace_id(orders[1]) == "A2EUQ1WTGCTBG2"
    kept = filter_orders_by_marketplace(orders, "ATVPDKIKX0DER")
    assert [o["amazonOrderId"] for o in kept] == ["US"]
    kept_ca = filter_orders_by_marketplace(orders, "A2EUQ1WTGCTBG2")
    assert [o["amazonOrderId"] for o in kept_ca] == ["CA"]


def test_infer_currency_from_amazon_ca_channel():
    assert infer_line_currency({"salesChannel": "Amazon.ca"}, None) == "CAD"
    assert infer_line_currency({"salesChannel": "Amazon.com"}, None) == "USD"


def test_cad_fba_labelled_cad_is_converted():
    """If fulfillmentFee is actually CAD, convert it; don't leave C$ as $."""
    orders = [
        {
            "amazonOrderId": "ca-1",
            "orderStatus": "Shipped",
            "salesChannel": "Amazon.ca",
            "purchaseDate": JAN,
            "orderItems": [
                _line(amount=17.98, currency="CAD", fba=6.10, fba_ccy="CAD"),
            ],
        }
    ]
    sku_data, _, _ = aggregate_sku_metrics_from_orders(orders, fx=_fx())
    assert round(sku_data[SKU]["fba_total"], 2) == round_money(6.10 * CAD_USD)
