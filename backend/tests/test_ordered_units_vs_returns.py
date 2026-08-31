"""Ordered units must match Amazon All Orders quantity.

A 3-unit order counts as 3 (sum of quantityOrdered), never as 1 order.
Returns reduce returned_units / net_units by return quantity, not by
order count, and must not change the displayed Units (ordered) figure.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

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
    # Referral / FBA net the refunded line (Andexports $1.76 / $4.35).
    # Revenue is what nets the refunded dollars. Units stay ordered qty.
    assert abs(sku_data["SKU-A"]["referral_total"] - 58.14) < 0.01
    assert abs(sku_data["SKU-A"]["fba_total"] - 147.90) < 0.01
    # Applying the same returns map again must not subtract twice
    # (that produced Blink Jan $519.75 / $78.03 / $146.16 instead of
    # $537.71 / $80.73 / $151.20).
    _apply_returns_to_sku_data(sku_data, returns, {"SKU-A": 0.15})
    assert abs(sku_data["SKU-A"]["revenue"] - 387.92) < 0.001
    assert abs(sku_data["SKU-A"]["referral_total"] - 58.14) < 0.01
    assert abs(sku_data["SKU-A"]["fba_total"] - 147.90) < 0.01


def test_physical_or_refund_counts_as_returned():
    """FBA physical returns count even when Finances refunds are not synced yet."""
    from aurora_data import returned_qty_for_sku

    assert returned_qty_for_sku(physical=1, refunded=0) == 1
    assert returned_qty_for_sku(physical=3, refunded=1) == 3
    assert returned_qty_for_sku(physical=0, refunded=2) == 2
    assert returned_qty_for_sku(physical=0, refunded=0) == 0


def test_returns_net_us_units_and_keep_mexico_unit():
    sku_data = {
        "SKU-A": {
            "units": 62,
            "revenue": 555.66,
            "referral_total": 83.43,
            "fba_total": 156.24,
            "units_by_marketplace": {
                "ATVPDKIKX0DER": 61,
                "A1AM78C64UM0Y8": 1,
            },
            "units_by_usd_price": {
                8.75: 61,
                14.69: 1,
            },
        }
    }
    returns = {
        "SKU-A": {
            "returned_units": 2,
            "refunded_revenue": 17.95,
            "refunded_referral": 2.70,
            "refunded_fulfillment": 5.04,
            "returned_units_by_marketplace": {"ATVPDKIKX0DER": 2},
            "returned_units_by_usd_price": {8.75: 2},
        }
    }
    _apply_returns_to_sku_data(sku_data, returns)
    assert sku_data["SKU-A"]["net_units"] == 60
    assert sku_data["SKU-A"]["units_by_marketplace"]["ATVPDKIKX0DER"] == 59
    assert sku_data["SKU-A"]["units_by_marketplace"]["A1AM78C64UM0Y8"] == 1
    assert sku_data["SKU-A"]["units_by_usd_price"][8.75] == 59
    assert sku_data["SKU-A"]["units_by_usd_price"][14.69] == 1
    assert sku_data["SKU-A"]["ordered_units_by_usd_price"][8.75] == 61
    assert sku_data["SKU-A"]["ordered_units_by_usd_price"][14.69] == 1


def test_phase_card_mexico_stays_in_net_fees():
    """B07H4S83D8: 75 US + 2 MX shipped, 2 US returns.

    MXN converts to USD first (not pesos-as-dollars). Fees then follow
    net units: FBA $4.20 × 75 = $315. Referral is 15% of net converted
    sales (~$201.26–$201.71), not $4.20 × 77 shipped.
    """
    from agent import apply_sale_price_fba
    from aurora_data import line_referral_fee

    sku = "ASG - PHASE CARD GAME PO2"
    sku_data = {
        sku: {
            "units": 77,
            "revenue": 1378.60,
            "referral_total": 206.79,
            "fba_total": 334.95,
            "asin": "B07H4S83D8",
            "units_by_usd_price": {17.96: 75, 21.40: 2},
            "units_by_marketplace": {
                "ATVPDKIKX0DER": 75,
                "A1AM78C64UM0Y8": 2,
            },
        }
    }
    returns = {
        sku: {
            "returned_units": 2,
            "refunded_revenue": 34.11,
            "returned_units_by_marketplace": {"ATVPDKIKX0DER": 2},
            "returned_units_by_usd_price": {17.96: 2},
        }
    }
    _apply_returns_to_sku_data(sku_data, returns, {sku: 0.15})
    d = sku_data[sku]
    assert d["units"] == 77
    assert d["ordered_units"] == 77
    assert d["returned_units"] == 2
    assert d["net_units"] == 75
    assert abs(d["revenue"] - 1344.49) < 0.001
    assert abs(d["units_by_marketplace"]["A1AM78C64UM0Y8"] - 2) < 0.001
    want_ref = round(206.79 - line_referral_fee(34.11, 0.15, 2), 2)
    assert abs(d["referral_total"] - want_ref) < 0.02
    assert abs(d["referral_total"] - 201.26) < 0.5
    rebuilt = apply_sale_price_fba(
        bill_units=75,
        units_by_usd_price=d["units_by_usd_price"],
        fba_per_usd_price={},
        catalog_fba_per_unit=4.20,
        include_fuel=False,
    )
    assert rebuilt == (315.00, 0.0)


def test_listing_under_10_uses_10_50_fba_on_higher_sale_prices():
    """B07TJT135V: listed $9.92 (lt10 FBA $3.38), sold at $11–$15.

    Mexico 21 units in the $10–$50 band are $4.20 each → $88.20, not
    $3.38 × 21 = $70.98. Referral stays 15% of converted USD (not the
    US $14.63 preview × 21 = $46.08).
    """
    from agent import apply_sale_price_fba

    rebuilt = apply_sale_price_fba(
        bill_units=21,
        units_by_usd_price={11.13: 21},
        fba_per_usd_price={},
        catalog_fba_per_unit=3.38,
        include_fuel=False,
        fba_per_band={"10_50": 4.20, "lt10": 3.38},
        fuel_per_band={"10_50": 0.15, "lt10": 0.12},
    )
    assert rebuilt == (88.20, 0.0)

    mixed = apply_sale_price_fba(
        bill_units=98,
        units_by_usd_price={14.70: 77, 11.13: 21},
        fba_per_usd_price={},
        catalog_fba_per_unit=3.38,
        include_fuel=False,
        fba_per_band={"10_50": 4.20, "lt10": 3.38},
        fuel_per_band={"10_50": 0.15, "lt10": 0.12},
    )
    assert mixed == (411.60, 0.0)


def test_referral_fallback_bills_net_converted_not_shipped():
    """Missing line-referral uses 15% of net converted USD × net units.

    B07H4S83D8 expected FBA $315 = $4.20 × 75 kept units. Gross shipped
    $1378.60 / 77 would overstate after the 2 US returns.
    """
    from agent import resolve_sku_referral_fba_fuel

    pf = {
        "listing_price": 17.96,
        "referral_per_unit": 2.69,
        "fba_per_unit": 4.20,
        "fuel_per_unit": 0.15,
        "fulfillment_per_unit": 4.35,
    }
    ref_net, fba, fuel, _ = resolve_sku_referral_fba_fuel(
        line_referral=0,
        line_fba=0,
        bill_units=75,
        revenue=1344.49,
        product_fees=pf,
        fee_estimate=None,
        include_fuel=False,
    )
    assert abs(ref_net - 201.67) < 1.0
    assert abs(fba - 315.00) < 0.05
    assert fuel == 0.0


def test_sp_api_aggregate_accrues_referral_on_converted_lines():
    """Live Amazon fallback must not skip referral the way a units-only
    defaultdict used to. MXN is converted, then 15% of USD."""
    from datetime import datetime, timezone

    from agent import _aggregate_sku_from_sp_api
    from currency_fx import UsdFx

    fx = UsdFx(source="test")
    fx.rates[("MXN", datetime(2026, 1, 15).date())] = 0.0534
    orders = [
        {
            "AmazonOrderId": "mx-1",
            "OrderStatus": "Shipped",
            "PurchaseDate": "2026-01-15T12:00:00Z",
            "MarketplaceId": "A1AM78C64UM0Y8",
        }
    ]
    items = [
        (
            "mx-1",
            {
                "payload": {
                    "OrderItems": [
                        {
                            "SellerSKU": "ASG - PHASE CARD GAME PO2",
                            "ASIN": "B07H4S83D8",
                            "QuantityOrdered": 2,
                            "ItemPrice": {"Amount": "801.18", "CurrencyCode": "MXN"},
                            "PromotionDiscount": {"Amount": "0", "CurrencyCode": "MXN"},
                        }
                    ]
                }
            },
        )
    ]
    sku_data, count = _aggregate_sku_from_sp_api(
        orders, items, [], [], fx=fx,
    )
    assert count == 1
    d = sku_data["ASG - PHASE CARD GAME PO2"]
    assert d["units"] == 2
    usd = 801.18 * 0.0534
    assert abs(d["revenue"] - usd) < 0.01
    assert d["referral_total"] > 0
    assert abs(d["referral_total"] - usd * 0.15) < 0.05
