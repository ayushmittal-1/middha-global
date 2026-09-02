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
from currency_fx import UsdFx, infer_line_currency, infer_marketplace_id, is_us_marketplace, FRANKFURTER_URL


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
    raw = 10 * 6.99 + 2 * 17.98
    assert round(raw, 2) == 105.86
    # Safety net: even an FX table with no CAD rate still converts via
    # marketplace quotes + fallback (must never ship 105.86).
    sku_data, _, _ = aggregate_sku_metrics_from_orders(
        _jan_asin_orders(), fx=UsdFx(source="identity", fallback={"USD": 1.0}),
    )
    d = sku_data[SKU]
    assert d["units"] == 12
    assert round(d["revenue"], 2) < 100
    assert 17.98 not in d["units_by_usd_price"]


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
    assert d["units_by_usd_price"][6.99] == 10
    assert d["units_by_usd_price"][cad_unit_usd] == 2


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
    assert is_us_marketplace("ATVPDKIKX0DER")
    assert is_us_marketplace("")
    assert not is_us_marketplace("A1AM78C64UM0Y8")
    assert infer_marketplace_id(orders[1]) == "A2EUQ1WTGCTBG2"
    kept = filter_orders_by_marketplace(orders, "ATVPDKIKX0DER")
    assert [o["amazonOrderId"] for o in kept] == ["US"]
    kept_ca = filter_orders_by_marketplace(orders, "A2EUQ1WTGCTBG2")
    assert [o["amazonOrderId"] for o in kept_ca] == ["CA"]


def test_infer_currency_from_amazon_ca_channel():
    assert infer_line_currency({"salesChannel": "Amazon.ca"}, None) == "CAD"
    assert infer_line_currency({"salesChannel": "Amazon.com"}, None) == "USD"


def test_mx_channel_overrides_usd_stamp():
    """US fee-map currency on an Amazon.com.mx line is still Mexican pesos."""
    item = {"itemPrice": {"amount": 267.13, "currencyCode": "USD"}}
    order = {"salesChannel": "Amazon.com.mx", "orderItems": [item]}
    assert infer_line_currency(order, item) == "MXN"


def test_order_total_mxn_beats_usd_item_price_stamp():
    """itemPrice labelled USD but orderTotal MXN — still pesos."""
    item = {"itemPrice": {"amount": 287.37, "currencyCode": "USD"}}
    order = {
        "orderTotal": {"amount": 287.37, "currencyCode": "MXN"},
        "orderItems": [item],
    }
    assert infer_line_currency(order, item) == "MXN"


def test_b08hgll647_march24_mxn_not_added_as_dollars():
    """B08HGLL647 24 Mar 2026: Amazon.com $15.95 + Amazon.com.mx MXN 287.37.

    Order sync stored referral 43.11 labelled USD (15% of pesos). Raw sum
    is $303.32 revenue / $45.50 referral. Convert MXN first; referral is
    15% of converted USD, never the 43.11 stamp. FBA stays US catalog
    dollars (not 287-as-$287 gt50 Fees API).
    """
    from aurora_data import repair_unconverted_foreign_price_buckets
    from currency_fx import looks_like_unconverted_foreign_face

    sku = "ASG-UNO FLIP+PHASE 10"
    asin = "B08HGLL647"
    mxn_usd = 0.05591
    day = datetime(2026, 3, 24, 3, 37, 4, tzinfo=timezone.utc)
    fx = UsdFx(source="test")
    fx.rates[("MXN", date(2026, 3, 24))] = mxn_usd

    us = {
        "amazonOrderId": "111-0337300-8003429",
        "orderStatus": "Shipped",
        "salesChannel": "Amazon.com",
        "purchaseDate": day,
        "orderTotal": {"amount": 15.95, "currencyCode": "USD"},
        "orderItems": [{
            "sellerSku": sku,
            "asin": asin,
            "quantityOrdered": 1,
            "itemPrice": {"amount": 15.95, "currencyCode": "USD"},
            "itemSubtotal": {"amount": 15.95, "currencyCode": "USD"},
            "referralFee": {"amount": 2.39, "currencyCode": "USD"},
            "fulfillmentFee": {"amount": 4.35, "currencyCode": "USD"},
        }],
    }
    mx = {
        "amazonOrderId": "702-0581238-9233808",
        "orderStatus": "Shipped",
        "salesChannel": "Amazon.com.mx",
        "purchaseDate": day,
        "orderTotal": {"amount": 287.37, "currencyCode": "MXN"},
        "orderItems": [{
            "sellerSku": sku,
            "asin": asin,
            "quantityOrdered": 1,
            "itemPrice": {"amount": 287.37, "currencyCode": "MXN"},
            "itemSubtotal": {"amount": 287.37, "currencyCode": "MXN"},
            "referralFee": {"amount": 43.11, "currencyCode": "USD"},
            "fulfillmentFee": {"amount": 4.35, "currencyCode": "USD"},
        }],
    }
    sku_data, _, _ = aggregate_sku_metrics_from_orders([us, mx], fx=fx)
    d = sku_data[sku]
    mxn_usd_rev = round_money(287.37 * mxn_usd)
    assert mxn_usd_rev == 16.07
    assert round(d["revenue"], 2) == round_money(15.95 + mxn_usd_rev)
    assert round(d["revenue"], 2) == 32.02
    assert round(d["revenue"], 2) < 50
    expected_ref = round_money(
        line_referral_fee(15.95, 0.15, 1) + line_referral_fee(mxn_usd_rev, 0.15, 1)
    )
    assert round(d["referral_total"], 2) == expected_ref
    assert round(d["referral_total"], 2) < 10
    # Must not keep the 43.11 USD stamp or 15% of pesos-as-dollars.
    assert round(d["referral_total"], 2) != 45.50
    assert round(d["fba_total"], 2) == 8.70
    assert 287.37 not in d["units_by_usd_price"]
    assert d["units_by_usd_price"][15.95] == 1
    assert d["units_by_usd_price"][mxn_usd_rev] == 1
    assert looks_like_unconverted_foreign_face(287.37, 13.49)
    assert not looks_like_unconverted_foreign_face(16.07, 13.49)

    broken = UsdFx(source="identity", fallback={"USD": 1.0})
    sku_raw, _, _ = aggregate_sku_metrics_from_orders([us, mx], fx=broken)
    d_raw = sku_raw[sku]
    assert 287.37 not in d_raw["units_by_usd_price"]
    assert round(d_raw["revenue"], 2) < 50
    repair_unconverted_foreign_price_buckets(sku_raw, broken)
    assert round(sku_raw[sku]["revenue"], 2) < 50


def test_december_cad_uses_december_rate_not_today():
    """Same CAD face value in December vs March must convert at that day's rate.

    A static / 'today' rate would make C$100 identical in both months.
    """
    sku = "FX-DATE-CAD"
    dec_rate = 0.70
    mar_rate = 0.73
    fx = UsdFx(source="test", fallback={"USD": 1.0, "CAD": 0.99})
    fx.rates[("CAD", date(2025, 12, 15))] = dec_rate
    fx.rates[("CAD", date(2026, 3, 24))] = mar_rate

    assert fx.rate("CAD", date(2025, 12, 15)) == dec_rate
    assert fx.rate("CAD", date(2026, 3, 24)) == mar_rate
    assert fx.rate("CAD", date(2025, 12, 15)) != fx.fallback["CAD"]
    dec_usd = fx.to_usd(100, "CAD", date(2025, 12, 15))
    mar_usd = fx.to_usd(100, "CAD", date(2026, 3, 24))
    assert dec_usd == round_money(100 * dec_rate)
    assert mar_usd == round_money(100 * mar_rate)
    assert dec_usd != mar_usd

    def _cad_order(when, oid):
        return {
            "amazonOrderId": oid,
            "orderStatus": "Shipped",
            "salesChannel": "Amazon.ca",
            "purchaseDate": when,
            "orderTotal": {"amount": 100.0, "currencyCode": "CAD"},
            "orderItems": [{
                "sellerSku": sku,
                "asin": "B0FXDATECAD",
                "quantityOrdered": 1,
                "itemPrice": {"amount": 100.0, "currencyCode": "CAD"},
                "itemSubtotal": {"amount": 100.0, "currencyCode": "CAD"},
                "referralFee": {"amount": 15.0, "currencyCode": "USD"},
                "fulfillmentFee": {"amount": 4.35, "currencyCode": "USD"},
            }],
        }

    sku_data, _, _ = aggregate_sku_metrics_from_orders(
        [
            _cad_order(datetime(2025, 12, 15, 18, 0, tzinfo=timezone.utc), "701-dec"),
            _cad_order(datetime(2026, 3, 24, 18, 0, tzinfo=timezone.utc), "701-mar"),
        ],
        fx=fx,
    )
    d = sku_data[sku]
    assert round(d["revenue"], 2) == round_money(dec_usd + mar_usd)
    assert round(d["revenue"], 2) != round_money(200 * 0.99)


def test_weekend_order_uses_prior_business_day_not_fallback():
    """Saturday / holiday should walk back to Friday, not jump to 'today'."""
    fx = UsdFx(source="test", fallback={"USD": 1.0, "CAD": 0.99})
    fx.rates[("CAD", date(2025, 12, 19))] = 0.71  # Friday
    assert fx.rate("CAD", date(2025, 12, 20)) == 0.71
    assert fx.rate("CAD", date(2025, 12, 21)) == 0.71
    assert fx.rate("CAD", datetime(2025, 12, 20, 16, 45, tzinfo=timezone.utc)) == 0.71


def test_fill_calendar_covers_christmas_weekend():
    from currency_fx import _fill_calendar
    daily = {
        date(2025, 12, 24): 0.72,
        date(2025, 12, 29): 0.73,
    }
    filled = _fill_calendar(daily, date(2025, 12, 24), date(2025, 12, 29))
    assert filled[date(2025, 12, 25)] == 0.72
    assert filled[date(2025, 12, 26)] == 0.72
    assert filled[date(2025, 12, 29)] == 0.73


def test_frankfurter_uses_current_host():
    """api.frankfurter.app 301s to .dev and httpx was not following redirects."""
    assert "frankfurter.dev" in FRANKFURTER_URL


def test_wise_history_url_matches_seller_page():
    from currency_fx import wise_history_url, parse_wise_history_html
    url = wise_history_url("CAD", date(2026, 1, 30))
    assert url.endswith("/cad-to-usd-rate/history/30-01-2026")
    assert "wise.com" in url
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"model":{"historicalRate":{"value":0.741152},'
        '"rate":{"value":0.718056}}}}}</script>'
    )
    assert parse_wise_history_html(html) == 0.741152
    # Prefer the dated historicalRate, not today's live rate on the same page.
    assert parse_wise_history_html(html) != 0.718056


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


def test_january_blink_mxn_not_added_as_dollars():
    """B0037W5Y2W January: Amazon.com USD + Amazon.com.mx MXN 267.13.

    Raw sum treated 267.13 pesos as $267.13. Convert MXN first. Referral
    stored as 15% of pesos labelled USD must not stay $40.07.
    """
    blink = "ASG - Blink Game PO1"
    asin = "B0037W5Y2W"
    mxn_usd = 0.049
    jan = datetime(2026, 1, 10, tzinfo=timezone.utc)
    fx = UsdFx(source="test")
    fx.rates[("MXN", date(2026, 1, 10))] = mxn_usd

    us = {
        "amazonOrderId": "111-us",
        "orderStatus": "Shipped",
        "salesChannel": "Amazon.com",
        "purchaseDate": jan,
        "orderTotal": {"amount": 8.75, "currencyCode": "USD"},
        "orderItems": [{
            "sellerSku": blink,
            "asin": asin,
            "quantityOrdered": 1,
            "itemPrice": {"amount": 8.75, "currencyCode": "USD"},
            "itemSubtotal": {"amount": 8.75, "currencyCode": "USD"},
            "referralFee": {"amount": 1.31, "currencyCode": "USD"},
            "fulfillmentFee": {"amount": 2.52, "currencyCode": "USD"},
        }],
    }
    mx = {
        "amazonOrderId": "702-9519082-9158651",
        "orderStatus": "Shipped",
        "salesChannel": "Amazon.com.mx",
        "purchaseDate": jan,
        "orderTotal": {"amount": 267.13, "currencyCode": "MXN"},
        "orderItems": [{
            "sellerSku": blink,
            "asin": asin,
            "quantityOrdered": 1,
            "itemPrice": {"amount": 267.13, "currencyCode": "MXN"},
            "itemSubtotal": {"amount": 267.13, "currencyCode": "MXN"},
            "referralFee": {"amount": 40.07, "currencyCode": "USD"},
            "fulfillmentFee": {"amount": 2.52, "currencyCode": "USD"},
        }],
    }
    sku_data, _, _ = aggregate_sku_metrics_from_orders([us, mx], fx=fx)
    d = sku_data[blink]
    mxn_usd_rev = round_money(267.13 * mxn_usd)
    assert mxn_usd_rev == 13.09
    assert round(d["revenue"], 2) == round_money(8.75 + mxn_usd_rev)
    assert round(d["revenue"], 2) == 21.84
    # Must not be 8.75 + 267.13
    assert round(d["revenue"], 2) < 50
    expected_ref = round_money(
        line_referral_fee(8.75, 0.15, 1) + line_referral_fee(mxn_usd_rev, 0.15, 1)
    )
    assert round(d["referral_total"], 2) == expected_ref
    assert round(d["referral_total"], 2) < 10
    # US catalog FBA stays dollars on the MX line at aggregation time
    # (2.52 + 2.52). Profitability later peels January fuel and replaces
    # the Mexico unit with Mexico FBA.
    assert round(d["fba_total"], 2) == 5.04
    assert d["units_by_marketplace"]["ATVPDKIKX0DER"] == 1
    assert d["units_by_marketplace"]["A1AM78C64UM0Y8"] == 1
    assert d["units_by_usd_price"][8.75] == 1
    assert d["units_by_usd_price"][mxn_usd_rev] == 1
    assert d["marketplace_fee_quotes"]["A1AM78C64UM0Y8"]["amount"] == 267.13
    assert d["marketplace_fee_quotes"]["A1AM78C64UM0Y8"]["currency"] == "MXN"


def test_eleet_january_fba_uses_under_10_band_not_listing():
    """B09JZL4J8S January: listing $16.99 FBA is $4.20, but 10 US units
    sold at $6.99 (under $10 → $3.38). Canada ~$13 stays $4.20.

    Catalog × 12 = $50.40 (the wrong UI row). Correct is
    $3.38 × 10 + $4.20 × 2 = $42.20.
    """
    from agent import apply_sale_price_fba, usd_fba_price_band

    assert usd_fba_price_band(6.99) == "lt10"
    assert usd_fba_price_band(16.99) == "10_50"
    assert usd_fba_price_band(12.93) == "10_50"

    rebuilt = apply_sale_price_fba(
        bill_units=12,
        units_by_usd_price={6.99: 10, 12.93: 2},
        fba_per_usd_price={6.99: 3.38},
        catalog_fba_per_unit=4.20,
        include_fuel=False,
    )
    assert rebuilt == (42.20, 0.0)
    assert rebuilt[0] != round(4.20 * 12, 2)

    # Same result via band so $6.99 and $7.49 share one under-$10 rate.
    by_band = apply_sale_price_fba(
        bill_units=12,
        units_by_usd_price={6.99: 7, 7.49: 3, 12.93: 2},
        fba_per_usd_price={},
        catalog_fba_per_unit=4.20,
        include_fuel=False,
        fba_per_band={"lt10": 3.38},
    )
    assert by_band == (42.20, 0.0)


def test_eleet_january_referral_uses_8pct_under_10():
    """Listing $16.99 is 15% ($14.48 on $96.48). $6.99 units are 8%.

    10 × $0.56 + 2 × 15% of ~$13.29 = $5.60 + $3.98 = $9.58
    (user $9.56 is the same split; cents follow CAD FX / rounding).
    """
    from agent import apply_sale_price_referral
    from aurora_data import line_referral_fee, round_money

    under10 = round_money(6.99 * 0.08)
    assert under10 == 0.56
    rebuilt = apply_sale_price_referral(
        bill_units=12,
        units_by_usd_price={6.99: 10, 13.29: 2},
        category_rate=0.15,
        referral_per_band={"lt10": under10},
    )
    cad_ref = line_referral_fee(13.29 * 2, 0.15, 2)
    assert rebuilt == round(0.56 * 10 + cad_ref, 2)
    assert rebuilt == 9.58
    assert rebuilt != round_money(96.48 * 0.15)


def test_february_fba_peels_live_bundle_after_sale_price_quote():
    """B07Z3TD8XG Feb: listing FBA $5.61 includes fuel; Fuel column is $0.

    13 net units at $19.99/$25.94 (same $10–$50 band as listing $30).
    Live Fees API still returns $5.61. Must not show $5.61 × 13 = $72.93.
    """
    from agent import apply_sale_price_fba
    from amazon_sp import split_bundled_fulfillment_total

    assert split_bundled_fulfillment_total(5.61) == (5.42, 0.19)
    rebuilt = apply_sale_price_fba(
        bill_units=13,
        units_by_usd_price={19.99: 12, 25.94: 1},
        fba_per_usd_price={},
        catalog_fba_per_unit=5.42,
        include_fuel=False,
        fba_per_band={"10_50": 5.61},
        fuel_per_band={"10_50": 0.19},
    )
    assert rebuilt == (70.46, 0.0)
    assert rebuilt[0] != round(5.61 * 13, 2)


def test_same_fba_band_does_not_store_live_fba_quote():
    """Listing $30 vs sale $19.99: referral tier changes, FBA band does not.

    Live $5.61 must not be stored on ``10_50`` or every SKU in that band
    gets February fuel stuffed back into FBA.
    """
    import asyncio
    import amazon_sp
    from agent import fetch_fba_rates_by_sale_price

    async def fake_batch(items, **kwargs):
        return {
            "B07Z3TD8XG": {
                "fba": 5.61,
                "fuel_surcharge": 0.19,
                "referral": 3.00,
            }
        }

    original = amazon_sp.get_fees_estimates_batch
    amazon_sp.get_fees_estimates_batch = fake_batch
    try:
        out = asyncio.run(
            fetch_fba_rates_by_sale_price(
                {
                    "EP - Green Knob Set of 12": {
                        "asin": "B07Z3TD8XG",
                        "units_by_usd_price": {19.99: 13},
                    }
                },
                {
                    "EP - Green Knob Set of 12": {
                        "listing_price": 30.0,
                        "is_fba": True,
                        "asin": "B07Z3TD8XG",
                    }
                },
            )
        )
    finally:
        amazon_sp.get_fees_estimates_batch = original

    sku = "EP - Green Knob Set of 12"
    assert "10_50" not in out[sku]
    assert out[sku]["le20"][2] == 3.00


def test_fetch_quotes_when_referral_tier_differs_same_fba_band():
    """Grocery-style: listing $19.99 and sale $12.99 are both $10–$50 FBA.

    Old skip (FBA band only) would keep 15%. Must still call Fees API.
    """
    from agent import needs_sale_price_fee_quote, referral_price_tier, usd_fba_price_band

    assert usd_fba_price_band(12.99) == usd_fba_price_band(19.99) == "10_50"
    assert referral_price_tier(12.99) != referral_price_tier(19.99)
    assert needs_sale_price_fee_quote(12.99, 19.99)
    assert needs_sale_price_fee_quote(6.99, 16.99)
    assert not needs_sale_price_fee_quote(16.50, 16.99)


def test_fetch_fba_rates_only_for_off_listing_band():
    """Fees API is called for every off-listing FBA or referral tier."""
    import asyncio
    import amazon_sp
    from agent import fetch_fba_rates_by_sale_price

    calls = []

    async def fake_batch(items, **kwargs):
        out = {}
        for entry in items:
            asin = entry[0]
            price = round(float(entry[1]), 2)
            calls.append((asin, price))
            if price < 10:
                out[asin] = {"fba": 3.38, "fuel_surcharge": 0.12, "referral": 0.56}
            else:
                out[asin] = {"fba": 4.20, "fuel_surcharge": 0.15, "referral": 1.94}
        return out

    original = amazon_sp.get_fees_estimates_batch
    amazon_sp.get_fees_estimates_batch = fake_batch
    try:
        sku_data = {
            SKU: {
                "asin": ASIN,
                "units_by_usd_price": {6.99: 10, 12.93: 2},
            }
        }
        product_fees = {
            SKU: {
                "listing_price": 16.99,
                "fba_per_unit": 4.20,
                "is_fba": True,
                "asin": ASIN,
            }
        }
        out = asyncio.run(fetch_fba_rates_by_sale_price(sku_data, product_fees))
    finally:
        amazon_sp.get_fees_estimates_batch = original

    prices = sorted(p for _, p in calls)
    assert prices == [6.99, 12.93]
    assert out[SKU]["lt10"][0] == 3.38
    assert out[SKU]["le10"][2] == 0.56
    # $12.93 is the same FBA band as listing $16.99 — referral quote only.
    assert "10_50" not in out[SKU]
    assert out[SKU]["le15"][2] == 1.94


def test_fetch_fba_rates_runs_for_every_asin_in_the_window():
    """No per-ASIN special case — Blink and Eleet in the same month both adjust."""
    import asyncio
    import amazon_sp
    from agent import fetch_fba_rates_by_sale_price

    blink_sku = "ASG - Blink Game PO1"
    blink_asin = "B0037W5Y2W"

    async def fake_batch(items, **kwargs):
        out = {}
        for entry in items:
            asin = entry[0]
            price = round(float(entry[1]), 2)
            if asin == ASIN:
                if price < 10:
                    out[asin] = {"fba": 3.38, "fuel_surcharge": 0.12, "referral": 0.56}
                else:
                    out[asin] = {"fba": 4.20, "fuel_surcharge": 0.15, "referral": 1.94}
            elif asin == blink_asin:
                out[asin] = {"fba": 3.32, "fuel_surcharge": 0.12, "referral": 2.20}
                assert 10 <= price < 50
        return out

    original = amazon_sp.get_fees_estimates_batch
    amazon_sp.get_fees_estimates_batch = fake_batch
    try:
        sku_data = {
            SKU: {
                "asin": ASIN,
                "units_by_usd_price": {6.99: 10, 12.93: 2},
            },
            blink_sku: {
                "asin": blink_asin,
                "units_by_usd_price": {8.75: 57, 14.69: 1},
            },
        }
        product_fees = {
            SKU: {"listing_price": 16.99, "is_fba": True, "asin": ASIN},
            blink_sku: {"listing_price": 9.99, "is_fba": True, "asin": blink_asin},
        }
        out = asyncio.run(fetch_fba_rates_by_sale_price(sku_data, product_fees))
    finally:
        amazon_sp.get_fees_estimates_batch = original

    assert out[SKU]["lt10"][0] == 3.38
    assert out[blink_sku]["10_50"][0] == 3.32



