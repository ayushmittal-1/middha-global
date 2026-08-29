"""Mixed sale prices: referral = sum of round(line_rev × category %, 2)."""

from agent import resolve_sku_referral_fba_fuel
from aurora_data import (
    line_referral_fee,
    recompute_referral_totals,
    snap_referral_rate,
)


def test_snap_referral_rate_uses_15_not_listing_ratio():
    # Fees API: $0.73 at listing $4.89 → must be 15%, not 0.73/4.89
    assert snap_referral_rate(4.89, 0.73) == 0.15
    assert snap_referral_rate(4.89, 0.73) != (0.73 / 4.89)


def test_multi_qty_line_uses_per_unit_round():
    # One line qty 209 @ $5.07 must be 0.76×209, not round(1059.63×0.15, 2)
    assert line_referral_fee(round(5.07 * 209, 2), 0.15, 209) == round(0.76 * 209, 2)
    assert line_referral_fee(round(5.07 * 209, 2), 0.15, 209) != round(
        round(5.07 * 209, 2) * 0.15, 2
    )


def test_nov_b007_mixed_price_referral_159_74():
    # 209 × $5.07 + 1 × $5.99 → 0.76×209 + 0.90×1 = $159.74
    sku = "AM - Kiwi Shoe Shine Sponge 7ml"
    orders = []
    for _ in range(209):
        orders.append(
            {
                "orderStatus": "Shipped",
                "orderItems": [
                    {
                        "sellerSku": sku,
                        "quantityOrdered": 1,
                        "itemPrice": {"amount": 5.07},
                        "itemSubtotal": {"amount": 5.07},
                        "promotionDiscount": {"amount": 0},
                        "referralFee": {"amount": 0.76},
                    }
                ],
            }
        )
    orders.append(
        {
            "orderStatus": "Shipped",
            "orderItems": [
                {
                    "sellerSku": sku,
                    "quantityOrdered": 1,
                    "itemPrice": {"amount": 5.99},
                    "itemSubtotal": {"amount": 5.99},
                    "promotionDiscount": {"amount": 0},
                    # Stale ratio-based fee from old sync
                    "referralFee": {"amount": 0.89},
                }
            ],
        }
    )

    sku_data = {sku: {"referral_total": 159.46, "units": 210, "revenue": 1065.62}}
    rate = snap_referral_rate(4.89, 0.73)
    assert rate == 0.15
    recompute_referral_totals(orders, sku_data, {sku: rate})

    assert round(0.76 * 209 + 0.90 * 1, 2) == 159.74
    assert round(sku_data[sku]["referral_total"], 2) == 159.74

    referral, _, _, source = resolve_sku_referral_fba_fuel(
        line_referral=sku_data[sku]["referral_total"],
        line_fba=3.01 * 210,
        bill_units=210,
        revenue=1065.62,
        product_fees={
            "referral_per_unit": 0.73,
            "fba_per_unit": 3.01,
            "listing_price": 4.89,
        },
    )
    assert source == "order_lines"
    assert referral == 159.74


def test_nov_b007_bulk_line_plus_one_also_159_74():
    """Same $159.74 when 209 units are one order line."""
    sku = "AM - Kiwi Shoe Shine Sponge 7ml"
    orders = [
        {
            "orderStatus": "Shipped",
            "orderItems": [
                {
                    "sellerSku": sku,
                    "quantityOrdered": 209,
                    "itemPrice": {"amount": round(5.07 * 209, 2)},
                    "itemSubtotal": {"amount": round(5.07 * 209, 2)},
                    "promotionDiscount": {"amount": 0},
                }
            ],
        },
        {
            "orderStatus": "Shipped",
            "orderItems": [
                {
                    "sellerSku": sku,
                    "quantityOrdered": 1,
                    "itemPrice": {"amount": 5.99},
                    "itemSubtotal": {"amount": 5.99},
                    "promotionDiscount": {"amount": 0},
                }
            ],
        },
    ]
    sku_data = {sku: {"referral_total": 0.0}}
    recompute_referral_totals(orders, sku_data, {sku: 0.15})
    assert round(sku_data[sku]["referral_total"], 2) == 159.74


def test_line_referral_fee_unit_math():
    assert line_referral_fee(5.07, 0.15) == 0.76
    assert line_referral_fee(5.99, 0.15) == 0.90
    # Half-up: 15.30 × 0.15 = 2.295 → $2.30 (Python round → $2.29)
    assert line_referral_fee(15.30, 0.15) == 2.30


def test_nov_b016_half_up_referral_218_50():
    """AM B016715XXY Nov 2025: 59@$15.30 + 35@$15.32 + 1@$15.34 → $218.50."""
    sku = "AM SKIP BO Phase 10 Bundle"
    orders = []
    for unit, n in ((15.30, 59), (15.32, 35), (15.34, 1)):
        for _ in range(n):
            orders.append(
                {
                    "orderStatus": "Shipped",
                    "orderItems": [
                        {
                            "sellerSku": sku,
                            "quantityOrdered": 1,
                            "itemPrice": {"amount": unit},
                            "itemSubtotal": {"amount": unit},
                            "promotionDiscount": {"amount": 0},
                        }
                    ],
                }
            )
    sku_data = {sku: {"referral_total": 0.0}}
    recompute_referral_totals(orders, sku_data, {sku: 0.15})
    assert round(sku_data[sku]["referral_total"], 2) == 218.50


def test_pre_april_2026_fuel_is_zero():
    """Amazon FBA fuel surcharge starts April 2026 — earlier months Fuel=$0."""
    from agent import fuel_surcharge_applies_for_window, resolve_sku_referral_fba_fuel

    assert fuel_surcharge_applies_for_window(
        display_start="2026-01-01", display_end="2026-01-31",
    ) is False
    assert fuel_surcharge_applies_for_window(
        display_start="2026-03-01", display_end="2026-03-31",
    ) is False
    assert fuel_surcharge_applies_for_window(
        display_start="2026-04-01", display_end="2026-04-30",
    ) is True

    referral, fba, fuel, source = resolve_sku_referral_fba_fuel(
        line_referral=159.74,
        line_fba=3.01,
        bill_units=210,
        revenue=1065.62,
        product_fees={
            "referral_per_unit": 0.73,
            "fba_per_unit": 2.91,
            "fuel_per_unit": 0.10,
            "fulfillment_per_unit": 3.01,
            "listing_price": 4.89,
        },
        include_fuel=False,
    )
    assert fuel == 0.0
    assert fba == 611.10


def test_pre_april_peels_bundled_live_fba_fee():
    """January must use base $2.43, not today's bundled $2.52 (base+fuel)."""
    from amazon_sp import split_bundled_fulfillment_total

    assert split_bundled_fulfillment_total(2.52) == (2.43, 0.09)
    referral, fba, fuel, source = resolve_sku_referral_fba_fuel(
        line_referral=78.03,
        line_fba=round(2.52 * 60, 2),
        bill_units=60,
        revenue=520.0,
        product_fees={
            "referral_per_unit": 1.50,
            "fba_per_unit": 2.43,
            "fuel_per_unit": 0.09,
            "fulfillment_per_unit": 2.52,
            "listing_price": 9.99,
        },
        include_fuel=False,
    )
    assert fuel == 0.0
    assert fba == 145.80  # 2.43 × 60, not 2.52 × 60 = 151.20

    # Unpacked catalog (fbaFee stored as the live bundle, fuel=0) must peel too.
    _, fba_unpacked, fuel_unpacked, _ = resolve_sku_referral_fba_fuel(
        line_referral=0,
        line_fba=0,
        bill_units=60,
        revenue=520.0,
        product_fees={
            "referral_per_unit": 1.50,
            "fba_per_unit": 2.52,
            "fuel_per_unit": 0.0,
            "fulfillment_per_unit": 2.52,
            "listing_price": 9.99,
        },
        include_fuel=False,
    )
    assert fuel_unpacked == 0.0
    assert fba_unpacked == 145.80


def test_mexico_unit_uses_us_price_band_fba_not_mxn():
    """Mexico unit at ~$15 USD uses Amazon's $10–$50 US FBA ($3.32).

    Listing $9.99 is the under-$10 band ($2.43). Converting Mexico Fees
    API 33 MXN to USD ($1.82) was the $140.33 row; Seller Central is
    $2.43 × 57 + $3.32 = $141.83 when 2 of 60 units are returned.
    """
    from agent import apply_foreign_marketplace_fba

    referral, fba, fuel, source = resolve_sku_referral_fba_fuel(
        line_referral=78.29,
        line_fba=round(2.52 * 58, 2),
        bill_units=58,
        revenue=521.48,
        product_fees={
            "referral_per_unit": 1.50,
            "fba_per_unit": 2.43,
            "fuel_per_unit": 0.09,
            "fulfillment_per_unit": 2.52,
            "listing_price": 9.99,
        },
        include_fuel=False,
    )
    fba, fuel = apply_foreign_marketplace_fba(
        fba,
        fuel,
        bill_units=58,
        units_by_marketplace={
            "ATVPDKIKX0DER": 57,
            "A1AM78C64UM0Y8": 1,
        },
        foreign_fba_per_unit_usd={"A1AM78C64UM0Y8": 3.32},
        include_fuel=False,
    )
    assert fuel == 0.0
    assert fba == round(2.43 * 57 + 3.32, 2)
    assert fba == 141.83


def test_sale_price_fba_covers_us_and_foreign_off_band():
    """Production path: per-line USD price, not only non-US marketplaces.

    Eleet $6.99 US units were missed by marketplace-only refetch because
    they are Amazon.com. Blink Mexico is the same $10–$50 band jump.
    """
    from agent import apply_sale_price_fba

    eleet = apply_sale_price_fba(
        bill_units=12,
        units_by_usd_price={6.99: 10, 13.13: 2},
        fba_per_usd_price={6.99: 3.38},
        catalog_fba_per_unit=4.20,
        include_fuel=False,
    )
    assert eleet == (42.20, 0.0)

    blink = apply_sale_price_fba(
        bill_units=58,
        units_by_usd_price={8.75: 57, 14.69: 1},
        fba_per_usd_price={14.69: 3.32},
        catalog_fba_per_unit=2.43,
        include_fuel=False,
    )
    assert blink == (141.83, 0.0)

    blink_band = apply_sale_price_fba(
        bill_units=58,
        units_by_usd_price={8.75: 57, 14.69: 1},
        fba_per_usd_price={},
        catalog_fba_per_unit=2.43,
        include_fuel=False,
        fba_per_band={"10_50": 3.32},
    )
    assert blink_band == (141.83, 0.0)


def test_order_line_referral_preferred_over_single_price_estimate():
    line_ref = 159.74
    line_fba = 3.01 * 210
    referral, fba, fuel, source = resolve_sku_referral_fba_fuel(
        line_referral=line_ref,
        line_fba=line_fba,
        bill_units=210,
        revenue=1065.62,
        product_fees={
            "referral_per_unit": 0.73,
            "fba_per_unit": 2.91,
            "fuel_per_unit": 0.10,
            "fulfillment_per_unit": 3.01,
            "listing_price": 4.89,
        },
        fee_estimate={"referral": 0.76, "fba": 2.91, "fuel_surcharge": 0.10},
    )
    assert source == "order_lines"
    assert referral == 159.74
    # Must use per-unit FBA/fuel ($2.91 / $0.10), not split(632.10)
    assert fba == 611.10
    assert fuel == 21.00


def test_aggregate_fulfillment_split_matches_revenue_calculator():
    from amazon_sp import (
        split_bundled_fulfillment_for_units,
        split_bundled_fulfillment_total,
    )

    assert split_bundled_fulfillment_total(3.01) == (2.91, 0.10)
    # Wrong path: split the aggregate
    assert split_bundled_fulfillment_total(632.10) == (610.72, 21.38)
    # Correct path: per-unit then × units
    assert split_bundled_fulfillment_for_units(632.10, 210) == (611.10, 21.00)


def test_sparse_order_fulfillment_uses_product_fees():
    """Andexports-style: only a few lines have fulfillmentFee — don't understate."""
    # 210 units, but only ~10 lines stored $3.01 → line_ful ≈ 30.10
    line_ful = round(3.01 * 10, 2)
    referral, fba, fuel, source = resolve_sku_referral_fba_fuel(
        line_referral=159.74,
        line_fba=line_ful,
        bill_units=210,
        revenue=1065.62,
        product_fees={
            "referral_per_unit": 0.73,
            "fba_per_unit": 2.91,
            "fuel_per_unit": 0.10,
            "fulfillment_per_unit": 3.01,
            "listing_price": 4.89,
        },
    )
    assert source == "products_fees"
    assert referral == 159.74
    assert fba == 611.10
    assert fuel == 21.00
    # Must NOT split the sparse $30.10 across 210 units
    assert fba + fuel != line_ful


def test_full_order_fulfillment_still_uses_line_split():
    """Allmart-style: every unit has fulfillmentFee — keep line-based split."""
    line_ful = round(3.01 * 210, 2)
    referral, fba, fuel, source = resolve_sku_referral_fba_fuel(
        line_referral=159.74,
        line_fba=line_ful,
        bill_units=210,
        revenue=1065.62,
        product_fees={
            "referral_per_unit": 0.73,
            "fba_per_unit": 2.91,
            "fuel_per_unit": 0.10,
            "fulfillment_per_unit": 3.01,
            "listing_price": 4.89,
        },
    )
    assert source == "order_lines"
    assert fba == 611.10
    assert fuel == 21.00


def test_products_fees_uses_snapped_15_percent():
    referral, fba, fuel, source = resolve_sku_referral_fba_fuel(
        line_referral=0,
        line_fba=0,
        bill_units=210,
        revenue=1065.62,
        product_fees={
            "referral_per_unit": 0.73,
            "fba_per_unit": 3.01,
            "fuel_per_unit": 0.0,
            "fulfillment_per_unit": 3.01,
            "listing_price": 4.89,
        },
    )
    assert source == "products_fees"
    # Per-unit half-up × units (not a single round of total revenue)
    assert referral == round(0.76 * 210, 2)
    assert referral != round(1065.62 * (0.73 / 4.89), 2)
    # Bundled $3.01 with fuel=0 is split to $2.91 + $0.10 before × units
    assert fba == 611.10
    assert fuel == 21.00
