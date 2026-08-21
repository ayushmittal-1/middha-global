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
    # Single total round (no per-line) — still snaps to 15%, not 0.73/4.89
    assert referral == round(1065.62 * 0.15, 2)
    assert referral != round(1065.62 * (0.73 / 4.89), 2)
    # Bundled $3.01 with fuel=0 is split to $2.91 + $0.10 before × units
    assert fba == 611.10
    assert fuel == 21.00
