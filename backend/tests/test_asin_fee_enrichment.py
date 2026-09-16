"""Eleet-style ASIN sibling FBA enrichment."""
from __future__ import annotations

import asyncio

from aurora_data import enrich_product_fees_from_asin_siblings


def test_enrich_copies_fba_from_merchant_sibling():
    sku_data = {
        "amzn.gr.EP_-_Grey_Knob_Set_of_-Ug6Yfy-VG": {
            "asin": "B07T8HD6R8",
            "units": 6,
            "revenue": 127.5,
        },
        "EP - Grey Knob Set of 12": {
            "asin": "B07T8HD6R8",
            "units": 6,
            "revenue": 99.95,
        },
    }
    fee_map = {
        "amzn.gr.EP_-_Grey_Knob_Set_of_-Ug6Yfy-VG": {
            "referral_per_unit": 3.825,
            "fba_per_unit": 0.0,
            "fuel_per_unit": 0.0,
            "fulfillment_per_unit": 0.0,
            "listing_price": 25.5,
            "asin": "B07T8HD6R8",
        },
        "EP - Grey Knob Set of 12": {
            "referral_per_unit": 4.5,
            "fba_per_unit": 5.42,
            "fuel_per_unit": 0.19,
            "fulfillment_per_unit": 5.61,
            "listing_price": 30.0,
            "asin": "B07T8HD6R8",
        },
    }
    user = {"_id": "6a75ade8bd9368b95bd1d30e"}

    out = asyncio.run(
        enrich_product_fees_from_asin_siblings(user, sku_data, fee_map)
    )

    gr = out["amzn.gr.EP_-_Grey_Knob_Set_of_-Ug6Yfy-VG"]
    assert gr["fba_per_unit"] == 5.42
    assert gr["fuel_per_unit"] == 0.19
    assert gr["fulfillment_per_unit"] == 5.61
    assert out["EP - Grey Knob Set of 12"]["fba_per_unit"] == 5.42
