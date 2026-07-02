"""One-shot seed: insert a TEST campaign into Aurora's `ads` collection
covering every SellerSKU in `products` for the given user, so we can
exercise the per-SKU ad-allocation path on Profitability.

Aurora's Ad schema doesn't natively have a `skus` field — we add it
here. Mongoose ignores unknown fields on read, so this doesn't break
Aurora's own writes to the doc; our Python side reads it via
`get_ad_spend_for_range`.

Run:
    cd backend && python _seed_test_campaign.py
"""

import asyncio
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv("/Users/ayushmittal/Downloads/middha-global/.env")

from bson import ObjectId  # noqa: E402
from auth import _db  # noqa: E402

USER_ID = ObjectId("69e0c751361814d89de3fa0b")
PROFILE_ID = "TEST_PROFILE"
CAMPAIGN_ID = "TEST_CAMPAIGN_ALL_SKUS"


async def main() -> None:
    db = _db()
    now = datetime.now(timezone.utc)
    end = now
    start = end - timedelta(days=60)  # last 2 months

    skus = await db.products.distinct("sku", {"sellerId": USER_ID})
    skus = [s for s in skus if s]
    print(f"found {len(skus)} SellerSKUs on Aurora products for this user")

    doc = {
        "sellerId": USER_ID,
        "profileId": PROFILE_ID,
        "campaignId": CAMPAIGN_ID,
        "campaignName": "TEST Sponsored Products — All Catalog",
        "status": "Enabled",
        "country": "US",
        "campaignType": "Sponsored Products",
        "portfolio": None,
        "startDate": start,
        "endDate": None,
        "budget": {"amount": 50, "currencyCode": "USD"},
        "spend": {"amount": 600, "currencyCode": "USD"},
        "cpc": 0.4,
        "impressions": 12000,
        "clicks": 1500,
        "ctr": 12.5,
        "orders": 90,
        "sales": {"amount": 1800, "currencyCode": "USD"},
        "conversionRate": 6.0,
        "unitsSold": 90,
        "acos": 33.3,
        "roas": 3.0,
        "tacos": None,
        "metricsStartDate": start,
        "metricsEndDate": end,
        "lastSynced": now,
        "updatedAt": now,
        # Non-standard field for our per-SKU attribution path.
        "skus": skus,
    }
    res = await db.ads.update_one(
        {"sellerId": USER_ID, "profileId": PROFILE_ID, "campaignId": CAMPAIGN_ID},
        {"$set": doc, "$setOnInsert": {"createdAt": now}},
        upsert=True,
    )
    action = "inserted" if res.upserted_id else "updated"
    print(f"{action} TEST campaign — {CAMPAIGN_ID}")
    print(f"  window:  {start.date()} → {end.date()} (60d)")
    print(f"  spend:   $600, sales: $1800 (ACOS ~33%)")
    print(f"  skus:    {len(skus)} SellerSKUs attached")


if __name__ == "__main__":
    asyncio.run(main())
