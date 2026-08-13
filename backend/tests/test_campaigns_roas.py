"""Campaign performance output must expose a properly-named ROAS field.

Review issue #5: `roi_pct` is misnamed — the formula (sales-spend)/spend*100
is a revenue-based ratio, not profit ROI. This test locks in the addition of
a `roas_pct` field (= sales/spend*100, standard ROAS %) alongside the legacy
`roi_pct` (kept for backward compatibility)."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import campaigns
from auth import current_user


@pytest.fixture(autouse=True)
def _seed_user_and_campaigns(request):
    user = {"_id": "u-test", "email": "t@example.test"}
    token = current_user.set(user)
    key = str(user["_id"])
    fake_campaigns = getattr(request, "param", None) or [
        {
            "campaignName": "SP Auto — Rugs",
            "status": "Enabled",
            "campaignType": "sponsoredProducts",
            "country": "US",
            "spend": {"amount": 100.0},
            "sales": {"amount": 400.0},
            "budget": {"amount": 20.0},
        },
        {
            "campaignName": "SP Zero Sales",
            "status": "Enabled",
            "campaignType": "sponsoredProducts",
            "country": "US",
            "spend": {"amount": 50.0},
            "sales": {"amount": 0.0},
            "budget": {"amount": 20.0},
        },
    ]
    campaigns._user_campaigns[key] = fake_campaigns
    campaigns._user_summary[key] = ""
    try:
        yield fake_campaigns
    finally:
        current_user.reset(token)
        campaigns._user_campaigns.pop(key, None)
        campaigns._user_summary.pop(key, None)


@pytest.mark.asyncio
async def test_row_exposes_both_roi_and_roas():
    result = await campaigns.analyze_performance_data(full=True)
    row = next(r for r in result["campaigns"] if r["name"] == "SP Auto — Rugs")

    # roas_pct = sales/spend*100 = 400/100*100 = 400.0
    assert row["roas_pct"] == 400.0
    # legacy roi_pct = (sales-spend)/spend*100 = 300.0
    assert row["roi_pct"] == 300.0
    # Invariant: roi_pct == roas_pct - 100 for any spend>0 row.
    assert abs((row["roas_pct"] - 100.0) - row["roi_pct"]) < 0.05


@pytest.mark.asyncio
async def test_zero_sales_row_has_defined_roas():
    """A campaign with spend but no sales must have roas_pct=0.0, not
    missing/None — downstream tables need a numeric value to sort by."""
    result = await campaigns.analyze_performance_data(full=True)
    row = next(r for r in result["campaigns"] if r["name"] == "SP Zero Sales")
    assert row["roas_pct"] == 0.0
    assert row["roi_pct"] == -100.0  # (0-50)/50*100


@pytest.mark.asyncio
async def test_legacy_roi_pct_still_present():
    """Backward-compat guard: existing frontends still read `roi_pct`; the
    new field is additive, not a rename."""
    result = await campaigns.analyze_performance_data(full=True)
    for row in result["campaigns"]:
        assert "roi_pct" in row, "roi_pct removed — frontend contract breakage"
        assert "roas_pct" in row, "roas_pct not added"
