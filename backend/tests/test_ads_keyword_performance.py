"""Tests for turning an Amazon spTargeting report into keyword verdicts."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

import ads_bidding as ab  # noqa: E402
import ads_keyword_performance as akp  # noqa: E402


def _row(keyword, campaign_id="1", **kw):
    base = {
        "campaignId": campaign_id,
        "campaignName": "Incense — Manual",
        "keyword": keyword,
        "matchType": "BROAD",
        "impressions": 500,
        "clicks": 10,
        "cost": 6.0,
        "purchases14d": 0,
        "sales14d": 0.0,
    }
    base.update(kw)
    return base


# ── Row parsing ─────────────────────────────────────────────────────────────


def test_product_target_rows_are_skipped():
    """Rows with no keyword are ASIN targets — a different feature."""
    assert akp._row_to_stats(_row("")) is None
    assert akp._row_to_stats({"campaignId": "1", "clicks": 5}) is None


def test_keywords_are_lowercased_so_lookups_match_the_picker():
    stats = akp._row_to_stats(_row("Backflow Incense Cones"))
    assert stats.keyword == "backflow incense cones"


def test_missing_or_junk_numeric_fields_become_zero_not_a_crash():
    stats = akp._row_to_stats(
        {"keyword": "incense", "cost": None, "clicks": "", "purchases14d": "x"}
    )
    assert (stats.clicks, stats.spend, stats.orders) == (0, 0.0, 0)


# ── Aggregation across campaigns and match types ────────────────────────────


@pytest.mark.asyncio
async def test_one_keyword_across_match_types_is_judged_on_its_combined_record():
    """The seller is choosing a keyword, not a (keyword, match type) pair.
    Split across BROAD and EXACT, each half may look too small to judge
    while the combined record is clearly wasting."""
    rows = [
        _row("incense", matchType="BROAD", clicks=6, cost=4.0, purchases14d=0),
        _row("incense", matchType="EXACT", clicks=7, cost=5.0, purchases14d=0),
    ]
    with patch.object(akp, "campaigns_advertising", AsyncMock(return_value={"1"})), \
         patch.object(akp, "fetch_keyword_rows", AsyncMock(return_value=rows)):
        out = await akp.evaluate_for_product(["B07SHH2RJX"])
    v = out["verdicts"]["incense"]
    assert v["clicks"] == 13 and v["spend"] == 9.0
    assert v["verdict"] == ab.VERDICT_WASTING


@pytest.mark.asyncio
async def test_keywords_from_other_products_campaigns_are_ignored():
    """A term that lost money on a rug says nothing about it on incense."""
    rows = [
        _row("incense", campaign_id="1"),
        _row("jute rug", campaign_id="99"),  # a different product's campaign
    ]
    with patch.object(akp, "campaigns_advertising", AsyncMock(return_value={"1"})), \
         patch.object(akp, "fetch_keyword_rows", AsyncMock(return_value=rows)):
        out = await akp.evaluate_for_product(["B07SHH2RJX"])
    assert set(out["verdicts"]) == {"incense"}


@pytest.mark.asyncio
async def test_no_existing_campaign_says_so_without_fetching_a_report():
    """Building the report takes minutes; there is nothing to compare
    against, so it must not be requested at all."""
    fetch = AsyncMock()
    with patch.object(akp, "campaigns_advertising", AsyncMock(return_value=set())), \
         patch.object(akp, "fetch_keyword_rows", fetch):
        out = await akp.evaluate_for_product(["B07SHH2RJX"])
    fetch.assert_not_awaited()
    assert out["scope"] == "none"
    assert out["verdicts"] == {}
    assert "No existing campaign" in out["note"]


@pytest.mark.asyncio
async def test_the_summary_reports_what_the_chat_should_say():
    rows = [
        _row("incense wasting", clicks=12, cost=8.0, purchases14d=0),
        _row("incense good", clicks=12, cost=6.0, purchases14d=3, sales14d=45.0),
        _row("incense new", clicks=1, cost=0.3, purchases14d=0),
    ]
    with patch.object(akp, "campaigns_advertising", AsyncMock(return_value={"1"})), \
         patch.object(akp, "fetch_keyword_rows", AsyncMock(return_value=rows)):
        out = await akp.evaluate_for_product(["B07SHH2RJX"])
    assert out["summary"] == {
        "evaluated": 3, "wasting": 1, "ok": 1, "unproven": 1, "wasted_spend": 8.0,
    }


# ── Report window ───────────────────────────────────────────────────────────


def test_the_window_ends_yesterday_and_covers_the_attribution_period():
    """Today is still accumulating, and Amazon attributes sales to a click
    for 14 days — a window shorter than that would call live keywords dead."""
    from datetime import date

    start, end = akp._window()
    assert end == (date.today() - __import__("datetime").timedelta(days=1)).isoformat()
    span = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    assert span == akp.REPORT_WINDOW_DAYS
    assert span >= 14
