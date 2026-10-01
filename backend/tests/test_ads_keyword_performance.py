"""Tests for turning an Amazon spTargeting report into keyword verdicts."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
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


# ── 425 duplicate-report recovery ───────────────────────────────────────────
#
# Amazon returns 425 when a report with this exact configuration was already
# requested, handing back the original id. The window is derived from the
# date, so every retry posts a byte-identical configuration — without this
# the feature could never recover on its own, which is what the seller hit:
# a permanent "Could not check existing keyword performance" banner.


def _http_error(status, payload=None, text=None):
    """An httpx.HTTPStatusError carrying a real response, as _post_json raises."""
    request = httpx.Request("POST", "https://advertising-api.amazon.com/reporting/reports")
    if payload is not None:
        response = httpx.Response(status, json=payload, request=request)
    else:
        response = httpx.Response(status, text=text or "", request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_report_id_read_from_a_json_field():
    err = _http_error(425, {"reportId": "amzn1.rpt.abc-123-def-456"})
    assert akp._duplicate_report_id(err) == "amzn1.rpt.abc-123-def-456"


def test_report_id_read_from_prose_in_detail():
    """Amazon has also returned it inside the human-readable `detail`."""
    err = _http_error(425, {
        "detail": "Duplicate report request. Please use reportId: "
                  "8f14e45f-ceea-467a-9f47-b6c1d8a2e331",
    })
    assert akp._duplicate_report_id(err) == "8f14e45f-ceea-467a-9f47-b6c1d8a2e331"


def test_report_id_read_from_a_non_json_body():
    err = _http_error(425, text="report_id 0c5a1b2c3d4e5f67")
    assert akp._duplicate_report_id(err) == "0c5a1b2c3d4e5f67"


def test_a_425_with_no_recoverable_id_is_not_guessed():
    """Polling an invented id would hang for the full backoff and then lie
    about why. Better to surface the failure."""
    err = _http_error(425, {"detail": "Too many requests in flight."})
    assert akp._duplicate_report_id(err) is None


def test_other_statuses_are_never_treated_as_duplicates():
    """A 400 body can legitimately mention a reportId; only 425 means
    'reuse this one'."""
    err = _http_error(400, {"reportId": "amzn1.rpt.should-not-be-used"})
    assert akp._duplicate_report_id(err) is None


@pytest.mark.asyncio
async def test_request_report_returns_the_existing_id_on_425():
    err = _http_error(425, {"reportId": "amzn1.rpt.existing-999"})
    with patch.object(akp, "_post_json", AsyncMock(side_effect=err)):
        assert await akp._request_report("2026-09-01", "2026-09-30") == \
            "amzn1.rpt.existing-999"


@pytest.mark.asyncio
async def test_request_report_surfaces_amazons_own_words_when_it_cannot_recover():
    """The seller used to see httpx's 'Client error 425 ... check MDN',
    which says nothing. Amazon's body does."""
    err = _http_error(425, {"detail": "Report quota exhausted for this profile."})
    with patch.object(akp, "_post_json", AsyncMock(side_effect=err)):
        with pytest.raises(RuntimeError) as caught:
            await akp._request_report("2026-09-01", "2026-09-30")
    message = str(caught.value)
    assert "Report quota exhausted for this profile." in message
    assert "425" in message
    assert "developer.mozilla.org" not in message


@pytest.mark.asyncio
async def test_a_normal_create_still_returns_the_new_report_id():
    with patch.object(akp, "_post_json", AsyncMock(return_value={"reportId": "new-1"})):
        assert await akp._request_report("2026-09-01", "2026-09-30") == "new-1"


def test_dotted_resource_ids_in_prose_are_recovered():
    """Amazon's ids are often dotted resource names, not bare UUIDs. An
    id-shaped pattern that only allowed [A-Za-z0-9-] silently dropped them
    and the 425 dead-ended exactly as before the fix."""
    err = _http_error(425, {
        "detail": "A report with this configuration is already being "
                  "generated. reportId: amzn1.rpt.already-building-7f3a",
    })
    assert akp._duplicate_report_id(err) == "amzn1.rpt.already-building-7f3a"


def test_a_sentence_ending_period_is_not_part_of_the_id():
    """Dots are legal inside the id, so the trailing one has to be trimmed
    or we would poll a report that does not exist."""
    err = _http_error(425, {"detail": "Already building, see reportId 8f14e45f-ceea-467a."})
    assert akp._duplicate_report_id(err) == "8f14e45f-ceea-467a"
