"""Tests for POST /campaigns/create.

This endpoint spends real advertising budget, so its job is to reject a
request that does not describe a servable campaign rather than quietly fix it
up. The cases below are the ways the Amazon Ads API accepts a request and
then silently does nothing useful with it — a campaign with no keywords or no
product ad is created successfully and never serves an impression, which the
seller only discovers a day later.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

import main  # noqa: E402


def _body(**overrides):
    payload = {
        "campaign_name": "Incense — Manual Exact",
        "budget": 20.0,
        "country": "us",
        "targeting_type": "MANUAL",
        "keywords": ["incense cones", "backflow incense"],
        "sku": "FG-2F35-W25S",
    }
    payload.update(overrides)
    return main.CreateCampaignRequest(**payload)


async def _call(body):
    """Invoke the endpoint with `create_campaign` stubbed, returning
    (response, the payload the endpoint would have sent to Amazon)."""
    with patch.object(main, "create_campaign", new=AsyncMock(return_value="ok")) as spy:
        resp = await main.campaigns_create(body, user={"_id": "u1"})
    return resp, (spy.await_args.args[0] if spy.await_args else None)


# ── Requests that must not reach Amazon ─────────────────────────────────────


@pytest.mark.asyncio
async def test_manual_campaign_without_keywords_is_rejected():
    """Amazon accepts this and creates a campaign that can never serve."""
    with pytest.raises(HTTPException) as exc:
        await _call(_body(keywords=[]))
    assert exc.value.status_code == 400
    assert "keyword" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_keywords_that_are_only_whitespace_count_as_none():
    with pytest.raises(HTTPException) as exc:
        await _call(_body(keywords=["  ", "\t", ""]))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_campaign_without_a_product_ad_is_rejected():
    """No SKU and no ASIN means no product ad, which means no impressions."""
    with pytest.raises(HTTPException) as exc:
        await _call(_body(sku=None, asin=None))
    assert exc.value.status_code == 400
    assert "sku" in exc.value.detail.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, -5])
async def test_non_positive_budget_is_rejected(budget):
    with pytest.raises(HTTPException) as exc:
        await _call(_body(budget=budget))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_blank_campaign_name_is_rejected():
    with pytest.raises(HTTPException) as exc:
        await _call(_body(campaign_name="   "))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_unknown_targeting_type_is_rejected():
    with pytest.raises(HTTPException) as exc:
        await _call(_body(targeting_type="SEMI_AUTO"))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_absurd_keyword_count_is_rejected():
    many = [f"keyword {i}" for i in range(main._MAX_CAMPAIGN_KEYWORDS + 1)]
    with pytest.raises(HTTPException) as exc:
        await _call(_body(keywords=many))
    assert exc.value.status_code == 400


# ── Requests that should go through, and in what shape ──────────────────────


@pytest.mark.asyncio
async def test_an_auto_campaign_needs_no_keywords():
    resp, sent = await _call(_body(targeting_type="AUTO", keywords=[]))
    assert sent["targeting_type"] == "AUTO"
    assert resp["keyword_count"] == 0


@pytest.mark.asyncio
async def test_the_selection_reaches_amazon_verbatim():
    """The whole reason this endpoint exists: what the user ticked is what
    gets sent, with no model retyping the list on the way through."""
    picked = ["incense cones", "backflow incense burner", "dhoop cones"]
    _resp, sent = await _call(_body(keywords=picked))
    assert sent["keywords"] == picked


@pytest.mark.asyncio
async def test_duplicate_keywords_collapse_but_keep_their_casing_and_order():
    _resp, sent = await _call(
        _body(keywords=["Incense Cones", "incense cones", "Backflow", "  incense cones  "])
    )
    assert sent["keywords"] == ["Incense Cones", "Backflow"]


@pytest.mark.asyncio
async def test_country_is_normalized_and_defaults_are_applied():
    _resp, sent = await _call(_body(country="us"))
    assert sent["country"] == "US"
    assert sent["campaign_type"] == "Sponsored Products"


@pytest.mark.asyncio
async def test_response_reports_the_count_actually_sent():
    resp, sent = await _call(_body(keywords=["a keyword", "A Keyword", "another"]))
    assert resp["keyword_count"] == len(sent["keywords"]) == 2


# ── Per-keyword bids ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bids_reach_amazon_keyed_to_their_keyword():
    _resp, sent = await _call(
        _body(keywords=["incense", "incense burner"],
              keyword_bids={"incense": 0.75, "incense burner": 1.2})
    )
    assert sent["keyword_bids"] == {"incense": 0.75, "incense burner": 1.2}


@pytest.mark.asyncio
async def test_keywords_without_a_bid_are_simply_absent_from_the_map():
    """They fall back to the ad group default rather than to zero."""
    _resp, sent = await _call(_body(keywords=["a", "b"], keyword_bids={"a": 0.5}))
    assert sent["keyword_bids"] == {"a": 0.5}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, 0.01, 250, -1])
async def test_an_out_of_range_bid_is_rejected_not_clamped(bad):
    """A mistyped bid is charged per click. Silently turning 250 into 2.50
    would be worse than refusing it."""
    with pytest.raises(HTTPException) as exc:
        await _call(_body(keyword_bids={"incense cones": bad}))
    assert exc.value.status_code == 400
    assert "incense cones" in exc.value.detail


def test_a_non_numeric_bid_is_rejected_by_the_request_model():
    """Caught a layer earlier than the range check — pydantic refuses to
    parse it, which FastAPI turns into a 422 before the handler runs."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _body(keyword_bids={"incense cones": "one dollar"})


@pytest.mark.asyncio
async def test_nan_slips_past_parsing_and_is_caught_by_the_range_check():
    """Pydantic accepts NaN as a float. It must not reach Amazon as a bid."""
    with pytest.raises(HTTPException) as exc:
        await _call(_body(keyword_bids={"incense cones": float("nan")}))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_bids_are_rounded_to_cents():
    _resp, sent = await _call(_body(keyword_bids={"incense cones": 0.7777}))
    assert sent["keyword_bids"]["incense cones"] == 0.78


# ── Bid strategy follows the budget ─────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("budget,expected", [
    (5, "LEGACY_FOR_SALES"),
    (10, "MANUAL"),
    (20, "MANUAL"),
    (25, "MANUAL"),
    (40, "LEGACY_FOR_SALES"),
])
async def test_the_response_reports_the_strategy_the_budget_implies(budget, expected):
    resp, _sent = await _call(_body(budget=budget))
    assert resp["bidding_strategy"] == expected
