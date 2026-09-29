"""Keyword-level performance for the campaigns a product already runs.

The picker needs to answer "have I bid on this term before, and did it work?"
before a seller commits budget to it again. Nothing in the system could
answer that: `admetricsdailies` and the `ads` collection are both
campaign-level, so a campaign could look fine in aggregate while three of its
keywords quietly ate the budget.

This module fills the gap with Amazon's Ads Reporting v3 `spTargeting`
report, which breaks spend and sales down per keyword. The report is
asynchronous — request it, poll until Amazon builds it, then download a
gzipped JSON payload — and can take minutes, so results are cached per
(user, profile, window) and shared across every ASIN the seller asks about
in that window.
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import date, datetime, timedelta, timezone

import httpx

from ads_bidding import KeywordStats, KeywordVerdict, classify_keyword_performance
from amazon_ads import (
    ADS_LWA_CLIENT_ID,
    _ads_base,
    _ads_profile_id,
    _post_json,
    get_ads_access_token,
)
from auth import _db, require_user

# The report covers this many days back from yesterday. Amazon attributes a
# sale to the click that caused it for 14 days, so a window shorter than that
# would count spend whose sales have not landed yet and call live keywords
# dead. 30 days gives every click in the window its full attribution period
# and still reflects how the campaign behaves now.
REPORT_WINDOW_DAYS = 30

# Amazon's own report generation is the slow part; caching the parsed rows
# means the second ASIN a seller asks about in a session is instant. Half a
# day is short enough that yesterday's spend shows up the next morning.
_CACHE_COLLECTION = "adsKeywordPerformanceCache"
_CACHE_TTL_SECONDS = 12 * 3600

_REPORT_CREATE_CT = "application/vnd.createasyncreportrequest.v3+json"
_REPORT_CREATE_ACCEPT = "application/vnd.createasyncreportresponse.v3+json"

# Poll schedule in seconds. Amazon usually has a targeting report ready in
# under two minutes, but a cold profile can take far longer; the schedule
# backs off and then gives up rather than holding a request open forever.
_POLL_WAITS = (5, 5, 10, 10, 15, 15, 20, 30, 30, 40, 40, 60)


def _cache_coll():
    return _db()[_CACHE_COLLECTION]


def _window() -> tuple[str, str]:
    """(start, end) for the report. Ends yesterday — today is still
    accumulating and Amazon's data for it is incomplete all day."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=REPORT_WINDOW_DAYS - 1)
    return start.isoformat(), end.isoformat()


# ── Which campaigns advertise this product ──────────────────────────────────


async def campaigns_advertising(asins: list[str], sku: str | None = None) -> set[str]:
    """Campaign ids whose product ads cover any of these ASINs or this SKU.

    Used to scope the warning to the product being built. A keyword that lost
    money advertising a rug says nothing about the same keyword on incense,
    and flagging it there would be noise the seller learns to skip past.
    """
    wanted_asins = {a.strip().upper() for a in (asins or []) if a and a.strip()}
    wanted_sku = (sku or "").strip()
    if not wanted_asins and not wanted_sku:
        return set()

    found: set[str] = set()
    for state in ("ENABLED", "PAUSED"):
        try:
            data = await _post_json(
                "productAds/list",
                "/sp/productAds/list",
                "application/vnd.spProductAd.v3+json",
                {"stateFilter": {"include": [state]}, "maxResults": 100},
            )
        except Exception as e:
            print(f"[ads_kw_perf] productAds lookup failed ({state}): {e}")
            continue
        for ad in (data or {}).get("productAds") or []:
            asin = (ad.get("asin") or "").strip().upper()
            ad_sku = (ad.get("sku") or "").strip()
            if (asin and asin in wanted_asins) or (wanted_sku and ad_sku == wanted_sku):
                cid = ad.get("campaignId")
                if cid:
                    found.add(str(cid))
    return found


# ── Report fetch ────────────────────────────────────────────────────────────


async def _request_report(start: str, end: str) -> str:
    """Ask Amazon to build a keyword-level report; return its id."""
    body = {
        "name": f"Aurora spTargeting {start}..{end}",
        "startDate": start,
        "endDate": end,
        "configuration": {
            "adProduct": "SPONSORED_PRODUCTS",
            "reportTypeId": "spTargeting",
            # `targeting` is the keyword/product-target grain — the whole
            # point of this report. Grouping by campaign would reproduce the
            # aggregate data the system already has.
            "groupBy": ["targeting"],
            "columns": [
                "campaignId",
                "campaignName",
                "adGroupId",
                "keyword",
                "keywordId",
                "matchType",
                "keywordType",
                "impressions",
                "clicks",
                "cost",
                "purchases14d",
                "sales14d",
            ],
            "timeUnit": "SUMMARY",
            "format": "GZIP_JSON",
        },
    }
    data = await _post_json("report", "/reporting/reports", _REPORT_CREATE_CT, body)
    report_id = (data or {}).get("reportId") or (data or {}).get("id")
    if not report_id:
        raise RuntimeError(f"Amazon did not return a report id: {data}")
    return str(report_id)


async def _await_report_url(report_id: str) -> str:
    """Poll until the report is generated; return its download URL."""
    import asyncio

    user = require_user()
    token = await get_ads_access_token(user)
    headers = {
        "Authorization": f"Bearer {token}",
        "Amazon-Advertising-API-ClientId": ADS_LWA_CLIENT_ID,
        "Amazon-Advertising-API-Scope": _ads_profile_id(user),
        "Accept": _REPORT_CREATE_ACCEPT,
    }
    url = f"{_ads_base(user)}/reporting/reports/{report_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        for wait in _POLL_WAITS:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            status = (data.get("status") or "").upper()
            if status in ("COMPLETED", "SUCCESS"):
                location = data.get("url") or data.get("location")
                if not location:
                    raise RuntimeError("Report completed with no download URL")
                return location
            if status in ("FAILURE", "FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"Amazon failed to build the report: "
                    f"{data.get('statusDetails') or status}"
                )
            await asyncio.sleep(wait)
    raise RuntimeError(
        "Amazon did not finish the keyword report in time. It is still "
        "building — try again in a few minutes."
    )


async def _download_rows(location: str) -> list[dict]:
    """Fetch and unpack the gzipped JSON report body.

    The download URL is pre-signed, so it must be fetched WITHOUT the Ads
    auth headers — sending them makes S3 reject the request.
    """
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(location)
        resp.raise_for_status()
        raw = resp.content
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
            payload = fh.read()
    except OSError:
        # Amazon occasionally serves the body already decompressed.
        payload = raw
    parsed = json.loads(payload or b"[]")
    return parsed if isinstance(parsed, list) else []


def _row_to_stats(row: dict) -> KeywordStats | None:
    """Narrow one report row. Rows without a keyword are product targets
    (ASIN targeting), which this feature does not speak to."""
    keyword = (row.get("keyword") or "").strip()
    if not keyword:
        return None
    def _num(key, cast=float):
        try:
            return cast(row.get(key) or 0)
        except (TypeError, ValueError):
            return cast(0)

    return KeywordStats(
        keyword=keyword.lower(),
        match_type=(row.get("matchType") or "").upper(),
        campaign_name=row.get("campaignName") or "",
        impressions=_num("impressions", int),
        clicks=_num("clicks", int),
        spend=round(_num("cost"), 2),
        orders=_num("purchases14d", int),
        sales=round(_num("sales14d"), 2),
    )


# ── Cached fetch ────────────────────────────────────────────────────────────


async def _cached_rows(cache_key: str) -> list[dict] | None:
    doc = await _cache_coll().find_one({"_id": cache_key})
    return (doc or {}).get("rows") if doc else None


async def _store_rows(cache_key: str, rows: list[dict]) -> None:
    try:
        await _cache_coll().create_index(
            "cached_at", expireAfterSeconds=_CACHE_TTL_SECONDS, background=True
        )
        await _cache_coll().replace_one(
            {"_id": cache_key},
            {
                "_id": cache_key,
                "rows": rows,
                "cached_at": datetime.now(timezone.utc),
            },
            upsert=True,
        )
    except Exception as e:
        # A cache miss next time costs minutes, not correctness.
        print(f"[ads_kw_perf] cache write failed: {e}")


async def fetch_keyword_rows() -> list[dict]:
    """All keyword rows for the account over the window, cached."""
    user = require_user()
    start, end = _window()
    cache_key = f"{user.get('_id')}:{_ads_profile_id(user)}:{start}:{end}"

    cached = await _cached_rows(cache_key)
    if cached is not None:
        print(f"[ads_kw_perf] cache HIT {cache_key} ({len(cached)} rows)")
        return cached

    print(f"[ads_kw_perf] requesting spTargeting report {start}..{end}")
    report_id = await _request_report(start, end)
    location = await _await_report_url(report_id)
    rows = await _download_rows(location)
    print(f"[ads_kw_perf] downloaded {len(rows)} keyword rows")
    await _store_rows(cache_key, rows)
    return rows


# ── Public entry point ──────────────────────────────────────────────────────


async def evaluate_for_product(
    asins: list[str], sku: str | None = None
) -> dict:
    """Judge every keyword already running for this product.

    Returns `{scope, window, verdicts: {keyword: {...}}, summary, note}`.
    `verdicts` is keyed by lowercased keyword so the picker can look up a
    candidate term directly.
    """
    start, end = _window()
    campaign_ids = await campaigns_advertising(asins, sku)
    if not campaign_ids:
        return {
            "scope": "none",
            "window": {"start": start, "end": end},
            "verdicts": {},
            "summary": {"evaluated": 0, "wasting": 0, "ok": 0, "unproven": 0,
                        "wasted_spend": 0.0},
            "note": "No existing campaign advertises this product yet — "
                    "nothing to compare against.",
        }

    rows = await fetch_keyword_rows()

    # Sum across campaigns and match types: the seller is choosing a keyword,
    # not a (keyword, match type, campaign) triple, so the question "did this
    # term work for this product" is answered by its combined record.
    merged: dict[str, KeywordStats] = {}
    for row in rows:
        if str(row.get("campaignId") or "") not in campaign_ids:
            continue
        stats = _row_to_stats(row)
        if not stats:
            continue
        existing = merged.get(stats.keyword)
        if existing is None:
            merged[stats.keyword] = stats
            continue
        existing.impressions += stats.impressions
        existing.clicks += stats.clicks
        existing.spend = round(existing.spend + stats.spend, 2)
        existing.orders += stats.orders
        existing.sales = round(existing.sales + stats.sales, 2)

    verdicts = [classify_keyword_performance(s) for s in merged.values()]
    from ads_bidding import summarize_verdicts

    return {
        "scope": "product",
        "window": {"start": start, "end": end},
        "campaign_count": len(campaign_ids),
        "verdicts": {v.stats.keyword: _verdict_payload(v) for v in verdicts},
        "summary": summarize_verdicts(verdicts),
    }


def _verdict_payload(v: KeywordVerdict) -> dict:
    return {
        "verdict": v.verdict,
        "reason": v.reason,
        "impressions": v.stats.impressions,
        "clicks": v.stats.clicks,
        "spend": v.stats.spend,
        "orders": v.stats.orders,
        "sales": v.stats.sales,
        "cpc": v.stats.cpc,
        "acos": v.stats.acos,
        "campaign_name": v.stats.campaign_name,
    }
