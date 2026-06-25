"""
Per-(SKU, date) ad spend / impressions / clicks via the Amazon Advertising
API v3 reporting endpoints.

We use the Sponsored Products `spAdvertisedProduct` report at DAILY
granularity. The response is a gzipped JSON array of
`{ date, advertisedAsin, advertisedSku, spend, impressions, clicks }`
rows — which is exactly what `salesDaily` needs.

Reporting v3 caps each request at 31 days, so we chunk longer windows.

This activates the Prophet `ad_spend` regressor automatically: the model
adds it as a feature once ≥ 10% of training days have non-zero spend.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

import httpx
from bson import ObjectId

from amazon_ads import (
    ADS_LWA_CLIENT_ID,
    _ads_base,
    _ads_profile_id,
    get_ads_access_token,
)
from auth import require_user
from database import upsert_sales_daily

log = logging.getLogger("forecasting.ads_ingest")

REPORT_CT = "application/vnd.createasyncreportrequest.v3+json"
MAX_WINDOW_DAYS = 31

# The columns we actually persist. groupBy=advertiser yields one row per
# (campaign × ad × date) — we sum down to (sku, date) ourselves.
COLUMNS = [
    "date",
    "advertisedAsin",
    "advertisedSku",
    "spend",
    "impressions",
    "clicks",
    "sales1d",
    "purchases1d",
]


# ── Low-level reporting calls ──────────────────────────────────────────────


def _headers(access_token: str, profile_id: str | None) -> dict:
    h = {
        "Authorization": f"Bearer {access_token}",
        "Amazon-Advertising-API-ClientId": ADS_LWA_CLIENT_ID,
        "Content-Type": REPORT_CT,
        "Accept": REPORT_CT,
    }
    if profile_id:
        h["Amazon-Advertising-API-Scope"] = str(profile_id)
    return h


def _profile_ids(user: dict) -> list[str]:
    """All ads profiles the user has connected — we loop over them so a
    seller with one Sponsored Products profile + one Sponsored Display
    profile (etc.) gets full coverage."""
    profiles = user.get("amazonAdsProfileIds") or []
    return [str(p) for p in profiles if p]


async def _create_report(user: dict, profile_id: str,
                         start: datetime, end: datetime) -> str:
    """POST /reporting/reports; return reportId."""
    token = await get_ads_access_token(user)
    body = {
        "name": f"sp-advertised-product-{start.date()}-{end.date()}",
        "startDate": start.date().isoformat(),
        "endDate": end.date().isoformat(),
        "configuration": {
            "adProduct": "SPONSORED_PRODUCTS",
            "reportTypeId": "spAdvertisedProduct",
            "format": "GZIP_JSON",
            "groupBy": ["advertiser"],
            "columns": COLUMNS,
            "timeUnit": "DAILY",
        },
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_ads_base(user)}/reporting/reports",
            headers=_headers(token, profile_id),
            json=body,
        )
        if resp.is_error:
            log.warning(
                "ads report create FAILED profile=%s status=%s body=%s",
                profile_id, resp.status_code, resp.text[:300],
            )
            resp.raise_for_status()
        data = resp.json()
    rid = data.get("reportId")
    if not rid:
        raise RuntimeError(f"ads create_report missing reportId: {data}")
    return rid


async def _poll_report(user: dict, profile_id: str, report_id: str,
                       max_polls: int = 30, poll_interval: int = 10) -> str:
    """Poll until status reaches a terminal state. Returns the download URL."""
    token = await get_ads_access_token(user)
    url = f"{_ads_base(user)}/reporting/reports/{report_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(max_polls):
            resp = await client.get(url, headers=_headers(token, profile_id))
            if resp.is_error:
                # Token may expire across polls — re-mint and retry once.
                token = await get_ads_access_token(user)
                resp = await client.get(url, headers=_headers(token, profile_id))
            if resp.is_error:
                log.warning("ads poll FAILED %s: %s", resp.status_code, resp.text[:200])
                resp.raise_for_status()
            data = resp.json()
            status = (data.get("status") or "").upper()
            if status in ("COMPLETED", "SUCCESS"):
                location = data.get("url") or data.get("location")
                if not location:
                    raise RuntimeError(f"ads report {report_id} done but no url: {data}")
                return location
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"ads report {report_id} {status}: {data.get('failureReason')}")
            await asyncio.sleep(poll_interval)
    raise TimeoutError(f"ads report {report_id} still pending after {max_polls * poll_interval}s")


async def _download_report(url: str) -> list[dict]:
    """We requested `format: GZIP_JSON` on report creation, so the body is
    always gzip-compressed JSON regardless of what headers S3 sets."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    raw = resp.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


# ── Aggregate + persist ────────────────────────────────────────────────────


def _aggregate_rows(raw: list[dict]) -> list[dict]:
    """Sum rows down to (sku, date). The v3 response groups by ad/campaign;
    one SKU can appear on multiple ads in the same day, so we sum."""
    by_key: dict[tuple[str, datetime], dict] = {}
    for r in raw:
        sku = (r.get("advertisedSku") or "").strip()
        if not sku:
            continue
        d_str = r.get("date") or ""
        try:
            d = datetime.fromisoformat(d_str).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        d = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        key = (sku, d)
        agg = by_key.setdefault(key, {
            "sku": sku, "date": d,
            "ad_spend": 0.0, "ad_impressions": 0, "ad_clicks": 0,
        })
        try: agg["ad_spend"] += float(r.get("spend") or 0)
        except (TypeError, ValueError): pass
        try: agg["ad_impressions"] += int(r.get("impressions") or 0)
        except (TypeError, ValueError): pass
        try: agg["ad_clicks"] += int(r.get("clicks") or 0)
        except (TypeError, ValueError): pass
    # Round spend to cents so the DB doesn't accumulate float dust.
    for v in by_key.values():
        v["ad_spend"] = round(v["ad_spend"], 2)
    return list(by_key.values())


async def _pull_window(user: dict, user_id: ObjectId,
                       start: datetime, end: datetime) -> int:
    """Pull the window across every connected ads profile and merge rows.

    A seller can have separate profiles per ad product (Sponsored
    Products / Brands / Display) or even per region. Querying only the
    first one misses spend that lives on the others — we saw exactly
    that during testing (first profile returned 22 bytes / no rows)."""
    profiles = _profile_ids(user)
    if not profiles:
        log.warning("ads window %s→%s skipped — no profile ids", start.date(), end.date())
        return 0

    all_rows: list[dict] = []
    for pid in profiles:
        label = f"profile={pid} {start.date()}→{end.date()}"
        log.warning("ads window %s — fetching", label)  # warning so it shows by default
        try:
            report_id = await _create_report(user, pid, start, end)
            url = await _poll_report(user, pid, report_id)
            raw = await _download_report(url)
        except Exception as e:
            log.error("ads window %s failed: %s", label, e)
            continue
        log.warning("ads window %s — got %d raw rows", label, len(raw))
        all_rows.extend(raw)

    if not all_rows:
        return 0
    rows = _aggregate_rows(all_rows)
    return await upsert_sales_daily(user_id, rows)


# ── Public entry points ────────────────────────────────────────────────────


def _user_has_ads(user: dict) -> bool:
    return bool(user.get("amazonAdsRefreshToken")) and bool(user.get("amazonAdsProfileIds"))


async def pull_ads_window(user: dict, days_back: int = 7) -> int:
    """Recent window for the nightly job. Defaults to 7 days to absorb
    late-reporting attribution."""
    if not _user_has_ads(user):
        log.info("ads ingest skipped — no ads refresh token / profile on user")
        return 0
    user_id = ObjectId(str(user["_id"]))
    # Setting current_user is the caller's job (ingest.py does it).
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - timedelta(days=days_back)
    end = today - timedelta(days=1)
    return await _pull_window(user, user_id, start, end)


async def pull_ads_backfill(user: dict, days_back: int = 540) -> int:
    """Full backfill, chunked to 31-day windows."""
    if not _user_has_ads(user):
        log.info("ads backfill skipped — no ads refresh token / profile on user")
        return 0
    user_id = ObjectId(str(user["_id"]))
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    earliest = today - timedelta(days=days_back)
    written = 0
    cursor = earliest
    while cursor < today:
        window_end = min(cursor + timedelta(days=MAX_WINDOW_DAYS - 1), today - timedelta(days=1))
        written += await _pull_window(user, user_id, cursor, window_end)
        cursor = window_end + timedelta(days=1)
    return written
