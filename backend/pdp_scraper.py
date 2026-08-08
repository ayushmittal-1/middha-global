"""Amazon PDP scraper — fills the gap where SP-API doesn't return
bullets / description for a listing.

Routing:
- If SCRAPER_API_KEY env is set, requests go through ScraperAPI so we
  get a rotating residential IP and CAPTCHA handling.
- Otherwise falls back to a direct fetch with a browser User-Agent —
  works from a local dev machine, will get 503/CAPTCHA'd on cloud IPs.

Results are cached in the `listingPdpCache` Mongo collection for 30
days per (asin, marketplace_domain), so repeat opens of the same SKU
don't burn a scrape quota.

Backend keywords are deliberately NOT scraped — those are seller-private
and never appear on the PDP.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger("pdp_scraper")

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "sec-ch-ua-mobile": "?0",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "upgrade-insecure-requests": "1",
}

# Marketplace ID → domain to hit. Add more as new marketplaces enable.
_MARKETPLACE_TO_DOMAIN = {
    "ATVPDKIKX0DER": "amazon.com",   # US
    "A21TJRUUN4KGV": "amazon.in",    # India
    "A1F83G8C2ARO7P": "amazon.co.uk",  # UK
    "A1PA6795UKMFR9": "amazon.de",   # Germany
    "A13V1IB3VIYZZH": "amazon.fr",   # France
    "APJ6JRA9NG5V4": "amazon.it",    # Italy
    "A1RKKUPIHCS9HS": "amazon.es",   # Spain
    "A1AM78C64UM0Y8": "amazon.com.mx",  # Mexico
    "A2EUQ1WTGCTBG2": "amazon.ca",   # Canada
    "A1VC38T7YXB528": "amazon.co.jp",  # Japan
}

_CACHE_TTL_DAYS = 30
_TIMEOUT = 25.0


# ── Cache helpers ──────────────────────────────────────────────────────────


def _cache_coll():
    from database import _db
    return _db().listingPdpCache


async def _cache_get(asin: str, domain: str) -> dict | None:
    doc = await _cache_coll().find_one(
        {"asin": asin, "domain": domain},
        {"_id": 0, "bullets": 1, "description": 1, "scraped_at": 1},
    )
    if not doc:
        return None
    scraped_at = doc.get("scraped_at")
    if scraped_at:
        if scraped_at.tzinfo is None:
            scraped_at = scraped_at.replace(tzinfo=timezone.utc)
        ttl_cutoff = datetime.now(timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)
        if scraped_at < ttl_cutoff:
            return None
    return doc


async def _cache_set(asin: str, domain: str, bullets: list[str], description: str) -> None:
    await _cache_coll().update_one(
        {"asin": asin, "domain": domain},
        {"$set": {
            "asin": asin,
            "domain": domain,
            "bullets": bullets,
            "description": description,
            "scraped_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )


# ── HTTP fetch — direct or via ScraperAPI ─────────────────────────────────


async def _fetch_pdp_html(url: str) -> str | None:
    """Fetch the PDP HTML. Routes through ScraperAPI when SCRAPER_API_KEY
    is set (needed on any cloud IP — Amazon blocks datacenter IPs)."""
    api_key = os.getenv("SCRAPER_API_KEY")
    if api_key:
        # ScraperAPI wraps the target URL — same auth for all requests.
        # We ask them to render country=us so US IPs are used and the
        # PDP layout matches what a US buyer sees.
        target = (
            "https://api.scraperapi.com/"
            f"?api_key={api_key}"
            f"&url={httpx.URL(url)}"
            f"&country_code=us"
        )
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                r = await c.get(target)
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            log.warning("scraperapi fetch failed for %s: %s", url, e)
            return None
        if r.status_code != 200:
            log.warning("scraperapi HTTP %s for %s: %s",
                        r.status_code, url, r.text[:200])
            return None
        return r.text

    # Direct fetch — dev only. Cloud IPs get 503'd immediately.
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS,
                                     follow_redirects=True) as c:
            r = await c.get(url)
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        log.warning("direct pdp fetch failed for %s: %s", url, e)
        return None
    if r.status_code != 200:
        log.warning("direct pdp HTTP %s for %s", r.status_code, url)
        return None
    return r.text


# ── HTML parsing ──────────────────────────────────────────────────────────


# Do NOT include #detailBullets_feature_div — that holds product-info
# specs (dimensions, ASIN, date first available), not feature bullets.
_BULLET_SELECTORS = [
    "#feature-bullets ul li span.a-list-item",
    "#featurebullets_feature_div ul li span.a-list-item",
    "#pqv-feature-bullets ul li span.a-list-item",
]

_DESC_SELECTORS = [
    "#productDescription",
    "#productDescription_feature_div",
    "#pqv-description",
    "#aplus_feature_div",
]

_CAPTCHA_MARKERS = (
    "api-services-support@amazon.com",
    "Type the characters you see",
    "Robot Check",
    "To discuss automated access to Amazon data",
)


def _parse_pdp(html: str) -> dict | None:
    """Extract bullets + description from PDP HTML. Returns None on
    CAPTCHA / bot-block pages so callers can distinguish "scrape blocked"
    from "listing genuinely has no bullets"."""
    if not html:
        return None
    if any(marker in html for marker in _CAPTCHA_MARKERS):
        log.info("pdp scrape hit CAPTCHA / bot-check page")
        return None
    soup = BeautifulSoup(html, "lxml")

    bullets: list[str] = []
    for sel in _BULLET_SELECTORS:
        for el in soup.select(sel):
            text = " ".join(el.get_text(" ", strip=True).split())
            if text and not text.lower().startswith("make sure this fits"):
                bullets.append(text)
        if bullets:
            break

    description = ""
    for sel in _DESC_SELECTORS:
        el = soup.select_one(sel)
        if el:
            text = " ".join(el.get_text(" ", strip=True).split())
            if text:
                description = text
                break
    return {"bullets": bullets, "description": description}


# ── Public API ────────────────────────────────────────────────────────────


async def scrape_pdp_for_asin(
    asin: str, marketplace_id: str | None = None,
) -> dict | None:
    """Fetch bullets + description for an ASIN from the Amazon PDP.

    Returns {bullets: [...], description: "..."} on success (either
    freshly scraped or cache-hit), None if the fetch failed or the
    PDP couldn't be parsed. Marketplace defaults to US when the id
    isn't recognised.
    """
    asin = (asin or "").strip().upper()
    if not asin:
        return None
    domain = _MARKETPLACE_TO_DOMAIN.get(
        marketplace_id or "ATVPDKIKX0DER", "amazon.com",
    )

    cached = await _cache_get(asin, domain)
    if cached is not None:
        return {
            "bullets": cached.get("bullets") or [],
            "description": cached.get("description") or "",
            "source": "cache",
        }

    url = f"https://www.{domain}/dp/{asin}"
    html = await _fetch_pdp_html(url)
    parsed = _parse_pdp(html) if html else None
    if parsed is None:
        return None

    await _cache_set(asin, domain, parsed["bullets"], parsed["description"])
    return {**parsed, "source": "scrape"}
