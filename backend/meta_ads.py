"""
Meta (Facebook) Ad Library scraper using Playwright.

Navigates to the public Ad Library, intercepts API/GraphQL responses for
structured data, and falls back to DOM parsing when interception yields nothing.

Browser lifecycle: lazy singleton Chromium — launches on first call, reuses for
subsequent calls.  Each scrape gets a fresh BrowserContext (cheap & isolated).
Call shutdown_browser() on server shutdown to clean up.
"""

import asyncio
import json
import re
from urllib.parse import quote_plus

# Playwright is optional — import errors are handled gracefully so the rest of
# the backend still works if Playwright isn't installed.
try:
    from playwright.async_api import async_playwright, Browser, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ── Lazy singleton browser ──────────────────────────────────────────────────

_playwright: "Playwright | None" = None
_browser: "Browser | None" = None
_lock = asyncio.Lock()

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


async def _ensure_browser() -> "Browser":
    """Launch (or re-launch) the shared Chromium instance."""
    global _playwright, _browser
    async with _lock:
        if _browser and _browser.is_connected():
            return _browser
        # Clean up stale handles
        if _browser:
            try:
                await _browser.close()
            except Exception:
                pass
        if _playwright:
            try:
                await _playwright.stop()
            except Exception:
                pass
        pw = await async_playwright().start()
        _playwright = pw
        _browser = await pw.chromium.launch(headless=True)
        return _browser


async def shutdown_browser() -> None:
    """Close the shared browser — call from server lifespan teardown."""
    global _playwright, _browser
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
    if _playwright:
        try:
            await _playwright.stop()
        except Exception:
            pass
        _playwright = None


# ── Ad Library URL builder ──────────────────────────────────────────────────

def _build_url(query: str, country: str, active_only: bool) -> str:
    status = "active" if active_only else ""
    params = {
        "active_status": status,
        "ad_type": "all",
        "country": country.upper(),
        "q": query,
        "search_type": "keyword_unordered",
    }
    qs = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items() if v)
    return f"https://www.facebook.com/ads/library/?{qs}"


# ── Response interceptor helpers ────────────────────────────────────────────

def _walk_for_ads(obj, results: list, max_results: int) -> None:
    """Recursively walk a JSON structure looking for ad-shaped dicts."""
    if len(results) >= max_results:
        return
    if isinstance(obj, dict):
        # Heuristic: a dict with a body/snapshot/ad_creative_bodies key is an ad
        has_body = any(
            k in obj
            for k in ("body", "ad_creative_bodies", "snapshot", "ad_snapshot_url")
        )
        has_page = any(k in obj for k in ("page_name", "byline", "page_id"))
        if has_body or has_page:
            results.append(obj)
            if len(results) >= max_results:
                return
        for v in obj.values():
            _walk_for_ads(v, results, max_results)
    elif isinstance(obj, list):
        for item in obj:
            _walk_for_ads(item, results, max_results)


def _extract_ads_from_response(data, max_results: int) -> list[dict]:
    """Pull ad objects out of an intercepted API response."""
    raw: list[dict] = []
    _walk_for_ads(data, raw, max_results)
    ads = []
    for r in raw:
        ad: dict = {}
        # Advertiser name
        ad["advertiser"] = (
            r.get("page_name")
            or r.get("byline", "")
            or ""
        )
        # Ad text
        body = r.get("body") or {}
        if isinstance(body, dict):
            body = body.get("text", "") or body.get("markup", {}).get("__html", "")
        if isinstance(body, list):
            body = " ".join(str(b) for b in body)
        if not body:
            bodies = r.get("ad_creative_bodies") or []
            body = bodies[0] if bodies else ""
        ad["text"] = str(body)[:300]
        # Platform
        platforms = r.get("publisher_platforms") or r.get("platforms") or []
        ad["platforms"] = platforms if isinstance(platforms, list) else [str(platforms)]
        # Start date
        ad["start_date"] = (
            r.get("ad_delivery_start_time")
            or r.get("start_date")
            or r.get("creation_time")
            or ""
        )
        # Status
        ad["active"] = r.get("is_active", r.get("ad_delivery_stop_time") is None)
        if ad["advertiser"] or ad["text"]:
            ads.append(ad)
    return ads


# ── DOM fallback parser ─────────────────────────────────────────────────────

async def _fallback_dom_parse(page, max_results: int) -> list[dict]:
    """Parse ads from the DOM using JS-based extraction.

    Strategy: find all text nodes containing "Library ID:", walk up to the
    card container, then extract structured fields from each card's text.
    """
    raw = await page.evaluate("""(maxResults) => {
        const ads = [];
        // Walk all text nodes to find "Library ID:" anchors
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        const anchors = [];
        while (walker.nextNode()) {
            if (walker.currentNode.textContent.includes('Library ID:')) {
                anchors.push(walker.currentNode);
            }
        }
        for (const node of anchors.slice(0, maxResults)) {
            let el = node.parentElement;
            // Walk up to find a container with "Sponsored" text (the ad card)
            for (let i = 0; i < 20; i++) {
                if (!el || !el.parentElement) break;
                el = el.parentElement;
                const t = el.innerText || '';
                if (t.length > 80 && t.includes('Sponsored')) {
                    ads.push(t);
                    break;
                }
            }
        }
        return ads;
    }""", max_results)

    ads: list[dict] = []
    for card_text in raw:
        lines = [l.strip() for l in card_text.split("\n") if l.strip()]
        ad: dict = {"advertiser": "", "text": "", "platforms": [], "start_date": "", "active": True}

        # Status (Active / Inactive)
        if "Inactive" in lines:
            ad["active"] = False

        # Start date — line like "Started running on 31 Oct 2025"
        for line in lines:
            if line.startswith("Started running on"):
                ad["start_date"] = line.replace("Started running on ", "")
                break

        # Advertiser name — the line right before "Sponsored"
        try:
            sp_idx = lines.index("Sponsored")
            if sp_idx > 0:
                ad["advertiser"] = lines[sp_idx - 1]
            # Ad copy — everything after "Sponsored" until a known stop marker
            stop_markers = {"Shop Now", "Shop now", "Learn More", "Learn more",
                            "Sign Up", "Sign up", "Download", "See ad details",
                            "See summary details", "Open Drop-down"}
            copy_lines = []
            for line in lines[sp_idx + 1:]:
                if line in stop_markers or line.startswith("0:00"):
                    break
                copy_lines.append(line)
            ad["text"] = " ".join(copy_lines)[:300]
        except ValueError:
            # No "Sponsored" found — grab whatever text we can
            ad["text"] = " ".join(lines[:3])[:300]

        if ad["advertiser"] or ad["text"]:
            ads.append(ad)

    return ads


# ── Formatter ───────────────────────────────────────────────────────────────

def _format_ads_for_llm(ads: list[dict], query: str) -> str:
    if not ads:
        return f"No ads found for \"{query}\". The query may be too specific, or there are no matching active ads in the Meta Ad Library."
    lines = [f"Found {len(ads)} ads for \"{query}\":\n"]
    for i, ad in enumerate(ads, 1):
        parts = [f"**Ad {i}**"]
        if ad.get("advertiser"):
            parts.append(f"  Advertiser: {ad['advertiser']}")
        if ad.get("text"):
            parts.append(f"  Text: {ad['text']}")
        if ad.get("platforms"):
            parts.append(f"  Platforms: {', '.join(ad['platforms'])}")
        if ad.get("start_date"):
            parts.append(f"  Started: {ad['start_date']}")
        parts.append(f"  Active: {'Yes' if ad.get('active') else 'No'}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


# ── Public entry point ──────────────────────────────────────────────────────

async def search_meta_ads(
    query: str,
    country: str = "US",
    active_only: bool = True,
    max_results: int = 10,
) -> str:
    """Scrape the Meta Ad Library and return a formatted summary of ads.

    Returns a human-readable string suitable for LLM consumption.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return (
            "Playwright is not installed. Run:\n"
            "  pip install playwright && playwright install chromium\n"
            "to enable Meta Ad Library searches."
        )

    url = _build_url(query, country, active_only)

    try:
        browser = await _ensure_browser()
    except Exception as e:
        return f"Failed to launch browser: {e}"

    intercepted_ads: list[dict] = []

    async def _on_response(response):
        """Capture API responses that contain ad data."""
        req_url = response.url
        if not any(
            p in req_url
            for p in ("/api/graphql/", "/ads/library/async/search_ads/", "ad_library")
        ):
            return
        try:
            body = await response.text()
            # Handle multi-line JSON (batched GraphQL)
            for line in body.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    found = _extract_ads_from_response(data, max_results)
                    intercepted_ads.extend(found)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    context = None
    try:
        context = await browser.new_context(user_agent=USER_AGENT, locale="en-US")
        page = await context.new_page()
        page.on("response", _on_response)

        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Wait for the Ad Library content to render (JS SPA)
        try:
            await page.wait_for_selector(
                'div[role="article"], div[class*="xrvj5dj"], div._7jvw',
                timeout=15000,
            )
        except Exception:
            pass  # fallback: proceed with whatever loaded
        # Extra buffer for API responses to arrive
        await page.wait_for_timeout(5000)

        # Check for CAPTCHA / checkpoint
        if "checkpoint" in page.url:
            return (
                "Meta is showing a CAPTCHA/checkpoint page. "
                "This may happen after too many requests. Try again later."
            )

        # Use intercepted ads if we got any, otherwise fall back to DOM
        ads = intercepted_ads[:max_results]
        if not ads:
            ads = await _fallback_dom_parse(page, max_results)

        return _format_ads_for_llm(ads, query)

    except Exception as e:
        error_str = str(e)
        if "Timeout" in error_str:
            return (
                f"Page load timed out while searching for \"{query}\". "
                "The Meta Ad Library may be slow or blocking automated access. "
                "Try again shortly."
            )
        return f"Error scraping Meta Ad Library: {error_str}"
    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass
