"""Google Search autocomplete — buyer-intent keyword source.

Wraps the same endpoint every SEO tool uses under the hood
(`suggestqueries.google.com/complete/search`). Returns real user-typed
query completions for a seed, which makes it a much better keyword
source than broad audience-targeting APIs like Meta's `/search?type=adinterest`.

The endpoint is unofficial but has been stable for 15+ years. It will
throttle if hammered — the matrix job caps concurrency at 4 for this
reason. A User-Agent header is REQUIRED; bare requests get 403.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

log = logging.getLogger("google_autocomplete")

_ENDPOINT = "https://suggestqueries.google.com/complete/search"
# Locked to US English so the same seller sees the same suggestions
# regardless of where the backend is deployed. If you ever localize
# for non-US marketplaces, thread hl/gl through as params.
_DEFAULT_PARAMS = {"client": "firefox", "hl": "en", "gl": "us"}
# Mimic a real browser — Google rejects bare programmatic UAs.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
_TIMEOUT = 8.0

# Small, single-owner semaphore so the matrix job (which fires many seeds
# in parallel) can't accidentally DDoS Google and trigger a 429/block.
# Tuned conservatively — Google's soft limit sits somewhere around 5-10
# rps per IP for this endpoint. 4 concurrent + ~200-500ms per call keeps
# us well under.
_CONCURRENCY = asyncio.Semaphore(4)


async def fetch_google_suggestions(seed: str) -> list[str]:
    """Return Google's autocomplete suggestions for a seed query.

    Returns [] on any failure (throttled, blocked, empty seed, malformed
    response) — the matrix job is best-effort and shouldn't fail because
    one seed 429s.
    """
    seed = (seed or "").strip()
    if not seed:
        return []

    params = {**_DEFAULT_PARAMS, "q": seed[:100]}
    async with _CONCURRENCY:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT,
                                         headers={"User-Agent": _UA}) as c:
                resp = await c.get(_ENDPOINT, params=params)
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            log.info("google suggest failed for '%s': %s", seed[:40], e)
            return []

    if resp.status_code != 200:
        log.info("google suggest %s for '%s'", resp.status_code, seed[:40])
        return []

    # Response shape: [query, [suggestions...], ...] — JSON but returned
    # as text/javascript so `.json()` sometimes chokes on the content-type
    # header. Parse manually.
    try:
        payload = json.loads(resp.text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    suggestions = payload[1] if isinstance(payload[1], list) else []
    return [s.strip() for s in suggestions if isinstance(s, str) and s.strip()]
