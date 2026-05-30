import asyncio

import httpx

_SUGGESTIONS_URL = "https://completion.amazon.com/api/2017/suggestions"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _suggestion_params(prefix: str) -> dict:
    return {
        "mid": "ATVPDKIKX0DER",
        "alias": "aps",
        "prefix": prefix,
        "event": "onKeyPress",
        "limit": "11",
        "fb": "1",
        "suggestion-type": "KEYWORD",
    }


async def _fetch_keywords_for_seed(seed: str) -> list[str]:
    """Fetch keywords for a single seed, expanding with a-z."""
    keywords: set[str] = set()
    prefixes = [seed] + [f"{seed} {chr(c)}" for c in range(ord("a"), ord("z") + 1)]

    async def _fetch_one(client: httpx.AsyncClient, prefix: str):
        try:
            resp = await client.get(
                _SUGGESTIONS_URL, headers=_HEADERS, params=_suggestion_params(prefix)
            )
            resp.raise_for_status()
            data = resp.json()
            for s in data.get("suggestions", []):
                if s.get("type") == "KEYWORD" and s.get("value"):
                    keywords.add(s["value"])
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=5) as client:
        await asyncio.gather(*[_fetch_one(client, p) for p in prefixes])

    return sorted(keywords)


async def fetch_amazon_keywords(seed_keyword: str) -> list[str]:
    """Fetch keyword suggestions from Amazon Autocomplete for a seed keyword.

    If the full phrase returns no results, progressively tries shorter
    versions (dropping words from the end) until results are found.
    Returns a deduplicated, sorted list capped at 25.
    """
    words = seed_keyword.strip().split()

    # Try full phrase first, then progressively shorter
    for end in range(len(words), 0, -1):
        shortened = " ".join(words[:end])
        results = await _fetch_keywords_for_seed(shortened)
        if results:
            return results[:25]

    return []


# ── Common negative keyword patterns ────────────────────────────────────────
# These are terms that often appear in autocomplete but indicate irrelevant
# buyer intent (e.g. looking for free stuff, DIY, repairs, jobs).
_NEGATIVE_SUFFIXES = [
    "free", "cheap", "used", "refurbished", "broken", "repair", "fix",
    "diy", "homemade", "manual", "instructions", "tutorial", "how to",
    "jobs", "career", "salary", "wholesale", "bulk", "scam", "complaint",
    "return", "refund", "recall", "lawsuit", "alternative to", "vs",
    "reddit", "review", "coupon", "discount code",
]


async def suggest_negative_keywords(seed_keyword: str) -> list[dict]:
    """Suggest negative keywords by checking Amazon Autocomplete for irrelevant modifiers.

    Returns a list of {keyword, reason} dicts so the LLM can explain each suggestion.
    """
    negatives: list[dict] = []
    seen: set[str] = set()

    async def _fetch_neg(client: httpx.AsyncClient, suffix: str):
        query = f"{seed_keyword} {suffix}"
        try:
            resp = await client.get(
                _SUGGESTIONS_URL, headers=_HEADERS, params=_suggestion_params(query)
            )
            resp.raise_for_status()
            data = resp.json()
            for s in data.get("suggestions", []):
                if s.get("type") == "KEYWORD" and s.get("value"):
                    kw = s["value"]
                    kw_lower = kw.lower()
                    if kw_lower not in seen and kw_lower != seed_keyword.lower():
                        seen.add(kw_lower)
                        negatives.append({"keyword": kw, "reason": f"Contains '{suffix}' — likely non-buyer intent"})
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=5) as client:
        await asyncio.gather(*[_fetch_neg(client, s) for s in _NEGATIVE_SUFFIXES])

    # Also add the raw suffixes themselves as broad negatives
    for suffix in _NEGATIVE_SUFFIXES:
        if suffix not in seen:
            seen.add(suffix)
            negatives.append({"keyword": suffix, "reason": "Common non-buyer intent modifier"})

    return negatives
