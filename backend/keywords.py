import httpx


async def fetch_amazon_keywords(seed_keyword: str) -> list[str]:
    """Fetch keyword suggestions from Amazon Autocomplete for a seed keyword.

    Expands by appending a-z to the seed to get broader suggestions.
    Returns a deduplicated, sorted list.
    """
    url = "https://completion.amazon.com/search/complete"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    keywords: set[str] = set()

    prefixes = [seed_keyword] + [f"{seed_keyword} {chr(c)}" for c in range(ord("a"), ord("z") + 1)]

    async with httpx.AsyncClient(timeout=10) as client:
        for prefix in prefixes:
            params = {
                "search-alias": "aps",
                "client": "amazon-search-ui",
                "mkt": "1",
                "q": prefix,
            }
            try:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
                    keywords.update(data[1])
            except Exception:
                continue

    return sorted(keywords)


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
    url = "https://completion.amazon.com/search/complete"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    negatives: list[dict] = []
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=10) as client:
        for suffix in _NEGATIVE_SUFFIXES:
            query = f"{seed_keyword} {suffix}"
            params = {
                "search-alias": "aps",
                "client": "amazon-search-ui",
                "mkt": "1",
                "q": query,
            }
            try:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
                    for kw in data[1]:
                        kw_lower = kw.lower()
                        if kw_lower not in seen and kw_lower != seed_keyword.lower():
                            seen.add(kw_lower)
                            negatives.append({"keyword": kw, "reason": f"Contains '{suffix}' — likely non-buyer intent"})
            except Exception:
                continue

    # Also add the raw suffixes themselves as broad negatives
    for suffix in _NEGATIVE_SUFFIXES:
        if suffix not in seen:
            seen.add(suffix)
            negatives.append({"keyword": suffix, "reason": "Common non-buyer intent modifier"})

    return negatives
