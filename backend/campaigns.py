import os
import httpx

AURORA_API_URL = "https://aurorabackend-is4p.onrender.com/api/ads"
AURORA_API_TOKEN = os.getenv(
    "AURORA_API_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY5ZTBjNzUxMzYxODE0ZDg5ZGUzZmEwYiIsImlhdCI6MTc3OTUzMTEzMywiZXhwIjoxNzgyMTIzMTMzfQ.Imk0CJm2-ocMtuJz1kPRPQ15l6PlT4AkzcCCAuAcsc0",
)

_campaigns: list[dict] = []
_campaigns_summary: str = ""


async def fetch_all_campaigns():
    """Fetch all campaigns from Aurora API with pagination."""
    global _campaigns, _campaigns_summary

    headers = {
        "accept": "*/*",
        "authorization": f"Bearer {AURORA_API_TOKEN}",
        "origin": "https://www.auroratest.in",
        "referer": "https://www.auroratest.in/",
    }

    all_campaigns = []
    page = 1
    limit = 100

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {
                "page": page,
                "limit": limit,
                "sortBy": "campaignName",
                "sortOrder": "asc",
            }
            resp = await client.get(AURORA_API_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            ads = data.get("ads", [])
            all_campaigns.extend(ads)

            total = data.get("pagination", {}).get("total", 0)
            if len(all_campaigns) >= total or not ads:
                break
            page += 1

    _campaigns = all_campaigns
    _campaigns_summary = _build_summary(all_campaigns)
    print(f"Loaded {len(_campaigns)} campaigns")


def _build_summary(campaigns: list[dict]) -> str:
    """Build an ultra-compact pipe-delimited summary of campaigns for LLM context."""
    # Use short column names and pipe delimiter to minimize token count
    lines = ["nm|tp|st|co|bgt|spd|sal|dt"]
    for c in campaigns:
        budget = c.get("budget") or {}
        b = budget.get("amount", "")
        spend = c.get("spend") or {}
        sp = spend.get("amount", "")
        sales = c.get("sales") or {}
        sa = sales.get("amount", "")
        sd = (c.get("startDate") or "")[:10]
        # Abbreviate type
        ctype = c.get("campaignType", "")
        if ctype == "Sponsored Products":
            ctype = "SP"
        elif ctype == "Sponsored Brands":
            ctype = "SB"
        elif ctype == "Sponsored Display":
            ctype = "SD"
        # Abbreviate status
        status = c.get("status", "")
        smap = {"Enabled": "E", "Paused": "P", "Archived": "A"}
        status = smap.get(status, status)
        name = c.get("campaignName", "").replace("|", "/")
        line = f"{name}|{ctype}|{status}|{c.get('country','')}|{b}|{sp}|{sa}|{sd}"
        lines.append(line)
    return "\n".join(lines)


def get_campaigns() -> list[dict]:
    return _campaigns


def get_campaigns_summary() -> str:
    return _campaigns_summary
