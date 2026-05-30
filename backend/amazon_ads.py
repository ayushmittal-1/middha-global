"""
Amazon Advertising API integration — OAuth + Sponsored Products campaign creation.
"""

import os
from pathlib import Path

import httpx

# ── Config ───────────────────────────────────────────────────────────────────
LWA_CLIENT_ID = os.getenv("AMAZON_LWA_CLIENT_ID", "")
LWA_CLIENT_SECRET = os.getenv("AMAZON_LWA_CLIENT_SECRET", "")
REFRESH_TOKEN = os.getenv("AMAZON_REFRESH_TOKEN", "")
PROFILE_ID = os.getenv("AMAZON_PROFILE_ID", "")

TOKEN_URL = "https://api.amazon.com/auth/o2/token"
ADS_API_BASE = "https://advertising-api.amazon.com"

_cached_access_token: str | None = None


# ── OAuth helpers ────────────────────────────────────────────────────────────

async def exchange_auth_code(code: str, redirect_uri: str) -> dict:
    """Exchange an authorization code for access + refresh tokens."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": LWA_CLIENT_ID,
            "client_secret": LWA_CLIENT_SECRET,
        })
        resp.raise_for_status()
        return resp.json()


async def get_access_token() -> str:
    """Get a fresh access token using the stored refresh token."""
    global _cached_access_token

    refresh = REFRESH_TOKEN or os.getenv("AMAZON_REFRESH_TOKEN", "")
    if not refresh:
        raise RuntimeError(
            "No Amazon refresh token configured. "
            "Visit /amazon/login to complete the OAuth flow first."
        )

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": LWA_CLIENT_ID,
            "client_secret": LWA_CLIENT_SECRET,
        })
        resp.raise_for_status()
        data = resp.json()
        _cached_access_token = data["access_token"]
        return _cached_access_token


def _headers(access_token: str) -> dict:
    """Standard headers for Amazon Ads API requests."""
    profile_id = PROFILE_ID or os.getenv("AMAZON_PROFILE_ID", "")
    h = {
        "Authorization": f"Bearer {access_token}",
        "Amazon-Advertising-API-ClientId": LWA_CLIENT_ID,
        "Content-Type": "application/vnd.spCampaign.v3+json",
        "Accept": "application/vnd.spCampaign.v3+json",
    }
    if profile_id:
        h["Amazon-Advertising-API-Scope"] = profile_id
    return h


def _simple_headers(access_token: str) -> dict:
    """Headers for simple endpoints (profiles, ad groups, keywords)."""
    profile_id = PROFILE_ID or os.getenv("AMAZON_PROFILE_ID", "")
    h = {
        "Authorization": f"Bearer {access_token}",
        "Amazon-Advertising-API-ClientId": LWA_CLIENT_ID,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if profile_id:
        h["Amazon-Advertising-API-Scope"] = profile_id
    return h


# ── API calls ────────────────────────────────────────────────────────────────

async def get_profiles() -> list[dict]:
    """List all advertising profiles for the authenticated account."""
    token = await get_access_token()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{ADS_API_BASE}/v2/profiles",
            headers=_simple_headers(token),
        )
        resp.raise_for_status()
        return resp.json()


async def create_sp_campaign(
    name: str,
    budget: float,
    start_date: str,
    state: str = "enabled",
) -> dict:
    """Create a Sponsored Products campaign.

    Args:
        name: Campaign name.
        budget: Daily budget in the marketplace currency.
        start_date: YYYYMMDD format.
        state: 'enabled' or 'paused'.
    """
    token = await get_access_token()
    payload = {
        "campaigns": [
            {
                "name": name,
                "targetingType": "MANUAL",
                "state": state,
                "dynamicBidding": {
                    "strategy": "LEGACY_FOR_SALES",
                },
                "budget": {
                    "budgetType": "DAILY",
                    "budget": budget,
                },
                "startDate": start_date,
            }
        ]
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{ADS_API_BASE}/sp/campaigns",
            headers=_headers(token),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def create_ad_group(
    campaign_id: str,
    name: str,
    default_bid: float = 0.75,
) -> dict:
    """Create an ad group inside a campaign."""
    token = await get_access_token()
    payload = {
        "adGroups": [
            {
                "campaignId": campaign_id,
                "name": name,
                "state": "enabled",
                "defaultBid": default_bid,
            }
        ]
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{ADS_API_BASE}/sp/adGroups",
            headers=_simple_headers(token),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def add_keywords(
    campaign_id: str,
    ad_group_id: str,
    keywords: list[str],
    match_type: str = "BROAD",
    bid: float = 0.75,
) -> dict:
    """Add keyword targets to an ad group."""
    token = await get_access_token()
    payload = {
        "keywords": [
            {
                "campaignId": campaign_id,
                "adGroupId": ad_group_id,
                "state": "enabled",
                "keywordText": kw,
                "matchType": match_type,
                "bid": bid,
            }
            for kw in keywords
        ]
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{ADS_API_BASE}/sp/keywords",
            headers=_simple_headers(token),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def add_negative_keywords(
    campaign_id: str,
    ad_group_id: str,
    keywords: list[str],
) -> dict:
    """Add campaign-level negative keywords."""
    token = await get_access_token()
    payload = {
        "negativeKeywords": [
            {
                "campaignId": campaign_id,
                "adGroupId": ad_group_id,
                "state": "enabled",
                "keywordText": kw,
                "matchType": "NEGATIVE_EXACT",
            }
            for kw in keywords
        ]
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{ADS_API_BASE}/sp/negativeKeywords",
            headers=_simple_headers(token),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ── Persist refresh token to .env ────────────────────────────────────────────

def save_refresh_token(token: str) -> None:
    """Write AMAZON_REFRESH_TOKEN back to .env file."""
    global REFRESH_TOKEN
    REFRESH_TOKEN = token
    os.environ["AMAZON_REFRESH_TOKEN"] = token

    env_path = Path(__file__).parent.parent / ".env"
    content = env_path.read_text()
    if "AMAZON_REFRESH_TOKEN=" in content:
        lines = content.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("AMAZON_REFRESH_TOKEN="):
                new_lines.append(f"AMAZON_REFRESH_TOKEN={token}")
            else:
                new_lines.append(line)
        env_path.write_text("\n".join(new_lines) + "\n")
    else:
        with open(env_path, "a") as f:
            f.write(f"\nAMAZON_REFRESH_TOKEN={token}\n")


def save_profile_id(profile_id: str) -> None:
    """Write AMAZON_PROFILE_ID back to .env file."""
    global PROFILE_ID
    PROFILE_ID = profile_id
    os.environ["AMAZON_PROFILE_ID"] = profile_id

    env_path = Path(__file__).parent.parent / ".env"
    content = env_path.read_text()
    if "AMAZON_PROFILE_ID=" in content:
        lines = content.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("AMAZON_PROFILE_ID="):
                new_lines.append(f"AMAZON_PROFILE_ID={profile_id}")
            else:
                new_lines.append(line)
        env_path.write_text("\n".join(new_lines) + "\n")
    else:
        with open(env_path, "a") as f:
            f.write(f"\nAMAZON_PROFILE_ID={profile_id}\n")
