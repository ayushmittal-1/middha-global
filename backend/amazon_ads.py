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

# Region-aware endpoints (mirrors Aurora's amazonAPI.js). Set AMAZON_REGION to
# NA (North America), EU (Europe), or FE (Far East). Defaults to NA.
_TOKEN_URLS = {
    "NA": "https://api.amazon.com/auth/o2/token",
    "EU": "https://api.amazon.co.uk/auth/o2/token",
    "FE": "https://api.amazon.co.jp/auth/o2/token",
}
_ADS_BASE_URLS = {
    "NA": "https://advertising-api.amazon.com",
    "EU": "https://advertising-api-eu.amazon.com",
    "FE": "https://advertising-api-fe.amazon.com",
}

# SP v3 endpoints require versioned content types per resource.
SP_CAMPAIGN_CT = "application/vnd.spCampaign.v3+json"
SP_AD_GROUP_CT = "application/vnd.spAdGroup.v3+json"
SP_KEYWORD_CT = "application/vnd.spKeyword.v3+json"
SP_NEGATIVE_KEYWORD_CT = "application/vnd.spNegativeKeyword.v3+json"
SP_PRODUCT_AD_CT = "application/vnd.spProductAd.v3+json"

_cached_access_token: str | None = None


def _region() -> str:
    return os.getenv("AMAZON_REGION", "NA").upper()


def _token_url() -> str:
    return _TOKEN_URLS.get(_region(), _TOKEN_URLS["NA"])


def _ads_base() -> str:
    return _ADS_BASE_URLS.get(_region(), _ADS_BASE_URLS["NA"])


# ── OAuth helpers ────────────────────────────────────────────────────────────

async def exchange_auth_code(code: str, redirect_uri: str) -> dict:
    """Exchange an authorization code for access + refresh tokens."""
    print(
        f"[exchange_auth_code] POST {_token_url()} redirect_uri={redirect_uri!r} "
        f"client_id={LWA_CLIENT_ID[:25]}... code={code[:8]}..."
    )
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(_token_url(), data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": LWA_CLIENT_ID,
            "client_secret": LWA_CLIENT_SECRET,
        })
        if resp.is_error:
            print(f"[exchange_auth_code] FAILED status={resp.status_code} body={resp.text}")
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")
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
        resp = await client.post(_token_url(), data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": LWA_CLIENT_ID,
            "client_secret": LWA_CLIENT_SECRET,
        })
        resp.raise_for_status()
        data = resp.json()
        _cached_access_token = data["access_token"]
        return _cached_access_token


def _versioned_headers(access_token: str, content_type: str) -> dict:
    """Headers for SP v3 endpoints, which require a per-resource versioned media type."""
    profile_id = PROFILE_ID or os.getenv("AMAZON_PROFILE_ID", "")
    h = {
        "Authorization": f"Bearer {access_token}",
        "Amazon-Advertising-API-ClientId": LWA_CLIENT_ID,
        "Content-Type": content_type,
        "Accept": content_type,
    }
    if profile_id:
        h["Amazon-Advertising-API-Scope"] = profile_id
    return h


def _simple_headers(access_token: str) -> dict:
    """Headers for plain-JSON endpoints (e.g. v2 profiles)."""
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

async def _post_json(label: str, path: str, content_type: str, payload: dict) -> dict:
    """POST to an Ads API endpoint with logging of the request and response."""
    token = await get_access_token()
    url = f"{_ads_base()}{path}"
    print(f"[amazon_ads] -> POST {label} {url}")
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            url,
            headers=_versioned_headers(token, content_type),
            json=payload,
        )
        if resp.is_error:
            print(f"[amazon_ads] <- {label} FAILED status={resp.status_code} body={resp.text}")
            resp.raise_for_status()
        print(f"[amazon_ads] <- {label} OK status={resp.status_code}")
        return resp.json()


async def get_profiles() -> list[dict]:
    """List all advertising profiles for the authenticated account.

    The /v2/profiles listing must NOT include an Amazon-Advertising-API-Scope
    header (that header selects a single profile and an invalid value 400s).
    """
    token = await get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Amazon-Advertising-API-ClientId": LWA_CLIENT_ID,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{_ads_base()}/v2/profiles", headers=headers)
        if resp.is_error:
            print(f"[get_profiles] FAILED status={resp.status_code} body={resp.text}")
            resp.raise_for_status()
        return resp.json()


async def create_sp_campaign(
    name: str,
    budget: float,
    start_date: str,
    state: str = "ENABLED",
) -> dict:
    """Create a Sponsored Products campaign.

    Args:
        name: Campaign name.
        budget: Daily budget in the marketplace currency.
        start_date: YYYY-MM-DD format.
        state: 'ENABLED', 'PAUSED', or 'PROPOSED'.
    """
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
    return await _post_json("campaign", "/sp/campaigns", SP_CAMPAIGN_CT, payload)


async def create_ad_group(
    campaign_id: str,
    name: str,
    default_bid: float = 0.75,
) -> dict:
    """Create an ad group inside a campaign."""
    payload = {
        "adGroups": [
            {
                "campaignId": campaign_id,
                "name": name,
                "state": "ENABLED",
                "defaultBid": default_bid,
            }
        ]
    }
    return await _post_json("adGroup", "/sp/adGroups", SP_AD_GROUP_CT, payload)


async def add_keywords(
    campaign_id: str,
    ad_group_id: str,
    keywords: list[str],
    match_type: str = "BROAD",
    bid: float = 0.75,
) -> dict:
    """Add keyword targets to an ad group."""
    payload = {
        "keywords": [
            {
                "campaignId": campaign_id,
                "adGroupId": ad_group_id,
                "state": "ENABLED",
                "keywordText": kw,
                "matchType": match_type,
                "bid": bid,
            }
            for kw in keywords
        ]
    }
    return await _post_json("keywords", "/sp/keywords", SP_KEYWORD_CT, payload)


async def add_negative_keywords(
    campaign_id: str,
    ad_group_id: str,
    keywords: list[str],
) -> dict:
    """Add campaign-level negative keywords."""
    payload = {
        "negativeKeywords": [
            {
                "campaignId": campaign_id,
                "adGroupId": ad_group_id,
                "state": "ENABLED",
                "keywordText": kw,
                "matchType": "NEGATIVE_EXACT",
            }
            for kw in keywords
        ]
    }
    return await _post_json(
        "negativeKeywords", "/sp/negativeKeywords", SP_NEGATIVE_KEYWORD_CT, payload
    )


async def create_product_ad(
    campaign_id: str,
    ad_group_id: str,
    sku: str | None = None,
    asin: str | None = None,
    state: str = "ENABLED",
) -> dict:
    """Create a Sponsored Products product ad linking a SKU or ASIN to an ad group.

    A campaign needs at least one product ad to actually serve. Provide either a
    seller SKU (FBA/seller-fulfilled) or an ASIN (vendor / catalog item).
    """
    if not sku and not asin:
        raise ValueError("create_product_ad requires either a sku or an asin.")

    ad: dict = {
        "campaignId": campaign_id,
        "adGroupId": ad_group_id,
        "state": state,
    }
    if sku:
        ad["sku"] = sku
    else:
        ad["asin"] = asin

    payload = {"productAds": [ad]}
    return await _post_json("productAd", "/sp/productAds", SP_PRODUCT_AD_CT, payload)


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
