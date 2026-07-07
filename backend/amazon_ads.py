"""
Amazon Advertising API integration — OAuth + Sponsored Products campaign creation.

App-level secrets (LWA client id/secret) live in .env. Per-user secrets
(refresh tokens, profile id, region/marketplace) come from the authenticated
user document in MongoDB — loaded by `auth.protect` / `auth.authenticate_ws`
and pulled from the ContextVar by `auth.require_user()`.
"""

import os

import httpx

from auth import require_user
from bson import ObjectId
from datetime import datetime, timezone

# ── App-level config (stays in env) ──────────────────────────────────────────
#
# Aurora registers TWO separate LWA apps with Amazon — one for SP-API and one
# for the Advertising API. Refresh tokens minted by the Ads app can ONLY be
# exchanged using the Ads client_id/secret; trying to use the SP-API pair
# returns a generic 400 from /auth/o2/token. We mirror Aurora's split.
LWA_CLIENT_ID = os.getenv("AMAZON_LWA_CLIENT_ID", "")
LWA_CLIENT_SECRET = os.getenv("AMAZON_LWA_CLIENT_SECRET", "")
ADS_LWA_CLIENT_ID = os.getenv("AMAZON_ADVERTISING_CLIENT_ID", "") or LWA_CLIENT_ID
ADS_LWA_CLIENT_SECRET = os.getenv("AMAZON_ADVERTISING_CLIENT_SECRET", "") or LWA_CLIENT_SECRET

# Region-aware endpoints (mirrors Aurora's amazonAPI.js).
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


def _user_region(user: dict) -> str:
    return (user.get("marketplace") or "NA").upper()


def _token_url(user: dict) -> str:
    return _TOKEN_URLS.get(_user_region(user), _TOKEN_URLS["NA"])


def _ads_base(user: dict) -> str:
    return _ADS_BASE_URLS.get(_user_region(user), _ADS_BASE_URLS["NA"])


def _ads_profile_id(user: dict) -> str:
    profiles = user.get("amazonAdsProfileIds") or []
    return str(profiles[0]) if profiles else ""


# ── OAuth helpers ────────────────────────────────────────────────────────────

async def exchange_auth_code(code: str, redirect_uri: str) -> dict:
    """Exchange an authorization code for access + refresh tokens.

    Token exchange uses app-level LWA creds and does not need user context.
    Default to the NA token URL — Amazon LWA accepts cross-region exchanges.
    """
    url = os.getenv("AMAZON_TOKEN_URL", _TOKEN_URLS["NA"])
    print(
        f"[exchange_auth_code] POST {url} redirect_uri={redirect_uri!r} "
        f"client_id={LWA_CLIENT_ID[:25]}... code={code[:8]}..."
    )
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, data={
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


async def _get_access_token(user: dict, refresh_field: str) -> str:
    from token_encryption import decrypt_token, is_encrypted

    refresh = user.get(refresh_field)
    if not refresh:
        raise RuntimeError(
            f"User {user.get('email')} has no {refresh_field}. "
            "Complete the Amazon OAuth flow in Aurora first."
        )
    # Aurora's Node backend stores these tokens AES-256-GCM encrypted with
    # the `enc:v1:` prefix (auroraBackend/src/models/User.js). Decrypt
    # transparently — otherwise we'd send the ciphertext to LWA and get
    # `invalid_grant`. Plaintext tokens (legacy or dev-seeded) pass through.
    if is_encrypted(refresh):
        try:
            refresh = decrypt_token(refresh)
        except Exception as e:
            raise RuntimeError(
                f"Failed to decrypt {refresh_field} for {user.get('email')}: "
                f"{e}. Check TOKEN_ENCRYPTION_KEY / SELLER_APP_ENCRYPTION_KEY "
                "matches the Node backend."
            )
    # Refresh tokens are app-scoped — Ads tokens MUST be exchanged with the
    # Ads LWA client_id/secret; SP-API tokens with the SP-API ones. Mixing
    # them up gives a generic 400 from /auth/o2/token.
    if refresh_field == "amazonAdsRefreshToken":
        # Ads uses a single shared LWA app (env-only — Aurora's
        # SellerApplication has no ads-LWA fields).
        client_id, client_secret = ADS_LWA_CLIENT_ID, ADS_LWA_CLIENT_SECRET
    else:
        # SP-API LWA is per-organization — prefer the user's SellerApplication
        # creds, env fallback. Matches Aurora's getSellerAppCredentials.
        from seller_app import get_seller_app_credentials  # local import avoids circular
        creds = await get_seller_app_credentials(user)
        client_id = creds.get("amazonLwaClientId") or LWA_CLIENT_ID
        client_secret = creds.get("amazonLwaClientSecret") or LWA_CLIENT_SECRET
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(_token_url(user), data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": client_id,
            "client_secret": client_secret,
        })
        if resp.is_error:
            # Amazon prefixes the body with a long `error_index` correlation
            # ID (~280 chars) — a 300-char truncation eats the actual
            # `error` / `error_description` we need. Log the whole body,
            # and try to surface the diagnostic error code in the exception
            # message so callers (and the FE via /profitability's `error`
            # field) can see `invalid_grant` / `invalid_client` etc.
            body_text = resp.text
            print(
                f"[lwa] refresh FAILED for {refresh_field} "
                f"status={resp.status_code} client_id={client_id[:35]}... "
                f"body={body_text[:1500]}"
            )
            err_code = err_desc = ""
            try:
                j = resp.json()
                err_code = j.get("error") or ""
                err_desc = j.get("error_description") or ""
            except Exception:
                pass
            if err_code or err_desc:
                raise RuntimeError(
                    f"LWA refresh failed ({refresh_field}): "
                    f"{err_code}: {err_desc}"
                )
        resp.raise_for_status()
        return resp.json()["access_token"]


async def get_ads_access_token(user: dict | None = None) -> str:
    """Fresh access token for the Advertising API."""
    user = user or require_user()
    return await _get_access_token(user, "amazonAdsRefreshToken")


async def get_sp_access_token(user: dict | None = None) -> str:
    """Fresh access token for the Selling Partner API."""
    user = user or require_user()
    return await _get_access_token(user, "amazonRefreshToken")


# Backwards-compat alias — historical callers expected the SP-API token here.
async def get_access_token() -> str:
    return await get_sp_access_token()


def _versioned_headers(user: dict, access_token: str, content_type: str) -> dict:
    """Headers for SP v3 endpoints, which require a per-resource versioned media type."""
    h = {
        "Authorization": f"Bearer {access_token}",
        "Amazon-Advertising-API-ClientId": ADS_LWA_CLIENT_ID,
        "Content-Type": content_type,
        "Accept": content_type,
    }
    profile_id = _ads_profile_id(user)
    if profile_id:
        h["Amazon-Advertising-API-Scope"] = profile_id
    return h


# ── API calls ────────────────────────────────────────────────────────────────

async def _post_json(label: str, path: str, content_type: str, payload: dict) -> dict:
    """POST to an Ads API endpoint with logging of the request and response."""
    user = require_user()
    token = await get_ads_access_token(user)
    url = f"{_ads_base(user)}{path}"
    print(f"[amazon_ads] -> POST {label} {url}")
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            url,
            headers=_versioned_headers(user, token, content_type),
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
    user = require_user()
    token = await get_ads_access_token(user)
    headers = {
        "Authorization": f"Bearer {token}",
        "Amazon-Advertising-API-ClientId": ADS_LWA_CLIENT_ID,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{_ads_base(user)}/v2/profiles", headers=headers)
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
    """Create a Sponsored Products campaign."""
    payload = {
        "campaigns": [
            {
                "name": name,
                "targetingType": "MANUAL",
                "state": state,
                "dynamicBidding": {"strategy": "LEGACY_FOR_SALES"},
                "budget": {"budgetType": "DAILY", "budget": budget},
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
    """Create a Sponsored Products product ad linking a SKU or ASIN to an ad group."""
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


# ── Persist OAuth results back to the user document ──────────────────────────

async def save_refresh_token(token: str, scope: str = "sp") -> None:
    """Write the refresh token into the authenticated user's Mongo doc.

    scope='sp' writes amazonRefreshToken, scope='ads' writes amazonAdsRefreshToken.
    """
    user = require_user()
    field = "amazonAdsRefreshToken" if scope == "ads" else "amazonRefreshToken"
    expires_field = "amazonAdsTokenExpiresAt" if scope == "ads" else "amazonTokenExpiresAt"

    from auth import _db  # local import to avoid circular at module load
    await _db().users.update_one(
        {"_id": ObjectId(str(user["_id"]))},
        {"$set": {field: token, expires_field: None, "updatedAt": datetime.now(timezone.utc)}},
    )
    user[field] = token


async def save_profile_id(profile_id: str) -> None:
    """Append profile_id onto the user's amazonAdsProfileIds (and dedupe)."""
    user = require_user()
    existing = list(user.get("amazonAdsProfileIds") or [])
    if profile_id not in existing:
        existing.insert(0, profile_id)

    from auth import _db
    await _db().users.update_one(
        {"_id": ObjectId(str(user["_id"]))},
        {"$set": {"amazonAdsProfileIds": existing, "updatedAt": datetime.now(timezone.utc)}},
    )
    user["amazonAdsProfileIds"] = existing
