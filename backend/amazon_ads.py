"""
Amazon Advertising API integration — OAuth + Sponsored Products campaign creation.

App-level secrets (LWA client id/secret) live in .env. Per-user secrets
(refresh tokens, profile id, region/marketplace) come from the authenticated
user document in MongoDB — loaded by `auth.protect` / `auth.authenticate_ws`
and pulled from the ContextVar by `auth.require_user()`.
"""

import asyncio
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


async def fetch_suggested_keywords(
    asin: str,
    max_suggestions: int = 100,
) -> dict:
    """Return Amazon-recommended keywords for a product (Sponsored Products).

    Wraps GET /v2/sp/asins/{asin}/suggested/keywords. Fast (<1s), no report
    generation. Amazon's response is a raw JSON list of
    `{"keywordText": "...", "matchType": "..."}` items — we normalize to
    `{"asin": ..., "suggestedKeywords": [...]}` here so callers don't have
    to deal with the top-level list shape (which broke code that assumed
    the newer wrapped shape documented in some Amazon guides).
    Bid data is not included here — use POST /sp/targets/bid/recommendations
    with an adGroupId if you need per-keyword bid ranges.
    """
    user = require_user()
    token = await get_ads_access_token(user)
    profile_id = _ads_profile_id(user) or os.getenv("AMAZON_PROFILE_ID", "")
    if not profile_id:
        raise RuntimeError(
            "No Ads profile id available for this user. Complete the Ads OAuth "
            "flow (POST /amazon/profiles/{profile_id}/select) or set AMAZON_PROFILE_ID."
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Amazon-Advertising-API-ClientId": ADS_LWA_CLIENT_ID,
        "Amazon-Advertising-API-Scope": profile_id,
        "Accept": "application/json",
    }
    params = {"maxNumSuggestions": max(1, min(max_suggestions, 1000))}
    url = f"{_ads_base(user)}/v2/sp/asins/{asin}/suggested/keywords"
    print(f"[amazon_ads] -> GET suggestedKeywords {url} params={params}")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.is_error:
            print(f"[amazon_ads] <- suggestedKeywords FAILED status={resp.status_code} body={resp.text}")
            resp.raise_for_status()
        raw = resp.json()
    if isinstance(raw, list):
        suggestions = raw
    elif isinstance(raw, dict):
        suggestions = raw.get("suggestedKeywords") or []
    else:
        suggestions = []
    return {"asin": asin, "suggestedKeywords": suggestions}


_SP_BID_REC_MEDIA = "application/vnd.spthemebasedbidrecommendation.v4+json"


async def fetch_keyword_bid_recommendations(
    keywords: list[str],
    ad_group_id: str,
    campaign_id: str,
    match_type: str = "BROAD",
) -> dict:
    """Return low/medium/high suggested bids per keyword for an existing ad group.

    Wraps POST /sp/targets/bid/recommendations (v4 media type). The legacy
    /v2/sp/keywords/bidRecommendations endpoint was retired — it now 404s.
    v4 requires BOTH the ad group id AND its campaign id, plus a
    `recommendationType` and `targetingExpressions` list.

    Amazon returns three bid values per targeting expression, ordered
    low → medium → high. We normalize the response to the same shape the
    caller previously consumed:
      {"adGroupId": "...", "recommendations": [
          {"keyword": "...", "matchType": "BROAD",
           "suggestedBid": {"low": 0.33, "suggested": 0.58, "high": 0.64}},
          ...
      ]}
    Cap is 100 targeting expressions per call; larger batches are chunked.
    """
    user = require_user()
    token = await get_ads_access_token(user)
    profile_id = _ads_profile_id(user) or os.getenv("AMAZON_PROFILE_ID", "")
    if not profile_id:
        raise RuntimeError(
            "No Ads profile id available for this user. Complete the Ads OAuth "
            "flow (POST /amazon/profiles/{profile_id}/select) or set AMAZON_PROFILE_ID."
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Amazon-Advertising-API-ClientId": ADS_LWA_CLIENT_ID,
        "Amazon-Advertising-API-Scope": profile_id,
        "Content-Type": _SP_BID_REC_MEDIA,
        "Accept": _SP_BID_REC_MEDIA,
    }
    # v4 uses one targeting-expression type per match type.
    expr_type = f"KEYWORD_{match_type.upper()}_MATCH"
    url = f"{_ads_base(user)}/sp/targets/bid/recommendations"
    recs: list[dict] = []
    # Retry schedule for 429s. Ads API has a per-endpoint bucket that refills
    # in a few seconds — three attempts (~1.5s + ~4s + ~10s) covers the usual
    # throttle without dragging the pipeline for a full minute on a hard cap.
    retry_waits = (1.5, 4.0, 10.0)
    async with httpx.AsyncClient(timeout=20) as client:
        for i in range(0, len(keywords), 100):
            chunk = keywords[i : i + 100]
            payload = {
                "campaignId": campaign_id,
                "adGroupId": ad_group_id,
                "recommendationType": "BIDS_FOR_EXISTING_AD_GROUP",
                "targetingExpressions": [
                    {"type": expr_type, "value": k} for k in chunk
                ],
            }
            print(f"[amazon_ads] -> POST bidRecommendations {url} n={len(chunk)}")
            resp = None
            for attempt in range(len(retry_waits) + 1):
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 429:
                    break
                if attempt >= len(retry_waits):
                    break
                # Respect Retry-After when Amazon sets it, else fall back to
                # our schedule. Retry-After is typically seconds.
                ra = resp.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else retry_waits[attempt]
                except ValueError:
                    wait = retry_waits[attempt]
                print(
                    f"[amazon_ads] <- bidRecommendations 429 — "
                    f"waiting {wait:.1f}s (attempt {attempt + 1}/{len(retry_waits)})"
                )
                await asyncio.sleep(wait)
            if resp is None or resp.is_error:
                status = resp.status_code if resp else "no response"
                body = resp.text if resp else ""
                print(
                    f"[amazon_ads] <- bidRecommendations FAILED "
                    f"status={status} body={body}"
                )
                if resp is not None:
                    resp.raise_for_status()
                raise RuntimeError("bidRecommendations: no response")
            data = resp.json()
            # v4 groups bids by "theme"; a single ad group gets one theme
            # (CONVERSION_OPPORTUNITIES). Flatten it back into a per-keyword list.
            for theme in data.get("bidRecommendations", []):
                for entry in theme.get("bidRecommendationsForTargetingExpressions", []):
                    expr = entry.get("targetingExpression") or {}
                    kw = expr.get("value")
                    bids = [b.get("suggestedBid") for b in entry.get("bidValues", [])]
                    if not kw or not bids:
                        continue
                    low = bids[0] if len(bids) > 0 else None
                    mid = bids[len(bids) // 2] if bids else None
                    high = bids[-1] if bids else None
                    recs.append({
                        "keyword": kw,
                        "matchType": match_type,
                        "suggestedBid": {
                            "low": low,
                            "suggested": mid,
                            "high": high,
                        },
                    })
    return {"adGroupId": ad_group_id, "recommendations": recs}


async def find_default_ad_group(preferred_asin: str | None = None) -> dict | None:
    """Return any valid ENABLED (campaignId, adGroupId) pair the account owns.

    Used by the keyword-matrix flow to auto-pick an ad group for bid-recs when
    the caller didn't supply one — bid-recs needs both IDs and requires the ad
    group to actually exist in the account.

    Prefers ad groups whose product ads include `preferred_asin` (so bids come
    back scoped to that product's competitive context), then falls back to any
    ENABLED ad group. NOTE: as of 2026-07, Amazon's `asinFilter` on
    `/sp/productAds/list` is silently ignored — the filter is sent but Amazon
    returns the same ordering for any ASIN. We keep the filter in the payload
    (harmless, forward-compatible if Amazon fixes it) and scan up to 100 results
    for a client-side match; if none matches we fall back to the first result.
    """
    for state in ("ENABLED", "PAUSED"):
        payload: dict = {
            "stateFilter": {"include": [state]},
            "maxResults": 100,
        }
        if preferred_asin:
            payload["asinFilter"] = {
                "queryTermMatchType": "EXACT_MATCH",
                "include": [preferred_asin],
            }
        try:
            data = await _post_json(
                "productAds/list", "/sp/productAds/list", SP_PRODUCT_AD_CT, payload
            )
        except Exception as e:
            print(f"[amazon_ads] productAds lookup failed state={state}: {e}")
            continue
        ads = data.get("productAds") if isinstance(data, dict) else None
        if not ads:
            continue
        # Client-side match on preferred ASIN if Amazon's filter didn't narrow.
        matched_by_asin = False
        pick = ads[0]
        if preferred_asin:
            for a in ads:
                if (a.get("asin") or "").upper() == preferred_asin.upper():
                    pick = a
                    matched_by_asin = True
                    break
        cid = pick.get("campaignId")
        gid = pick.get("adGroupId")
        if cid and gid:
            return {
                "campaignId": str(cid),
                "adGroupId": str(gid),
                "state": state,
                "matched_asin": matched_by_asin,
            }
    return None


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
