"""
Amazon Selling Partner API (SP-API) integration — Orders, Inventory, Reports.

Uses the same LWA credentials as amazon_ads.py for token exchange.
AWS IAM credentials are used for SigV4 request signing.
"""

import asyncio
import csv
import gzip
import hashlib
import hmac
import io
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlencode, urlparse

import httpx

from amazon_ads import get_sp_access_token
from auth import require_user

# In-memory cache for Product Fees API estimates. Amazon caps this endpoint
# at 1 req/s (2 burst); the profitability endpoint calls it once per SKU,
# which alone can trip 429s and definitely does on repeated "Apply" clicks.
# Estimates depend only on (ASIN, price, is_fba, marketplace), and Amazon's
# fee schedules don't change intra-day, so a 30-min per-process cache is safe.
_FEES_ESTIMATE_CACHE: dict[tuple, tuple[float, dict]] = {}
_FEES_ESTIMATE_TTL_S = 30 * 60

# In-memory cache for paginated getOrders. Orders API is 1 req/min sustained
# — the tightest limit we hit. A completed date window's orders don't change,
# so re-clicks (or the LLM tool + FE hitting profitability back-to-back)
# should reuse the last result instead of re-paging.
_ORDERS_CACHE: dict[tuple, tuple[float, dict]] = {}
_ORDERS_CACHE_TTL_S = 30 * 60

# ── App-level config (stays in env) ──────────────────────────────────────────

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Region-aware SP-API hosts.
_SP_API_BASES = {
    "NA": ("https://sellingpartnerapi-na.amazon.com", "us-east-1"),
    "EU": ("https://sellingpartnerapi-eu.amazon.com", "eu-west-1"),
    "FE": ("https://sellingpartnerapi-fe.amazon.com", "us-west-2"),
}
SP_API_SERVICE = "execute-api"


def _sp_base_and_region(user: dict) -> tuple[str, str]:
    region = (user.get("marketplace") or "NA").upper()
    return _SP_API_BASES.get(region, _SP_API_BASES["NA"])


_US_MARKETPLACE_ID = "ATVPDKIKX0DER"

# Human-readable names for the marketplace IDs Amazon publishes. Used by the
# new `get_marketplaces` tool so the LLM can present a friendly picker.
MARKETPLACE_NAMES = {
    "ATVPDKIKX0DER": "United States",
    "A2EUQ1WTGCTBG2": "Canada",
    "A1AM78C64UM0Y8": "Mexico",
    "A2Q3Y263D00KWC": "Brazil",
    "A1F83G8C2ARO7P": "United Kingdom",
    "A1PA6795UKMFR9": "Germany",
    "A13V1IB3VIYZZH": "France",
    "APJ6JRA9NG5V4": "Italy",
    "A1RKKUPIHCS9HS": "Spain",
    "A1805IZSGTT6HS": "Netherlands",
    "A2NODRKZP88ZB9": "Sweden",
    "A1C3SOZRARQ6R3": "Poland",
    "ARBP9OOSHTCHU": "Egypt",
    "A33AVAJ2PDY3EV": "Turkey",
    "A17E79C6D8DWNP": "Saudi Arabia",
    "A2VIGQ35RCS4UG": "United Arab Emirates",
    "A21TJRUUN4KGV": "India",
    "A19VAU5U5O7RUS": "Singapore",
    "A39IBJ37TRP1C6": "Australia",
    "A1VC38T7YXB528": "Japan",
    # Common mis-mapped ones that appear in NA accounts (verify in your seller central):
    "A1MQXOICRS2Z7M": "Canada (FBA)",
    "A2ZV50J4W1RKNI": "Saudi Arabia",
    "A3H6HPSLHAK3XG": "Egypt",
    "AHRY1CZE9ZY4H": "Singapore",
}


def _user_marketplace_ids(user: dict) -> list[str]:
    """All marketplaces the user is registered in. Fall back to US."""
    ids = user.get("amazonMarketplaceIds") or []
    return [str(x) for x in ids] if ids else [_US_MARKETPLACE_ID]


def _user_primary_marketplace_id(user: dict) -> str:
    """For endpoints that accept only a single marketplace id (inventory
    granularity, single-marketplace reports). Prefer US when available so
    the chatbot shows the active warehouse rather than an empty regional
    sub-marketplace."""
    ids = _user_marketplace_ids(user)
    return _US_MARKETPLACE_ID if _US_MARKETPLACE_ID in ids else ids[0]


def list_marketplaces() -> list[dict]:
    """Return the current user's marketplaces with human-readable names."""
    user = require_user()
    primary = _user_primary_marketplace_id(user)
    return [
        {
            "id": mid,
            "name": MARKETPLACE_NAMES.get(mid, "Unknown"),
            "is_primary": mid == primary,
        }
        for mid in _user_marketplace_ids(user)
    ]


# ISO-style short codes → list of canonical marketplace ids. Some countries
# have multiple historical ids (e.g. Saudi Arabia is A17E79C6D8DWNP on some
# seller central regions and A2ZV50J4W1RKNI on others); resolve_marketplace
# picks whichever one the *user* actually has.
_SHORT_CODES = {
    "us": ["ATVPDKIKX0DER"], "usa": ["ATVPDKIKX0DER"],
    "ca": ["A2EUQ1WTGCTBG2", "A1MQXOICRS2Z7M"],
    "mx": ["A1AM78C64UM0Y8"],
    "br": ["A2Q3Y263D00KWC"],
    "uk": ["A1F83G8C2ARO7P"], "gb": ["A1F83G8C2ARO7P"],
    "de": ["A1PA6795UKMFR9"],
    "fr": ["A13V1IB3VIYZZH"],
    "it": ["APJ6JRA9NG5V4"],
    "es": ["A1RKKUPIHCS9HS"],
    "nl": ["A1805IZSGTT6HS"],
    "se": ["A2NODRKZP88ZB9"],
    "pl": ["A1C3SOZRARQ6R3"],
    "tr": ["A33AVAJ2PDY3EV"],
    "eg": ["ARBP9OOSHTCHU", "A3H6HPSLHAK3XG"],
    "sa": ["A17E79C6D8DWNP", "A2ZV50J4W1RKNI"],
    "ae": ["A2VIGQ35RCS4UG"], "uae": ["A2VIGQ35RCS4UG"],
    "in": ["A21TJRUUN4KGV"],
    "sg": ["A19VAU5U5O7RUS", "AHRY1CZE9ZY4H"],
    "au": ["A39IBJ37TRP1C6"],
    "jp": ["A1VC38T7YXB528"],
}


def resolve_marketplace(
    user: dict,
    requested: str | list[str] | None,
    *,
    multiple: bool,
) -> list[str] | str:
    """Turn the LLM's `marketplace` arg into a clean marketplace id list (or
    single id). Accepts an id, full country name ("United States"), short
    code ("US", "SA", "UK"), a comma-separated string, a list, or None.
    If None: use all (multiple=True) or the primary (multiple=False)."""
    available = _user_marketplace_ids(user)
    full_name_lookup = {v.lower(): k for k, v in MARKETPLACE_NAMES.items()}

    def normalize(item: str) -> str | None:
        item = (item or "").strip()
        if not item:
            return None
        if item in available:
            return item
        # Full country name ("United States", "Saudi Arabia") — picks whichever
        # canonical id matches first; verify it's actually one this user has.
        full = full_name_lookup.get(item.lower())
        if full and full in available:
            return full
        # ISO-ish short code — multiple candidates possible, pick the first
        # one the user actually has registered.
        for candidate in _SHORT_CODES.get(item.lower(), []):
            if candidate in available:
                return candidate
        return None

    if requested is None or requested == "":
        return available if multiple else _user_primary_marketplace_id(user)

    if isinstance(requested, str):
        parts = [normalize(p) for p in requested.split(",")]
    else:
        parts = [normalize(p) for p in requested]

    cleaned = [p for p in parts if p]
    if not cleaned:
        # Nothing matched — fall back so the call doesn't hard-fail, but
        # this usually means the LLM passed a bad code. Behavior is the
        # same as omitting the arg.
        return available if multiple else _user_primary_marketplace_id(user)

    return cleaned if multiple else cleaned[0]

# ── SigV4 signing ────────────────────────────────────────────────────────────


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = _sign(("AWS4" + secret).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


def _sigv4_headers(
    method: str,
    url: str,
    headers: dict,
    region: str,
    body: str = "",
) -> dict:
    """Add SigV4 Authorization header to the request headers dict (in-place + returned)."""
    access_key = AWS_ACCESS_KEY_ID or os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = AWS_SECRET_ACCESS_KEY or os.getenv("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        # Aurora's amazon-sp-api npm client uses LWA access token only.
        return headers

    parsed = urlparse(url)
    host = parsed.hostname
    canonical_uri = quote(parsed.path or "/", safe="/")
    canonical_querystring = parsed.query  # already encoded by caller

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    headers["x-amz-date"] = amz_date
    headers["host"] = host

    # Canonical headers — must be sorted by lowercase key
    signed_header_keys = sorted(headers.keys())
    canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in signed_header_keys)
    signed_headers = ";".join(signed_header_keys)

    payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    canonical_request = "\n".join([
        method,
        canonical_uri,
        canonical_querystring,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    credential_scope = f"{date_stamp}/{region}/{SP_API_SERVICE}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    signing_key = _get_signature_key(secret_key, date_stamp, region, SP_API_SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


# ── HTTP helpers ─────────────────────────────────────────────────────────────


async def _sp_request(
    method: str,
    path: str,
    params: dict | None = None,
    body: dict | None = None,
    *,
    max_429_retries: int = 8,
) -> dict | list | str:
    """Make a signed SP-API request on behalf of the current authenticated user.

    Retries 429 (QuotaExceeded) with exponential backoff so a single throttled
    call doesn't fail the whole endpoint. Honors `x-amzn-RateLimit-Limit`
    (requests/sec) when present to pick a wait floor; otherwise falls back to
    exponential backoff starting at 1.5s."""
    user = require_user()
    access_token = await get_sp_access_token(user)
    sp_base, sp_region = _sp_base_and_region(user)

    query_string = urlencode(params, doseq=True) if params else ""
    url = f"{sp_base}{path}"
    if query_string:
        url = f"{url}?{query_string}"

    body_str = json.dumps(body) if body else ""

    print(f"[sp-api] -> {method} {path} params={params}")

    attempt = 0
    while True:
        # Re-sign every attempt: SigV4 signatures include a per-request
        # timestamp, so reusing headers across retries fails auth if we wait
        # more than 15 minutes (and is technically incorrect anyway).
        headers = {"content-type": "application/json"}
        _sigv4_headers(method, url, headers, sp_region, body=body_str)
        if "x-amz-date" not in headers:
            headers["x-amz-date"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H:%M%SZ")
        if "host" not in headers:
            headers["host"] = urlparse(url).hostname or ""
        headers["x-amz-access-token"] = access_token
        headers["user-agent"] = "MiddhaGlobal/1.0 (Language=Python)"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(
                method,
                url,
                headers=headers,
                content=body_str if body_str else None,
            )
            if resp.status_code == 429 and attempt < max_429_retries:
                # Prefer the advertised rate: `x-amzn-RateLimit-Limit` is
                # req/s, so 1/rate is the minimum spacing (60s for the
                # Orders API's 0.0167/s bucket, 2s for Order Items' 0.5/s,
                # etc.). Exponential growth on repeated hits, but cap
                # per-attempt at `max(90s, 1.5x rate window)` so we
                # actually wait long enough for the tight buckets to
                # refill instead of burning the retry budget on 30s waits.
                rate_hdr = resp.headers.get("x-amzn-RateLimit-Limit")
                try:
                    rate = float(rate_hdr) if rate_hdr else 0.0
                except ValueError:
                    rate = 0.0
                base = (1.0 / rate) if rate > 0 else 1.5
                per_attempt_cap = max(90.0, base * 1.5)
                wait = min(base * (2 ** attempt), per_attempt_cap)
                attempt += 1
                print(
                    f"[sp-api] <- 429 QuotaExceeded on {path}; "
                    f"retry {attempt}/{max_429_retries} in {wait:.1f}s "
                    f"(rate={rate_hdr})"
                )
                await asyncio.sleep(wait)
                continue
            if resp.is_error:
                print(f"[sp-api] <- FAILED {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
            print(f"[sp-api] <- OK {resp.status_code}")
            try:
                return resp.json()
            except Exception:
                return resp.text


# ── Orders API (v0) ─────────────────────────────────────────────────────────


async def get_orders(
    created_after: str | None = None,
    created_before: str | None = None,
    statuses: list[str] | None = None,
    max_results: int = 20,
    marketplace: str | list[str] | None = None,
    paginate: bool = False,
) -> dict:
    """List orders across the requested marketplaces (default: all the user
    is registered in). created_after / created_before are ISO-8601 (e.g.
    '2024-01-01T00:00:00Z').

    When paginate=True, follow `payload.NextToken` until the window is
    exhausted and return a merged payload (Orders concatenated, NextToken
    dropped). SP-API's contract: continuations send only `MarketplaceIds`
    + `NextToken`."""
    user = require_user()
    marketplace_ids = resolve_marketplace(user, marketplace, multiple=True)
    base_params = {
        "MarketplaceIds": ",".join(marketplace_ids),
        "MaxResultsPerPage": str(min(max_results, 100)),
    }
    if created_after:
        base_params["CreatedAfter"] = created_after
    if created_before:
        base_params["CreatedBefore"] = created_before
    if statuses:
        base_params["OrderStatuses"] = ",".join(statuses)

    if not paginate:
        return await _sp_request("GET", "/orders/v0/orders", params=base_params)

    # In-memory cache for paginated getOrders. Orders API is 0.0167 req/s
    # (1/min) — brutal on multi-page catalogs. A user re-clicking Apply on
    # the same window shouldn't repay that cost. Cache is per-process,
    # 30 min TTL, keyed by the query params that define the window.
    user_id = str(user.get("_id") or user.get("id") or "")
    cache_key = (
        user_id, base_params["MarketplaceIds"],
        base_params.get("CreatedAfter"), base_params.get("CreatedBefore"),
        base_params.get("OrderStatuses"),
    )
    now_ts = time.time()
    cached = _ORDERS_CACHE.get(cache_key)
    if cached and now_ts - cached[0] < _ORDERS_CACHE_TTL_S:
        print(f"[sp-api] getOrders cache HIT ({len(cached[1].get('payload',{}).get('Orders',[]))} orders)")
        return cached[1]

    merged: dict = {}
    orders: list = []
    page_params = dict(base_params)
    page_num = 0
    truncated_reason: str | None = None
    while True:
        # Pace pagination pages so we don't exhaust the 20-burst bucket in
        # one shot on a heavy catalog. 3s spacing = ~7 pages before we
        # start biting into the sustained rate; retries in _sp_request
        # cover the tail.
        if page_num > 0:
            await asyncio.sleep(3.0)
        page_num += 1
        try:
            resp = await _sp_request("GET", "/orders/v0/orders", params=page_params)
        except Exception as e:
            # If we've collected AT LEAST one page, degrade to partial
            # results instead of losing all of it. Fresh call from a new
            # process would just restart from page 1 on the same depleted
            # bucket and fail the same way — better to hand back what we
            # have plus a warning the FE can surface.
            if orders:
                truncated_reason = (
                    f"pagination halted at page {page_num} after "
                    f"{len(orders)} order(s): {str(e)[:200]}"
                )
                print(f"[sp-api] getOrders {truncated_reason}")
                break
            raise
        if not merged:
            merged = resp
        payload = resp.get("payload") or {}
        orders.extend(payload.get("Orders") or [])
        next_token = payload.get("NextToken")
        if not next_token:
            break
        page_params = {
            "MarketplaceIds": base_params["MarketplaceIds"],
            "NextToken": next_token,
        }

    if "payload" not in merged:
        merged["payload"] = {}
    merged["payload"]["Orders"] = orders
    merged["payload"].pop("NextToken", None)
    if truncated_reason:
        merged["_partial"] = truncated_reason
    # Only cache complete results; caching a partial page count would let
    # a bad-luck 429 poison the window for 30 minutes.
    if not truncated_reason:
        _ORDERS_CACHE[cache_key] = (now_ts, merged)
    return merged


async def get_order(order_id: str) -> dict:
    """Get details for a single order."""
    return await _sp_request("GET", f"/orders/v0/orders/{order_id}")


async def get_order_items(order_id: str) -> dict:
    """Get line items for an order."""
    return await _sp_request("GET", f"/orders/v0/orders/{order_id}/orderItems")


# ── Product Fees API (v0) ────────────────────────────────────────────────────


def _fee_line_amount(entry: dict) -> float:
    for key in ("FinalFee", "FeeAmount", "feeAmount"):
        block = entry.get(key)
        if block is not None:
            try:
                return float(block.get("Amount") or block.get("amount") or 0)
            except (TypeError, ValueError, AttributeError):
                pass
    try:
        return float(entry.get("amount") or 0)
    except (TypeError, ValueError):
        return 0.0


def split_bundled_fulfillment_total(total: float) -> tuple[float, float]:
    """Split a bundled FBA fulfillment total into base + fuel (Revenue Calculator)."""
    if total <= 0:
        return 0.0, 0.0
    base = round(total / 1.035, 4)
    fuel = round(total - base, 4)
    return base, fuel


def parse_fee_detail_lines(detail_list: list) -> dict:
    """Parse Product Fees API / Aurora `fees.breakdown` into referral, base FBA, fuel."""
    referral = 0.0
    fba = 0.0
    fuel = 0.0
    variable_closing = 0.0
    has_explicit_fuel = False
    has_base_fba = False
    bundled_fba = 0.0
    breakdown: list[dict] = []

    for entry in detail_list or []:
        ftype_raw = entry.get("FeeType") or entry.get("feeType") or ""
        ftype = str(ftype_raw).lower()
        amount = _fee_line_amount(entry)
        breakdown.append({"type": ftype_raw, "amount": amount})

        if ftype in ("fbafees", "fulfillmentfees"):
            bundled_fba += amount
            continue
        if ftype in ("commission",) or "referral" in ftype:
            referral += amount
        elif "fuel" in ftype or "inflation" in ftype:
            fuel += amount
            has_explicit_fuel = True
        elif "fbaperunitfulfillmentfee" in ftype or "fbaperorderfulfillmentfee" in ftype:
            fba += amount
            has_base_fba = True
        elif ("fba" in ftype or "fulfillment" in ftype) and "fuel" not in ftype:
            fba += amount
            has_base_fba = True
        elif "variableclosingfee" in ftype:
            variable_closing += amount

    if bundled_fba > 0 and not has_base_fba:
        base, bundled_fuel = split_bundled_fulfillment_total(bundled_fba)
        fba += base
        if not has_explicit_fuel:
            fuel += bundled_fuel
            has_explicit_fuel = bundled_fuel > 0
        has_base_fba = True
    elif fba > 0 and fuel == 0 and has_base_fba and not has_explicit_fuel:
        fuel = round(fba * 0.035, 4)

    total = referral + fba + fuel + variable_closing
    return {
        "referral": referral,
        "fba": fba,
        "fuel_surcharge": fuel,
        "variable_closing": variable_closing,
        "total": round(total, 4),
        "breakdown": breakdown,
    }


def _parse_fees_result(result: dict) -> dict:
    """Normalize one FeesEstimateResult into the shape callers expect
    (referral / fba / fuel_surcharge / total etc.). Same logic whether
    the result came from the singleton or batch endpoint."""
    estimate = (result.get("FeesEstimate") or {})
    detail_list = (estimate.get("FeeDetailList") or [])
    total = (estimate.get("TotalFeesEstimate") or {}).get("Amount") or 0
    parsed = parse_fee_detail_lines(detail_list)
    out = {
        **parsed,
        "total": float(total) if total else parsed["total"],
        "status": (result.get("Status") or "").lower(),
        "error": result.get("Error"),
    }
    return out


async def get_fees_estimate(
    asin: str,
    price: float,
    *,
    is_fba: bool = True,
    marketplace: str | None = None,
    currency: str = "USD",
) -> dict:
    """Estimate Amazon fees Amazon would charge if this ASIN sold at `price`.
    Returns {referral, fba, fuel_surcharge, total, breakdown:[...]} where each
    field is in `currency`. Caller multiplies by units to get the per-SKU
    fee total over a window.

    Per the PDF: referral, FBA fulfilment fee, and 3.5%-of-FBA fuel surcharge
    are all returned by Amazon as line items here, so we don't need to
    maintain a category percentages table or size-tier formulas ourselves.
    """
    user = require_user()
    marketplace_id = resolve_marketplace(user, marketplace, multiple=False)
    price_r = round(price, 2)
    cache_key = (asin, price_r, bool(is_fba), marketplace_id, currency)
    now_ts = time.time()
    cached = _FEES_ESTIMATE_CACHE.get(cache_key)
    if cached and now_ts - cached[0] < _FEES_ESTIMATE_TTL_S:
        return cached[1]
    body = {
        "FeesEstimateRequest": {
            "MarketplaceId": marketplace_id,
            "IsAmazonFulfilled": bool(is_fba),
            "PriceToEstimateFees": {
                "ListingPrice": {"Amount": price_r, "CurrencyCode": currency},
            },
            "Identifier": f"est-{asin}",
        }
    }
    resp = await _sp_request("POST", f"/products/fees/v0/items/{asin}/feesEstimate", body=body)
    payload = resp.get("payload") or {}
    result = payload.get("FeesEstimateResult") or {}
    out = _parse_fees_result(result)
    # Only cache successful estimates — Amazon returns Status="ClientError"
    # for un-listable ASINs; don't pin those in-memory in case the listing
    # comes back live within the TTL.
    if (out.get("status") or "").lower() == "success" or out["total"] > 0:
        _FEES_ESTIMATE_CACHE[cache_key] = (now_ts, out)
    return out


# Amazon caps the batch endpoint at 20 requests per call.
_FEES_BATCH_MAX = 20
# Batch endpoint is 0.5 req/s sustained (2 burst) — 20 ASINs per call means
# ~10x the throughput of the singleton (1 req/s × 1 ASIN). Pace batches at
# 2.1s spacing so we stay under the sustained limit.
_FEES_BATCH_MIN_SPACING_S = 2.1
_last_fees_batch_ts = 0.0


async def get_fees_estimates_batch(
    items: list[tuple[str, float]],
    *,
    is_fba: bool = True,
    marketplace: str | None = None,
    currency: str = "USD",
) -> dict[str, dict]:
    """Batch variant of get_fees_estimate — one HTTP call per 20 ASINs via
    /products/fees/v0/feesEstimate. Cache-aware: skips ASINs already in
    `_FEES_ESTIMATE_CACHE`, so a partial re-request only hits Amazon for
    the misses.

    Returns a {asin: normalized_estimate_dict} map. ASINs whose batch
    request errored (unlisted, price out of range, etc.) get a zero-fee
    dict with `status`/`error` populated so the caller can distinguish
    "no fees found" from "not queried".

    `items` is a list of (asin, price) tuples. Duplicate ASINs are
    de-duplicated by (asin, rounded price) — same as the singleton cache
    key — since Amazon's fee estimate depends on price."""
    global _last_fees_batch_ts
    if not items:
        return {}
    user = require_user()
    marketplace_id = resolve_marketplace(user, marketplace, multiple=False)
    now_ts = time.time()
    out: dict[str, dict] = {}
    # De-dupe by (asin, rounded price) so a repeated ASIN doesn't waste
    # a slot in the 20-item batch.
    seen: set[tuple[str, float]] = set()
    to_fetch: list[tuple[str, float]] = []
    for asin, price in items:
        if not asin or price is None or price <= 0:
            continue
        pr = round(float(price), 2)
        key = (asin, pr)
        if key in seen:
            continue
        seen.add(key)
        cache_key = (asin, pr, bool(is_fba), marketplace_id, currency)
        cached = _FEES_ESTIMATE_CACHE.get(cache_key)
        if cached and now_ts - cached[0] < _FEES_ESTIMATE_TTL_S:
            out[asin] = cached[1]
        else:
            to_fetch.append((asin, pr))

    if not to_fetch:
        return out

    for i in range(0, len(to_fetch), _FEES_BATCH_MAX):
        chunk = to_fetch[i : i + _FEES_BATCH_MAX]
        # Pace against the sustained batch limit (0.5/s = 2s spacing);
        # module-level `_last_fees_batch_ts` keeps concurrent callers
        # honest across requests.
        elapsed = time.time() - _last_fees_batch_ts
        if _last_fees_batch_ts and elapsed < _FEES_BATCH_MIN_SPACING_S:
            await asyncio.sleep(_FEES_BATCH_MIN_SPACING_S - elapsed)

        # Body is an array of FeesEstimateByIdRequest per the SP-API docs.
        # `Identifier` echoes back on the response so we can match ASINs
        # even if Amazon changes the response order.
        body = [
            {
                "FeesEstimateRequest": {
                    "MarketplaceId": marketplace_id,
                    "IsAmazonFulfilled": bool(is_fba),
                    "PriceToEstimateFees": {
                        "ListingPrice": {"Amount": pr, "CurrencyCode": currency},
                    },
                    "Identifier": f"est-{asin}-{pr}",
                },
                "IdType": "ASIN",
                "IdValue": asin,
            }
            for asin, pr in chunk
        ]
        try:
            resp = await _sp_request(
                "POST", "/products/fees/v0/feesEstimate", body=body,
            )
        except Exception as e:
            # Whole batch failed — fall back to the singleton loop for
            # this chunk so one broken ASIN doesn't lose all 20 estimates.
            print(f"[sp-api] batch feesEstimate failed ({e}); falling back to per-ASIN")
            for asin, pr in chunk:
                try:
                    out[asin] = await get_fees_estimate(
                        asin, pr, is_fba=is_fba,
                        marketplace=marketplace, currency=currency,
                    )
                except Exception as e2:
                    out[asin] = {
                        "referral": 0.0, "fba": 0.0, "fuel_surcharge": 0.0,
                        "total": 0.0, "status": "error", "error": str(e2)[:200],
                        "breakdown": [],
                    }
            _last_fees_batch_ts = time.time()
            continue

        _last_fees_batch_ts = time.time()
        # SP-API's batch feesEstimate returns the result list in one of
        # three shapes depending on marketplace / API era:
        #   1. bare JSON array of FeesEstimateResult (observed on NA)
        #   2. {"payload": [ ... ]} (older wrapping)
        #   3. {"payload": {"FeesEstimateResultList": [ ... ]}} (docs)
        # Normalize to a list before matching.
        if isinstance(resp, list):
            results = resp
        elif isinstance(resp, dict):
            payload = resp.get("payload")
            if isinstance(payload, list):
                results = payload
            elif isinstance(payload, dict):
                results = payload.get("FeesEstimateResultList") or []
            else:
                results = resp.get("FeesEstimateResultList") or []
        else:
            results = []
        if isinstance(results, dict):
            # Occasionally a single-item batch returns as an object; wrap.
            results = [results]
        # Match by echoed SellerInputIdentifier (`Identifier`) since order
        # is not guaranteed. Fall back to positional if the field is missing.
        by_ident: dict[str, dict] = {}
        for r in results:
            ident_obj = r.get("FeesEstimateIdentifier") or {}
            ident = ident_obj.get("SellerInputIdentifier")
            if ident:
                by_ident[ident] = r

        for idx, (asin, pr) in enumerate(chunk):
            ident = f"est-{asin}-{pr}"
            r = by_ident.get(ident)
            if r is None and idx < len(results):
                r = results[idx]
            if r is None:
                out[asin] = {
                    "referral": 0.0, "fba": 0.0, "fuel_surcharge": 0.0,
                    "total": 0.0, "status": "error",
                    "error": "no result returned in batch",
                    "breakdown": [],
                }
                continue
            parsed = _parse_fees_result(r)
            out[asin] = parsed
            if (parsed.get("status") or "").lower() == "success" or parsed["total"] > 0:
                _FEES_ESTIMATE_CACHE[
                    (asin, pr, bool(is_fba), marketplace_id, currency)
                ] = (time.time(), parsed)

    return out


# ── FBA Storage Fees report (per-SKU monthly storage) ───────────────────────


async def fetch_storage_fees_per_sku(months_back: int = 2) -> tuple[dict, list[str]]:
    """Pull GET_FBA_STORAGE_FEE_CHARGES_DATA for the last `months_back` months
    and return ({asin: avg_monthly_fee}, months_covered).

    The report is keyed by **ASIN** (also has fnsku + product_name; there is
    no seller_sku column). One ASIN can appear in multiple rows for the same
    month — one per fulfillment center / FNSKU pair — so we first sum
    estimated_monthly_storage_fee within each (asin, month), then average
    across months.

    The profitability calc joins by ASIN (which we already track on every
    order item). Caller is responsible for caching — the report takes
    30–120 s to generate."""
    now = datetime.now(timezone.utc)
    start = (now.replace(day=1) - timedelta(days=months_back * 31)).replace(day=1)
    end = now
    create_resp = await create_report(
        "GET_FBA_STORAGE_FEE_CHARGES_DATA",
        start_date=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_date=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        single_marketplace=True,
    )
    report_id = create_resp.get("reportId")
    if not report_id:
        raise RuntimeError(f"Storage report create returned no id: {create_resp}")
    text = await download_report_raw(report_id, max_polls=24, poll_interval=10)

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    qty_keys = (
        "average_quantity_on_hand",
        "average-quantity-on-hand",
        "Average quantity on hand",
    )
    # (asin, month) -> {fee, qty}
    by_asin_month: dict[tuple[str, str], dict] = {}
    months: set[str] = set()
    for row in reader:
        asin = (row.get("asin") or "").strip()
        if not asin:
            continue
        try:
            fee = float(row.get("estimated_monthly_storage_fee") or 0)
        except (TypeError, ValueError):
            fee = 0.0
        qty = 0.0
        for key in qty_keys:
            raw = row.get(key)
            if raw not in (None, ""):
                try:
                    qty = float(raw)
                except (TypeError, ValueError):
                    qty = 0.0
                break
        month = (row.get("month_of_charge") or "").strip()
        if not month:
            continue
        months.add(month)
        key = (asin, month)
        bucket = by_asin_month.setdefault(key, {"fee": 0.0, "qty": 0.0})
        bucket["fee"] += fee
        bucket["qty"] += qty

    # Average monthly fee + quantity per ASIN, then per-unit storage like Revenue Calculator.
    by_asin_monthly: dict[str, list[dict]] = defaultdict(list)
    for (asin, _month), bucket in by_asin_month.items():
        by_asin_monthly[asin].append(bucket)

    per_asin: dict[str, dict] = {}
    for asin, rows in by_asin_monthly.items():
        monthly_fees = [r["fee"] for r in rows if r["fee"] > 0]
        monthly_qtys = [r["qty"] for r in rows if r["qty"] > 0]
        avg_monthly_fee = sum(monthly_fees) / len(monthly_fees) if monthly_fees else 0.0
        avg_qty = sum(monthly_qtys) / len(monthly_qtys) if monthly_qtys else 0.0
        if avg_monthly_fee <= 0:
            continue
        divisor = max(avg_qty, 1.0)
        per_asin[asin] = {
            "monthly_fee": round(avg_monthly_fee, 4),
            "avg_quantity_on_hand": round(avg_qty, 2),
            "storage_per_unit": round(avg_monthly_fee / divisor, 4),
        }
    return per_asin, sorted(months)


# ── Finances API (v0) ────────────────────────────────────────────────────────


_FEE_TYPE_BUCKETS = [
    # (bucket key, list of substrings — matched case-insensitively against
    # Finances API FeeType strings, which have drifted over time)
    ("return_processing", ("returnfee", "refundcommission", "returnprocessingfee")),
    ("low_inventory", ("lowinventorylevelfee", "lowinventoryfee", "lowinventory")),
    # FBAInboundConvenienceFee is how the inbound placement service fee posts
    # in the Finances API — a shipment-level lump sum with NO SellerSKU.
    ("inbound_placement", ("inboundplacement", "inboundconvenience",
                           "inboundtransportationfee", "inboundplacementservice",
                           "fbainboundplacementservice", "placementservice",
                           "placementfee")),
    ("aged_inventory", ("agedinventorysurcharge", "longtermstoragefee",
                        "agedinventory", "inventoryagesurcharge",
                        "agedinventoryfee")),
]

_REMOVAL_ADJUSTMENT_HINTS = ("removal", "disposal")
_PLACEMENT_ADJUSTMENT_HINTS = (
    "inboundplacement", "inbound placement", "placementservice",
    "placement service", "placement fee",
)
_AGED_ADJUSTMENT_HINTS = (
    "agedinventory", "aged inventory", "longtermstorage", "long-term storage",
    "inventory age",
)


def _classify_fee_type(fee_type: str) -> str | None:
    ft = (fee_type or "").lower()
    for bucket, hints in _FEE_TYPE_BUCKETS:
        if any(h in ft for h in hints):
            return bucket
    return None


def _empty_fee_bucket() -> dict:
    return {
        "return_processing": 0.0,
        "low_inventory": 0.0,
        "inbound_placement": 0.0,
        "aged_inventory": 0.0,
        "removal": 0.0,
    }


def _fees_from_lists(*lists) -> list[tuple[str, float]]:
    """Flatten one or more ItemFeeList / ChargeList arrays into
    [(fee_type, amount)] tuples. Amazon uses `FeeType` in ItemFeeList and
    sometimes nests amount under `FeeAmount` / `ChargeAmount` / `Amount`."""
    out: list[tuple[str, float]] = []
    for lst in lists:
        if not lst:
            continue
        for it in lst:
            ftype = it.get("FeeType") or it.get("ChargeType") or ""
            for amount_key in ("FeeAmount", "ChargeAmount"):
                amt = it.get(amount_key)
                if amt is not None:
                    try:
                        out.append((ftype, float(amt.get("CurrencyAmount", 0) or 0)))
                    except (TypeError, ValueError, AttributeError):
                        pass
                    break
    return out


async def get_financial_events(
    posted_after: str,
    posted_before: str | None = None,
    paginate: bool = True,
    max_pages: int = 20,
) -> dict:
    """Pull ListFinancialEvents for the window and normalize into per-SKU
    fee buckets. Returns:

        {
          "by_sku":        {sku: {return_processing, low_inventory,
                                  inbound_placement, aged_inventory,
                                  removal}},
          "unattributed":  {…same keys… — fees we couldn't map to a SKU},
          "totals":        {…same keys, summed across all…},
          "pages":         int,
          "posted_after":  str,
        }

    Covers the 5 fees the FBA calculator PDF lists that aren't in Product
    Fees API: return processing, low inventory, inbound placement, aged
    inventory surcharge, and removal fees.

    Rate-limited: Finances API is 0.5 req/s sustained (2 burst). We sleep
    2 s between pages so a busy quota doesn't drop us. `max_pages` caps
    the walk so a very long window can't stall the request forever."""
    from collections import defaultdict

    base_params = {
        "PostedAfter": posted_after,
        "MaxResultsPerPage": "100",
    }
    if posted_before:
        base_params["PostedBefore"] = posted_before

    by_sku: dict[str, dict] = defaultdict(_empty_fee_bucket)
    unattributed = _empty_fee_bucket()
    pages = 0
    page_params = dict(base_params)

    while True:
        resp = await _sp_request(
            "GET", "/finances/v0/financialEvents", params=page_params,
        )
        pages += 1
        payload = resp.get("payload") or {}
        # SP-API nests all *EventList fields under `FinancialEvents`.
        # `NextToken` stays at the top level of `payload`.
        events = payload.get("FinancialEvents") or {}

        for evt in events.get("ShipmentEventList") or []:
            for item in evt.get("ShipmentItemList") or []:
                sku = (item.get("SellerSKU") or "").strip()
                for ftype, amt in _fees_from_lists(
                    item.get("ItemFeeList"), item.get("ItemChargeList"),
                ):
                    bucket = _classify_fee_type(ftype)
                    if not bucket:
                        continue
                    target = by_sku[sku] if sku else unattributed
                    # Amazon fees show up as negative (charges to seller);
                    # we want a positive cost figure.
                    target[bucket] += abs(amt)

        for evt in events.get("RefundEventList") or []:
            for item in (evt.get("ShipmentItemAdjustmentList")
                         or evt.get("ShipmentItemList") or []):
                sku = (item.get("SellerSKU") or "").strip()
                for ftype, amt in _fees_from_lists(
                    item.get("ItemFeeAdjustmentList") or item.get("ItemFeeList"),
                ):
                    bucket = _classify_fee_type(ftype)
                    if not bucket:
                        continue
                    target = by_sku[sku] if sku else unattributed
                    target[bucket] += abs(amt)

        for evt in events.get("ServiceFeeEventList") or []:
            sku = (evt.get("SellerSKU") or "").strip()
            fee_desc = (
                evt.get("FeeDescription") or evt.get("FeeReason") or ""
            ).lower()
            for ftype, amt in _fees_from_lists(evt.get("FeeList")):
                bucket = _classify_fee_type(ftype)
                if not bucket and any(h in fee_desc for h in _PLACEMENT_ADJUSTMENT_HINTS):
                    bucket = "inbound_placement"
                if not bucket and any(h in fee_desc for h in _AGED_ADJUSTMENT_HINTS):
                    bucket = "aged_inventory"
                if not bucket:
                    continue
                target = by_sku[sku] if sku else unattributed
                target[bucket] += abs(amt)

        for evt in events.get("AdjustmentEventList") or []:
            adj_type = (evt.get("AdjustmentType") or "").lower()
            if any(h in adj_type for h in _PLACEMENT_ADJUSTMENT_HINTS):
                # Prefer per-item amounts (may carry a SellerSKU); only fall
                # back to the event-level total when no item amounts exist —
                # counting both double-counts the same charge.
                item_total = 0.0
                for item in evt.get("AdjustmentItemList") or []:
                    sku = (item.get("SellerSKU") or "").strip()
                    amt_obj = item.get("PerUnitAmount") or item.get("TotalAmount") or {}
                    try:
                        item_amt = float(amt_obj.get("CurrencyAmount", 0) or 0)
                    except (TypeError, ValueError, AttributeError):
                        item_amt = 0.0
                    if item_amt:
                        target = by_sku[sku] if sku else unattributed
                        target["inbound_placement"] += abs(item_amt)
                        item_total += abs(item_amt)
                if item_total == 0.0:
                    adj_amt = evt.get("AdjustmentAmount") or {}
                    try:
                        amt = float(adj_amt.get("CurrencyAmount", 0) or 0)
                    except (TypeError, ValueError, AttributeError):
                        amt = 0.0
                    if amt:
                        unattributed["inbound_placement"] += abs(amt)
                continue
            if any(h in adj_type for h in _AGED_ADJUSTMENT_HINTS):
                item_total = 0.0
                for item in evt.get("AdjustmentItemList") or []:
                    sku = (item.get("SellerSKU") or "").strip()
                    amt_obj = item.get("PerUnitAmount") or item.get("TotalAmount") or {}
                    try:
                        item_amt = float(amt_obj.get("CurrencyAmount", 0) or 0)
                    except (TypeError, ValueError, AttributeError):
                        item_amt = 0.0
                    if item_amt:
                        target = by_sku[sku] if sku else unattributed
                        target["aged_inventory"] += abs(item_amt)
                        item_total += abs(item_amt)
                if item_total == 0.0:
                    adj_amt = evt.get("AdjustmentAmount") or {}
                    try:
                        amt = float(adj_amt.get("CurrencyAmount", 0) or 0)
                    except (TypeError, ValueError, AttributeError):
                        amt = 0.0
                    if amt:
                        unattributed["aged_inventory"] += abs(amt)
                continue
            if not any(h in adj_type for h in _REMOVAL_ADJUSTMENT_HINTS):
                continue
            for item in evt.get("AdjustmentItemList") or []:
                sku = (item.get("SellerSKU") or "").strip()
                amt_obj = item.get("PerUnitAmount") or item.get("TotalAmount") or {}
                try:
                    amt = float(amt_obj.get("CurrencyAmount", 0) or 0)
                except (TypeError, ValueError, AttributeError):
                    amt = 0.0
                target = by_sku[sku] if sku else unattributed
                target["removal"] += abs(amt)

        next_token = payload.get("NextToken")
        if not paginate or not next_token or pages >= max_pages:
            break
        # SP-API continuations: only NextToken (rest of the query is
        # remembered by Amazon).
        page_params = {"NextToken": next_token}
        await asyncio.sleep(2.0)

    totals = _empty_fee_bucket()
    for bucket in by_sku.values():
        for k in totals:
            totals[k] += bucket[k]
    for k in totals:
        totals[k] = round(totals[k] + unattributed[k], 2)

    return {
        "by_sku": {sku: {k: round(v, 2) for k, v in bucket.items()}
                   for sku, bucket in by_sku.items()},
        "unattributed": {k: round(v, 2) for k, v in unattributed.items()},
        "totals": totals,
        "pages": pages,
        "posted_after": posted_after,
    }


async def fetch_placement_service_fees_by_shipment(
    days_back: int = 365,
    max_pages: int = 40,
) -> dict[str, float]:
    """Scan Finances ServiceFeeEventList for inbound placement service fees
    (posted as FBAInboundConvenienceFee) grouped by FBA shipment id.

    Placement fees post ~45 days after receipt as shipment-level lump sums
    with NO SellerSKU — AmazonOrderId holds the FBA shipment id instead.
    The caller joins these against Aurora's `shipments` collection (which
    has per-SKU units received) to rebuild the per-SKU per-unit rates shown
    in Seller Central's placement fee report.

    Slow (Finances is 0.5 req/s); caller must cache the derived rates.
    """
    posted_after = (
        datetime.now(timezone.utc) - timedelta(days=days_back)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    params: dict = {"PostedAfter": posted_after, "MaxResultsPerPage": 100}
    by_shipment: dict[str, float] = defaultdict(float)
    pages = 0
    while pages < max_pages:
        resp = await _sp_request(
            "GET", "/finances/v0/financialEvents", params=params,
        )
        pages += 1
        payload = resp.get("payload") or {}
        events = payload.get("FinancialEvents") or {}
        for evt in events.get("ServiceFeeEventList") or []:
            if (evt.get("SellerSKU") or "").strip():
                continue  # SKU-attributed fees flow through get_financial_events
            shipment_id = (evt.get("AmazonOrderId") or "").strip()
            for ftype, amt in _fees_from_lists(evt.get("FeeList")):
                if _classify_fee_type(ftype) == "inbound_placement" and amt:
                    by_shipment[shipment_id or "_unknown"] += abs(amt)
        next_token = payload.get("NextToken")
        if not next_token:
            break
        params = {"NextToken": next_token}
        await asyncio.sleep(2.1)
    return {k: round(v, 2) for k, v in by_shipment.items()}


async def fetch_inbound_placement_fees_per_sku(
    months_back: int = 12,
) -> tuple[dict, list[str]]:
    """Pull GET_FBA_INBOUND_PLACEMENT_SERVICE_FEE_INVOICE_DATA for the
    last `months_back` months and return
    ({sku: {"fee_total": $, "units_received": N}}, months_covered).

    This is the same data the seller sees in their Amazon dashboard
    (SKU × units received × per-unit rate × total charge). We accumulate
    fees + units per SKU across all inbound shipments in the window so
    the caller can compute a weighted per-unit rate and amortize like
    COGS: units_sold_in_window × avg_fee_per_unit.

    We avoid the Finances API / shipment-items reconstruction path
    because it double-counts non-fee-qualifying SKUs in the same
    shipment's denominator, diluting the per-unit rate 15-20×.

    Caller is responsible for caching — the report takes 30-120 s to
    generate.
    """
    now = datetime.now(timezone.utc)
    start = (now.replace(day=1) - timedelta(days=months_back * 31)).replace(day=1)
    end = now
    # Report ID matches the Seller Central Report Central path
    # (/reportcentral/INBOUND_PLACEMENT_FEES_CHARGES/1).
    create_resp = await create_report(
        "GET_FBA_INBOUND_PLACEMENT_FEES_CHARGES_DATA",
        start_date=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_date=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        single_marketplace=True,
    )
    report_id = create_resp.get("reportId")
    if not report_id:
        raise RuntimeError(
            f"Placement fee report create returned no id: {create_resp}"
        )
    text = await download_report_raw(report_id, max_polls=24, poll_interval=10)

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    per_sku: dict[str, dict] = {}
    months: set[str] = set()
    # Amazon exports have drifted between snake_case and "friendly" column
    # names; accept common variants so we don't break if they rename.
    sku_keys = ("sku", "seller_sku", "SKU")
    asin_keys = ("asin", "ASIN", "Asin")
    qty_keys = ("actual_received_quantity", "received_quantity",
                "Actual received quantity")
    fee_keys = ("total_fba_inbound_placement_service_fee_charge",
                "total_charge", "total_charges",
                "Total FBA inbound placement service fee charge")
    date_keys = ("transaction_date", "Transaction date")
    for row in reader:
        def _get(keys):
            for k in keys:
                v = row.get(k)
                if v is not None and v != "":
                    return v
            return None
        sku = (_get(sku_keys) or "").strip()
        asin = (_get(asin_keys) or "").strip().upper()
        if not sku:
            continue
        try:
            units = int(float(_get(qty_keys) or 0))
        except (TypeError, ValueError):
            units = 0
        try:
            fee = float(_get(fee_keys) or 0)
        except (TypeError, ValueError):
            fee = 0.0
        if units <= 0 and fee == 0:
            continue
        bucket = per_sku.setdefault(
            sku,
            {
                "fee_total": 0.0,
                "units_received": 0,
                # Only units that incurred a placement charge — used for the
                # Revenue Calculator-style per-unit rate. Including $0-fee
                # Amazon-optimized inbound units diluted the rate ~4×+.
                "fee_bearing_units": 0,
                "asin": asin or None,
            },
        )
        bucket["fee_total"] += fee
        bucket["units_received"] += units
        if fee > 0 and units > 0:
            bucket["fee_bearing_units"] += units
        if asin and not bucket.get("asin"):
            bucket["asin"] = asin
        dt = (_get(date_keys) or "").strip()
        if dt:
            months.add(dt[:7])  # YYYY-MM prefix

    per_sku_rounded = {
        sku: {
            "fee_total": round(v["fee_total"], 2),
            "units_received": v["units_received"],
            "fee_bearing_units": v["fee_bearing_units"],
            "asin": v.get("asin"),
        }
        for sku, v in per_sku.items()
    }
    return per_sku_rounded, sorted(months)


async def fetch_aged_inventory_fees_per_sku() -> dict:
    """Pull GET_FBA_INVENTORY_PLANNING_DATA (snapshot) and sum Amazon's
    per-SKU aged-inventory-surcharge projections into a monthly per-SKU
    total.

    The report lists Amazon's own `estimated-ais-<bucket>-days` columns
    per SKU — the aged inventory surcharge Amazon will charge that SKU
    this month, already segmented by age bucket. We sum the buckets to
    get the SKU's projected monthly aged fee. The caller amortizes over
    the sales window (× months_in_window) the same way we do for
    storage.

    Works on Draft SP-API apps (unlike GET_FBA_INVENTORY_AGE_DATA and
    the LONGTERM_STORAGE_FEE_CHARGES report, which are Published-only).

    Returns {sku: {"monthly_fee": $, "total_aged_units": N}}. Caller is
    responsible for caching — the report takes 30-120 s to generate.
    """
    create_resp = await create_report(
        "GET_FBA_INVENTORY_PLANNING_DATA",
        single_marketplace=True,
    )
    report_id = create_resp.get("reportId")
    if not report_id:
        raise RuntimeError(
            f"Inventory planning report create returned no id: {create_resp}"
        )
    text = await download_report_raw(report_id, max_polls=30, poll_interval=10)

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    ais_fee_cols = (
        "estimated-ais-181-210-days",
        "estimated-ais-211-240-days",
        "estimated-ais-241-270-days",
        "estimated-ais-271-300-days",
        "estimated-ais-301-330-days",
        "estimated-ais-331-365-days",
        "estimated-ais-366-455-days",
        "estimated-ais-456-plus-days",
    )
    ais_qty_cols = (
        "quantity-to-be-charged-ais-181-210-days",
        "quantity-to-be-charged-ais-211-240-days",
        "quantity-to-be-charged-ais-241-270-days",
        "quantity-to-be-charged-ais-271-300-days",
        "quantity-to-be-charged-ais-301-330-days",
        "quantity-to-be-charged-ais-331-365-days",
        "quantity-to-be-charged-ais-366-455-days",
        "quantity-to-be-charged-ais-456-plus-days",
    )

    def _f(v):
        try:
            return float(v) if v not in (None, "") else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _i(v):
        try:
            return int(float(v)) if v not in (None, "") else 0
        except (TypeError, ValueError):
            return 0

    per_sku: dict[str, dict] = {}
    for row in reader:
        sku = (row.get("sku") or row.get("seller-sku") or "").strip()
        asin = (row.get("asin") or row.get("ASIN") or "").strip().upper()
        if not sku:
            continue
        monthly_fee = sum(_f(row.get(c)) for c in ais_fee_cols)
        aged_units = sum(_i(row.get(c)) for c in ais_qty_cols)
        # Amazon's own restock signals from the same report — used by the
        # Restock dashboard's HDOS and "recommended ship qty" columns.
        hdos_raw = row.get("historical-days-of-supply")
        rec_qty_raw = row.get("recommended-ship-in-quantity")
        rec_date_raw = (row.get("recommended-ship-in-date") or "").strip() or None
        has_supplement = any(
            v not in (None, "") for v in (hdos_raw, rec_qty_raw, rec_date_raw)
        )
        if monthly_fee == 0 and aged_units == 0 and not has_supplement:
            continue
        bucket = per_sku.setdefault(
            sku,
            {
                "monthly_fee": 0.0,
                "total_aged_units": 0,
                "asin": asin or None,
                "historical_days_of_supply": None,
                "recommended_ship_in_quantity": None,
                "recommended_ship_in_date": None,
            },
        )
        # A SKU can appear once per (fnsku, marketplace) — sum defensively
        # for fees/units. Take the max HDOS and the sum of recommended-ship
        # quantities so a multi-FC SKU still surfaces a sensible signal.
        bucket["monthly_fee"] += monthly_fee
        bucket["total_aged_units"] += aged_units
        if asin and not bucket.get("asin"):
            bucket["asin"] = asin
        if hdos_raw not in (None, ""):
            v = _f(hdos_raw)
            prev = bucket["historical_days_of_supply"]
            bucket["historical_days_of_supply"] = v if prev is None else max(prev, v)
        if rec_qty_raw not in (None, ""):
            v = _i(rec_qty_raw)
            prev = bucket["recommended_ship_in_quantity"] or 0
            bucket["recommended_ship_in_quantity"] = prev + v
        if rec_date_raw and not bucket["recommended_ship_in_date"]:
            bucket["recommended_ship_in_date"] = rec_date_raw

    return {
        sku: {
            "monthly_fee": round(v["monthly_fee"], 2),
            "total_aged_units": v["total_aged_units"],
            "asin": v.get("asin"),
            "historical_days_of_supply": (
                round(v["historical_days_of_supply"], 1)
                if v["historical_days_of_supply"] is not None else None
            ),
            "recommended_ship_in_quantity": v["recommended_ship_in_quantity"],
            "recommended_ship_in_date": v["recommended_ship_in_date"],
        }
        for sku, v in per_sku.items()
    }


async def fetch_aged_surcharge_charges_per_sku(
    start: datetime,
    end: datetime,
) -> dict[str, dict]:
    """Pull GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA — the same
    report Seller Central shows as "Aged Inventory Surcharge report".

    Unlike GET_FBA_INVENTORY_PLANNING_DATA (estimated-ais-* projections),
    this report has the actual `amount-charged` Amazon billed per SKU /
    age tier for the event month. Sum those into a per-SKU total so
    Profitability can match Seller Central dollar-for-dollar.

    Amazon requires dataStartTime→dataEndTime to equal **exactly one
    calendar month** per request (otherwise the report goes FATAL). We
    issue one request per overlapping month and merge.

    Returns {sku: {charged_total, qty_charged, asin}}.
    """
    from calendar import monthrange

    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    # Walk inclusive calendar months that overlap [start, end].
    months: list[tuple[datetime, datetime]] = []
    cursor = start_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = end_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor <= end_month:
        last_day = monthrange(cursor.year, cursor.month)[1]
        month_start = cursor
        month_end = cursor.replace(
            day=last_day, hour=23, minute=59, second=59, microsecond=0,
        )
        months.append((month_start, month_end))
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)

    def _f(row: dict, *keys: str) -> float:
        for k in keys:
            raw = row.get(k)
            if raw in (None, ""):
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _i(row: dict, *keys: str) -> int:
        return int(_f(row, *keys))

    def _accumulate(text: str, per_sku: dict[str, dict]) -> None:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        for row in reader:
            sku = (
                row.get("sku")
                or row.get("seller-sku")
                or row.get("merchant-sku")
                or ""
            ).strip()
            if not sku:
                continue
            charged = _f(row, "amount-charged", "amount_charged")
            if charged <= 0:
                charged = _f(row, "long-time-range-long-term-storage-fee") + _f(
                    row, "short-time-range-long-term-storage-fee",
                )
            qty = _i(
                row,
                "qty-charged",
                "qty_charged",
                "qty-charged-long-time-range-long-term-storage-fee",
                "qty-charged-short-time-range-long-term-storage-fee",
            )
            if charged <= 0 and qty <= 0:
                continue
            asin = (row.get("asin") or row.get("ASIN") or "").strip().upper() or None
            bucket = per_sku.setdefault(
                sku,
                {"charged_total": 0.0, "qty_charged": 0, "asin": asin},
            )
            bucket["charged_total"] += charged
            bucket["qty_charged"] += max(qty, 0)
            if asin and not bucket.get("asin"):
                bucket["asin"] = asin

    per_sku: dict[str, dict] = {}
    errors: list[str] = []
    for month_start, month_end in months:
        try:
            create_resp = await create_report(
                "GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA",
                start_date=month_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end_date=month_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                single_marketplace=True,
            )
            report_id = create_resp.get("reportId")
            if not report_id:
                errors.append(
                    f"{month_start.strftime('%Y-%m')}: create returned no id"
                )
                continue
            text = await download_report_raw(
                report_id, max_polls=30, poll_interval=10,
            )
            _accumulate(text, per_sku)
        except Exception as e:
            errors.append(f"{month_start.strftime('%Y-%m')}: {e}")

    if not per_sku and errors:
        raise RuntimeError(
            "Aged surcharge charges report failed for every month: "
            + "; ".join(errors)
        )
    return {
        sku: {
            "charged_total": round(v["charged_total"], 2),
            "qty_charged": int(v["qty_charged"]),
            "asin": v.get("asin"),
        }
        for sku, v in per_sku.items()
        if v["charged_total"] > 0
    }


# ── FBA Inventory API (v1) ───────────────────────────────────────────────────


async def get_inventory_summaries(
    skus: list[str] | None = None,
    details: bool = True,
    marketplace: str | None = None,
) -> dict:
    """Get FBA inventory levels. SP-API requires a single granularityId per
    call, so this scopes to one marketplace at a time (default: user's
    primary, US-preferred).

    Follows pagination — the first response carries ~50 SKUs and a
    `pagination.nextToken`; we keep fetching until the token is gone so
    every fulfillable SKU lands in the result. SP-API's contract for
    paginated continuations: send ONLY `nextToken` (no other query
    params), otherwise it returns InvalidInput.
    """
    user = require_user()
    marketplace_id = resolve_marketplace(user, marketplace, multiple=False)
    base_params: dict = {
        "details": str(details).lower(),
        "granularityType": "Marketplace",
        "granularityId": marketplace_id,
        "marketplaceIds": marketplace_id,
    }
    if skus:
        base_params["sellerSkus"] = ",".join(skus)

    # FBA inventory pagination requires the granularity params on EVERY
    # call, including continuations — only `details` and `sellerSkus`
    # get dropped after page 1.
    page_params = dict(base_params)
    next_token: str | None = None
    merged: dict = {}
    summaries: list = []
    while True:
        if next_token:
            page_params = {
                "granularityType": "Marketplace",
                "granularityId": marketplace_id,
                "marketplaceIds": marketplace_id,
                "nextToken": next_token,
            }
        resp = await _sp_request(
            "GET", "/fba/inventory/v1/summaries", params=page_params
        )
        if not merged:
            merged = resp
        payload = resp.get("payload") or {}
        summaries.extend(payload.get("inventorySummaries") or [])
        next_token = (resp.get("pagination") or {}).get("nextToken")
        if not next_token:
            break
        if skus:
            # Caller asked for a specific list — first page covers it.
            break

    if "payload" not in merged:
        merged["payload"] = {}
    merged["payload"]["inventorySummaries"] = summaries
    merged.pop("pagination", None)
    return merged


# ── Reports API (2021-06-30) ────────────────────────────────────────────────


async def create_report(
    report_type: str,
    start_date: str | None = None,
    end_date: str | None = None,
    marketplace: str | list[str] | None = None,
    report_options: dict | None = None,
    single_marketplace: bool = False,
) -> dict:
    """Request a new report. Returns reportId for polling. By default, covers
    all the user's marketplaces. Some report types (Sales & Traffic, FBA
    inventory planning) reject multi-marketplace requests — pass
    single_marketplace=True for those."""
    user = require_user()
    marketplace_ids = resolve_marketplace(
        user, marketplace, multiple=not single_marketplace
    )
    body: dict = {
        "reportType": report_type,
        "marketplaceIds": marketplace_ids if isinstance(marketplace_ids, list) else [marketplace_ids],
    }
    if start_date or end_date:
        body["dataStartTime"] = start_date
        body["dataEndTime"] = end_date
    if report_options:
        body["reportOptions"] = report_options
    return await _sp_request("POST", "/reports/2021-06-30/reports", body=body)


async def get_report(report_id: str) -> dict:
    """Check report processing status."""
    return await _sp_request("GET", f"/reports/2021-06-30/reports/{report_id}")


async def get_report_document(document_id: str) -> dict:
    """Get the download URL for a completed report document."""
    return await _sp_request("GET", f"/reports/2021-06-30/documents/{document_id}")


async def download_report_raw(report_id: str, max_polls: int = 30, poll_interval: int = 10) -> str:
    """Poll until a report is done, then return the FULL decoded text.

    Used by the ingest pipeline, which needs every row (not the
    LLM-friendly truncated summary that `download_report` returns).
    """
    for _ in range(max_polls):
        status = await get_report(report_id)
        processing_status = status.get("processingStatus", "")
        if processing_status == "DONE":
            doc_id = status.get("reportDocumentId")
            if not doc_id:
                raise RuntimeError(f"Report {report_id} done but no document id")
            doc_info = await get_report_document(doc_id)
            url = doc_info.get("url")
            if not url:
                raise RuntimeError(f"Report doc {doc_id} has no download url")
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            content = resp.content
            if doc_info.get("compressionAlgorithm") == "GZIP":
                content = gzip.decompress(content)
            return content.decode("utf-8", errors="replace")
        if processing_status in ("CANCELLED", "FATAL"):
            raise RuntimeError(f"Report {report_id} failed: {processing_status}")
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Report {report_id} did not finish within {max_polls * poll_interval}s")


async def download_report(report_id: str, max_polls: int = 12, poll_interval: int = 10) -> str:
    """Poll until a report is done, then download and return its content as text.

    Returns a compact summary suitable for the LLM context window.
    """
    for _ in range(max_polls):
        status = await get_report(report_id)
        processing_status = status.get("processingStatus", "")
        print(f"[sp-api] report {report_id} status: {processing_status}")

        if processing_status == "DONE":
            doc_id = status.get("reportDocumentId")
            if not doc_id:
                return "Report completed but no document ID returned."
            doc_info = await get_report_document(doc_id)
            download_url = doc_info.get("url")
            if not download_url:
                return "Report document has no download URL."

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(download_url)
                resp.raise_for_status()

            # Handle gzip-compressed reports
            content_bytes = resp.content
            compression = doc_info.get("compressionAlgorithm", "")
            if compression == "GZIP":
                content_bytes = gzip.decompress(content_bytes)

            text = content_bytes.decode("utf-8")

            # Parse tab-delimited report into readable format
            try:
                reader = csv.DictReader(io.StringIO(text), delimiter="\t")
                rows = list(reader)
                if rows:
                    return json.dumps(rows[:100], indent=2)  # Cap at 100 rows
            except Exception:
                pass

            # Return raw text (capped)
            return text[:5000]

        if processing_status in ("CANCELLED", "FATAL"):
            return f"Report failed with status: {processing_status}"

        await asyncio.sleep(poll_interval)

    return "Report timed out — still processing. Try again later."


# ── Brand Analytics ─────────────────────────────────────────────────────────

async def fetch_brand_analytics_search_terms(
    start_date: str,
    end_date: str,
    period: str = "WEEK",
    marketplace: str | None = None,
) -> list[dict]:
    """
    Requests, polls, and downloads the Amazon Brand Analytics Search Terms report.
    """
    create_resp = await create_report(
        report_type="GET_BRAND_ANALYTICS_SEARCH_TERMS_REPORT",
        start_date=start_date,
        end_date=end_date,
        marketplace=marketplace,
        report_options={"reportPeriod": period},
        single_marketplace=True,
    )

    report_id = create_resp.get("reportId")
    if not report_id:
        raise RuntimeError(f"Brand Analytics report creation failed: {create_resp}")

    raw_text = await download_report_raw(report_id, max_polls=30, poll_interval=10)

    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
        return data.get("dataByDepartmentAndSearchTerm", [])
    except json.JSONDecodeError:
        delimiter = "\t" if "\t" in raw_text else ","
        reader = csv.DictReader(io.StringIO(raw_text), delimiter=delimiter)
        return list(reader)

def check_keyword_match_types(target_keywords: list[str], brand_analytics_data: list[dict]) -> dict:
    """
    Filters the Brand Analytics report data into Exact, Phrase, and Broad matches.
    """
    results = {}

    for keyword in target_keywords:
        kw_lower = keyword.lower().strip()
        kw_words = set(kw_lower.split())

        exact = []
        phrase = []
        broad = []

        for row in brand_analytics_data:
            search_term = row.get("searchTerm", row.get("search_term", "")).lower().strip()
            if not search_term:
                continue

            if search_term == kw_lower:
                exact.append(row)
            elif kw_lower in search_term:
                phrase.append(row)
            else:
                st_words = set(search_term.split())
                if kw_words.issubset(st_words):
                    broad.append(row)

        results[keyword] = {
            "exact_match": exact,
            "phrase_match": phrase,
            "broad_match": broad
        }

    return results

async def process_brand_analytics_keywords(
    keywords: list[str],
    start_date: str,
    end_date: str,
    period: str = "WEEK",
    marketplace: str | None = None,
) -> dict:
    """
    Orchestrator function to fetch report and run match checks.
    """
    report_data = await fetch_brand_analytics_search_terms(start_date, end_date, period, marketplace)
    match_data = check_keyword_match_types(keywords, report_data)

    return match_data