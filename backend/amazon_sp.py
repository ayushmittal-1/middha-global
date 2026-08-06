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
import re
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

# DONE report bodies never change — cache by reportId so profitability does
# not re-hit getReportDocument (tight quota) on every Apply / every candidate.
_REPORT_TEXT_CACHE: dict[str, str] = {}
_REPORT_TEXT_CACHE_MAX = 64
# Space document GETs so we don't burn the Reports document rate limit.
_DOC_GET_LOCK: asyncio.Lock | None = None
_DOC_GET_LAST_TS = 0.0
_DOC_GET_MIN_GAP_S = 1.25


def _doc_get_lock() -> asyncio.Lock:
    global _DOC_GET_LOCK
    if _DOC_GET_LOCK is None:
        _DOC_GET_LOCK = asyncio.Lock()
    return _DOC_GET_LOCK

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
    """Split a bundled FBA fulfillment total into base + fuel (Revenue Calculator).

    Amazon displays these at cent precision (e.g. $3.86 → base $3.73 + fuel $0.13).
    Rounding to 4dp first (3.7295 / 0.1305) then × units drifts by a few cents
    vs the calculator even though FBA+Fuel still matches.
    """
    if total <= 0:
        return 0.0, 0.0
    total = round(float(total), 2)
    base = round(total / 1.035, 2)
    fuel = round(total - base, 2)
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
        fuel = round(round(fba, 2) * 0.035, 2)
        # Keep base+fuel consistent with Amazon cent display when we invented fuel.
        fba = round(float(fba), 2)

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
    items: list[tuple],
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

    `items` is a list of (asin, price) or (asin, price, is_fba) tuples.
    Per-item `is_fba` overrides the keyword default (needed so FBM SKUs
    are not estimated as Amazon-fulfilled). Duplicate ASINs are
    de-duplicated by (asin, rounded price, is_fba)."""
    global _last_fees_batch_ts
    if not items:
        return {}
    user = require_user()
    marketplace_id = resolve_marketplace(user, marketplace, multiple=False)
    now_ts = time.time()
    out: dict[str, dict] = {}
    # De-dupe by (asin, rounded price, is_fba).
    seen: set[tuple[str, float, bool]] = set()
    to_fetch: list[tuple[str, float, bool]] = []
    for entry in items:
        if not entry or len(entry) < 2:
            continue
        asin = entry[0]
        price = entry[1]
        item_is_fba = bool(entry[2]) if len(entry) >= 3 else bool(is_fba)
        if not asin or price is None or price <= 0:
            continue
        pr = round(float(price), 2)
        key = (asin, pr, item_is_fba)
        if key in seen:
            continue
        seen.add(key)
        cache_key = (asin, pr, item_is_fba, marketplace_id, currency)
        cached = _FEES_ESTIMATE_CACHE.get(cache_key)
        if cached and now_ts - cached[0] < _FEES_ESTIMATE_TTL_S:
            out[asin] = cached[1]
        else:
            to_fetch.append((asin, pr, item_is_fba))

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
        # even if Amazon changes the response order. Include is_fba in the
        # identifier so FBA/FBM estimates for the same ASIN don't collide.
        body = [
            {
                "FeesEstimateRequest": {
                    "MarketplaceId": marketplace_id,
                    "IsAmazonFulfilled": bool(item_fba),
                    "PriceToEstimateFees": {
                        "ListingPrice": {"Amount": pr, "CurrencyCode": currency},
                    },
                    "Identifier": f"est-{asin}-{pr}-{'fba' if item_fba else 'fbm'}",
                },
                "IdType": "ASIN",
                "IdValue": asin,
            }
            for asin, pr, item_fba in chunk
        ]
        try:
            resp = await _sp_request(
                "POST", "/products/fees/v0/feesEstimate", body=body,
            )
        except Exception as e:
            # Whole batch failed — fall back to the singleton loop for
            # this chunk so one broken ASIN doesn't lose all 20 estimates.
            print(f"[sp-api] batch feesEstimate failed ({e}); falling back to per-ASIN")
            for asin, pr, item_fba in chunk:
                try:
                    out[asin] = await get_fees_estimate(
                        asin, pr, is_fba=item_fba,
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

        for idx, (asin, pr, item_fba) in enumerate(chunk):
            ident = f"est-{asin}-{pr}-{'fba' if item_fba else 'fbm'}"
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
                    (asin, pr, bool(item_fba), marketplace_id, currency)
                ] = (time.time(), parsed)

    return out


# ── FBA Storage Fees report (per-ASIN monthly storage) ───────────────────────


def _storage_row_get(row: dict, *keys: str) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key]).strip()
        dashed = key.replace("_", "-")
        if dashed in row and row[dashed] not in (None, ""):
            return str(row[dashed]).strip()
    return ""


def _storage_row_float(row: dict, *keys: str) -> float:
    raw = _storage_row_get(row, *keys)
    if not raw or raw in ("--", "N/A", "n/a", "-"):
        return 0.0
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _coerce_float(value, default: float = 0.0) -> float:
    """Safe float for Mongo/cache values that may arrive as str/Decimal/None."""
    if value is None or value == "" or value in ("--", "N/A", "n/a", "-"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return default


def _normalize_month_of_charge(raw: str) -> str:
    """Normalize month_of_charge to YYYY-MM (handles 2026-4, 2026/04, etc.)."""
    s = (raw or "").strip().strip("'\"")
    if not s:
        return ""
    m = re.match(r"^(\d{4})[-/](\d{1,2})$", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    return s if re.match(r"^\d{4}-\d{2}$", s) else ""


def normalize_storage_fee_map(
    per_asin: dict,
) -> dict[str, dict[str, dict]]:
    """Ensure {ASIN: {YYYY-MM: {monthly_fee: float, ...}}} with numeric fields."""
    out: dict[str, dict[str, dict]] = {}
    if not isinstance(per_asin, dict):
        return out
    for asin, by_month in per_asin.items():
        if not isinstance(by_month, dict):
            continue
        asin_key = str(asin or "").strip().upper()
        if not asin_key:
            continue
        kept: dict[str, dict] = {}
        for month, bucket in by_month.items():
            month_key = _normalize_month_of_charge(str(month))
            if not month_key or not isinstance(bucket, dict):
                continue
            fee = _coerce_float(bucket.get("monthly_fee"))
            qty = _coerce_float(bucket.get("avg_quantity_on_hand"))
            if fee <= 0:
                continue
            kept[month_key] = {
                "monthly_fee": round(fee, 4),
                "avg_quantity_on_hand": round(qty, 2),
                "storage_per_unit": round(
                    fee / max(qty, 1.0), 6,
                ),
            }
        if kept:
            out[asin_key] = kept
    return out


def parse_storage_fee_report(text: str) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Parse GET_FBA_STORAGE_FEE_CHARGES_DATA (TSV or CSV).

    Amazon emits one row per ASIN × FNSKU × fulfillment center × month.
    We roll up to ASIN × month (sum fees and avg qty on hand), matching
    Seller Central's Monthly Storage Fees report totals.
    """
    if not (text or "").strip():
        return {}, []

    # Strip BOM; detect delimiter from header (SP-API is usually TSV).
    cleaned = text.lstrip("\ufeff")
    first = cleaned.splitlines()[0]
    delim = "\t" if "\t" in first else ","
    reader = csv.DictReader(io.StringIO(cleaned), delimiter=delim)

    by_asin_month: dict[tuple[str, str], dict] = {}
    months: set[str] = set()
    for row in reader:
        # Normalize keys: SP-API / Seller Central sometimes vary separators.
        norm_row = {
            (k or "").strip().lstrip("\ufeff").lower().replace("-", "_").replace(" ", "_"): v
            for k, v in row.items()
        }
        asin = _storage_row_get(norm_row, "asin").upper()
        if not asin:
            continue
        fee = _storage_row_float(norm_row, "estimated_monthly_storage_fee")
        qty = _storage_row_float(
            norm_row,
            "average_quantity_on_hand",
            "average_quantity_on_hand",
        )
        month = _normalize_month_of_charge(
            _storage_row_get(norm_row, "month_of_charge"),
        )
        if not month:
            continue
        months.add(month)
        key = (asin, month)
        bucket = by_asin_month.setdefault(
            key, {"monthly_fee": 0.0, "avg_quantity_on_hand": 0.0},
        )
        bucket["monthly_fee"] = _coerce_float(bucket["monthly_fee"]) + fee
        bucket["avg_quantity_on_hand"] = (
            _coerce_float(bucket["avg_quantity_on_hand"]) + qty
        )

    per_asin: dict[str, dict[str, dict]] = defaultdict(dict)
    for (asin, month), bucket in by_asin_month.items():
        fee = _coerce_float(bucket["monthly_fee"])
        qty = _coerce_float(bucket["avg_quantity_on_hand"])
        if fee <= 0:
            continue
        divisor = max(qty, 1.0)
        per_asin[asin][month] = {
            "monthly_fee": round(fee, 4),
            "avg_quantity_on_hand": round(qty, 2),
            "storage_per_unit": round(fee / divisor, 6),
        }
    return dict(per_asin), sorted(months)


def calendar_months_in_window(
    start_dt: datetime,
    end_dt: datetime,
    time_zone: str = "UTC",
) -> list[str]:
    """Calendar YYYY-MM keys overlapping [start_dt, end_dt] in marketplace TZ."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(time_zone or "UTC")
    local_start = start_dt.astimezone(tz).date()
    local_end = end_dt.astimezone(tz).date()
    months: list[str] = []
    year, month = local_start.year, local_start.month
    while (year, month) < (local_end.year, local_end.month) or (
        year == local_end.year and month == local_end.month
    ):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def storage_per_unit_for_window(
    asin_by_month: dict[str, dict],
    months_in_window: list[str],
) -> float:
    """Blend monthly storage rates for the months in the profitability window.

    Seller Central's Revenue Calculator uses each month's
    (estimated_monthly_storage_fee ÷ average_quantity_on_hand). When the
    window spans multiple months we sum fees and qty across those months
    (same ASIN rollup SC uses in the monthly report).

    Prefer ``storage_fee_for_asin_months`` when matching the Monthly Storage
    Fees report dollar total (fee is inventory-based, not units-sold × rate).
    """
    total_fee = 0.0
    total_qty = 0.0
    for month in months_in_window:
        bucket = asin_by_month.get(month) or {}
        if not isinstance(bucket, dict):
            continue
        total_fee += _coerce_float(bucket.get("monthly_fee"))
        total_qty += _coerce_float(bucket.get("avg_quantity_on_hand"))
    if total_fee <= 0:
        return 0.0
    return round(total_fee / max(total_qty, 1.0), 6)


def month_overlap_fraction(
    month_key: str,
    window_start: datetime,
    window_end: datetime,
    time_zone: str = "UTC",
) -> float:
    """Fraction of calendar month overlapping [window_start, window_end] (UTC)."""
    m_start, m_end_excl = month_start_end_excl(month_key, time_zone)
    # Treat end as inclusive instant: extend by 1µs so end-of-month matches full month.
    w_start = window_start
    w_end_excl = window_end + timedelta(microseconds=1)
    overlap_start = max(m_start, w_start)
    overlap_end = min(m_end_excl, w_end_excl)
    if overlap_end <= overlap_start:
        return 0.0
    month_seconds = (m_end_excl - m_start).total_seconds()
    if month_seconds <= 0:
        return 0.0
    frac = (overlap_end - overlap_start).total_seconds() / month_seconds
    return max(0.0, min(1.0, frac))


def storage_fee_for_asin_months(
    asin_by_month: dict[str, dict],
    months_in_window: list[str],
    month_fractions: dict[str, float] | None = None,
) -> float:
    """Actual Monthly Storage Fees $ for one ASIN (optionally day-prorated)."""
    total = 0.0
    for month in months_in_window:
        bucket = asin_by_month.get(month) or {}
        if not isinstance(bucket, dict):
            continue
        fee = _coerce_float(bucket.get("monthly_fee"))
        frac = 1.0 if not month_fractions else float(month_fractions.get(month, 0.0))
        total += fee * frac
    return round(total, 4)


def storage_fees_by_asin_for_window(
    storage_by_asin_month: dict[str, dict[str, dict]],
    months_in_window: list[str],
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    time_zone: str = "UTC",
) -> tuple[dict[str, float], float, dict[str, float]]:
    """ASIN → prorated monthly storage fee matching Seller Central report.

    Returns (fees_by_asin, report_total, month_fractions).
    Full-month filters (e.g. June 1–30) yield fraction 1.0 so totals match
    the downloaded Monthly Storage Fees CSV exactly.
    """
    fractions: dict[str, float] = {}
    for month in months_in_window:
        if window_start is not None and window_end is not None:
            fractions[month] = month_overlap_fraction(
                month, window_start, window_end, time_zone,
            )
        else:
            fractions[month] = 1.0

    fees_by_asin: dict[str, float] = {}
    for asin, by_month in (storage_by_asin_month or {}).items():
        fee = storage_fee_for_asin_months(by_month, months_in_window, fractions)
        if fee > 0:
            fees_by_asin[str(asin).upper()] = fee
    report_total = round(sum(fees_by_asin.values()), 2)
    return fees_by_asin, report_total, fractions


def is_per_asin_by_month_storage_cache(cached: dict) -> bool:
    """True when cache holds {asin: {YYYY-MM: {...}}} not legacy averages."""
    if not cached:
        return False
    for value in cached.values():
        if not isinstance(value, dict):
            return False
        if any(re.match(r"^\d{4}-\d{2}$", key) for key in value.keys()):
            return True
        if "monthly_fee" in value and "storage_per_unit" in value:
            return False
    return False


def merge_storage_by_asin_month(
    base: dict[str, dict[str, dict]],
    extra: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """Merge ASIN×month storage maps (new months overwrite same month keys)."""
    merged = {asin: dict(months) for asin, months in (base or {}).items()}
    for asin, months in (extra or {}).items():
        bucket = merged.setdefault(asin, {})
        bucket.update(months)
    return normalize_storage_fee_map(merged)


def filter_storage_to_months(
    per_asin: dict[str, dict[str, dict]],
    months: list[str],
) -> dict[str, dict[str, dict]]:
    allowed = set(months)
    out: dict[str, dict[str, dict]] = {}
    for asin, by_month in per_asin.items():
        kept = {m: b for m, b in by_month.items() if m in allowed}
        if kept:
            out[asin] = kept
    return out


def month_start_end_excl(month_key: str, time_zone: str) -> tuple[datetime, datetime]:
    """UTC [start, end) for a YYYY-MM month_of_charge in marketplace TZ."""
    from marketplace_timezone import zoned_time_to_utc

    year, month = (int(p) for p in month_key.split("-"))
    start = zoned_time_to_utc(
        {"year": year, "month": month, "day": 1, "hour": 0, "minute": 0, "second": 0, "microsecond": 0},
        time_zone,
    )
    if month == 12:
        end_excl = zoned_time_to_utc(
            {"year": year + 1, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0, "microsecond": 0},
            time_zone,
        )
    else:
        end_excl = zoned_time_to_utc(
            {"year": year, "month": month + 1, "day": 1, "hour": 0, "minute": 0, "second": 0, "microsecond": 0},
            time_zone,
        )
    return start, end_excl


def storage_report_range_for_months(
    months: list[str],
    time_zone: str,
) -> tuple[datetime, datetime]:
    """SP-API dataStartTime/dataEndTime covering all requested month_of_charge keys."""
    if not months:
        now = datetime.now(timezone.utc)
        return now, now
    sorted_months = sorted(months)
    start, _ = month_start_end_excl(sorted_months[0], time_zone)
    _, end_excl = month_start_end_excl(sorted_months[-1], time_zone)
    now = datetime.now(timezone.utc)
    # Amazon rejects future end times; historical months end before now.
    end = min(end_excl - timedelta(microseconds=1), now)
    if end <= start:
        end = now
    return start, end


async def _list_storage_fee_reports(created_since: datetime) -> list[dict]:
    """Recent GET_FBA_STORAGE_FEE_CHARGES_DATA reports (Seller Central downloads too)."""
    resp = await _sp_request(
        "GET",
        "/reports/2021-06-30/reports",
        params={
            "reportTypes": "GET_FBA_STORAGE_FEE_CHARGES_DATA",
            "pageSize": "100",
            "createdSince": created_since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    return list(resp.get("reports") or [])


async def fetch_storage_fees_for_months(
    months: list[str],
    time_zone: str = "UTC",
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Pull storage for specific calendar months via SP-API (no manual CSV).

    Reuses a recent DONE report when Seller Central / Aurora already requested
    the same month. Otherwise creates GET_FBA_STORAGE_FEE_CHARGES_DATA for the
    range — works for January (or any past month) as long as Amazon still
    exposes that month_of_charge in the report API.
    """
    months = sorted({m for m in months if re.match(r"^\d{4}-\d{2}$", m)})
    if not months:
        return {}, []

    report_start, report_end = storage_report_range_for_months(months, time_zone)
    now = datetime.now(timezone.utc)
    # List reports back to the earliest requested month (not just 89d) so a
    # January view can reuse a DONE report created when that month closed.
    lookback = report_start - timedelta(days=1)
    max_lookback = now - timedelta(days=730)
    if lookback < max_lookback:
        lookback = max_lookback

    text: str | None = None
    try:
        existing = await _list_storage_fee_reports(lookback)
        candidates = [
            r for r in existing
            if r.get("processingStatus") == "DONE"
            and r.get("reportDocumentId")
            and _report_covers_month(r, report_start, report_end + timedelta(seconds=1))
        ]
        candidates.sort(key=lambda r: r.get("createdTime") or "", reverse=True)
        # Fallback: any recent DONE storage report — Amazon often stamps a
        # wider data window; we filter month_of_charge after download.
        if not candidates:
            candidates = [
                r for r in existing
                if r.get("processingStatus") == "DONE" and r.get("reportDocumentId")
            ]
            candidates.sort(key=lambda r: r.get("createdTime") or "", reverse=True)
            candidates = candidates[:3]
        if candidates:
            for cand in candidates:
                try:
                    text = await download_report_raw(
                        cand["reportId"], max_polls=6, poll_interval=5,
                    )
                    if text:
                        break
                except Exception:
                    text = None
                    continue
        if text is None:
            # Prefer waiting on an in-flight storage report over creating
            # another (Amazon often CANCELLED's duplicates).
            in_flight = [
                r for r in existing
                if r.get("processingStatus") in ("IN_QUEUE", "IN_PROGRESS")
                and _report_covers_month(
                    r, report_start, report_end + timedelta(seconds=1),
                )
            ]
            in_flight.sort(key=lambda r: r.get("createdTime") or "", reverse=True)
            for r in in_flight[:2]:
                try:
                    text = await download_report_raw(
                        r["reportId"], max_polls=30, poll_interval=10,
                    )
                    if text is not None:
                        break
                except Exception:
                    continue
    except Exception:
        text = None

    if text is None:
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                create_resp = await create_report(
                    "GET_FBA_STORAGE_FEE_CHARGES_DATA",
                    start_date=report_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end_date=report_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    single_marketplace=True,
                )
                report_id = create_resp.get("reportId")
                if not report_id:
                    raise RuntimeError(
                        f"Storage report create returned no id: {create_resp}"
                    )
                text = await download_report_raw(
                    report_id, max_polls=24, poll_interval=10,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                # Amazon occasionally CANCELLED/FATAL's a create when another
                # storage report is in flight — wait, reuse any new DONE, retry.
                if "CANCELLED" not in msg and "FATAL" not in msg:
                    raise
                await asyncio.sleep(8)
                try:
                    existing = await _list_storage_fee_reports(lookback)
                    candidates = [
                        r for r in existing
                        if r.get("processingStatus") == "DONE"
                        and r.get("reportDocumentId")
                        and _report_covers_month(
                            r, report_start, report_end + timedelta(seconds=1),
                        )
                    ]
                    candidates.sort(
                        key=lambda r: r.get("createdTime") or "", reverse=True,
                    )
                    if candidates:
                        text = await download_report_raw(
                            candidates[0]["reportId"], max_polls=6, poll_interval=5,
                        )
                        if text is not None:
                            last_err = None
                            break
                    in_flight = [
                        r for r in existing
                        if r.get("processingStatus") in ("IN_QUEUE", "IN_PROGRESS")
                        and _report_covers_month(
                            r, report_start, report_end + timedelta(seconds=1),
                        )
                    ]
                    for r in in_flight[:1]:
                        try:
                            text = await download_report_raw(
                                r["reportId"], max_polls=24, poll_interval=10,
                            )
                            if text is not None:
                                last_err = None
                                break
                        except Exception:
                            continue
                    if text is not None:
                        break
                except Exception:
                    pass
        if text is None and last_err is not None:
            raise last_err
        if text is None:
            raise RuntimeError("Storage report unavailable after retries")

    per_asin, parsed_months = parse_storage_fee_report(text)
    filtered = filter_storage_to_months(per_asin, months)
    # Return only months we actually got rows for (may be subset if Amazon omitted data).
    found_months = sorted(
        {m for by_month in filtered.values() for m in by_month.keys()}
    )
    if not found_months and parsed_months:
        # Amazon sometimes returns a neighboring month_of_charge when the
        # requested month isn't published yet — keep whatever we parsed so
        # cache can still grow; caller filters to the profitability window.
        return normalize_storage_fee_map(per_asin), sorted(parsed_months)
    return normalize_storage_fee_map(filtered), found_months or [
        m for m in months if m in parsed_months
    ]


async def fetch_storage_fees_by_asin_month(
    window_start: datetime,
    window_end: datetime,
    time_zone: str = "UTC",
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Pull storage report for calendar months overlapping the profitability window."""
    months = calendar_months_in_window(window_start, window_end, time_zone)
    return await fetch_storage_fees_for_months(months, time_zone)


async def fetch_storage_fees_per_sku(months_back: int = 2) -> tuple[dict, list[str]]:
    """Legacy wrapper — prefer fetch_storage_fees_by_asin_month."""
    now = datetime.now(timezone.utc)
    start = (now.replace(day=1) - timedelta(days=months_back * 31)).replace(day=1)
    per_asin, months = await fetch_storage_fees_by_asin_month(start, now)
    # Flatten to legacy averaged shape for any old callers.
    flat: dict[str, dict] = {}
    for asin, by_month in per_asin.items():
        fees = [b["monthly_fee"] for b in by_month.values() if b.get("monthly_fee")]
        qtys = [b["avg_quantity_on_hand"] for b in by_month.values() if b.get("avg_quantity_on_hand")]
        if not fees:
            continue
        avg_fee = sum(fees) / len(fees)
        avg_qty = sum(qtys) / len(qtys) if qtys else 1.0
        flat[asin] = {
            "monthly_fee": round(avg_fee, 4),
            "avg_quantity_on_hand": round(avg_qty, 2),
            "storage_per_unit": round(avg_fee / max(avg_qty, 1.0), 4),
        }
    return flat, months


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
    # Payments → Transactions "FBA Removal Order: Return/Disposal Fee"
    # posts as Service Fees (not AdjustmentEventList). Match FeeType and
    # FeeDescription text from Seller Central.
    ("removal", (
        "removalorder", "removalfee", "disposalfee", "fbaremover",
        "fbaremoverorder", "disposalorder",
    )),
]

_REMOVAL_ADJUSTMENT_HINTS = ("removal", "disposal")
_REMOVAL_SERVICE_HINTS = (
    "removal order", "removal fee", "disposal fee", "fba removal",
    "disposal order",
)
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
        # Count of units Amazon posted a refund for in the window (from
        # RefundEventList[].ShipmentItemAdjustmentList[].QuantityShipped).
        # Kept for reporting / diagnostics.
        "returned_units": 0,
        # Sum of the referral commission Amazon posted on refund events for
        # this SKU (Commission entries in RefundEventList[].ShipmentItem
        # AdjustmentList[].ItemFeeAdjustmentList). This is the actual
        # referral fee that was reversed on the returned units — client's
        # rule computes return_processing_fee = 20% × this value.
        # Independent of the current window's average price, which is why
        # the earlier `0.20 × window_referral_per_unit × returned_units`
        # formulation was wrong for refunds of pre-window sales.
        "refunded_referral": 0.0,
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
    max_pages: int = 40,
    refund_posted_after: str | None = None,
    removal_posted_after: str | None = None,
    placement_posted_after: str | None = None,
    placement_posted_before: str | None = None,
) -> dict:
    """Pull ListFinancialEvents for the window and normalize into per-SKU
    fee buckets. Returns:

        {
          "by_sku":        {sku: {return_processing, low_inventory,
                                  inbound_placement, aged_inventory,
                                  removal}},
          "unattributed":  {…same keys… — fees we couldn't map to a SKU},
          "totals":        {…same keys, summed across all…},
          "placement_window_total": float,  # inbound_placement in PostedDate window
          "pages":         int,
          "posted_after":  str,
        }

    Covers the 5 fees the FBA calculator PDF lists that aren't in Product
    Fees API: return processing, low inventory, inbound placement, aged
    inventory surcharge, and removal fees.

    When ``placement_posted_after`` / ``placement_posted_before`` are set,
    inbound_placement ServiceFee/Adjustment rows are kept only if PostedDate
    falls in that half-open window (profitability filter) — same idea as
    removal's PostedDate bound.

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
    removal_by_order: dict[str, float] = defaultdict(float)
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
            # Bound refund aggregation to the caller's true window. The
            # outer 45-day pre-window is meant for late-posting fees like
            # placement/aged (which post ~45d after shipment receipt),
            # NOT refunds — a return posted in mid-May shouldn't inflate
            # the June profitability row's return_processing_fee.
            if refund_posted_after:
                posted = evt.get("PostedDate") or ""
                if posted and posted < refund_posted_after:
                    continue
            for item in (evt.get("ShipmentItemAdjustmentList")
                         or evt.get("ShipmentItemList") or []):
                sku = (item.get("SellerSKU") or "").strip()
                # Count returned units per SKU so /profitability can apply
                # the 20% × referral_per_unit × returned_units rule.
                # ShipmentItemAdjustmentList entries can carry negative
                # QuantityShipped (Amazon reports refunds as negative);
                # abs() and int() so we always get a positive count.
                qty_raw = (
                    item.get("QuantityShipped")
                    or item.get("QuantityAdjusted")
                    or 0
                )
                try:
                    returned_qty = abs(int(qty_raw))
                except (TypeError, ValueError):
                    returned_qty = 0
                if sku and returned_qty > 0:
                    by_sku[sku]["returned_units"] += returned_qty
                elif returned_qty > 0:
                    unattributed["returned_units"] += returned_qty

                for ftype, amt in _fees_from_lists(
                    item.get("ItemFeeAdjustmentList") or item.get("ItemFeeList"),
                ):
                    # `Commission` (exact) inside a refund event is Amazon
                    # reversing the ORIGINAL referral fee back to the seller.
                    # That's the referral-per-returned-unit input to the
                    # client's 20% return-proc rule — capture it separately
                    # from the bucketed fee categories. Distinguish from
                    # `RefundCommission` (the return-proc fee itself).
                    if (ftype or "").strip().lower() == "commission":
                        target = by_sku[sku] if sku else unattributed
                        target["refunded_referral"] += abs(amt)
                        continue
                    bucket = _classify_fee_type(ftype)
                    if not bucket:
                        continue
                    target = by_sku[sku] if sku else unattributed
                    target[bucket] += abs(amt)

        for evt in events.get("ServiceFeeEventList") or []:
            sku = (evt.get("SellerSKU") or "").strip()
            fee_desc = " ".join(
                str(evt.get(k) or "")
                for k in ("FeeDescription", "FeeReason", "FeeType")
            ).lower()
            # On removal service fees AmazonOrderId is the Removal Order ID
            # (Payments → Transactions "Order ID"), not a customer order.
            removal_order_id = (evt.get("AmazonOrderId") or "").strip()
            posted = evt.get("PostedDate") or ""
            for ftype, amt in _fees_from_lists(evt.get("FeeList")):
                if not amt:
                    continue
                ftype_l = (ftype or "").lower()
                bucket = _classify_fee_type(ftype)
                if not bucket and any(h in fee_desc for h in _PLACEMENT_ADJUSTMENT_HINTS):
                    bucket = "inbound_placement"
                if not bucket and any(h in fee_desc for h in _AGED_ADJUSTMENT_HINTS):
                    bucket = "aged_inventory"
                # Payments → Transactions rows like
                # "FBA Removal Order: Return Fee" / "Disposal Fee".
                if not bucket and (
                    any(h in fee_desc for h in _REMOVAL_SERVICE_HINTS)
                    or any(h in ftype_l for h in _REMOVAL_ADJUSTMENT_HINTS)
                ):
                    bucket = "removal"
                if not bucket:
                    continue
                # Bound removal to the profitability PostedDate window so we
                # match Seller Central Payments → Transactions "Date", not
                # Removal Order Detail request-date (those diverge a lot).
                if bucket == "removal" and removal_posted_after:
                    if posted and posted < removal_posted_after:
                        continue
                # Inbound placement: only count posts inside the profitability
                # filter (PostedDate), not the wider 45-day Finances lookback.
                if bucket == "inbound_placement":
                    if placement_posted_after and posted and posted < placement_posted_after:
                        continue
                    if placement_posted_before and posted and posted >= placement_posted_before:
                        continue
                target = by_sku[sku] if sku else unattributed
                target[bucket] += abs(amt)
                if bucket == "removal" and removal_order_id:
                    removal_by_order[removal_order_id] += abs(amt)

        for evt in events.get("AdjustmentEventList") or []:
            adj_type = (evt.get("AdjustmentType") or "").lower()
            adj_posted = evt.get("PostedDate") or ""
            if any(h in adj_type for h in _PLACEMENT_ADJUSTMENT_HINTS):
                if placement_posted_after and adj_posted and adj_posted < placement_posted_after:
                    continue
                if placement_posted_before and adj_posted and adj_posted >= placement_posted_before:
                    continue
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
            if removal_posted_after:
                posted = evt.get("PostedDate") or ""
                if posted and posted < removal_posted_after:
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
    placement_window_total = round(float(totals.get("inbound_placement") or 0), 2)

    return {
        "by_sku": {sku: {k: round(v, 2) for k, v in bucket.items()}
                   for sku, bucket in by_sku.items()},
        "unattributed": {k: round(v, 2) for k, v in unattributed.items()},
        "removal_by_order": {k: round(v, 2) for k, v in removal_by_order.items()},
        "totals": totals,
        "placement_window_total": placement_window_total,
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


def _placement_row_get(row: dict, *needles: str) -> str | None:
    """Case/space-insensitive column lookup for Amazon placement exports."""
    if not row:
        return None
    lowered = {
        str(k).strip().lower().replace("_", " "): v for k, v in row.items()
    }
    for needle in needles:
        key = needle.strip().lower().replace("_", " ")
        if key in lowered and lowered[key] not in (None, ""):
            return str(lowered[key]).strip()
    # Substring match only for long headers — never for "sku" (matches fnsku).
    for needle in needles:
        key = needle.strip().lower().replace("_", " ")
        if len(key) < 10 and " " not in key:
            continue
        for rk, rv in lowered.items():
            if key in rk and rv not in (None, ""):
                return str(rv).strip()
    return None


def _placement_row_float(row: dict, *needles: str) -> float:
    raw = _placement_row_get(row, *needles)
    if raw is None:
        return 0.0
    try:
        return float(str(raw).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _placement_row_int(row: dict, *needles: str) -> int:
    try:
        return int(float(_placement_row_float(row, *needles)))
    except (TypeError, ValueError):
        return 0


def parse_inbound_placement_fee_report(
    text: str,
) -> tuple[dict, list[str], list[dict]]:
    """Parse Seller Central / SP-API inbound placement fee report text.

    Seller Central Reports → Fulfillment → FBA inbound placement service fees
    exports CSV rows keyed by **FNSKU + ASIN** with a Transaction date.
    Profitability must sum `fee_total` for events whose transaction date
    falls in the selected filter window (same as SC Event Date filter) —
    not fee_rate × units sold.

    Returns (
      {key: {fee_total, units_received, fee_bearing_units, fee_rate, asin, fnsku, sku}},
      months_covered,
      events: [{transaction_date, shipment_id, fnsku, asin, sku, units, fee_rate, fee_total}],
    ).
    """
    if not (text or "").strip():
        return {}, [], []

    first = text.splitlines()[0]
    delim = "\t" if "\t" in first else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    acc: dict[str, dict] = {}
    months: set[str] = set()
    events: list[dict] = []

    for row in reader:
        sku = (
            _placement_row_get(
                row, "sku", "seller sku", "seller-sku", "msku", "merchant sku"
            )
            or ""
        ).strip()
        fnsku = (
            _placement_row_get(row, "fnsku", "fn sku", "fulfillment network sku")
            or ""
        ).strip().upper()
        asin = (_placement_row_get(row, "asin") or "").strip().upper()
        key = sku or fnsku or asin
        if not key:
            continue

        units = _placement_row_int(
            row,
            "actual received quantity",
            "received quantity",
            "quantity shipped",
            "unit quantity",
            "units",
            "quantity",
            "unit-count",
        )
        rate = _placement_row_float(
            row,
            "fba inbound placement service fee rate (per unit)",
            "fba inbound placement service fee rate",
            "placement service fee rate",
            "fee rate",
            "fee_rate",
            "inbound placement fee rate",
        )
        fee = _placement_row_float(
            row,
            "total fba inbound placement service fee charge",
            "total charge",
            "total charges",
            "placement fee",
            "placement service fee",
            "fee amount",
            "total fee",
        )
        if rate <= 0 and fee > 0 and units > 0:
            rate = fee / units
        if rate <= 0 and fee <= 0:
            continue
        if fee <= 0 and rate > 0 and units > 0:
            fee = rate * units
        weight = units if units > 0 else (1 if rate > 0 else 0)
        if weight <= 0:
            continue

        dt = (
            _placement_row_get(
                row, "transaction date", "event date", "charge date", "date"
            )
            or ""
        ).strip()
        shipment_id = (
            _placement_row_get(
                row, "fba shipment id", "shipment id", "amazon shipment id"
            )
            or ""
        ).strip()

        bucket = acc.setdefault(
            key,
            {
                "fee_total": 0.0,
                "units_received": 0,
                "fee_bearing_units": 0,
                "rate_weight": 0.0,
                "rate_weighted_sum": 0.0,
                "asin": asin or None,
                "fnsku": fnsku or None,
                "sku": sku or None,
            },
        )
        row_fee = fee if fee > 0 else rate * weight
        bucket["fee_total"] += row_fee
        bucket["units_received"] += units
        if rate > 0:
            bucket["fee_bearing_units"] += weight
            bucket["rate_weight"] += weight
            bucket["rate_weighted_sum"] += rate * weight
        if asin and not bucket.get("asin"):
            bucket["asin"] = asin
        if fnsku and not bucket.get("fnsku"):
            bucket["fnsku"] = fnsku
        if sku and not bucket.get("sku"):
            bucket["sku"] = sku

        if len(dt) >= 7:
            months.add(dt[:7])

        # Normalize to ISO-ish for window compares: "2026-04-27 17:13:02" → date
        tx_date = dt.replace("/", "-")
        if "T" not in tx_date and " " in tx_date:
            tx_date = tx_date.replace(" ", "T", 1)
        events.append(
            {
                "transaction_date": tx_date,
                "shipment_id": shipment_id or None,
                "fnsku": fnsku or None,
                "asin": asin or None,
                "sku": sku or None,
                "units": units,
                "fee_rate": round(rate, 6) if rate > 0 else 0.0,
                "fee_total": round(row_fee, 4),
            }
        )

    per_sku: dict[str, dict] = {}
    for key, v in acc.items():
        if v["rate_weight"] > 0:
            fee_rate = v["rate_weighted_sum"] / v["rate_weight"]
        elif v["fee_bearing_units"] > 0:
            fee_rate = v["fee_total"] / v["fee_bearing_units"]
        elif v["units_received"] > 0:
            fee_rate = v["fee_total"] / v["units_received"]
        else:
            fee_rate = 0.0
        if fee_rate <= 0:
            continue
        per_sku[key] = {
            "fee_total": round(v["fee_total"], 4),
            "units_received": int(v["units_received"]),
            "fee_bearing_units": int(v["fee_bearing_units"]),
            "fee_rate": round(fee_rate, 6),
            "asin": v.get("asin"),
            "fnsku": v.get("fnsku"),
            "sku": v.get("sku"),
        }
    return per_sku, sorted(months), events


async def fetch_inbound_placement_fees_per_sku(
    months_back: int = 3,
) -> tuple[dict, list[str], list[dict]]:
    """Pull GET_FBA_INBOUND_PLACEMENT_FEES_CHARGES_DATA — same report as
    Seller Central Reports → Fulfillment → FBA inbound placement service fees.

    Many Draft SP-API apps receive 403 for this type. Rows are FNSKU-keyed;
    callers should resolve to seller SKUs via products.

    Returns (per_sku, months_covered, events) — events carry transaction_date
    so profitability can filter by the same Event Date window as Seller Central.
    """
    now = datetime.now(timezone.utc)
    days = max(1, min(int(months_back * 30), 90))
    start = now - timedelta(days=days)
    end = now
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
    return parse_inbound_placement_fee_report(text)


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


async def _list_aged_surcharge_reports(created_since: datetime) -> list[dict]:
    """Recent GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA reports."""
    resp = await _sp_request(
        "GET",
        "/reports/2021-06-30/reports",
        params={
            "reportTypes": "GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA",
            "pageSize": "50",
            "createdSince": created_since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    return list(resp.get("reports") or [])


def _parse_report_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _report_covers_month(
    report: dict, month_start: datetime, month_end_excl: datetime,
) -> bool:
    """True if the report's data window overlaps [month_start, month_end_excl)."""
    ds = _parse_report_dt(report.get("dataStartTime"))
    de = _parse_report_dt(report.get("dataEndTime"))
    if ds is None or de is None:
        return False
    if ds.tzinfo is None:
        ds = ds.replace(tzinfo=timezone.utc)
    if de.tzinfo is None:
        de = de.replace(tzinfo=timezone.utc)
    return ds < month_end_excl and de > month_start


def _report_covers_month(
    report: dict, month_start: datetime, report_end: datetime,
) -> bool:
    """Loose overlap check for IN_PROGRESS reports whose dataEnd may not be set yet.
    Returns True if the report window overlaps with [month_start, report_end).
    Falls back to True if no timing info is present (assume it might cover it).
    """
    ds = _parse_report_dt(report.get("dataStartTime"))
    de = _parse_report_dt(report.get("dataEndTime"))
    if ds is None:
        return True  # No timing — assume it might be relevant
    if ds.tzinfo is None:
        ds = ds.replace(tzinfo=timezone.utc)
    if month_start.tzinfo is None:
        month_start = month_start.replace(tzinfo=timezone.utc)
    if report_end.tzinfo is None:
        report_end = report_end.replace(tzinfo=timezone.utc)
    # If dataEnd not known, check dataStart is within 45 days of month_start.
    if de is None:
        return abs((ds - month_start).days) <= 45
    if de.tzinfo is None:
        de = de.replace(tzinfo=timezone.utc)
    return ds < report_end and de > month_start


def _report_fully_covers_window(
    report: dict, start: datetime, end: datetime,
) -> bool:
    """True if report dataStart/dataEnd fully contains [start, end).

    Mere overlap is not enough for Removal Order Detail: a report that ends
    mid-month would under-count later request-dates if we treated it as done.
    Amazon often sets dataEnd to the last second of the range (end - 1s).
    """
    ds = _parse_report_dt(report.get("dataStartTime"))
    de = _parse_report_dt(report.get("dataEndTime"))
    if ds is None or de is None:
        return False
    if ds.tzinfo is None:
        ds = ds.replace(tzinfo=timezone.utc)
    if de.tzinfo is None:
        de = de.replace(tzinfo=timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return ds <= (start + timedelta(seconds=1)) and de >= (end - timedelta(seconds=2))


def _accumulate_aged_surcharge_rows(
    text: str,
    per_sku: dict[str, dict],
    months_filter: set[str] | None = None,
) -> list[str]:
    """Parse amount-charged rows from an AIS charges report into per_sku.

    Accepts Seller Central CSV downloads and SP-API TSV. Optionally keeps only
    rows whose snapshot-date month is in ``months_filter`` (YYYY-MM), matching
    Seller Central's Event Month filter.

    Returns the set of snapshot months observed (as a sorted list).
    """
    if not (text or "").strip():
        return []

    cleaned = text.lstrip("\ufeff")
    first = cleaned.splitlines()[0]
    delim = "\t" if "\t" in first else ","
    reader = csv.DictReader(io.StringIO(cleaned), delimiter=delim)
    seen_months: set[str] = set()

    for row in reader:
        norm = {
            (k or "").strip().lstrip("\ufeff").lower().replace("_", "-").replace(" ", "-"): v
            for k, v in row.items()
        }
        sku = (
            norm.get("sku")
            or norm.get("seller-sku")
            or norm.get("merchant-sku")
            or ""
        ).strip()
        if not sku:
            continue

        snapshot = (
            norm.get("snapshot-date")
            or norm.get("date")
            or norm.get("snapshotdate")
            or ""
        ).strip()
        month_key = ""
        if snapshot:
            # 2026-05-15T08:02:00+00:00 or 2026-05-15
            month_key = snapshot[:7] if re.match(r"^\d{4}-\d{2}", snapshot) else ""
            if month_key:
                seen_months.add(month_key)
            if months_filter is not None and month_key and month_key not in months_filter:
                continue

        charged = _coerce_float(
            norm.get("amount-charged") or norm.get("amountcharged"),
        )
        if charged <= 0:
            charged = _coerce_float(norm.get("long-time-range-long-term-storage-fee")) + _coerce_float(
                norm.get("short-time-range-long-term-storage-fee"),
            )
        qty = int(
            _coerce_float(
                norm.get("qty-charged")
                or norm.get("qtycharged")
                or norm.get("qty-charged-long-time-range-long-term-storage-fee")
                or norm.get("qty-charged-short-time-range-long-term-storage-fee"),
            )
        )
        if charged <= 0 and qty <= 0:
            continue
        asin = (norm.get("asin") or "").strip().upper() or None
        bucket = per_sku.setdefault(
            sku,
            {"charged_total": 0.0, "qty_charged": 0, "asin": asin},
        )
        bucket["charged_total"] = _coerce_float(bucket["charged_total"]) + charged
        bucket["qty_charged"] = int(bucket.get("qty_charged") or 0) + max(qty, 0)
        if asin and not bucket.get("asin"):
            bucket["asin"] = asin

    return sorted(seen_months)


def parse_aged_surcharge_charges_report(
    text: str,
    months_filter: list[str] | None = None,
) -> tuple[dict[str, dict], list[str]]:
    """Parse Aged Inventory Surcharge / LONGTERM_STORAGE_FEE_CHARGES report.

    Returns ({sku: {charged_total, qty_charged, asin}}, months_seen).
    """
    allowed = set(months_filter) if months_filter else None
    per_sku: dict[str, dict] = {}
    months = _accumulate_aged_surcharge_rows(text, per_sku, allowed)
    out = {
        sku: {
            "charged_total": round(_coerce_float(v["charged_total"]), 2),
            "qty_charged": int(v.get("qty_charged") or 0),
            "asin": v.get("asin"),
        }
        for sku, v in per_sku.items()
        if _coerce_float(v.get("charged_total")) > 0
    }
    return out, months


async def fetch_aged_surcharge_charges_per_sku(
    start: datetime,
    end: datetime,
    time_zone: str = "UTC",
) -> dict[str, dict]:
    """Pull GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA — the same
    report Seller Central shows as "Aged Inventory Surcharge report".

    Strategy (critical — Amazon FATALS duplicate creates):
      1. Calendar months are taken in marketplace TZ (not UTC day boundaries —
         March 31 PDT end-of-day is April 1 UTC and must NOT pull April).
      2. Prefer an existing DONE report, but only if its rows include the
         requested Event Month (snapshot-date). Blind date-range reuse was
         returning June/May files for March filters → $0 after month filter.
      3. Create a new month report when reuse yields no matching Event Month.
      4. Empty parsed result after a successful create = real SC "No results".

    Returns {sku: {charged_total, qty_charged, asin}}.
    """
    month_keys = calendar_months_in_window(start, end, time_zone)
    if not month_keys:
        return {}

    now = datetime.now(timezone.utc)
    # getReports rejects createdSince older than 90 days.
    lookback = now - timedelta(days=89)
    try:
        existing = await _list_aged_surcharge_reports(lookback)
    except Exception:
        existing = []
    existing_done = [
        r for r in existing
        if r.get("processingStatus") == "DONE" and r.get("reportDocumentId")
    ]

    per_sku: dict[str, dict] = {}
    errors: list[str] = []
    months_resolved: list[str] = []

    def _merge_parsed(parsed: dict[str, dict]) -> None:
        for sku, bucket in parsed.items():
            dest = per_sku.setdefault(
                sku,
                {"charged_total": 0.0, "qty_charged": 0, "asin": bucket.get("asin")},
            )
            dest["charged_total"] += float(bucket.get("charged_total") or 0)
            dest["qty_charged"] += int(bucket.get("qty_charged") or 0)
            if bucket.get("asin") and not dest.get("asin"):
                dest["asin"] = bucket["asin"]

    async def _try_text_for_month(text: str, label: str) -> bool:
        """True if report contains this Event Month (even when amount-charged is $0)."""
        parsed, seen = parse_aged_surcharge_charges_report(text, [label])
        if label not in seen and not parsed:
            # Wrong month file (e.g. reused June report while asking for March).
            return False
        _merge_parsed(parsed)
        return True

    for label in month_keys:
        month_start, month_end_excl = month_start_end_excl(label, time_zone)
        resolved = False

        # 1) Try a few DONE reports whose data window overlaps this month.
        #    Do NOT scan extras[:15] / [:20] — each getReportDocument hit
        #    burns the Reports quota and freezes the profitability UI on 429.
        candidates = [
            r for r in existing_done
            if _report_covers_month(r, month_start, month_end_excl)
        ]
        candidates.sort(key=lambda r: r.get("createdTime") or "", reverse=True)
        # Newest overlapping first; at most one non-overlapping newest as fallback.
        ordered = list(candidates[:2])
        if existing_done:
            newest = max(
                existing_done,
                key=lambda r: r.get("createdTime") or "",
            )
            if newest not in ordered:
                ordered.append(newest)

        for report in ordered:
            try:
                text = await download_report_raw(
                    report["reportId"], max_polls=3, poll_interval=5,
                )
            except Exception as e:
                errors.append(f"{label}: reuse {report.get('reportId')} failed ({e})")
                continue
            if await _try_text_for_month(text, label):
                resolved = True
                months_resolved.append(label)
                break

        if resolved:
            continue

        # 3) Create a fresh report for this Event Month.
        try:
            # Amazon wants [start, end) in UTC; end exclusive = next month start.
            end_for_api = min(month_end_excl, now)
            if end_for_api <= month_start:
                end_for_api = now
            create_resp = await create_report(
                "GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA",
                start_date=month_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end_date=end_for_api.strftime("%Y-%m-%dT%H:%M:%SZ"),
                single_marketplace=True,
            )
            report_id = create_resp.get("reportId")
            if not report_id:
                errors.append(f"{label}: create returned no id")
            else:
                text = await download_report_raw(
                    report_id, max_polls=24, poll_interval=8,
                )
                if await _try_text_for_month(text, label):
                    resolved = True
                    months_resolved.append(label)
                else:
                    # Successful download but no rows for this month = real $0.
                    # Mark resolved so we don't treat it as a hard failure.
                    parsed_all, seen_all = parse_aged_surcharge_charges_report(text)
                    if not seen_all or label in seen_all or not parsed_all:
                        resolved = True
                        months_resolved.append(label)
                    else:
                        errors.append(
                            f"{label}: created report had months {seen_all}, not {label}"
                        )
        except Exception as e:
            errors.append(f"{label}: create/download failed ({e})")
            # Last chance: refresh list and try at most 2 newest DONE reports.
            try:
                existing = await _list_aged_surcharge_reports(lookback)
                existing_done = [
                    r for r in existing
                    if r.get("processingStatus") == "DONE"
                    and r.get("reportDocumentId")
                ]
                for report in sorted(
                    existing_done,
                    key=lambda r: r.get("createdTime") or "",
                    reverse=True,
                )[:2]:
                    try:
                        text = await download_report_raw(
                            report["reportId"], max_polls=3, poll_interval=5,
                        )
                    except Exception:
                        continue
                    if await _try_text_for_month(text, label):
                        resolved = True
                        months_resolved.append(label)
                        errors.append(
                            f"{label}: recovered via existing DONE "
                            f"{report.get('reportId')}"
                        )
                        break
            except Exception as e2:
                errors.append(f"{label}: recovery failed ({e2})")

        if not resolved:
            errors.append(f"{label}: no Aged Inventory Surcharge data resolved")

    if not per_sku and errors and not months_resolved:
        raise RuntimeError(
            "Aged surcharge charges report failed for every month: "
            + "; ".join(errors)
        )
    # Empty per_sku with resolved months = valid "No results found".
    return {
        sku: {
            "charged_total": round(v["charged_total"], 2),
            "qty_charged": int(v["qty_charged"]),
            "asin": v.get("asin"),
        }
        for sku, v in per_sku.items()
        if v["charged_total"] > 0
    }


# ── FBA Removal Order Detail (removal / disposal fees) ───────────────────────


async def _list_removal_order_detail_reports(created_since: datetime) -> list[dict]:
    """Recent GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA reports."""
    params = {
        "reportTypes": "GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA",
        "pageSize": "50",
        "createdSince": created_since.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        resp = await _sp_request("GET", "/reports/2021-06-30/reports", params=params)
    except Exception:
        return []
    if isinstance(resp, dict):
        return list(resp.get("reports") or [])
    return []


def _parse_removal_request_dt(raw: str) -> datetime | None:
    """Parse request-date from Removal Order Detail (CSV/TSV)."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        if len(s) == 10 and re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return datetime(int(s[:4]), int(s[5:7]), int(s[8:10]), tzinfo=timezone.utc)
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def parse_removal_order_detail_report(
    text: str,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> tuple[dict[str, dict], float]:
    """Parse GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA (SC CSV or SP-API TSV).

    Seller Central's Removal Order Detail report bills by ``request-date``.
    When ``window_start`` / ``window_end`` are set, only rows with
    request-date in ``[window_start, window_end)`` are counted — same filter
    as the profitability date picker.

    Returns ({sku: {removal_fee, qty, order_ids}}, report_total).
    """
    if not (text or "").strip():
        return {}, 0.0

    cleaned = text.lstrip("\ufeff")
    first = cleaned.splitlines()[0]
    delim = "\t" if "\t" in first else ","
    reader = csv.DictReader(io.StringIO(cleaned), delimiter=delim)

    start = window_start
    end = window_end
    if start is not None and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    per_sku: dict[str, dict] = {}
    report_total = 0.0

    for row in reader:
        norm = {
            (k or "").strip().lstrip("\ufeff").lower().replace("_", "-").replace(" ", "-"): v
            for k, v in row.items()
        }
        sku = (
            norm.get("sku")
            or norm.get("seller-sku")
            or norm.get("merchant-sku")
            or ""
        ).strip()
        if not sku:
            continue

        req_raw = (
            norm.get("request-date")
            or norm.get("requestdate")
            or norm.get("order-date")
            or ""
        ).strip()
        req_dt = _parse_removal_request_dt(req_raw)
        if start is not None or end is not None:
            if req_dt is None:
                continue
            if start is not None and req_dt < start:
                continue
            if end is not None and req_dt >= end:
                continue

        fee = _coerce_float(norm.get("removal-fee") or norm.get("removalfee"))
        if fee <= 0:
            continue

        qty = int(
            _coerce_float(
                norm.get("requested-quantity")
                or norm.get("disposed-quantity")
                or norm.get("shipped-quantity")
                or 0,
            )
        )
        order_id = (norm.get("order-id") or norm.get("orderid") or "").strip()
        bucket = per_sku.setdefault(
            sku,
            {"removal_fee": 0.0, "qty": 0, "order_ids": []},
        )
        bucket["removal_fee"] = _coerce_float(bucket["removal_fee"]) + fee
        bucket["qty"] = int(bucket.get("qty") or 0) + max(qty, 0)
        if order_id and order_id not in bucket["order_ids"]:
            bucket["order_ids"].append(order_id)
        report_total += fee

    out = {
        sku: {
            "removal_fee": round(_coerce_float(v["removal_fee"]), 2),
            "qty": int(v.get("qty") or 0),
            "order_ids": list(v.get("order_ids") or []),
        }
        for sku, v in per_sku.items()
        if _coerce_float(v.get("removal_fee")) > 0
    }
    return out, round(report_total, 2)


async def fetch_removal_fees_per_sku(
    start: datetime,
    end: datetime,
    time_zone: str = "UTC",
) -> dict[str, dict]:
    """Pull Removal Order Detail fees for the profitability window.

    Strategy (matches Seller Central Event Date / request-date):
      1. Resolve calendar months overlapping the window in marketplace TZ.
      2. For each month, prefer an existing DONE report that fully covers it.
         If one is IN_PROGRESS, wait for it (never create a duplicate).
         Only create a new report when nothing useful exists.
      3. Filter rows by the customer's exact [start, end) date range.

    Returns {sku: {removal_fee, qty, order_ids}}.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end <= start:
        return {}

    month_keys = calendar_months_in_window(start, end, time_zone)
    if not month_keys:
        return {}

    now = datetime.now(timezone.utc)
    # 90-day lookback for the reports list.
    lookback = now - timedelta(days=89)

    def _refresh_lists(reports: list[dict]) -> tuple[list, list]:
        done = [
            r for r in reports
            if r.get("processingStatus") == "DONE" and r.get("reportDocumentId")
        ]
        in_prog = [
            r for r in reports
            if r.get("processingStatus") in ("IN_QUEUE", "IN_PROGRESS")
        ]
        return done, in_prog

    try:
        existing = await _list_removal_order_detail_reports(lookback)
    except Exception:
        existing = []
    existing_done, existing_inprog = _refresh_lists(existing)

    merged: dict[str, dict] = {}

    def _merge(parsed: dict[str, dict]) -> None:
        for sku, bucket in parsed.items():
            dest = merged.setdefault(
                sku, {"removal_fee": 0.0, "qty": 0, "order_ids": []}
            )
            dest["removal_fee"] = float(dest["removal_fee"]) + float(
                bucket.get("removal_fee") or 0
            )
            dest["qty"] = int(dest.get("qty") or 0) + int(bucket.get("qty") or 0)
            for oid in bucket.get("order_ids") or []:
                if oid and oid not in dest["order_ids"]:
                    dest["order_ids"].append(oid)

    for label in month_keys:
        month_start, month_end_excl = month_start_end_excl(label, time_zone)
        w_start = max(start, month_start)
        w_end = min(end, month_end_excl)
        if w_end <= w_start:
            continue

        report_end = min(month_end_excl, now)
        if report_end <= month_start:
            report_end = now

        text: str | None = None

        # 1) Try existing DONE reports that fully cover this month (1 best).
        covering = [
            r for r in existing_done
            if _report_fully_covers_window(r, month_start, report_end)
        ]
        covering.sort(key=lambda r: r.get("createdTime") or "", reverse=True)
        for report in covering[:1]:
            try:
                text = await download_report_raw(
                    report["reportId"], max_polls=3, poll_interval=5,
                )
                break
            except Exception:
                pass

        if text is not None:
            parsed, _ = parse_removal_order_detail_report(text, w_start, w_end)
            _merge(parsed)
            continue

        # 2) An IN_PROGRESS report for this month already exists — wait briefly.
        #    Never create a duplicate (Amazon will FATAL). Cap wait so the UI
        #    cannot hang for minutes; caller can refresh once Amazon finishes.
        in_prog_covering = [
            r for r in existing_inprog
            if _report_covers_month(r, month_start, report_end)
        ]
        in_prog_covering.sort(key=lambda r: r.get("createdTime") or "", reverse=True)
        waited_id: str | None = None
        if in_prog_covering:
            waited_id = in_prog_covering[0]["reportId"]
            try:
                # ~60s max (12 × 5s) — not 4 minutes.
                text = await download_report_raw(
                    waited_id, max_polls=12, poll_interval=5,
                )
            except Exception:
                text = None

        # 3) No existing report — create one.
        created_id: str | None = None
        if text is None and waited_id is None:
            try:
                create_resp = await create_report(
                    "GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA",
                    start_date=month_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end_date=report_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    single_marketplace=True,
                )
                created_id = create_resp.get("reportId")
                if created_id:
                    text = await download_report_raw(
                        created_id, max_polls=18, poll_interval=5,
                    )
            except Exception:
                created_id = None

        # 4) After create failure (FATAL = duplicate) refresh and find the DONE.
        if text is None:
            try:
                refreshed = await _list_removal_order_detail_reports(lookback)
                refreshed_done, _ = _refresh_lists(refreshed)
                existing_done = refreshed_done
                for r in sorted(
                    refreshed_done,
                    key=lambda r: r.get("createdTime") or "",
                    reverse=True,
                )[:2]:
                    if _report_fully_covers_window(r, month_start, report_end) or \
                       _report_covers_month(r, month_start, report_end):
                        try:
                            text = await download_report_raw(
                                r["reportId"], max_polls=3, poll_interval=5,
                            )
                            if text:
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        if text is None:
            # Still nothing — raise so the caller can show the warning.
            raise RuntimeError(
                f"Removal Order Detail unavailable for {label}: "
                "report still IN_PROGRESS or create failed. "
                "Refresh in ~1 minute once Amazon finishes processing."
            )

        parsed, _ = parse_removal_order_detail_report(text, w_start, w_end)
        _merge(parsed)
        # Update our cached lists for the next month iteration.
        try:
            existing = await _list_removal_order_detail_reports(lookback)
            existing_done, existing_inprog = _refresh_lists(existing)
        except Exception:
            pass

    return {
        sku: {
            "removal_fee": round(float(v["removal_fee"]), 2),
            "qty": int(v.get("qty") or 0),
            "order_ids": list(v.get("order_ids") or []),
        }
        for sku, v in merged.items()
        if float(v.get("removal_fee") or 0) > 0
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
    """Get the download URL for a completed report document.

    Documents endpoint has a tight quota — space calls and fail fast on 429
    so the profitability UI cannot hang for minutes on retries.
    """
    global _DOC_GET_LAST_TS
    async with _doc_get_lock():
        now = time.monotonic()
        gap = _DOC_GET_MIN_GAP_S - (now - _DOC_GET_LAST_TS)
        if gap > 0:
            await asyncio.sleep(gap)
        try:
            return await _sp_request(
                "GET",
                f"/reports/2021-06-30/documents/{document_id}",
                max_429_retries=3,
            )
        finally:
            _DOC_GET_LAST_TS = time.monotonic()


async def download_report_raw(report_id: str, max_polls: int = 30, poll_interval: int = 10) -> str:
    """Poll until a report is done, then return the FULL decoded text.

    Used by the ingest pipeline, which needs every row (not the
    LLM-friendly truncated summary that `download_report` returns).
    Caches DONE report bodies by reportId for the process lifetime.
    """
    cached = _REPORT_TEXT_CACHE.get(report_id)
    if cached is not None:
        return cached

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
            text = content.decode("utf-8", errors="replace")
            if len(_REPORT_TEXT_CACHE) >= _REPORT_TEXT_CACHE_MAX:
                # Drop an arbitrary oldest entry (insertion order in 3.7+).
                _REPORT_TEXT_CACHE.pop(next(iter(_REPORT_TEXT_CACHE)), None)
            _REPORT_TEXT_CACHE[report_id] = text
            return text
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