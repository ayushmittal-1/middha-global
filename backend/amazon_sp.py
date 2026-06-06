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
from datetime import datetime, timezone
from urllib.parse import quote, urlencode, urlparse

import httpx

from amazon_ads import get_access_token

# ── Config ───────────────────────────────────────────────────────────────────

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
SELLER_ID = os.getenv("AMAZON_SELLER_ID", "")
MARKETPLACE_ID = os.getenv("AMAZON_MARKETPLACE_ID", "ATVPDKIKX0DER")

SP_API_BASE = "https://sellingpartnerapi-na.amazon.com"
SP_API_REGION = "us-east-1"
SP_API_SERVICE = "execute-api"

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
    body: str = "",
) -> dict:
    """Add SigV4 Authorization header to the request headers dict (in-place + returned)."""
    access_key = AWS_ACCESS_KEY_ID or os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = AWS_SECRET_ACCESS_KEY or os.getenv("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        raise RuntimeError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for SP-API")

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

    credential_scope = f"{date_stamp}/{SP_API_REGION}/{SP_API_SERVICE}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    signing_key = _get_signature_key(secret_key, date_stamp, SP_API_REGION, SP_API_SERVICE)
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
) -> dict | list | str:
    """Make a signed SP-API request."""
    access_token = await get_access_token()

    query_string = urlencode(params, doseq=True) if params else ""
    url = f"{SP_API_BASE}{path}"
    if query_string:
        url = f"{url}?{query_string}"

    body_str = json.dumps(body) if body else ""

    # Only include headers that should be SigV4-signed
    headers = {
        "content-type": "application/json",
    }
    _sigv4_headers(method, url, headers, body=body_str)

    # Add non-signed headers after signing
    headers["x-amz-access-token"] = access_token
    headers["user-agent"] = "MiddhaGlobal/1.0 (Language=Python)"

    print(f"[sp-api] -> {method} {path} params={params}")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(
            method,
            url,
            headers=headers,
            content=body_str if body_str else None,
        )
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
    statuses: list[str] | None = None,
    max_results: int = 20,
) -> dict:
    """List orders. created_after is ISO-8601 (e.g. '2024-01-01T00:00:00Z')."""
    params = {
        "MarketplaceIds": MARKETPLACE_ID,
        "MaxResultsPerPage": str(min(max_results, 100)),
    }
    if created_after:
        params["CreatedAfter"] = created_after
    if statuses:
        params["OrderStatuses"] = ",".join(statuses)
    return await _sp_request("GET", "/orders/v0/orders", params=params)


async def get_order(order_id: str) -> dict:
    """Get details for a single order."""
    return await _sp_request("GET", f"/orders/v0/orders/{order_id}")


async def get_order_items(order_id: str) -> dict:
    """Get line items for an order."""
    return await _sp_request("GET", f"/orders/v0/orders/{order_id}/orderItems")


# ── FBA Inventory API (v1) ───────────────────────────────────────────────────


async def get_inventory_summaries(
    skus: list[str] | None = None,
    details: bool = True,
) -> dict:
    """Get FBA inventory levels."""
    params = {
        "details": str(details).lower(),
        "granularityType": "Marketplace",
        "granularityId": MARKETPLACE_ID,
        "marketplaceIds": MARKETPLACE_ID,
    }
    if skus:
        params["sellerSkus"] = ",".join(skus)
    return await _sp_request("GET", "/fba/inventory/v1/summaries", params=params)


# ── Reports API (2021-06-30) ────────────────────────────────────────────────


async def create_report(
    report_type: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Request a new report. Returns reportId for polling."""
    body: dict = {
        "reportType": report_type,
        "marketplaceIds": [MARKETPLACE_ID],
    }
    if start_date or end_date:
        body["dataStartTime"] = start_date
        body["dataEndTime"] = end_date
    return await _sp_request("POST", "/reports/2021-06-30/reports", body=body)


async def get_report(report_id: str) -> dict:
    """Check report processing status."""
    return await _sp_request("GET", f"/reports/2021-06-30/reports/{report_id}")


async def get_report_document(document_id: str) -> dict:
    """Get the download URL for a completed report document."""
    return await _sp_request("GET", f"/reports/2021-06-30/documents/{document_id}")


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
