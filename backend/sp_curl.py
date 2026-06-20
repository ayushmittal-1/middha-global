"""
Print a freshly SigV4-signed curl command for an SP-API call, using the same
LWA refresh token + AWS keys + marketplace the FastAPI backend uses for the
chosen user.

Examples:
    # default: last 1000 days of orders for rohit@example.com
    python backend/sp_curl.py

    # specific user + only the last 30 days
    python backend/sp_curl.py --email someone@example.com --days-back 30

    # different endpoint (FBA inventory) + override params
    python backend/sp_curl.py --path /fba/inventory/v1/summaries \
        -p granularityType=Marketplace -p details=true

    # bigger page size
    python backend/sp_curl.py -p MaxResultsPerPage=100

The signature is valid for ~15 minutes from the printed x-amz-date; the LWA
access token for ~60. Re-run when either expires.
"""

import argparse
import asyncio
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv(Path(__file__).parent.parent / ".env")

LWA_ID = os.getenv("AMAZON_LWA_CLIENT_ID", "")
LWA_SECRET = os.getenv("AMAZON_LWA_CLIENT_SECRET", "")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB_NAME") or "test"

# Same region table as amazon_sp.py
_SP_API_BASES = {
    "NA": ("https://sellingpartnerapi-na.amazon.com", "us-east-1"),
    "EU": ("https://sellingpartnerapi-eu.amazon.com", "eu-west-1"),
    "FE": ("https://sellingpartnerapi-fe.amazon.com", "us-west-2"),
}
SERVICE = "execute-api"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


async def _get_user(email: str) -> dict:
    if not MONGO_URI:
        raise SystemExit("MONGO_URI is not set in .env")
    client = AsyncIOMotorClient(MONGO_URI)
    user = await client[MONGO_DB].users.find_one({"email": email.lower().strip()})
    client.close()
    if not user:
        raise SystemExit(f"No user with email {email!r} in db {MONGO_DB!r}")
    if not user.get("amazonRefreshToken"):
        raise SystemExit(f"User {email!r} has no amazonRefreshToken — finish the SP-API OAuth flow first.")
    return user


async def _lwa_access_token(refresh: str, region: str) -> str:
    token_url = {
        "NA": "https://api.amazon.com/auth/o2/token",
        "EU": "https://api.amazon.co.uk/auth/o2/token",
        "FE": "https://api.amazon.co.jp/auth/o2/token",
    }.get(region, "https://api.amazon.com/auth/o2/token")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(token_url, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": LWA_ID,
            "client_secret": LWA_SECRET,
        })
        r.raise_for_status()
        return r.json()["access_token"]


def _sigv4_curl(method: str, host: str, path: str, query: str, access_token: str, region: str) -> str:
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    headers = {"content-type": "application/json", "host": host, "x-amz-date": amz_date}
    signed_keys = sorted(headers)
    canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in signed_keys)
    signed_header_str = ";".join(signed_keys)

    canonical_request = "\n".join([
        method,
        path,
        query,
        canonical_headers,
        signed_header_str,
        hashlib.sha256(b"").hexdigest(),
    ])
    scope = f"{date_stamp}/{region}/{SERVICE}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])

    k = _sign(("AWS4" + AWS_SECRET).encode(), date_stamp)
    k = _sign(k, region)
    k = _sign(k, SERVICE)
    k = _sign(k, "aws4_request")
    signature = hmac.new(k, string_to_sign.encode(), hashlib.sha256).hexdigest()

    auth = (
        f"AWS4-HMAC-SHA256 Credential={AWS_KEY}/{scope}, "
        f"SignedHeaders={signed_header_str}, Signature={signature}"
    )

    url = f"https://{host}{path}" + (f"?{query}" if query else "")
    return (
        f"# x-amz-date {amz_date} — signature valid ~15 min; LWA token ~60 min\n"
        f"curl -sS -X {method} '{url}' \\\n"
        f"  -H 'content-type: application/json' \\\n"
        f"  -H 'host: {host}' \\\n"
        f"  -H 'x-amz-date: {amz_date}' \\\n"
        f"  -H 'x-amz-access-token: {access_token}' \\\n"
        f"  -H 'user-agent: MiddhaGlobal/1.0 (Language=Python)' \\\n"
        f"  -H 'Authorization: {auth}'"
    )


def _parse_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"bad -p value {it!r}, expected key=value")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


async def amain():
    ap = argparse.ArgumentParser(description="Generate a SigV4-signed SP-API curl.")
    ap.add_argument("--email", default="rohit@example.com",
                    help="Mongo user to mint creds from (default: rohit@example.com)")
    ap.add_argument("--method", default="GET", choices=["GET", "POST", "PUT", "DELETE"])
    ap.add_argument("--path", default="/orders/v0/orders",
                    help="SP-API path (default: /orders/v0/orders)")
    ap.add_argument("--days-back", type=int, default=1000,
                    help="For /orders/v0/orders, sets CreatedAfter (default: 1000)")
    ap.add_argument("--marketplace", default=None,
                    help="Override marketplace id (default: user.amazonMarketplaceIds[0])")
    ap.add_argument("-p", "--param", action="append", default=[],
                    help="Extra query param as key=value (repeatable). Overrides defaults.")
    args = ap.parse_args()

    user = await _get_user(args.email)
    region = (user.get("marketplace") or "NA").upper()
    sp_base, sp_region = _SP_API_BASES.get(region, _SP_API_BASES["NA"])
    host = sp_base.replace("https://", "")
    marketplace = args.marketplace or (user.get("amazonMarketplaceIds") or [None])[0]

    # Default params per endpoint
    params: dict[str, str] = {}
    if args.path == "/orders/v0/orders":
        params["MarketplaceIds"] = marketplace or ""
        params["MaxResultsPerPage"] = "20"
        if args.days_back > 0:
            params["CreatedAfter"] = (
                datetime.now(timezone.utc) - timedelta(days=args.days_back)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif args.path.startswith("/fba/inventory/v1/summaries"):
        params["details"] = "true"
        params["granularityType"] = "Marketplace"
        params["granularityId"] = marketplace or ""
        params["marketplaceIds"] = marketplace or ""

    params.update(_parse_kv(args.param))
    query = urlencode(params)

    access_token = await _lwa_access_token(user["amazonRefreshToken"], region)
    print(f"# user: {user['email']}  region: {region}  marketplace: {marketplace}")
    print(_sigv4_curl(args.method, host, args.path, query, access_token, sp_region))


if __name__ == "__main__":
    asyncio.run(amain())
