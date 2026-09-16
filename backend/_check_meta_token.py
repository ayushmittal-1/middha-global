"""Verify a Meta access token is the right kind before wiring it up.

The keyword matrix's Meta column needs a token that (a) is valid, (b) never
expires, and (c) can actually run an ad-interest search. Short-lived user
tokens satisfy (a) and (c) for a few hours and then die at a fixed clock time,
which looks like a code failure but isn't — so check before handing it over.

Usage (no install needed beyond the project's venv):

    META_ACCESS_TOKEN='EAA...' python3 _check_meta_token.py

Exit code 0 means every check passed. Non-zero means do not use this token;
the output says which check failed and what to change.

The token is never printed, only its length and last 4 characters, so the
output is safe to paste into a ticket or chat.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

GRAPH = "https://graph.facebook.com/v19.0"
TOKEN = (os.getenv("META_ACCESS_TOKEN") or "").strip()

# The search term used for the live capability check. Any common term works;
# this one is known to exist in Meta's interest taxonomy.
PROBE_TERM = "incense"

OK, BAD, WARN = "  PASS", "  FAIL", "  WARN"
failures: list[str] = []


def call(path: str, **params) -> tuple[int, dict]:
    """GET a Graph endpoint. Token goes in the header, never the query string,
    so it can't leak into logs or error text."""
    url = f"{GRAPH}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read() or b"{}"
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": {"message": body.decode("utf-8", "replace")[:300]}}
    except Exception as e:  # network, DNS, timeout
        return 0, {"error": {"message": f"{type(e).__name__}: {e}"}}


def err_of(payload: dict) -> str:
    return ((payload.get("error") or {}).get("message") or "").strip() or "(no message)"


def main() -> int:
    if not TOKEN:
        print("FAIL: set META_ACCESS_TOKEN in the environment first, e.g.")
        print("      META_ACCESS_TOKEN='EAA...' python3 _check_meta_token.py")
        return 2

    print(f"Checking token: {len(TOKEN)} chars, ending ...{TOKEN[-4:]}\n")

    # ---- 1. Is it valid at all, and what is it? ---------------------------
    # debug_token introspects a token. An expired or revoked token cannot even
    # be introspected, so a failure here is itself the answer.
    print("1. Token is valid and introspectable")
    status, payload = call("debug_token", input_token=TOKEN)
    if status != 200:
        print(f"{BAD} Meta rejected the token: {err_of(payload)}")
        print("       -> The token is expired, revoked, or from another app.")
        print("          Generate a new one and re-run this script.")
        return 1
    data = payload.get("data") or {}
    if not data.get("is_valid"):
        print(f"{BAD} Meta reports is_valid=false")
        return 1
    print(f"{OK} valid (app_id={data.get('app_id')})")

    # ---- 2. System User, not a user session token -------------------------
    print("\n2. Token type is SYSTEM_USER")
    ttype = (data.get("type") or "").upper()
    if ttype == "SYSTEM_USER":
        print(f"{OK} type={ttype}")
    else:
        print(f"{BAD} type={ttype or '(unknown)'} — expected SYSTEM_USER")
        print("       -> A USER token dies when the person logs out or the")
        print("          session lapses. Create the token under")
        print("          Business Settings > Users > System Users.")
        failures.append("token type")

    # ---- 3. Never expires -------------------------------------------------
    print("\n3. Token never expires")
    exp = data.get("expires_at")
    if exp == 0:
        print(f"{OK} expires_at=0 (never)")
    else:
        import datetime
        when = ""
        try:
            when = datetime.datetime.fromtimestamp(
                int(exp), datetime.timezone.utc
            ).strftime(" (%Y-%m-%d %H:%M UTC)")
        except Exception:
            pass
        print(f"{BAD} expires_at={exp}{when}")
        print("       -> Re-generate with expiry set to 'Never'. A token with")
        print("          an expiry WILL break the Meta column when it lapses.")
        failures.append("expiry")

    # ---- 4. Has an ads scope ---------------------------------------------
    print("\n4. Has an ads read scope")
    scopes = data.get("scopes") or []
    if any(s in scopes for s in ("ads_read", "ads_management")):
        print(f"{OK} scopes include: {[s for s in scopes if s.startswith('ads')]}")
    else:
        print(f"{BAD} no ads_read / ads_management in {scopes}")
        print("       -> Re-generate the token with ads_read ticked.")
        failures.append("scopes")

    # ---- 5. The check that actually matters -------------------------------
    # Everything above can look right while the app still lacks Marketing API
    # access, so finish by running the exact call the app makes.
    print(f"\n5. Ad-interest search works (the call the app actually makes)")
    status, payload = call(
        "search", type="adinterest", q=PROBE_TERM, limit=5
    )
    if status != 200:
        print(f"{BAD} HTTP {status}: {err_of(payload)}")
        print("       -> If this mentions permissions rather than the token,")
        print("          the app needs Marketing API access for adinterest")
        print("          search. Everything above can pass and this still fail.")
        failures.append("adinterest search")
    else:
        names = [d.get("name") for d in (payload.get("data") or [])]
        if names:
            print(f"{OK} '{PROBE_TERM}' returned {len(names)}: {names}")
        else:
            print(f"{WARN} '{PROBE_TERM}' returned 0 interests")
            print("       -> Auth works but the search came back empty, which")
            print("          is unexpected for this term. Worth reporting.")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 62)
    if failures:
        print("RESULT: NOT READY — failed: " + ", ".join(failures))
        print("Fix the items marked FAIL above and re-run.")
        return 1
    print("RESULT: READY — this token is a non-expiring System User token")
    print("        with ads access, and ad-interest search works.")
    print("Set it as META_ACCESS_TOKEN in the Render dashboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
