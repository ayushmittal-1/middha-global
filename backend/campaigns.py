import os
import httpx

from auth import require_user

AURORA_API_URL = os.getenv("AURORA_API_URL", "https://aurorabackend-is4p.onrender.com/api/ads")

# Cache campaigns per-user so different sellers don't see each other's data.
_user_campaigns: dict[str, list[dict]] = {}
_user_summary: dict[str, str] = {}


def _user_key() -> str:
    return str(require_user()["_id"])


async def fetch_all_campaigns(user: dict | None = None) -> None:
    """Fetch all campaigns from Aurora API with pagination using the current
    user's JWT — same token that authenticated this request."""
    user = user or require_user()
    token = user.get("_token")
    if not token:
        raise RuntimeError("Authenticated user has no Bearer token attached")

    headers = {
        "accept": "*/*",
        "authorization": f"Bearer {token}",
        "origin": "https://www.auroratest.in",
        "referer": "https://www.auroratest.in/",
    }

    all_campaigns: list[dict] = []
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

    key = str(user["_id"])
    _user_campaigns[key] = all_campaigns
    _user_summary[key] = _build_summary(all_campaigns)
    print(f"Loaded {len(all_campaigns)} campaigns for user {user.get('email')}")


async def _ensure_loaded() -> None:
    key = _user_key()
    if key not in _user_campaigns:
        await fetch_all_campaigns()


def _build_summary(campaigns: list[dict]) -> str:
    """Build an ultra-compact pipe-delimited summary of campaigns for LLM context."""
    lines = ["nm|tp|st|co|bgt|spd|sal|dt"]
    for c in campaigns:
        budget = c.get("budget") or {}
        b = budget.get("amount", "")
        spend = c.get("spend") or {}
        sp = spend.get("amount", "")
        sales = c.get("sales") or {}
        sa = sales.get("amount", "")
        sd = (c.get("startDate") or "")[:10]
        ctype = c.get("campaignType", "")
        if ctype == "Sponsored Products":
            ctype = "SP"
        elif ctype == "Sponsored Brands":
            ctype = "SB"
        elif ctype == "Sponsored Display":
            ctype = "SD"
        status = c.get("status", "")
        smap = {"Enabled": "E", "Paused": "P", "Archived": "A"}
        status = smap.get(status, status)
        name = c.get("campaignName", "").replace("|", "/")
        line = f"{name}|{ctype}|{status}|{c.get('country','')}|{b}|{sp}|{sa}|{sd}"
        lines.append(line)
    return "\n".join(lines)


async def get_campaigns() -> list[dict]:
    await _ensure_loaded()
    return _user_campaigns[_user_key()]


async def get_campaigns_summary() -> str:
    await _ensure_loaded()
    return _user_summary[_user_key()]


async def search_campaigns(query: str) -> str:
    """Search campaigns by name (case-insensitive). Returns matching campaigns."""
    await _ensure_loaded()
    campaigns = _user_campaigns[_user_key()]
    query_lower = query.lower()
    matches = [c for c in campaigns if query_lower in (c.get("campaignName", "")).lower()]
    if not matches:
        return f"No campaigns found matching '{query}'."
    return f"Found {len(matches)} campaign(s) matching '{query}':\n" + _build_summary(matches)


async def analyze_performance_data(full: bool = False) -> dict:
    """Structured campaign performance: overview, top/bottom performers,
    recommendations. When full=True, also includes every campaign with its
    derived ACOS/ROI under `campaigns` — used by the FE table. The LLM tool
    uses full=False so the response stays compact."""
    await _ensure_loaded()
    campaigns = _user_campaigns[_user_key()]
    if not campaigns:
        return {"empty": True}

    top_performers: list[dict] = []
    underperformers: list[dict] = []
    paused_with_spend: list[dict] = []
    all_rows: list[dict] = []

    for c in campaigns:
        spend_amt = float((c.get("spend") or {}).get("amount", 0) or 0)
        sales_amt = float((c.get("sales") or {}).get("amount", 0) or 0)
        budget_amt = float((c.get("budget") or {}).get("amount", 0) or 0)
        name = c.get("campaignName", "Unnamed")
        status = c.get("status", "")
        ctype = c.get("campaignType", "")

        acos_val = None
        roi_val = 0.0
        if spend_amt > 0:
            acos_val = (spend_amt / sales_amt * 100) if sales_amt > 0 else None
            roi_val = ((sales_amt - spend_amt) / spend_amt * 100)

        entry = {
            "name": name,
            "type": ctype,
            "status": status,
            "country": c.get("country", ""),
            "spend": round(spend_amt, 2),
            "sales": round(sales_amt, 2),
            "budget": round(budget_amt, 2),
            "acos": round(acos_val, 1) if acos_val is not None else ("N/A (no sales)" if spend_amt > 0 else None),
            "roi_pct": round(roi_val, 1),
        }
        all_rows.append(entry)

        if spend_amt > 0:
            if sales_amt == 0:
                underperformers.append(entry)
            elif acos_val is not None and acos_val > 40:
                underperformers.append(entry)
            elif acos_val is not None and acos_val < 20 and sales_amt > 0:
                top_performers.append(entry)

            if status == "Paused":
                paused_with_spend.append(entry)

    total_spend = sum(float((c.get("spend") or {}).get("amount", 0) or 0) for c in campaigns)
    total_sales = sum(float((c.get("sales") or {}).get("amount", 0) or 0) for c in campaigns)
    overall_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
    enabled = sum(1 for c in campaigns if c.get("status") == "Enabled")
    paused = sum(1 for c in campaigns if c.get("status") == "Paused")

    analysis: dict = {
        "overview": {
            "total_campaigns": len(campaigns),
            "enabled": enabled,
            "paused": paused,
            "total_spend_usd": round(total_spend, 2),
            "total_sales_usd": round(total_sales, 2),
            "overall_acos_pct": round(overall_acos, 1),
        },
        "top_performers": sorted(top_performers, key=lambda x: x.get("roi_pct", 0), reverse=True)[:5],
        "underperformers": sorted(underperformers, key=lambda x: x.get("spend", 0), reverse=True)[:5],
        "recommendations": [],
    }
    if full:
        analysis["campaigns"] = all_rows

    if enabled == 0 and len(campaigns) > 0:
        analysis["recommendations"].append(
            f"All {len(campaigns)} campaigns are paused or archived — no active campaigns running."
        )
    if paused > 0:
        analysis["recommendations"].append(
            f"{paused} campaign(s) are paused — review and enable ones with good historical performance."
        )
    if underperformers:
        high_spend_losers = [u for u in underperformers if u["acos"] == "N/A (no sales)"]
        if high_spend_losers:
            analysis["recommendations"].append(
                f"{len(high_spend_losers)} campaign(s) have spend but ZERO sales — consider pausing or revising keywords."
            )
        high_acos = [u for u in underperformers if isinstance(u["acos"], (int, float)) and u["acos"] > 40]
        if high_acos:
            analysis["recommendations"].append(
                f"{len(high_acos)} campaign(s) have ACOS > 40% — review bids and targeting."
            )
    if top_performers:
        analysis["recommendations"].append(
            f"{len(top_performers)} campaign(s) performing well (ACOS < 20%) — consider increasing budget."
        )
    if overall_acos > 30:
        analysis["recommendations"].append(
            f"Overall ACOS is {round(overall_acos, 1)}% — above healthy threshold. Review underperformers."
        )
    if total_spend == 0 and len(campaigns) > 0:
        analysis["recommendations"].append(
            "No spend recorded across any campaign — campaigns may need to be enabled and given budget to start generating data."
        )

    return analysis


async def analyze_performance() -> str:
    """Markdown/JSON wrapper used by the LLM tool — compact (no full campaign list)."""
    import json as _json
    data = await analyze_performance_data(full=False)
    if data.get("empty"):
        return "No campaign data available."
    return _json.dumps(data)


def _extract_id(resp: dict, section_key: str, id_field: str) -> str:
    """Pull a created resource id out of an SP v3 batch-create response."""
    section = resp.get(section_key)
    if isinstance(section, dict):
        success = section.get("success", [])
        if success:
            return str(success[0].get(id_field, ""))
        errors = section.get("error", [])
        if errors:
            raise RuntimeError(f"{section_key} creation failed: {errors}")
    elif isinstance(section, list) and section:
        return str(section[0].get(id_field, ""))
    return ""


async def create_campaign(payload: dict) -> str:
    """Create a real Sponsored Products campaign via Amazon Ads API.

    Flow: campaign -> ad group -> product ad (SKU/ASIN) -> keywords -> negatives.
    """
    from datetime import datetime
    from amazon_ads import (
        create_sp_campaign,
        create_ad_group,
        create_product_ad,
        add_keywords,
        add_negative_keywords,
    )

    name = payload.get("campaign_name", "Untitled")
    budget = float(payload.get("budget", 10))
    keywords = payload.get("keywords", [])
    negative_kws = payload.get("negative_keywords", [])
    sku = payload.get("sku") or None
    asin = payload.get("asin") or None
    start_date = datetime.utcnow().strftime("%Y-%m-%d")

    print(
        f"[create_campaign] START name={name!r} budget={budget} "
        f"keywords={len(keywords)} negatives={len(negative_kws)} "
        f"sku={sku!r} asin={asin!r}"
    )

    try:
        print(f"[create_campaign] -> creating SP campaign {name!r}...")
        camp_resp = await create_sp_campaign(name, budget, start_date)
        campaign_id = _extract_id(camp_resp, "campaigns", "campaignId")
        if not campaign_id:
            print(f"[create_campaign] FAILED — no campaign id in response: {camp_resp}")
            return f"Failed to create campaign: {camp_resp}"
        print(f"[create_campaign] <- campaign created. campaignId={campaign_id}")

        print(f"[create_campaign] -> creating ad group for campaign {campaign_id}...")
        ag_resp = await create_ad_group(campaign_id, f"{name} - Ad Group")
        ad_group_id = _extract_id(ag_resp, "adGroups", "adGroupId")
        if not ad_group_id:
            print(f"[create_campaign] FAILED — no ad group id in response: {ag_resp}")
            return (
                f"Campaign '{name}' created (ID {campaign_id}), but ad group creation "
                f"failed: {ag_resp}"
            )
        print(f"[create_campaign] <- ad group created. adGroupId={ad_group_id}")

        results = [f"Campaign '{name}' created. Campaign ID: {campaign_id}"]

        if sku or asin:
            target = f"SKU {sku}" if sku else f"ASIN {asin}"
            print(f"[create_campaign] -> creating product ad ({target}) on ad group {ad_group_id}...")
            try:
                pa_resp = await create_product_ad(campaign_id, ad_group_id, sku=sku, asin=asin)
                pa_id = _extract_id(pa_resp, "productAds", "adId")
                print(f"[create_campaign] <- product ad created. adId={pa_id or '(unknown)'} resp={pa_resp}")
                results.append(
                    f"Linked product ad for {target}." if pa_id
                    else f"Product ad request sent for {target} (response: {pa_resp})."
                )
            except Exception as e:
                print(f"[create_campaign] !! product ad failed: {e}")
                results.append(f"Warning: product ad could not be created ({e}). The campaign will not serve until a product ad is added.")
        else:
            print("[create_campaign] -- no SKU/ASIN provided, skipping product ad")
            results.append("No SKU/ASIN provided — campaign will NOT serve until a product ad is added.")

        if keywords:
            print(f"[create_campaign] -> adding {len(keywords)} keyword(s)...")
            await add_keywords(campaign_id, ad_group_id, keywords)
            print(f"[create_campaign] <- keywords added")
            results.append(f"Added {len(keywords)} keyword(s).")

        if negative_kws:
            print(f"[create_campaign] -> adding {len(negative_kws)} negative keyword(s)...")
            await add_negative_keywords(campaign_id, ad_group_id, negative_kws)
            print(f"[create_campaign] <- negative keywords added")
            results.append(f"Added {len(negative_kws)} negative keyword(s).")

        print(f"[create_campaign] DONE campaignId={campaign_id}")
        return " ".join(results)

    except Exception as e:
        print(f"[create_campaign] ERROR: {e}")
        return f"Error creating campaign on Amazon: {e}"
