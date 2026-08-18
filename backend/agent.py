import asyncio
import inspect
import json
import os
import re
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from groq import AsyncGroq

from campaigns import get_campaigns_summary, create_campaign, analyze_performance, search_campaigns
from keywords import fetch_amazon_keywords, suggest_negative_keywords
from meta_ads import search_meta_ads
from database import create_session, get_messages, save_message, update_session_title, get_cogs, get_forecast_cache
import amazon_sp
import aurora_data
from aurora_data import aurora_db_enabled
import data_resolver
from auth import require_user
from marketplace_timezone import (
    inclusive_ymd_bounds,
    parse_date_range_for_query,
    parse_ymd_parts,
    resolve_dashboard_timezone,
    resolve_window_from_days_back,
    utc_instant_to_iso_z,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"

client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a campaign analyst and creation assistant for Amazon advertising. "
    "You can fetch campaign data, analyze performance, suggest keywords, and help users create new campaigns.\n\n"
    "## Tools at your disposal\n"
    "- **get_campaigns_summary**: Fetch campaign data. Pass a query parameter to filter by name (e.g. query='rug'). Omit query to get all.\n"
    "- **get_keywords**: Fetch keyword suggestions from Amazon Autocomplete for a seed keyword.\n"
    "- **get_negative_keywords**: Get negative keyword suggestions to exclude wasteful/irrelevant terms from a campaign.\n"
    "- **analyze_campaign_performance**: Analyze campaign health — ACOS, ROI, and actionable recommendations.\n"
    "- **create_campaign**: Create a campaign with a product ad (only after user approves keywords, ASIN/SKU & details).\n\n"
    "## Campaign creation flow (STRICT — follow every step)\n"
    "1. Ask about the product or seed keyword.\n"
    "2. Call get_keywords to fetch suggestions.\n"
    "3. **ALWAYS list the fetched keywords in your response** so the user can see them. Never skip this step.\n"
    "4. Ask the user which keywords to keep and whether to fetch negative keywords.\n"
    "5. Collect campaign details (name, type, budget, country) if not already provided.\n"
    "6. **ALWAYS ask the user for the product's seller SKU (merchant SKU).** This is a SELLER account, so the product ad MUST use the SKU — an ASIN will be rejected. A campaign cannot serve ads without a product ad, so you MUST collect the SKU before creating. Explain this if the user is unsure.\n"
    "7. **NEVER call create_campaign until the user explicitly approves the keywords, the SKU, and the campaign details.** When you call create_campaign, pass the approved sku (not asin).\n\n"
    "## Performance analysis\n"
    "When users ask about campaign performance, health, or what to optimize, call analyze_campaign_performance.\n"
    "IMPORTANT: analyze_campaign_performance returns an ACCOUNT-WIDE summary — it does NOT filter by campaign name. "
    "If the user asked about a specific campaign and the response's top_performers/underperformers don't mention it, "
    "do NOT say 'no data available for that campaign'. Instead, present the account totals and tell the user that "
    "per-campaign drill-down isn't supported by this tool yet.\n\n"
    "## Competitor research\n"
    "- **search_meta_ads**: Search the public Meta (Facebook) Ad Library for competitor ads. "
    "Use this when the user asks about competitor ads, wants to see what others are advertising, "
    "or wants ad copy inspiration. Pass a product/brand query and optionally a country code.\n\n"
    "## Selling Partner API (Orders, Inventory, Reports)\n"
    "- **get_orders**: Fetch recent Amazon orders. Use when the user asks about order status, recent sales, shipments, or order history. "
    "Pass days_back (default 7) and optionally a status filter (Unshipped, Shipped, PartiallyShipped, Canceled, Unfulfillable).\n"
    "- **get_order_items**: Fetch SKU-level line items (SKU, ASIN, price, qty, promo) for a single order id. Use whenever you need per-SKU revenue.\n"
    "- **get_inventory**: Check FBA inventory levels. Use when the user asks about stock, inventory, or supply levels. "
    "Optionally pass a list of SKUs to filter.\n"
    "- **get_cogs**: Look up unit_cost and inbound_shipping_per_unit per SKU from the user-supplied COGS table. "
    "Amazon's APIs do NOT expose what the seller paid to make or ship product, so this is the only source of cost data.\n"
    "- **get_report**: Generate and download an Amazon report. Use when the user asks for sales reports, settlement reports, "
    "or other data exports. Profitability-relevant types: GET_V2_SETTLEMENT_REPORT_DATA_FLAT_FILE (actual fees deducted), "
    "GET_FBA_ESTIMATED_FBA_FEES_TXT_DATA (per-SKU referral + FBA fee estimates), "
    "GET_FBA_STORAGE_FEE_CHARGES_DATA (monthly storage), "
    "GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA (aged surcharge charges), "
    "GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA (removal fees).\n\n"
    "## Profitability analysis\n"
    "When the user asks about profit, margin, P&L — or asks for BOTH 'performance AND profitability' (or 'performance AND profit') — "
    "call **analyze_profitability(days_back=N)** (N defaults to 7) WITHOUT asking permission first. "
    "For a compound 'performance and profitability' ask, call BOTH analyze_campaign_performance AND analyze_profitability in the same turn. "
    "The tool returns a fully-formatted markdown table + summary + caveats — present that text VERBATIM to the user, optionally with one short intro line. "
    "Do NOT recompute the math. Do NOT call get_orders / get_order_items / get_cogs manually — analyze_profitability already does all of that server-side. "
    "Do NOT add commentary that contradicts the table.\n"
    "If the user asks for fees / ad spend / fully-loaded mode, tell them that mode isn't built yet and offer the simple table.\n\n"
    "## Inventory forecasting / restock\n"
    "When the user asks about future demand, restock planning, when to reorder, days of cover, PO quantities, "
    "or whether they'll stock out:\n"
    "- **forecast_sku(sku, horizon_days)** — single-SKU forecast + reorder math\n"
    "- **restock_recommendations(top_n)** — ranked list of SKUs to reorder next\n"
    "- **days_until_stockout(sku)** — quick triage; omit sku for a riskiest-SKUs list\n"
    "These read a nightly-refreshed cache; they do NOT refit on the fly. If the response says 'no cached forecast', "
    "tell the user to hit POST /forecasting/refresh (or wait for the 03:00 UTC nightly job). "
    "Present the markdown the tool returns verbatim — do NOT recompute the math or invent SKUs not in the response.\n\n"
    "Be concise and actionable."
)

# ── Tool definitions ────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_campaigns_summary",
            "description": "Fetch campaign data. If query is provided, returns only campaigns matching that name (e.g. query='rug' for rug campaigns). If query is omitted, returns all campaigns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional search term to filter campaigns by name (e.g. 'rug', 'leather', 'incense'). Omit to get all campaigns.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_keywords",
            "description": "Fetch keyword suggestions from Amazon Autocomplete for a given seed keyword. Call this when the user wants keyword ideas for a campaign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seed_keyword": {
                        "type": "string",
                        "description": "The product or seed keyword (e.g. 'wireless earbuds').",
                    },
                },
                "required": ["seed_keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_negative_keywords",
            "description": "Get negative keyword suggestions — irrelevant or wasteful terms to exclude from a campaign. Call this to help reduce wasted ad spend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seed_keyword": {
                        "type": "string",
                        "description": "The product keyword to find negatives for (e.g. 'wireless earbuds').",
                    },
                },
                "required": ["seed_keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_campaign_performance",
            "description": "Analyze campaign performance: calculates ACOS, identifies top/bottom performers, and gives optimization recommendations. Call when user asks about performance, health, or what to optimize.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_campaign",
            "description": "Create a new advertising campaign. Only call this after the user has approved the keywords and provided campaign details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign_name": {
                        "type": "string",
                        "description": "Name of the campaign.",
                    },
                    "campaign_type": {
                        "type": "string",
                        "description": "Type: 'Sponsored Products', 'Sponsored Brands', or 'Sponsored Display'.",
                        "enum": ["Sponsored Products", "Sponsored Brands", "Sponsored Display"],
                    },
                    "budget": {
                        "type": "number",
                        "description": "Daily budget in USD.",
                    },
                    "country": {
                        "type": "string",
                        "description": "Target country code (e.g. 'US', 'IN').",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of approved keywords for the campaign.",
                    },
                    "negative_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of negative keywords to exclude from the campaign.",
                    },
                    "sku": {
                        "type": "string",
                        "description": "Seller/merchant SKU of the product to advertise. REQUIRED for seller accounts (Sponsored Products product ads on seller accounts must use the SKU). This is what makes the campaign actually serve.",
                    },
                    "asin": {
                        "type": "string",
                        "description": "ASIN of the product (e.g. 'B08XXXXXXX'). Only valid for VENDOR accounts. Seller accounts must use sku instead.",
                    },
                },
                "required": ["campaign_name", "campaign_type", "budget", "country", "keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_meta_ads",
            "description": "Search the public Meta (Facebook) Ad Library for competitor ads. Returns ad copy, advertiser names, platforms, and dates. Use for competitor research and ad copy inspiration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — product name, brand, or keyword (e.g. 'wireless earbuds', 'Nike').",
                    },
                    "country": {
                        "type": "string",
                        "description": "Two-letter country code (default 'US').",
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "If true (default), only return currently active ads.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "Fetch recent Amazon orders from the Selling Partner API. By default searches across ALL marketplaces the user is registered in; pass `marketplace` to scope to one (or several) — e.g. 'US' or 'A1F83G8C2ARO7P' or 'US,UK'. Returns order IDs, dates, statuses, and totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": ["integer", "string"],
                        "description": "Number of days to look back (default 7). Must be a number — '7' or 7 both fine.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by order status. One of: Unshipped, PartiallyShipped, Shipped, Canceled, Unfulfillable. Omit to get all.",
                    },
                    "marketplace": {
                        "type": ["string", "null"],
                        "description": "Optional. Marketplace id (e.g. ATVPDKIKX0DER), short name ('US', 'UK', 'UAE'), or comma-separated list. Omit / null to search all the user's marketplaces. Call get_marketplaces first if you're unsure what's available.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory",
            "description": "Check FBA inventory levels via the Selling Partner API. Returns fulfillable, inbound, reserved, and unfulfillable quantities per SKU. SP-API requires a single marketplace per call — defaults to the user's primary (US-preferred); pass `marketplace` to override.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skus": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional list of seller SKUs to filter. Omit, pass null, or pass [] to get ALL inventory.",
                    },
                    "marketplace": {
                        "type": ["string", "null"],
                        "description": "Optional. One marketplace id or short name ('US', 'UK', 'MX', etc). Omit / null for the user's primary marketplace.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_items",
            "description": "Fetch line items (SKU, ASIN, item price, quantity, promo discount) for a single Amazon order. Use this to break an order down to SKU-level revenue. NOT for SKUs/ASINs — those go to get_inventory or get_orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": ["string", "object"],
                        "description": "The AmazonOrderId returned by get_orders. Format is exactly '3-7-7' digits (e.g. '114-4871996-9329822'). Do NOT pass a SKU or ASIN here.",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cogs",
            "description": "Look up unit cost and inbound shipping cost per SKU from the user-supplied COGS table. Returns rows with sku, unit_cost, and inbound_shipping_per_unit. Required input for profitability — Amazon's APIs do not expose what the seller paid to acquire/ship product. If a SKU is missing, the user has not uploaded its cost yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skus": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional list of seller SKUs. Omit, pass null, or pass [] to get all stored COGS rows.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_profitability",
            "description": (
                "Compute per-SKU profitability for recent orders using the full "
                "Amazon FBA Profitability Calculator formula. Pulls orders, order "
                "items, Product Fees API (referral / FBA / fuel surcharge per ASIN), "
                "FBA storage report (per-ASIN monthly fee allocated to the window), "
                "Aurora ad spend (pro-rated, uniform per unit), and the user's "
                "uploaded COGS. Returns a markdown table with the full per-SKU fee "
                "breakdown — referral, FBA, fuel, storage, ad cost, product cost, "
                "inbound shipping, net, margin — plus totals and methodology caveats. "
                "Use this for ANY profit / margin / P&L question — do not chain "
                "get_orders + get_order_items + get_cogs manually. "
                "Note: returns processing fees, aged-inventory surcharge, and "
                "low-inventory level fee are not yet computed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": ["integer", "string"],
                        "description": "Days of order history to include (default 7).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_report",
            "description": "Generate and download an Amazon Seller report. Creates the report, waits for processing, and returns the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "description": "The SP-API report type identifier.",
                        "enum": [
                            "GET_FLAT_FILE_ALL_ORDERS_DATA_BY_ORDER_DATE_GENERAL",
                            "GET_FBA_MYI_UNSUPPRESSED_INVENTORY_DATA",
                            "GET_V2_SETTLEMENT_REPORT_DATA_FLAT_FILE",
                            "GET_SALES_AND_TRAFFIC_REPORT",
                            "GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA",
                            "GET_FBA_ESTIMATED_FBA_FEES_TXT_DATA",
                            "GET_FBA_STORAGE_FEE_CHARGES_DATA",
                            "GET_FBA_INVENTORY_AGED_DATA",
                        ],
                    },
                    "days_back": {
                        "type": ["integer", "string"],
                        "description": "Number of days to cover (default 30). Must be a number — '30' or 30 both fine.",
                    },
                },
                "required": ["report_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_marketplaces",
            "description": "List the Amazon marketplaces the user is registered in (id + human-readable country name + which is primary). Call this when the user asks 'which marketplaces do I have?', or before scoping another tool (get_orders / get_inventory) to a specific country.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_sku",
            "description": "Forecast future demand for one SKU. Returns the next N-day p50/p90 demand, the model used (prophet / prophet+ads / naive), drivers (recent_avg, growth_rate), and the reorder math (days of cover, reorder-by date, suggested PO qty). Reads from the cached forecast — does NOT refit. If no cache exists for the SKU, tell the user to run /forecasting/refresh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Seller SKU."},
                    "horizon_days": {
                        "type": "integer",
                        "description": "Days of forecast to summarize (default 30, max 90).",
                    },
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restock_recommendations",
            "description": "Top SKUs that need a PO soon, ranked by stockout urgency (lowest days-of-cover first). Returns a markdown table with on-hand, inbound, days of cover, reorder-by date, and suggested PO quantity. Use when the user asks 'what should I reorder?', 'what's running low?', or wants a restock plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "How many SKUs to show (default 10).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "days_until_stockout",
            "description": "Quick triage — days of cover remaining per SKU. Pass a SKU for one answer, omit for a sorted list of the riskiest SKUs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Optional. Single SKU to check."},
                },
            },
        },
    },
]

# ── Tool function wrappers ──────────────────────────────────────────────────

async def _get_keywords(seed_keyword: str) -> str:
    keywords = await fetch_amazon_keywords(seed_keyword)
    if not keywords:
        return "No keyword suggestions found. Try a different seed keyword."
    return "Keywords found:\n" + "\n".join(f"- {kw}" for kw in keywords)


async def _get_negative_keywords(seed_keyword: str) -> str:
    negatives = await suggest_negative_keywords(seed_keyword)
    if not negatives:
        return "No negative keyword suggestions found."
    lines = [f"- {n['keyword']} ({n['reason']})" for n in negatives[:20]]
    return "Negative keyword suggestions:\n" + "\n".join(lines)


async def _create_campaign(
    campaign_name: str,
    campaign_type: str,
    budget: float,
    country: str,
    keywords: list[str],
    negative_keywords: list[str] | None = None,
    sku: str | None = None,
    asin: str | None = None,
) -> str:
    return await create_campaign({
        "campaign_name": campaign_name,
        "campaign_type": campaign_type,
        "budget": budget,
        "country": country,
        "keywords": keywords,
        "negative_keywords": negative_keywords or [],
        "sku": sku,
        "asin": asin,
    })


async def _analyze_campaign_performance() -> str:
    return await analyze_performance()


async def _search_meta_ads(query: str, country: str = "US", active_only: bool = True) -> str:
    return await search_meta_ads(query, country=country, active_only=active_only)


async def _get_campaigns_summary(query: str = "") -> str:
    if query:
        return await search_campaigns(query)
    return await get_campaigns_summary()


async def _get_orders(days_back: int | str = 7, status: str | None = None, marketplace=None) -> str:
    try:
        days_back = int(days_back)
    except (TypeError, ValueError):
        days_back = 7
    valid_statuses = {"Unshipped", "PartiallyShipped", "Shipped", "Canceled", "Unfulfillable"}
    statuses = [status] if status in valid_statuses else None
    created_after = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        data = await data_resolver.list_orders_resolved(
            require_user(),
            created_after=created_after,
            statuses=statuses,
            marketplace=marketplace,
            paginate=False,
        )
        orders = data.get("orders") or []
    except Exception as e:
        return f"Error fetching orders: {e}"
    if not orders:
        return f"No orders found in the last {days_back} days."
    lines = []
    for o in orders[:30]:
        total = o.get("OrderTotal", {})
        total_str = f"{total.get('CurrencyCode', '')} {total.get('Amount', 'N/A')}" if total else "N/A"
        lines.append(
            f"- {o.get('AmazonOrderId')} | {o.get('PurchaseDate', '')[:10]} | "
            f"{o.get('OrderStatus')} | {total_str} | {o.get('NumberOfItemsUnshipped', 0)} unshipped"
        )
    return f"Orders (last {days_back} days, {len(orders)} total):\n" + "\n".join(lines)


async def _get_inventory(skus=None, marketplace=None) -> str:
    # Models occasionally emit `skus={}` or `skus="ABC,DEF"` instead of a list.
    # Coerce to a clean list-or-None so the SP-API call doesn't blow up.
    if isinstance(skus, str):
        skus = [s.strip() for s in skus.split(",") if s.strip()]
    elif not isinstance(skus, list):
        skus = None
    if skus is not None and not skus:
        skus = None
    try:
        summaries, _src = await data_resolver.fetch_inventory_resolved(
            require_user(), skus=skus, marketplace=marketplace,
        )
    except Exception as e:
        return f"Error fetching inventory: {e}"
    if not summaries:
        return "No FBA inventory found." + (" (filtered by SKUs: " + ", ".join(skus) + ")" if skus else "")
    lines = []
    for s in summaries[:50]:
        inv = s.get("inventoryDetails", {})
        lines.append(
            f"- {s.get('sellerSku', '?')} (ASIN: {s.get('asin', '?')}) | "
            f"Fulfillable: {inv.get('fulfillableQuantity', 0)} | "
            f"Inbound: {inv.get('inboundWorkingQuantity', 0)}+{inv.get('inboundShippedQuantity', 0)} | "
            f"Reserved: {inv.get('reservedQuantity', {}).get('totalReservedQuantity', 0)} | "
            f"Unfulfillable: {inv.get('unfulfillableQuantity', {}).get('totalUnfulfillableQuantity', 0)}"
        )
    return f"FBA Inventory ({len(summaries)} SKUs):\n" + "\n".join(lines)


_AMAZON_ORDER_ID_RE = __import__("re").compile(r"^\d{3}-\d{7}-\d{7}$")


async def _get_order_items(order_id=None) -> str:
    # Models sometimes wrap the id in an object like {"order_id": "..."}
    # or {"sku": "..."}. Coerce to a string and validate the shape.
    if isinstance(order_id, dict):
        order_id = order_id.get("order_id") or order_id.get("AmazonOrderId") or next(iter(order_id.values()), "")
    order_id = str(order_id or "").strip()
    if not order_id:
        return "Error: get_order_items needs an order_id. Get one from get_orders first."
    if not _AMAZON_ORDER_ID_RE.match(order_id):
        return (
            f"Error: '{order_id}' is not an AmazonOrderId (expected '3-7-7' digit format, "
            f"e.g. '114-4871996-9329822'). If this is a SKU or ASIN, call get_orders to "
            "find which orders contain it, then call get_order_items on those order ids."
        )
    try:
        data = await data_resolver.get_order_items_resolved(require_user(), order_id)
        items = data.get("items") or []
    except Exception as e:
        return f"Error fetching order items: {e}"
    if not items:
        return f"No items found for order {order_id}."
    lines = []
    for it in items:
        price = it.get("ItemPrice") or {}
        promo = it.get("PromotionDiscount") or {}
        lines.append(
            f"- {it.get('SellerSKU', '?')} (ASIN: {it.get('ASIN', '?')}) | "
            f"qty: {it.get('QuantityOrdered', 0)} | "
            f"price: {price.get('CurrencyCode', '')} {price.get('Amount', 'N/A')} | "
            f"promo: {promo.get('CurrencyCode', '')} {promo.get('Amount', '0.00')}"
        )
    return f"Order {order_id} items ({len(items)}):\n" + "\n".join(lines)


async def _get_cogs(skus=None) -> str:
    if isinstance(skus, str):
        skus = [s.strip() for s in skus.split(",") if s.strip()]
    elif not isinstance(skus, list):
        skus = None
    if skus is not None and not skus:
        skus = None
    try:
        rows = await get_cogs(skus)
    except Exception as e:
        return f"Error reading COGS: {e}"
    if not rows:
        if skus:
            return f"No COGS found for: {', '.join(skus)}. User must upload them via the COGS panel."
        return "No COGS uploaded yet. User must upload a CSV via the COGS panel before profit can be computed."
    lines = [
        f"- {r['sku']} | unit_cost: {r['unit_cost']} | inbound_shipping_per_unit: {r['inbound_shipping_per_unit']}"
        for r in rows
    ]
    out = f"COGS ({len(rows)} SKU{'s' if len(rows) != 1 else ''}):\n" + "\n".join(lines)
    if skus:
        missing = sorted(set(skus) - {r["sku"] for r in rows})
        if missing:
            out += f"\nMissing COGS for: {', '.join(missing)} (user must upload)."
    return out


def _sp_report_warning(report_label: str, err: Exception) -> str:
    """User-facing message for SP-API report failures (no raw stack traces)."""
    text = f"{err!r}"
    if "403" in text or "Forbidden" in text:
        return (
            f"{report_label} report unavailable — Amazon denied access (403). "
            "Your SP-API app may need the Reports role enabled in Seller Central, "
            "or this report type isn't enabled for your account. "
            "Using settlement-window estimates instead."
        )
    short = str(err).replace("\n", " ")
    if len(short) > 160:
        short = short[:157] + "..."
    return f"{report_label} report failed ({type(err).__name__}: {short})"


def _is_sp_access_denied(err: Exception) -> bool:
    text = f"{err!r}"
    return "403" in text or "Forbidden" in text


def _lookup_rate(by_sku: dict | None, by_asin: dict | None, sku: str, asin: str | None):
    """Look up a per-unit/monthly rate by seller SKU (case-insensitive) or ASIN."""
    if by_sku:
        if sku in by_sku:
            return by_sku[sku]
        sku_l = sku.lower()
        for key, val in by_sku.items():
            if str(key).lower() == sku_l:
                return val
    if by_asin and asin:
        return by_asin.get(str(asin).upper())
    return None


def _build_placement_rates(placement_per_sku: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Return (by_sku, by_asin) maps of fee-bearing placement $/unit."""
    by_sku: dict[str, float] = {}
    by_asin: dict[str, float] = {}
    for psku, b in (placement_per_sku or {}).items():
        if not isinstance(b, dict):
            continue
        # Prefer explicit fee_rate (Seller Central fee-rate column).
        rate = float(b.get("fee_rate") or 0)
        if rate <= 0:
            fee = float(b.get("fee_total") or 0)
            bearing = int(b.get("fee_bearing_units") or 0)
            received = int(b.get("units_received") or 0)
            denom = bearing if bearing > 0 else received
            if denom <= 0 or fee <= 0:
                continue
            rate = fee / denom
        if rate <= 0:
            continue
        by_sku[psku] = rate
        asin = (b.get("asin") or "").strip().upper()
        if asin:
            by_asin[asin] = max(by_asin.get(asin, 0.0), rate)
        if (psku or "").upper().startswith("B") and len(psku) == 10:
            by_asin[psku.upper()] = max(by_asin.get(psku.upper(), 0.0), rate)
    return by_sku, by_asin


async def _placement_rates_from_finances_join(use_db: bool) -> dict:
    """When the placement report is blocked (403), rebuild per-SKU rates from
    Finances shipment-level placement charges joined with Aurora shipments."""
    if not use_db:
        return {}
    from database import put_placement_fee_cache
    fees_by_shipment = await amazon_sp.fetch_placement_service_fees_by_shipment()
    placement_per_sku = await aurora_data.placement_rates_from_shipments(
        require_user(), fees_by_shipment,
    )
    await put_placement_fee_cache(placement_per_sku, [], access_denied=True)
    return placement_per_sku


def _build_aged_charges(charges_per_sku: dict) -> dict[str, float]:
    """SKU → actual amount-charged from the Aged Inventory Surcharge report.

    SKU-only (no ASIN map): applying the same ASIN total to every listing
    SKU (e.g. merchant + amzn.gr.*) double-counts the fee in Profitability.
    """
    by_sku: dict[str, float] = {}
    for asku, b in (charges_per_sku or {}).items():
        if not isinstance(b, dict):
            fee = float(b or 0)
        else:
            fee = float(b.get("charged_total") or 0)
        if fee > 0:
            by_sku[asku] = fee
    return by_sku


def _build_removal_fees(charges_per_sku: dict) -> dict[str, float]:
    """SKU → removal-fee from Seller Central Removal Order Detail."""
    by_sku: dict[str, float] = {}
    for asku, b in (charges_per_sku or {}).items():
        if not isinstance(b, dict):
            fee = float(b or 0)
        else:
            fee = float(b.get("removal_fee") or 0)
        if fee > 0:
            by_sku[str(asku)] = fee
    return by_sku


def _lookup_sku_amount(by_sku: dict[str, float] | None, sku: str) -> float:
    """Case-insensitive SKU lookup for absolute fee amounts."""
    if not by_sku:
        return 0.0
    if sku in by_sku:
        return float(by_sku[sku] or 0)
    sku_l = sku.lower()
    for key, val in by_sku.items():
        if str(key).lower() == sku_l:
            return float(val or 0)
    return 0.0


def _sp_item_line_revenue(item: dict) -> float | None:
    """Net line revenue from SP-API OrderItems (ItemPrice − PromotionDiscount)."""
    price = item.get("ItemPrice") or {}
    promo = item.get("PromotionDiscount") or {}
    try:
        amount_raw = price.get("Amount")
        if amount_raw is None:
            return None
        amount = float(amount_raw)
    except (TypeError, ValueError):
        return None
    try:
        promo_amt = float(promo.get("Amount") or 0)
    except (TypeError, ValueError):
        promo_amt = 0.0
    return max(0.0, amount - promo_amt)


# SP-API GetOrderItems sustained quota — Amazon's published figure. The
# per-slot pacing sleep is derived from this so bumping `concurrency`
# can't unintentionally push the effective sustained rate past quota.
SP_API_ORDER_ITEMS_SUSTAINED_REQ_PER_SEC = 0.5


def _pacing_sleep_for(concurrency: int, attempt: int) -> float:
    """Per-slot sleep that keeps `concurrency / sleep` at the sustained
    quota (with `attempt`-based backoff multiplier).

    Verification-review followup: the first fix used a fixed 2.1s per-slot
    sleep for every concurrency setting, which meant `concurrency=5`
    produced ~2.4 req/s — 5× quota. The formula below guarantees the
    effective sustained rate never exceeds
    SP_API_ORDER_ITEMS_SUSTAINED_REQ_PER_SEC regardless of `concurrency`.
    Attempt N applies an N× backoff, matching the original serial path."""
    concurrency = max(1, concurrency)
    return (concurrency / SP_API_ORDER_ITEMS_SUSTAINED_REQ_PER_SEC) * attempt


async def _fetch_order_items_paced(
    order_ids: list[str],
    concurrency: int = 5,
) -> list[tuple[str, dict]]:
    """SP-API GetOrderItems with bounded concurrency + backoff on 429.

    Review issue #3: originally serial with 2.1s per-call sleep, which made
    the fallback path effectively unusable for sellers with hundreds of
    unsynced orders. Now runs up to `concurrency` calls in flight against
    a shared asyncio.Semaphore. Per-slot pacing is computed by
    `_pacing_sleep_for` so effective sustained rate never exceeds
    SP-API's per-account quota (see review-verification followup — the
    prior implementation used a fixed 2.1s sleep and ran ~5× quota).

    The practical win is burst latency, not sustained throughput: a small
    window of `<= concurrency` orders finishes in ~1 call latency instead
    of `N × base_sleep`, while larger windows converge on the same total
    wall-clock the original serial path took. `concurrency` is a parameter
    so tests can pin it to a known value."""
    items_results: list[tuple[str, dict]] = []
    results_lock = asyncio.Lock()
    pending = list(order_ids)
    attempt = 1
    concurrency = max(1, concurrency)
    while pending and attempt <= 4:
        per_slot_sleep = _pacing_sleep_for(concurrency, attempt)
        sem = asyncio.Semaphore(concurrency)
        failed: list[str] = []
        failed_lock = asyncio.Lock()

        async def fetch_one(oid: str) -> None:
            async with sem:
                try:
                    resp = await amazon_sp.get_order_items(oid)
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "QuotaExceeded" in msg:
                        async with failed_lock:
                            failed.append(oid)
                        # 429 backoff: double the per-slot sleep for
                        # this slot only; asyncio.Semaphore holds the
                        # slot for that duration, throttling naturally.
                        await asyncio.sleep(per_slot_sleep * 2)
                        return
                    async with results_lock:
                        items_results.append((oid, {"_error": msg}))
                    return
                async with results_lock:
                    items_results.append((oid, resp))
                # Per-slot pacing: with `concurrency` slots each sleeping
                # `per_slot_sleep` after every call, effective sustained
                # rate is `concurrency / per_slot_sleep` req/s, which by
                # construction equals SP_API_ORDER_ITEMS_SUSTAINED_REQ_PER_SEC.
                await asyncio.sleep(per_slot_sleep)

        await asyncio.gather(*(fetch_one(oid) for oid in pending))
        pending = failed
        attempt += 1
        if pending:
            await asyncio.sleep(20)
    for oid in pending:
        items_results.append((oid, {"_error": "rate limited after 4 retries"}))
    return items_results


def _aggregate_sku_from_sp_api(
    orders: list[dict],
    items_results: list[tuple[str, dict]],
    fetch_errors: list[str],
    na_price_rows: list[dict],
) -> tuple[dict[str, dict], int]:
    """Build per-SKU units/revenue; skip Canceled / Cancelled / Unfulfillable orders."""
    excluded_ids = {
        oid
        for o in orders
        if (oid := o.get("AmazonOrderId"))
        and aurora_data.is_excluded_order_status(o.get("OrderStatus"))
    }
    orders_count = sum(
        1
        for o in orders
        if o.get("AmazonOrderId") and o.get("AmazonOrderId") not in excluded_ids
    )
    sku_data: dict[str, dict] = defaultdict(
        lambda: {"units": 0, "revenue": 0.0, "asin": None},
    )
    for oid, items_resp in items_results:
        if oid in excluded_ids:
            continue
        if "_error" in items_resp:
            fetch_errors.append(f"{oid}: {items_resp['_error']}")
            continue
        for it in items_resp.get("payload", {}).get("OrderItems", []):
            sku = it.get("SellerSKU")
            if not sku:
                continue
            qty = int(it.get("QuantityOrdered", 0) or 0)
            amount = _sp_item_line_revenue(it)
            if amount is None:
                na_price_rows.append({"order_id": oid, "sku": sku, "qty": qty})
                continue
            if amount <= 0 and qty > 0:
                na_price_rows.append({"order_id": oid, "sku": sku, "qty": qty})
                continue
            sku_data[sku]["units"] += qty
            sku_data[sku]["revenue"] += amount
            if not sku_data[sku]["asin"] and it.get("ASIN"):
                sku_data[sku]["asin"] = it["ASIN"]
    return sku_data, orders_count


def _lookup_returns_for_sku(
    returns_by_sku: dict[str, dict],
    sku: str,
) -> dict | None:
    """Exact then case-insensitive SKU match (no ASIN fallback)."""
    ret = returns_by_sku.get(sku)
    if ret or not sku:
        return ret
    sku_l = sku.lower()
    return next(
        (v for k, v in returns_by_sku.items() if str(k).lower() == sku_l),
        None,
    )


def _apply_returns_to_sku_data(
    sku_data: dict[str, dict],
    returns_by_sku: dict[str, dict],
) -> None:
    """Net units/revenue/line fees by returned quantity (in-place).

    Units become ordered − returned. Revenue and Aurora order-line
    referral/FBA totals scale by kept/ordered so Referral, FBA, Fuel,
    COGS, and per-unit allocations all follow remaining quantity.
    """
    if not returns_by_sku or not sku_data:
        return
    for sku, d in sku_data.items():
        ordered = int(d.get("units") or 0)
        if ordered <= 0:
            continue
        ret = _lookup_returns_for_sku(returns_by_sku, sku)
        returned = int((ret or {}).get("returned_units") or 0)
        if returned <= 0:
            continue
        returned = min(returned, ordered)
        kept = ordered - returned
        ratio = kept / ordered
        d["units"] = kept
        d["revenue"] = float(d.get("revenue") or 0.0) * ratio
        if "referral_total" in d:
            d["referral_total"] = float(d.get("referral_total") or 0.0) * ratio
        if "fba_total" in d:
            d["fba_total"] = float(d.get("fba_total") or 0.0) * ratio


# Storage report generation on Amazon's side runs 30-240s for months not
# yet published. Kick it off in the background, dedupe rapid callers on the
# same (user, months) key, and let the caller decide whether to wait. Keep
# strong references so the task isn't garbage-collected mid-flight.
_STORAGE_BG_TASKS: dict[tuple, asyncio.Task] = {}

# SP-API's /reports endpoint has a tight per-account quota (~0.02 req/s).
# Running the storage / aged / removal / placement loaders in parallel
# triggers 429 storms with exponential backoff up to 24s+ per call —
# strictly worse than sequential. Serialize the report-hitting parts.
#
# The quota is per-Amazon-account, so we key the semaphore by the current
# user's ObjectId: one seller's slow report call no longer queues another
# seller's request. Lazily populated; the outer dict-init is guarded by an
# asyncio.Lock to avoid two coroutines racing to insert the same key.
_REPORT_SEMS: dict[str, asyncio.Semaphore] = {}
_REPORT_SEMS_LOCK = asyncio.Lock()


def build_incompleteness_report(task_status: dict[str, bool]) -> dict:
    """Given {section_name: is_done}, return the machine-readable partial-
    result contract used in the profitability response (review issue #4).

    The free-text `warnings` array alone lets a forgetful frontend render
    an incomplete profit total as if it were final. A top-level `complete`
    bool + explicit `partial_sections` list is impossible to miss."""
    missing = sorted(name for name, done in task_status.items() if not done)
    return {"complete": not missing, "partial_sections": missing}


def _sp_api_order_currencies(orders: list[dict]) -> set[str]:
    """SP-API-shape sibling of aurora_data.detect_order_currencies.

    SP-API GetOrders returns OrderTotal.CurrencyCode at the top of each
    order — cheaper than scanning line items and consistent with what
    Seller Central shows for that order."""
    codes: set[str] = set()
    for o in orders or []:
        code = (o.get("OrderTotal") or {}).get("CurrencyCode")
        if isinstance(code, str) and code.strip():
            codes.add(code.strip().upper())
    return codes


async def _get_report_sem() -> asyncio.Semaphore:
    """Per-account single-slot limiter for SP-API /reports calls.

    Falls back to a shared '__anon__' slot if no user context is set — the
    profitability path always has one, but the helper stays safe elsewhere."""
    from auth import current_user
    user = current_user.get()
    key = str(user["_id"]) if user and user.get("_id") else "__anon__"
    sem = _REPORT_SEMS.get(key)
    if sem is None:
        async with _REPORT_SEMS_LOCK:
            sem = _REPORT_SEMS.get(key)
            if sem is None:
                sem = asyncio.Semaphore(1)
                _REPORT_SEMS[key] = sem
    return sem

# Strong references so tasks that outlive the request handler (report
# loaders that missed the soft deadline) aren't garbage-collected. The
# done_callback discards them once complete.
_BG_TASKS: set[asyncio.Task] = set()


def _fire_bg(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


async def compute_profitability_data(
    days_back: int = 7,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    paginate: bool = False,
    time_zone: str | None = None,
    marketplace_id: str | None = None,
) -> dict:
    """Per-SKU profitability following the FBA Profitability Calculator PDF:

      Net = Selling Price
            − Referral Fee
            − FBA Fulfilment Fee
            − Fuel Surcharge (Product Fees API — split from fulfillment when bundled)
            − Allocated Storage Fee (ASIN monthly storage × units sold for SKU ÷ total units sold for ASIN — sales-share proxy, not on-hand-based)
            − Product Cost (COGS)
            − Shipping Cost to Amazon (inbound)
            − Advertising Cost (allocated)

    Amazon fees (referral / FBA / fuel) come from Product Fees API per ASIN.
    Storage comes from GET_FBA_STORAGE_FEE_CHARGES_DATA (cached 24h — first
    cold call adds 30–120 s). Ads come from Aurora campaigns pro-rated to
    the window and allocated per SKU when a campaign lists its SKUs;
    otherwise spread uniformly across units sold. Returns / low-inv /
    inbound-placement come from Finances / Aurora events; aged-inv and
    removal / disposal come from their Seller Central charge reports.
    Units, revenue, Referral, FBA, Fuel, and COGS are net of returned/
    refunded units for orders purchased in the window; Return Proc still
    charges 20% of referral on those returned units.

    Window is defined by `start`/`end` (YYYY-MM-DD strings or datetimes)
    when either is provided; falls back to `days_back` for legacy callers
    (LLM tool). End-exclusive vs Amazon docs is orders-`CreatedBefore`;
    for the FE picker that's inclusive-of-end-date so we push end to
    23:59:59Z."""
    from campaigns import get_ad_spend_for_range
    from database import (
        get_aged_inventory_cache,
        get_aged_surcharge_charges_cache,
        get_placement_fee_cache,
        get_removal_fees_cache,
        get_reimbursements_cache,
        get_storage_cache,
        put_aged_inventory_cache,
        put_aged_surcharge_charges_cache,
        put_placement_fee_cache,
        put_removal_fees_cache,
        put_reimbursements_cache,
        merge_storage_cache,
    )
    from amazon_sp import (
        calendar_months_in_window,
        fetch_reimbursements_for_window,
        fetch_removal_fees_per_sku,
        fetch_storage_fees_for_months,
        is_per_asin_by_month_storage_cache,
        merge_storage_by_asin_month,
        split_bundled_fulfillment_total,
        storage_fees_by_asin_for_window,
    )

    # Normalize window using marketplace timezone (matches Aurora Orders / SC).
    user = require_user()
    mp_tz = resolve_dashboard_timezone(user, time_zone)
    now = datetime.now(timezone.utc)

    display_start: str | None = None
    display_end: str | None = None
    start_dt: datetime | None = None
    end_dt: datetime | None = None

    start_raw = start if start is None or isinstance(start, str) else None
    end_raw = end if end is None or isinstance(end, str) else None

    if isinstance(start, datetime):
        start_dt = start.replace(tzinfo=start.tzinfo or timezone.utc)
    if isinstance(end, datetime):
        end_dt = end.replace(tzinfo=end.tzinfo or timezone.utc)

    if start_raw is not None or end_raw is not None:
        if parse_ymd_parts(start_raw) or parse_ymd_parts(end_raw):
            mp_start, mp_end = parse_date_range_for_query(start_raw, end_raw, mp_tz)
            if start_dt is None:
                start_dt = mp_start
            if end_dt is None:
                end_dt = mp_end
            if parse_ymd_parts(start_raw):
                display_start = str(start_raw).strip()
            if parse_ymd_parts(end_raw):
                display_end = str(end_raw).strip()
        else:
            def _parse_iso_dt(v, end_of_day: bool = False) -> datetime | None:
                if v is None or v == "":
                    return None
                s = str(v).strip()
                if len(s) == 10:
                    mp_start, mp_end = parse_date_range_for_query(s, s, mp_tz)
                    return mp_end if end_of_day else mp_start
                return datetime.fromisoformat(s.replace("Z", "+00:00"))

            if start_dt is None:
                start_dt = _parse_iso_dt(start_raw)
            if end_dt is None:
                end_dt = _parse_iso_dt(end_raw, end_of_day=True)

    if start_dt is None and end_dt is None:
        start_dt, end_dt, display_start, display_end = resolve_window_from_days_back(
            now, days_back, mp_tz,
        )
    else:
        if start_dt is None:
            fallback_start, _, ds, _ = resolve_window_from_days_back(now, days_back, mp_tz)
            start_dt = fallback_start
            display_start = display_start or ds
        if end_dt is None:
            end_dt = now
            display_end = display_end or now.astimezone(ZoneInfo(mp_tz)).date().isoformat()

    # SP-API rejects CreatedBefore < 2 minutes before "now".
    api_ceiling = now - timedelta(minutes=3)
    if end_dt > api_ceiling:
        end_dt = api_ceiling

    created_after = utc_instant_to_iso_z(start_dt)
    created_before = utc_instant_to_iso_z(end_dt)
    window_days = max((end_dt - start_dt).total_seconds() / 86400.0, 0.5)
    if not display_start:
        display_start = start_dt.astimezone(ZoneInfo(mp_tz)).date().isoformat()
    if not display_end:
        display_end = end_dt.astimezone(ZoneInfo(mp_tz)).date().isoformat()

    # Placement Finances / report Transaction date bounds: picker calendar
    # days as UTC midnights (matches Amazon placement-fee report Transaction
    # date column — NOT marketplace Event Date). PostedBefore is exclusive.
    place_start_day, place_end_day = inclusive_ymd_bounds(
        start_dt, end_dt, mp_tz, display_start, display_end,
    )
    placement_posted_after = f"{place_start_day.isoformat()}T00:00:00Z"
    placement_posted_before = utc_instant_to_iso_z(
        datetime.combine(
            place_end_day + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
    )

    use_db = aurora_db_enabled()
    partial_warning: str | None = None
    na_price_rows: list[dict] = []
    fetch_errors: list[str] = []
    orders_count = 0
    # Review issue #1: the calc path assumes a single currency per response
    # but the app supports 10 marketplaces / 7+ currencies. Two guards:
    # (a) when `marketplace_id` is passed, filter to that marketplace before
    #     aggregating — guarantees a single-currency response;
    # (b) when it isn't, still scan loaded orders for distinct currency
    #     codes and, if >1 appears, emit a loud warning + top-level
    #     `mixed_currency` flag so frontends can refuse to display a
    #     mathematically-meaningless blended total.
    detected_currencies: set[str] = set()
    sku_data: dict[str, dict] = {}

    if use_db:
        db_orders = await aurora_data.fetch_orders_with_items(
            user,
            created_after=created_after,
            created_before=created_before,
        )
        if marketplace_id:
            db_orders = aurora_data.filter_orders_by_marketplace(
                db_orders, marketplace_id,
            )
        if not db_orders:
            # Window empty in DB — try live SP-API (unsynced recent orders).
            try:
                orders_resp = await amazon_sp.get_orders(
                    created_after=created_after,
                    created_before=created_before,
                    paginate=paginate,
                )
            except Exception as e:
                msg = str(e)
                return {"error": f"Error fetching orders: {msg}",
                        "error_kind": "rate_limited" if "429" in msg else "orders_fetch_failed"}
            orders = orders_resp.get("payload", {}).get("Orders", [])
            partial_warning = orders_resp.get("_partial")
            if marketplace_id:
                orders = aurora_data.filter_orders_by_marketplace(
                    orders, marketplace_id,
                )
            if not orders:
                return {
                    "orders_count": 0, "skus_count": 0, "rows": [], "totals": None,
                    "missing_cogs": [], "na_price_rows": [], "fetch_errors": [],
                    "days_back": round(window_days, 2), "window_days": round(window_days, 2),
                    "start": display_start, "end": display_end,
                    "created_after": created_after, "created_before": created_before,
                    "caveats": [], "warnings": [],
                }
            order_ids = [o.get("AmazonOrderId") for o in orders if o.get("AmazonOrderId")]
            items_results = await _fetch_order_items_paced(order_ids)
            sku_data, orders_count = _aggregate_sku_from_sp_api(
                orders, items_results, fetch_errors, na_price_rows,
            )
            detected_currencies = _sp_api_order_currencies(orders)
        else:
            db_orders, item_errors = await data_resolver.supplement_order_items_from_sp_api(
                db_orders,
            )
            fetch_errors.extend(item_errors)
            sku_data, na_price_rows, orders_count = (
                aurora_data.aggregate_sku_metrics_from_orders(db_orders)
            )
            detected_currencies = aurora_data.detect_order_currencies(db_orders)
    else:
        try:
            orders_resp = await amazon_sp.get_orders(
                created_after=created_after,
                created_before=created_before,
                paginate=paginate,
            )
        except Exception as e:
            msg = str(e)
            return {"error": f"Error fetching orders: {msg}",
                    "error_kind": "rate_limited" if "429" in msg else "orders_fetch_failed"}
        orders = orders_resp.get("payload", {}).get("Orders", [])
        partial_warning = orders_resp.get("_partial")
        if marketplace_id:
            orders = aurora_data.filter_orders_by_marketplace(
                orders, marketplace_id,
            )
        if not orders:
            return {
                "orders_count": 0, "skus_count": 0, "rows": [], "totals": None,
                "missing_cogs": [], "na_price_rows": [], "fetch_errors": [],
                "days_back": round(window_days, 2), "window_days": round(window_days, 2),
                "start": display_start, "end": display_end,
                "created_after": created_after, "created_before": created_before,
                "caveats": [], "warnings": [],
            }

        order_ids = [o.get("AmazonOrderId") for o in orders if o.get("AmazonOrderId")]
        items_results = await _fetch_order_items_paced(order_ids)
        sku_data, orders_count = _aggregate_sku_from_sp_api(
            orders, items_results, fetch_errors, na_price_rows,
        )
        detected_currencies = _sp_api_order_currencies(orders)

    skus = sorted(sku_data.keys())

    # ── Storage / finances / placement / aged — not in Aurora DB; SP-API ───
    warnings: list[str] = []
    if len(detected_currencies) > 1:
        # See declaration comment: the module currently sums raw floats
        # across currencies. Surface it loudly so the FE can refuse to
        # display the blended total.
        warnings.append(
            "Mixed currencies in this window: "
            + ", ".join(sorted(detected_currencies))
            + ". Totals below are the raw sum without FX conversion and are "
            "not comparable. Filter by a single marketplace to get a "
            "meaningful figure."
        )
    if partial_warning:
        warnings.append(
            "Amazon Orders API rate-limited us mid-fetch — showing partial results. "
            "Re-open this tab in a minute to see the full window."
        )

    # State populated by parallel loaders below.
    cogs_map: dict = {}
    cogs_error: str | None = None
    storage_by_asin_month: dict[str, dict[str, dict]] = {}
    storage_cached_at: str | None = None
    storage_months_in_window = calendar_months_in_window(start_dt, end_dt, mp_tz)
    covered_months: set[str] = set()

    product_fee_fallback: dict[str, dict] = {}
    fin_by_sku: dict[str, dict] = {}
    unattributed_fees = amazon_sp._empty_fee_bucket()
    placement_avg_per_unit: dict[str, float] | None = None
    placement_avg_per_asin: dict[str, float] = {}
    placement_charges_by_sku: dict[str, float] = {}
    placement_charges_by_asin: dict[str, float] = {}
    placement_meta: dict[str, object] = {"source": "none", "sku_count": 0}
    placement_blended = False
    placement_window_total = 0.0
    fin_placement_window_total = 0.0
    fin_storage_window_total = 0.0
    storage_meta: dict[str, object] = {"source": "none"}
    aged_charges_by_sku: dict[str, float] | None = {}
    aged_charges_meta: dict = {"source": None, "access_denied": False}
    aged_report_total = 0.0
    aged_charges_trusted = False  # True when SC charges report/cache was used
    fees_by_asin: dict[str, dict] = {}
    fee_errors_pre: list[str] = []
    returns_by_sku: dict[str, dict] = {}

    removal_fees_by_sku: dict[str, float] = {}
    removal_fees_trusted = False
    removal_blended = True
    removal_report_total = 0.0
    removal_months = calendar_months_in_window(start_dt, end_dt, mp_tz)
    removal_meta: dict = {
        "source": None,
        "access_denied": False,
        "blended": True,
        "months": removal_months,
    }

    reimbursement_trusted = False
    reimbursement_blended = True
    reimbursement_report_total = 0.0
    reimbursement_months = calendar_months_in_window(start_dt, end_dt, mp_tz)
    reimbursement_meta: dict = {
        "source": None,
        "access_denied": False,
        "blended": True,
        "months": reimbursement_months,
    }

    ad_window = {
        "total_window": 0.0, "campaign_count": 0,
        "by_sku": {}, "unattributed": 0.0,
    }

    charges_start_iso = utc_instant_to_iso_z(start_dt)
    charges_end_iso = utc_instant_to_iso_z(end_dt)

    # Each loader is fully async; asyncio.gather below overlaps all their
    # I/O so wall time collapses to the slowest single call instead of the
    # sum. Loaders write to enclosing-scope vars via `nonlocal` — safe
    # because asyncio is single-threaded and each loader touches its own
    # subset of state.
    async def _load_cogs():
        nonlocal cogs_map, cogs_error
        if not skus:
            return
        try:
            cogs_rows = await get_cogs(skus)
            cogs_map = {r["sku"]: r for r in cogs_rows}
        except Exception as e:
            cogs_error = f"Error reading COGS: {e}"

    async def _load_storage():
        nonlocal storage_by_asin_month, storage_cached_at, covered_months
        storage_cache_meta = await get_storage_cache(max_age_hours=24)
        if storage_cache_meta:
            cached_map = storage_cache_meta.get("per_sku_monthly") or {}
            if is_per_asin_by_month_storage_cache(cached_map):
                storage_by_asin_month = amazon_sp.normalize_storage_fee_map(cached_map)
                storage_cached_at = storage_cache_meta.get("updated_at")
                # Effective coverage from get_storage_cache: fee months plus
                # short-lived empty-checks. Phantom monthsCovered are ignored.
                covered_months = {
                    str(m) for m in (storage_cache_meta.get("months_covered") or [])
                    if re.match(r"^\d{4}-\d{2}$", str(m))
                }
        missing_months = [m for m in storage_months_in_window if m not in covered_months]
        if not missing_months:
            return
        try:
            report_sem = await _get_report_sem()
            async with report_sem:
                fetched, found_months = await fetch_storage_fees_for_months(
                    missing_months, mp_tz,
                )
            if fetched:
                # Always apply in-memory first so profitability works even if
                # the Mongo cache write fails.
                storage_by_asin_month = merge_storage_by_asin_month(
                    storage_by_asin_month, fetched,
                )
            found = list(found_months or [])
            # Only empty-check when Amazon returned NO month rows at all.
            # Neighbor-month responses (asked July, got June) must NOT mark
            # July covered — otherwise Storage stays $0 for hours for every
            # account. Short empty-check only for true empty reports.
            empty_months = (
                list(missing_months) if not found else []
            )
            try:
                merged = await merge_storage_cache(
                    fetched or {},
                    found,
                    empty_months=empty_months,
                )
                storage_by_asin_month = merged["per_sku_monthly"]
                storage_cached_at = merged["updated_at"]
                covered_months = set(merged["months_covered"])
            except Exception as cache_err:
                warnings.append(
                    f"Storage cache save failed ({type(cache_err).__name__}); "
                    "using freshly fetched report in-memory."
                )
                covered_months = covered_months | set(found)
                storage_cached_at = datetime.now(timezone.utc).isoformat()
            still_missing = [m for m in missing_months if m not in found]
            if still_missing:
                warnings.append(
                    f"Storage: no Monthly Storage Fees report rows yet for "
                    f"{', '.join(still_missing)} — using Finances FBAStorageFee "
                    "per SKU when SellerSKU is present (not BLENDED); report "
                    "retry on next load."
                )
        except Exception as e:
            warnings.append(_sp_report_warning("Storage", e))

    async def _load_finances():
        nonlocal fin_by_sku, unattributed_fees, fin_placement_window_total
        nonlocal fin_storage_window_total
        # Finances fees (low inv, placement settlements) often post weeks
        # after the related sale/inbound event — look back 45 days before
        # the window for those. But refunds must be bounded to the actual
        # window: counting May's refund events against a June 1–10 window
        # inflates return_processing_fee and returned_units for pre-window
        # sales that the seller doesn't see as returns "in" the window.
        # Placement / monthly storage are also bounded to the profitability
        # PostedDate window so July UI gets July posts only (not the 45-day
        # lookback).
        fin_posted_after = utc_instant_to_iso_z(start_dt - timedelta(days=45))
        refund_posted_after = utc_instant_to_iso_z(start_dt)
        try:
            fin = await amazon_sp.get_financial_events(
                posted_after=fin_posted_after,
                posted_before=created_before,
                paginate=True,
                refund_posted_after=refund_posted_after,
                placement_posted_after=placement_posted_after,
                placement_posted_before=placement_posted_before,
                storage_posted_after=placement_posted_after,
                storage_posted_before=placement_posted_before,
            )
            fin_by_sku = fin.get("by_sku") or {}
            unattributed_fees = fin.get("unattributed") or unattributed_fees
            fin_placement_window_total = float(fin.get("placement_window_total") or 0)
            fin_storage_window_total = float(fin.get("storage_window_total") or 0)
        except Exception as e:
            warnings.append(_sp_report_warning("Finances", e))

    async def _load_returns():
        # Returns come from Aurora orders — same sources as the Returned Orders
        # tab: Order.customerReturns (FBA physical returns) AND Order.refunds
        # (Finances "Refund applied", including returnless refunds). Bounded by
        # the ORDER's purchaseDate — matches the same sale-window the rest of
        # the tab uses. `orderItems[].referralFee` gives the exact referral
        # Amazon charged at sale time, so 0.20 × that × units is per-order-line
        # accurate rather than a window average.
        nonlocal returns_by_sku
        if not use_db:
            return
        try:
            returns_by_sku = await aurora_data.fba_returns_by_sku(
                require_user(), start_dt, end_dt,
            )
        except Exception as e:
            warnings.append(_sp_report_warning("Customer returns (Aurora)", e))

    # Started before placement so placement can await Finances for its
    # PostedDate-window fallback without a second Finances walk.
    finances_task = _fire_bg(_load_finances())

    async def _load_placement():
        # Prefer Aurora's fbainboundplacementfees dated charges filtered by the
        # profitability window (Transaction/Posted date — andexports July path).
        # If none match, use live Finances PostedDate-window total. If that is
        # also $0, lock Totals at $0. Never invent fee_rate × units sold from
        # other months.
        nonlocal placement_charges_by_sku, placement_charges_by_asin
        nonlocal placement_blended, placement_window_total
        nonlocal placement_avg_per_unit, placement_avg_per_asin

        def _lock_zero_placement(source: str) -> None:
            nonlocal placement_blended, placement_window_total
            placement_blended = True
            placement_window_total = 0.0
            placement_meta["source"] = source
            placement_meta["blended"] = True
            placement_meta["window_total"] = 0.0
            placement_meta["sku_count"] = 0

        if use_db:
            try:
                (
                    placement_charges_by_sku,
                    placement_charges_by_asin,
                    win_meta,
                ) = await aurora_data.fba_inbound_placement_charges_for_window(
                    require_user(), start_dt, end_dt, marketplace_tz=mp_tz,
                    display_start=display_start, display_end=display_end,
                )
                matched_events = int(win_meta.get("matched_events") or 0)
                placement_meta["matched_events"] = matched_events
                placement_meta["event_count"] = win_meta.get("event_count")
                placement_meta["charge_count"] = win_meta.get("charge_count")
                placement_meta["totals_from"] = win_meta.get("totals_from")
                placement_meta["window_start"] = win_meta.get("window_start")
                placement_meta["window_end"] = win_meta.get("window_end")
                if matched_events > 0:
                    placement_blended = True
                    placement_window_total = float(win_meta.get("window_total") or 0)
                    placement_meta["source"] = "aurora_events_window"
                    placement_meta["blended"] = True
                    placement_meta["window_total"] = placement_window_total
                    placement_meta["sku_count"] = 0
                    return
                else:
                    # Shared Finances walk uses start-45d + max_pages=40, so on
                    # busy accounts it never reaches the selected month's
                    # placement posts (asglobal March needs ~70 pages in-window).
                    # Do an exact-window walk so BLENDED totals aren't $0.
                    try:
                        await finances_task
                    except Exception:
                        pass
                    exact_placement_total = 0.0
                    if fin_placement_window_total <= 0:
                        try:
                            fin_exact = await amazon_sp.get_financial_events(
                                posted_after=placement_posted_after,
                                posted_before=placement_posted_before,
                                paginate=True,
                                max_pages=200,
                                placement_posted_after=placement_posted_after,
                                placement_posted_before=placement_posted_before,
                            )
                            exact_placement_total = float(
                                fin_exact.get("placement_window_total") or 0
                            )
                            placement_meta["finances_exact_window_pages"] = (
                                fin_exact.get("pages")
                            )
                        except Exception as e:
                            warnings.append(
                                _sp_report_warning(
                                    "Inbound placement (Finances window)", e,
                                )
                            )
                    use_placement_total = (
                        fin_placement_window_total
                        if fin_placement_window_total > 0
                        else exact_placement_total
                    )
                    if use_placement_total > 0:
                        placement_blended = True
                        placement_window_total = use_placement_total
                        placement_meta["source"] = "finances_posted_window"
                        placement_meta["blended"] = True
                        placement_meta["window_total"] = placement_window_total
                        placement_meta["sku_count"] = 0
                        return
                    _lock_zero_placement("zero_no_window_charges")
                    return
            except Exception as e:
                warnings.append(_sp_report_warning("Inbound placement (Aurora)", e))
                try:
                    await finances_task
                except Exception:
                    pass
                if fin_placement_window_total > 0:
                    placement_blended = True
                    placement_window_total = fin_placement_window_total
                    placement_meta["source"] = "finances_posted_window"
                    placement_meta["blended"] = True
                    placement_meta["window_total"] = placement_window_total
                    placement_meta["sku_count"] = 0
                    return
                _lock_zero_placement("zero_no_window_charges")
                return
        else:
            try:
                await finances_task
            except Exception:
                pass
            if fin_placement_window_total > 0:
                placement_blended = True
                placement_window_total = fin_placement_window_total
                placement_meta["source"] = "finances_posted_window"
                placement_meta["blended"] = True
                placement_meta["window_total"] = placement_window_total
                placement_meta["sku_count"] = 0
            else:
                _lock_zero_placement("zero_no_window_charges")
                return

        if placement_blended:
            return
        _lock_zero_placement("zero_no_window_charges")

    async def _load_aged():
        # Prefer Aurora's fbaagedinventoryfees snapshot (populated by
        # auroraBackend's fbaAgedInventorySyncService from the FBA Inventory
        # Planning report). This is the SAME source the legacy planning path
        # uses — Aurora just runs it on its side and hands us the result.
        # NOTE: this feeds planning_aged_by_sku only. The actual Aged Inventory
        # Surcharge dollars used in /profitability come from the CHARGES report
        # (further below), which Aurora doesn't ingest.
        nonlocal aged_charges_by_sku, aged_charges_meta, aged_report_total, aged_charges_trusted
        planning_aged_by_sku: dict[str, float] = {}
        aged_from_aurora: dict | None = None
        if use_db:
            try:
                aged_from_aurora = await aurora_data.fba_aged_inventory_by_sku(require_user())
            except Exception as e:
                warnings.append(_sp_report_warning("Aged inventory (Aurora)", e))

        aged_cache_meta = await get_aged_inventory_cache(max_age_hours=24)
        # Force a refetch when the cache pre-dates the HDoS / recommended-ship-qty
        # extraction — an old-shape cache is technically fresh (< 24h) but missing
        # the fields the Restock table needs. Detect it by looking for any entry
        # carrying the new keys; if none do, treat as stale.
        aged_needs_refresh = bool(aged_cache_meta) and not any(
            (
                "historical_days_of_supply" in v
                or "recommended_ship_in_quantity" in v
            )
            for v in (aged_cache_meta.get("per_sku") or {}).values()
        )
        try:
            aged_per_sku: dict = {}
            if aged_from_aurora is not None:
                # Aurora synced this planning report — use it, skip our fetch.
                aged_per_sku = aged_from_aurora
            else:
                need_aged_fetch = True
                if aged_cache_meta and not aged_needs_refresh:
                    aged_per_sku = aged_cache_meta.get("per_sku") or {}
                    if aged_per_sku:
                        need_aged_fetch = False
                if need_aged_fetch:
                    report_sem = await _get_report_sem()
                    async with report_sem:
                        aged_per_sku = await amazon_sp.fetch_aged_inventory_fees_per_sku()
                    await put_aged_inventory_cache(aged_per_sku)
            for psku, b in (aged_per_sku or {}).items():
                if not isinstance(b, dict):
                    continue
                fee = float(b.get("monthly_fee") or 0)
                if fee > 0:
                    planning_aged_by_sku[psku] = fee
            # Planning report is cached for Restock (HDOS / recommended qty).
            # Profitability uses the actual AIS charges report below.
        except Exception as e:
            warnings.append(_sp_report_warning("Aged inventory planning", e))
            if _is_sp_access_denied(e):
                await put_aged_inventory_cache({})

        # Actual Aged Inventory Surcharge charges (Seller Central report).
        try:
            charges_cache = await get_aged_surcharge_charges_cache(
                charges_start_iso, charges_end_iso, max_age_hours=24,
            )
            if charges_cache and not charges_cache.get("access_denied"):
                aged_charges_by_sku = _build_aged_charges(
                    charges_cache.get("per_sku") or {}
                )
                aged_charges_meta["source"] = "charges_cache"
                aged_charges_trusted = True
            else:
                report_sem = await _get_report_sem()
                async with report_sem:
                    charges_per_sku = await amazon_sp.fetch_aged_surcharge_charges_per_sku(
                        start_dt, end_dt, time_zone=mp_tz,
                    )
                await put_aged_surcharge_charges_cache(
                    charges_per_sku, charges_start_iso, charges_end_iso,
                )
                aged_charges_by_sku = _build_aged_charges(charges_per_sku)
                aged_charges_meta["source"] = "charges_report"
                aged_charges_trusted = True
            aged_report_total = round(
                sum(float(v or 0) for v in (aged_charges_by_sku or {}).values()), 2,
            )
            aged_charges_meta["report_total"] = aged_report_total
        except Exception as e:
            warnings.append(_sp_report_warning("Aged inventory surcharge charges", e))
            if _is_sp_access_denied(e):
                aged_charges_meta["access_denied"] = True
                await put_aged_surcharge_charges_cache(
                    {}, charges_start_iso, charges_end_iso, access_denied=True,
                )
            # Prefer planning estimated-ais (SKU-only) over Finances only when
            # BOTH create and DONE-report reuse failed. Finances posts AIS as a
            # lumped unattributed total (no SellerSKU), which left every row at $0.
            if planning_aged_by_sku:
                aged_charges_by_sku = dict(planning_aged_by_sku)
                aged_charges_meta["source"] = "planning_estimate_fallback"
                aged_charges_trusted = False
                warnings.append(
                    "Aged Inv: could not load the Aged Inventory Surcharge charges "
                    "report (create FATAL and no reusable DONE report). Showing "
                    "Inventory Planning estimated-ais totals — these will NOT match "
                    "Seller Central amount-charged. Retry later."
                )
            else:
                aged_charges_by_sku = None
                aged_charges_meta["source"] = "finances_fallback"
                aged_charges_trusted = False

    async def _load_removal():
        # Removal / Disposal: Seller Central Removal Order Detail only
        # (GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA), request-date / Event Date.
        # Always BLENDED on Totals (like inbound placement) — never Finances
        # ServiceFee spread across sold SKUs (that invents wrong per-row amounts).
        # Report is requested per calendar month in the filter, then clipped to the
        # customer's exact [start, end) date range.
        nonlocal removal_fees_by_sku, removal_fees_trusted, removal_report_total
        try:
            removal_cache = await get_removal_fees_cache(
                charges_start_iso, charges_end_iso, max_age_hours=24,
            )
            if removal_cache and not removal_cache.get("access_denied"):
                removal_fees_by_sku = _build_removal_fees(
                    removal_cache.get("per_sku") or {},
                )
                removal_report_total = round(
                    float(removal_cache.get("report_total") or 0)
                    or sum(removal_fees_by_sku.values()),
                    2,
                )
                removal_meta["source"] = "charges_cache"
                removal_fees_trusted = True
            else:
                report_sem = await _get_report_sem()
                async with report_sem:
                    removal_raw = await fetch_removal_fees_per_sku(
                        start_dt, end_dt, time_zone=mp_tz,
                    )
                await put_removal_fees_cache(
                    removal_raw, charges_start_iso, charges_end_iso,
                )
                removal_fees_by_sku = _build_removal_fees(removal_raw)
                removal_report_total = round(
                    sum(removal_fees_by_sku.values()), 2,
                )
                removal_meta["source"] = "charges_report"
                removal_fees_trusted = True
            removal_meta["report_total"] = removal_report_total
        except Exception as e:
            warnings.append(_sp_report_warning("Removal Order Detail", e))
            if _is_sp_access_denied(e):
                removal_meta["access_denied"] = True
                await put_removal_fees_cache(
                    {}, charges_start_iso, charges_end_iso, access_denied=True,
                )
            removal_fees_by_sku = {}
            removal_fees_trusted = False
            removal_report_total = 0.0
            removal_meta["source"] = "unavailable"

    async def _load_reimbursements():
        # FBA Reimbursements report (GET_FBA_REIMBURSEMENTS_DATA), approval-date
        # window. Always BLENDED account income on Totals — rows stay $0.
        # Month-wise report create/reuse, then clipped to exact [start, end).
        nonlocal reimbursement_trusted, reimbursement_report_total
        try:
            reimb_cache = await get_reimbursements_cache(
                charges_start_iso, charges_end_iso, max_age_hours=24,
            )
            if reimb_cache and not reimb_cache.get("access_denied"):
                reimbursement_report_total = round(
                    float(reimb_cache.get("report_total") or 0), 2,
                )
                reimbursement_meta["source"] = "charges_cache"
                reimbursement_trusted = True
            else:
                report_sem = await _get_report_sem()
                async with report_sem:
                    reimb_raw = await fetch_reimbursements_for_window(
                        start_dt, end_dt, time_zone=mp_tz,
                    )
                per_sku = (reimb_raw or {}).get("per_sku") or {}
                reimbursement_report_total = round(
                    float((reimb_raw or {}).get("report_total") or 0), 2,
                )
                await put_reimbursements_cache(
                    per_sku,
                    charges_start_iso,
                    charges_end_iso,
                    report_total=reimbursement_report_total,
                )
                reimbursement_meta["source"] = "charges_report"
                reimbursement_trusted = True
            reimbursement_meta["report_total"] = reimbursement_report_total
            reimbursement_meta["blended"] = True
        except Exception as e:
            warnings.append(_sp_report_warning("FBA Reimbursements", e))
            if _is_sp_access_denied(e):
                reimbursement_meta["access_denied"] = True
                await put_reimbursements_cache(
                    {},
                    charges_start_iso,
                    charges_end_iso,
                    report_total=0.0,
                    access_denied=True,
                )
            reimbursement_trusted = False
            reimbursement_report_total = 0.0
            reimbursement_meta["source"] = "unavailable"

    async def _load_product_fees_and_batch():
        # Fast path: Aurora products.fees (Mongo). Live Fees API only for
        # SKUs missing fees or whose sale price drifted from listing price —
        # never for every sold SKU (that was ~N/20 × 2.1s and blocked the UI).
        # Always pass correct IsAmazonFulfilled so FBM is not charged FBA.
        nonlocal product_fee_fallback, fees_by_asin, fee_errors_pre
        if use_db:
            product_fee_fallback = await aurora_data.product_fee_estimates_by_sku(
                require_user(), skus,
            )
            fee_skus = data_resolver.skus_needing_fees_api(
                sku_data, product_fee_fallback,
            )
            batch_items = data_resolver.fee_batch_items_for_skus(
                fee_skus, sku_data, product_fee_fallback,
            )
        else:
            product_fee_fallback = {}
            batch_items = data_resolver.fee_batch_items_for_skus(
                skus, sku_data, {},
            )

        if batch_items:
            try:
                fees_by_asin = await amazon_sp.get_fees_estimates_batch(batch_items)
            except Exception as e:
                fee_errors_pre = [f"batch fees fetch failed: {str(e)[:200]}"]

    async def _load_ads():
        # Per-SKU when a campaign lists its SKUs (Aurora Ad.skus[]); campaigns
        # without a SKU list fall into `unattributed` and get spread uniformly
        # per unit (legacy behavior).
        nonlocal ad_window
        try:
            ad_window = await get_ad_spend_for_range(start=start_dt, end=end_dt)
        except Exception as e:
            warnings.append(f"Ad spend fetch failed ({e}); ad cost set to 0.")


    # Storage report generation on Amazon's side can take 30-240s for months
    # not yet published. Dedupe concurrent callers on the same (user, months)
    # key so rapid reloads don't pile on Amazon.
    storage_bg_key = (
        str(user.get("_id")) if isinstance(user, dict) else str(user),
        tuple(storage_months_in_window),
    )
    existing_storage_task = _STORAGE_BG_TASKS.get(storage_bg_key)
    if existing_storage_task is not None and not existing_storage_task.done():
        storage_task = existing_storage_task
    else:
        storage_task = _fire_bg(_load_storage())
        _STORAGE_BG_TASKS[storage_bg_key] = storage_task
        storage_task.add_done_callback(
            lambda t, k=storage_bg_key: _STORAGE_BG_TASKS.pop(k, None)
        )

    # COGS is critical — the response is wrong without it. Everything else
    # is best-effort: give it a soft deadline and return whatever came back.
    # Slow loaders keep running via _fire_bg and warm the cache for the
    # next call, which will read them near-instantly.
    cogs_task = _fire_bg(_load_cogs())
    await cogs_task
    if cogs_error:
        return {"error": cogs_error}

    critical_tasks: dict[str, asyncio.Task] = {
        "finances": finances_task,
        "returns": _fire_bg(_load_returns()),
        "placement": _fire_bg(_load_placement()),
        "product fees": _fire_bg(_load_product_fees_and_batch()),
        # Storage / reimbursements report creates are slow; soft 5s left them
        # at $0 on first paint even when Seller Central had rows.
        "storage": storage_task,
        "reimbursements": _fire_bg(_load_reimbursements()),
    }
    soft_tasks: dict[str, asyncio.Task] = {
        "aged inventory": _fire_bg(_load_aged()),
        "removal": _fire_bg(_load_removal()),
        "ads": _fire_bg(_load_ads()),
    }
    # Two-tier wait:
    # 1) Soft reports (aged / removal / ads) get ~5s — keep running in bg.
    # 2) Critical fee sources (finances / placement / storage / reimbursements /
    #    product fees) await longer so first paint is usable.
    soft_deadline_s = 5.0
    critical_deadline_s = 55.0
    all_tasks = {**critical_tasks, **soft_tasks}
    await asyncio.wait(list(all_tasks.values()), timeout=soft_deadline_s)
    still_critical = [
        t for t in critical_tasks.values() if not t.done()
    ]
    if still_critical:
        await asyncio.wait(
            still_critical,
            timeout=max(0.0, critical_deadline_s - soft_deadline_s),
        )
    partial_status = build_incompleteness_report(
        {name: task.done() for name, task in all_tasks.items()}
    )
    if not partial_status["complete"]:
        warnings.append(
            "Still loading in background: "
            + ", ".join(partial_status["partial_sections"])
            + ". These sections may show as 0 in this response — reload in "
            "~30 seconds to pick up the warmed cache."
        )

    storage_fees_by_asin, storage_report_total, storage_month_fractions = (
        storage_fees_by_asin_for_window(
            storage_by_asin_month,
            storage_months_in_window,
            window_start=start_dt,
            window_end=end_dt,
            time_zone=mp_tz,
        )
    )
    # Storage is NEVER BLENDED (only placement / ads / removal / reimbursements
    # are). Primary: Monthly Storage Fees report → per-ASIN → split to sold SKUs.
    # Fallback when report has no rows for the filter months: Finances
    # FBAStorageFee attributed by SellerSKU (and unattributed residual on
    # Totals). Same dollars as Payments → Transactions, per-SKU when Amazon
    # sends a SKU — never a BLENDED column.
    storage_from_finances = False
    if float(storage_report_total or 0) <= 0 and fin_storage_window_total > 0:
        storage_from_finances = True
        storage_report_total = round(float(fin_storage_window_total), 2)
        storage_fees_by_asin = {}
        storage_meta["source"] = "finances_per_sku"
        storage_meta["window_total"] = storage_report_total
    elif float(storage_report_total or 0) > 0:
        storage_meta["source"] = "storage_report"
        storage_meta["window_total"] = round(float(storage_report_total), 2)

    # Net out returned/refunded units before fee × units math and storage /
    # ad allocation. Return Proc still uses full returned_units below.
    _apply_returns_to_sku_data(sku_data, returns_by_sku)

    # Units sold per ASIN — used to split that ASIN's report fee across SKUs.
    storage_units_by_asin: dict[str, float] = defaultdict(float)
    for _sku, _d in sku_data.items():
        _asin = (_d.get("asin") or "").upper()
        if _asin and _d.get("units"):
            storage_units_by_asin[_asin] += float(_d["units"])

    total_units_window = sum(d["units"] for d in sku_data.values())
    unattr_per_unit = {
        k: (unattributed_fees.get(k, 0.0) / total_units_window
            if total_units_window > 0 else 0.0)
        for k in ("return_processing", "low_inventory", "inbound_placement",
                  "aged_inventory", "removal", "storage")
    }

    ad_by_sku: dict[str, float] = ad_window.get("by_sku") or {}
    ad_unattributed: float = float(ad_window.get("unattributed") or 0.0)
    ad_unattr_per_unit = (
        ad_unattributed / total_units_window if total_units_window > 0 else 0.0
    )

    # ── Per-SKU enrichment ────────────────────────────────────────────────
    rows: list[dict] = []
    totals = defaultdict(float)
    missing_cogs: list[dict] = []
    fee_errors: list[str] = list(fee_errors_pre)

    for sku in skus:
        d = sku_data[sku]
        units = d["units"]
        revenue = round(d["revenue"], 2)
        avg_price = revenue / units if units else 0.0
        asin = d["asin"]

        # Referral / base FBA / fuel.
        # Priority (keeps UI fast while staying Amazon-accurate):
        #   1) live Fees API at sale price — only fetched for missing/diverged SKUs
        #   2) products.fees (listing-price sync — same as Products page)
        #   3) order line totals (last resort)
        referral_total = 0.0
        fba_total = 0.0
        fuel_total = 0.0

        est = fees_by_asin.get(asin) if asin else None
        est_ok = bool(
            est
            and not est.get("error")
            and (
                float(est.get("referral") or 0) > 0
                or float(est.get("fba") or 0) > 0
                or float(est.get("fuel_surcharge") or 0) > 0
            )
        )
        pf = product_fee_fallback.get(sku) if sku in product_fee_fallback else None
        if not pf and product_fee_fallback:
            pf = next(
                (
                    v for k, v in product_fee_fallback.items()
                    if str(k).lower() == str(sku).lower()
                ),
                None,
            )
        pf_ok = bool(
            pf
            and (
                float(pf.get("referral_per_unit") or 0) > 0
                or float(pf.get("fba_per_unit") or 0) > 0
                or float(pf.get("fuel_per_unit") or 0) > 0
                or float(pf.get("fulfillment_per_unit") or 0) > 0
            )
        )

        # Prefer live estimate when present (selective fetch only puts an ASIN
        # here when fees were missing or sale price drifted). Otherwise use
        # products.fees — the fast path for the common case.
        if est_ok:
            referral_total = round(float(est.get("referral", 0.0)) * units, 2)
            fba_total = round(float(est.get("fba", 0.0)) * units, 2)
            fuel_total = round(float(est.get("fuel_surcharge", 0.0)) * units, 2)
        elif pf_ok:
            # Same source as Aurora Products (`products.fees`). FBA + Fuel
            # equals the Products "FBA Fee" column (fulfillment total).
            listing_price = float(pf.get("listing_price") or 0)
            ref_unit = float(pf.get("referral_per_unit") or 0)
            if listing_price > 0 and ref_unit > 0 and revenue > 0:
                referral_total = round(revenue * (ref_unit / listing_price), 2)
            else:
                referral_total = round(ref_unit * units, 2)
            fba_unit = float(pf.get("fba_per_unit") or 0)
            fuel_unit = float(pf.get("fuel_per_unit") or 0)
            # Safety: if split drifted, re-align to Products fbaFee total.
            ful_unit = float(pf.get("fulfillment_per_unit") or 0)
            if ful_unit > 0 and abs((fba_unit + fuel_unit) - ful_unit) > 0.02:
                fba_unit, fuel_unit = split_bundled_fulfillment_total(ful_unit)
            fba_total = round(fba_unit * units, 2)
            fuel_total = round(fuel_unit * units, 2)
            if not asin and pf.get("asin"):
                asin = pf["asin"]
        else:
            line_ref = round(float(d.get("referral_total") or 0), 2)
            line_fba = round(float(d.get("fba_total") or 0), 2)
            if line_ref > 0 or line_fba > 0:
                referral_total = line_ref
                fba_total, fuel_total = split_bundled_fulfillment_total(line_fba)
            elif asin and avg_price > 0:
                fee_errors.append(f"{sku} ({asin}): no fee data in products.fees or Fees API")
            elif avg_price > 0:
                fee_errors.append(f"{sku}: no ASIN found in order items")
            if est and est.get("error"):
                fee_errors.append(f"{sku} ({asin}): {str(est['error'])[:120]}")

        amazon_fees = round(referral_total + fba_total + fuel_total, 2)

        # Ads: per-campaign attribution when the campaign lists this SKU,
        # plus a share of the unattributed pool spread per unit.
        ad_cost = round(
            float(ad_by_sku.get(sku, 0.0)) + ad_unattr_per_unit * units, 2,
        )

        # Finances API fees per SKU + share of unattributed pool
        fin_sku = fin_by_sku.get(sku, {})
        # Finances keys may differ by casing from order SKUs.
        if not fin_sku:
            fin_sku = next(
                (v for k, v in fin_by_sku.items() if str(k).lower() == sku.lower()),
                {},
            )

        # Storage: Monthly Storage Fees report $ for this ASIN (month_of_charge
        # in filter), split across SKUs that sold that ASIN. Never BLENDED —
        # only placement / ads / removal use that pattern. If the report has
        # no rows yet, use Finances FBAStorageFee for this SellerSKU (+ share
        # of unattributed storage posts).
        asin_key = (asin or "").upper()
        asin_storage_fee = float(storage_fees_by_asin.get(asin_key, 0.0) or 0.0)
        asin_sold_units = float(storage_units_by_asin.get(asin_key, 0.0) or 0.0)
        if asin_storage_fee > 0 and asin_sold_units > 0 and units > 0:
            storage = round(asin_storage_fee * (units / asin_sold_units), 2)
        elif storage_from_finances:
            storage = round(
                float(fin_sku.get("storage", 0.0) or 0)
                + unattr_per_unit.get("storage", 0.0) * units,
                2,
            )
        else:
            storage = 0.0

        # Return processing fee = 20% × the actual referral fee for the
        # returned units. Client-confirmed policy for this seller's
        # categories; matches Amazon's standard FBA Returns Processing
        # Fee rate card.
        #
        # Source: Aurora's Order.customerReturns / Order.refunds joined against
        # the same order's orderItems.referralFee (returned to us as
        # `refunded_referral`). Must match by exact seller SKU only — many
        # replenishment / amazon.gr SKUs share one ASIN, and falling back by
        # ASIN incorrectly charged Return Proc on never-returned SKUs.
        ret = _lookup_returns_for_sku(returns_by_sku, sku)
        refunded_referral = float((ret or {}).get("refunded_referral") or 0.0)
        returned_units = int((ret or {}).get("returned_units") or 0)
        return_processing_fee = round(0.20 * refunded_referral, 2)
        low_inventory_fee = round(
            (fin_sku.get("low_inventory", 0.0)
             + unattr_per_unit["low_inventory"] * units), 2)

        # Inbound placement: dated window charges (or Finances window total)
        # go on Totals as BLENDED. Rows stay $0. Never fee_rate × units sold.
        inbound_placement_fee = 0.0

        # Aged inventory: actual amount-charged from Seller Central's Aged
        # Inventory Surcharge report (GET_FBA_FULFILLMENT_LONGTERM_STORAGE_
        # FEE_CHARGES_DATA), matched by seller SKU only. Do NOT fall back to
        # ASIN — that double-counts when merchant + amzn.gr.* SKUs share an
        # ASIN. Do NOT use planning-report estimated-ais × months_in_window
        # (those are forward projections, not billed amounts).
        if aged_charges_trusted and aged_charges_by_sku is not None:
            # Trust the SC report: $0 when SKU has no amount-charged rows.
            aged_inventory_fee = round(_lookup_sku_amount(aged_charges_by_sku, sku), 2)
        elif aged_charges_by_sku is not None:
            charged = _lookup_sku_amount(aged_charges_by_sku, sku)
            if charged > 0:
                aged_inventory_fee = round(charged, 2)
            else:
                aged_inventory_fee = round(
                    float(fin_sku.get("aged_inventory", 0.0) or 0), 2,
                )
        else:
            # Charges report unavailable (403 / error) — Finances attributed
            # only; never spread the unattributed pool (inflates SKUs that
            # never incurred AIS).
            aged_inventory_fee = round(
                float(fin_sku.get("aged_inventory", 0.0) or 0), 2,
            )

        # Removal / Disposal: always BLENDED from Removal Order Detail.
        # Per-SKU rows stay $0; Totals carries the request-date report sum.
        removal_fee = 0.0
        # Reimbursements: always BLENDED account income; rows stay $0.
        reimbursement = 0.0

        # COGS components (only when uploaded)
        cogs_row = cogs_map.get(sku)
        if cogs_row:
            product_cost = round(cogs_row["unit_cost"] * units, 2)
            inbound = round(cogs_row["inbound_shipping_per_unit"] * units, 2)
        else:
            product_cost = 0.0
            inbound = 0.0
            missing_cogs.append({"sku": sku, "units": units, "revenue": revenue})

        net = round(
            revenue - amazon_fees - storage - product_cost - inbound - ad_cost
            - return_processing_fee - low_inventory_fee - inbound_placement_fee
            - aged_inventory_fee - removal_fee + reimbursement,
            2,
        )
        margin = round((net / revenue * 100), 1) if revenue > 0 else 0.0

        row = {
            "sku": sku,
            "asin": asin,
            "units": units,
            "avg_price": round(avg_price, 2),
            "revenue": revenue,
            "referral_fee": referral_total,
            "fba_fee": fba_total,
            "fuel_surcharge": fuel_total,
            "amazon_fees": amazon_fees,
            "storage_fee": storage,
            "ad_cost": ad_cost,
            "product_cost": product_cost,
            "inbound_shipping": inbound,
            "cogs_total": round(product_cost + inbound, 2),
            "unit_cost": float(cogs_row["unit_cost"]) if cogs_row else None,
            "inbound_shipping_per_unit": float(cogs_row["inbound_shipping_per_unit"]) if cogs_row else None,
            "return_processing_fee": return_processing_fee,
            "low_inventory_fee": low_inventory_fee,
            "inbound_placement_fee": inbound_placement_fee,
            "aged_inventory_fee": aged_inventory_fee,
            "removal_fee": removal_fee,
            "reimbursement": reimbursement,
            "net": net,
            "margin": margin,
            "cogs_uploaded": cogs_row is not None,
        }
        rows.append(row)

        totals["units"] += units
        totals["revenue"] += revenue
        totals["referral_fee"] += referral_total
        totals["fba_fee"] += fba_total
        totals["fuel_surcharge"] += fuel_total
        totals["amazon_fees"] += amazon_fees
        totals["storage_fee"] += storage
        totals["ad_cost"] += ad_cost
        totals["product_cost"] += product_cost
        totals["inbound_shipping"] += inbound
        totals["cogs_total"] += product_cost + inbound
        totals["return_processing_fee"] += return_processing_fee
        totals["low_inventory_fee"] += low_inventory_fee
        totals["inbound_placement_fee"] += inbound_placement_fee
        totals["aged_inventory_fee"] += aged_inventory_fee
        totals["removal_fee"] += removal_fee
        totals["reimbursement"] += reimbursement
        totals["net"] += net

    # Blended placement: rows carry $0 / BLENDED; apply the Amazon placement
    # fee report window total once on account totals (Transaction date sum).
    if placement_blended:
        fee_r = round(float(placement_window_total), 2)
        totals["inbound_placement_fee"] = fee_r
        totals["net"] -= fee_r

    # Storage fees on ASINs with inventory but no sales in the window still
    # appear on Seller Central's Monthly Storage Fees report — add residual
    # so the Storage KPI matches the CSV total. Same path applies the
    # Finances FBAStorageFee blended total when the report is empty.
    allocated_storage = round(float(totals.get("storage_fee") or 0), 2)
    storage_unallocated = round(
        max(0.0, float(storage_report_total or 0) - allocated_storage), 2,
    )
    if storage_unallocated > 0:
        totals["storage_fee"] = round(allocated_storage + storage_unallocated, 2)
        totals["net"] -= storage_unallocated

    # Aged Inv: SKUs with amount-charged but no sales in the window still
    # appear on Seller Central — add residual so Totals match the CSV.
    aged_unallocated = 0.0
    if aged_charges_trusted and aged_charges_by_sku:
        sold_skus_l = {str(s).lower() for s in skus}
        for asku, fee in aged_charges_by_sku.items():
            if str(asku).lower() not in sold_skus_l:
                aged_unallocated += float(fee or 0)
        aged_unallocated = round(aged_unallocated, 2)
        # Also catch penny drift between row sum and report total
        allocated_aged = round(float(totals.get("aged_inventory_fee") or 0), 2)
        target_aged = round(float(aged_report_total or 0), 2)
        gap = round(target_aged - allocated_aged - aged_unallocated, 2)
        if abs(gap) >= 0.01 and abs(gap) <= 0.05:
            aged_unallocated = round(aged_unallocated + gap, 2)
        if aged_unallocated > 0:
            totals["aged_inventory_fee"] = round(allocated_aged + aged_unallocated, 2)
            totals["net"] -= aged_unallocated

    # Removal: BLENDED account total from Removal Order Detail (request-date).
    removal_unallocated = 0.0
    if removal_fees_trusted:
        fee_r = round(float(removal_report_total or 0), 2)
        totals["removal_fee"] = fee_r
        # Rows carried $0; apply once on account totals (same as placement).
        if fee_r:
            totals["net"] -= fee_r
        removal_unallocated = fee_r
        removal_blended = True

    # Reimbursements: BLENDED account income from FBA Reimbursements report
    # (approval-date window). Credit increases net once on Totals.
    if reimbursement_trusted:
        credit = round(float(reimbursement_report_total or 0), 2)
        totals["reimbursement"] = credit
        if credit:
            totals["net"] += credit
        reimbursement_blended = True

    # Review issues #7 + #8: delegate row-derived total reconciliation to
    # aurora_data.reconcile_row_derived_totals — a pure extracted stage
    # that uses Decimal + ROUND_HALF_UP so binary-float drift over
    # thousands of rows can't produce totals that are off by a cent.
    # Blended/overwritten totals (placement, storage w/ unallocated, aged
    # w/ unallocated, removal, reimbursement) keep their existing values.
    for _key, _val in aurora_data.reconcile_row_derived_totals(rows).items():
        totals[_key] = _val
    # Verification-review followup on issue #7: also reconcile `net` from
    # the (now Decimal-accurate) component totals, so `net` == revenue -
    # fees ... at the penny level. The prior fix left `net` on the float-
    # accumulator path, opening a tie-out gap between components and net.
    totals["net"] = aurora_data.sum_money([
        totals["revenue"],
        -totals["amazon_fees"],
        -totals["storage_fee"],
        -totals["product_cost"],
        -totals["inbound_shipping"],
        -totals["ad_cost"],
        -totals["return_processing_fee"],
        -totals["low_inventory_fee"],
        -totals["inbound_placement_fee"],
        -totals["aged_inventory_fee"],
        -totals["removal_fee"],
        totals["reimbursement"],
    ])
    rev = totals["revenue"]
    totals_out = {k: round(v, 2) for k, v in totals.items()}
    totals_out["units"] = int(totals["units"])
    totals_out["margin"] = round((totals["net"] / rev * 100), 1) if rev > 0 else 0.0
    totals_out["storage_report_total"] = round(float(storage_report_total or 0), 2)
    totals_out["storage_unallocated"] = storage_unallocated
    totals_out["aged_report_total"] = round(float(aged_report_total or 0), 2)
    totals_out["aged_unallocated"] = aged_unallocated
    totals_out["removal_report_total"] = round(float(removal_report_total or 0), 2)
    totals_out["removal_unallocated"] = removal_unallocated
    totals_out["removal_blended"] = bool(removal_blended)
    totals_out["reimbursement_report_total"] = round(
        float(reimbursement_report_total or 0), 2,
    )
    totals_out["reimbursement_blended"] = bool(reimbursement_blended)

    caveats = [
        f"Date range uses marketplace timezone ({mp_tz}) — same day boundaries as Aurora Orders.",
        "Units and revenue exclude Canceled, Cancelled, and Unfulfillable orders "
        "(same rules as the Aurora dashboard), and subtract returned/refunded "
        "units so Referral, FBA, Fuel, and COGS use remaining quantity only.",
    ]
    if placement_blended:
        src = placement_meta.get("source")
        if src == "finances_posted_window":
            caveats.append(
                "Inbound placement is BLENDED from Amazon Finances for this date "
                "filter — Totals match the placement-fee report Transaction date "
                "(UTC calendar day of each charge). SKU rows show BLENDED."
            )
        elif src == "zero_no_window_charges":
            ws = placement_meta.get("window_start")
            we = placement_meta.get("window_end")
            caveats.append(
                "Inbound placement Totals is $0 — no Amazon placement charges "
                f"were posted in {ws or display_start} → {we or display_end}. "
                "Other months' rates are not applied."
            )
        else:
            caveats.append(
                "Inbound placement is BLENDED to match Amazon (placement-fee report "
                "Total charge sum for the Transaction date filter). SKU rows show "
                "BLENDED; see Totals / Inbound Placement."
            )
        if (
            float(placement_window_total or 0) <= 0
            and src not in ("finances_posted_window", "zero_no_window_charges")
        ):
            ws = placement_meta.get("window_start")
            we = placement_meta.get("window_end")
            n_ev = placement_meta.get("event_count") or 0
            caveats.append(
                f"Inbound placement Totals is $0 because no Amazon placement "
                f"charges have a date in {ws} → {we}. "
                f"({n_ev} charge rows are loaded — run inventory sync so Finances/"
                f"report data refreshes, or pick a month with posted placement fees.)"
            )
    else:
        caveats.append(
            "Inbound placement Totals is $0 because no dated Amazon placement "
            "charges were found for this filter."
        )
    caveats.extend([
        (
            "Storage uses Finances FBAStorageFee for this date filter (per SellerSKU "
            "when Amazon sends one; unattributed posts are spread per unit sold). "
            "Monthly Storage Fees report rows were not available for "
            f"{', '.join(storage_months_in_window) or 'this window'} yet — Storage "
            "is never BLENDED (unlike placement / ads / removal)."
            if storage_from_finances else
            "Storage uses GET_FBA_STORAGE_FEE_CHARGES_DATA for the calendar month(s) "
            f"in your filter ({', '.join(storage_months_in_window) or 'none'}): the "
            "estimated_monthly_storage_fee total from Seller Central's Monthly Storage "
            "Fees report (same as the CSV download). Each ASIN's fee is split across "
            "SKUs that sold that ASIN; ASINs with storage but no sales stay in Totals."
        ),
        "Aged Inv uses GET_FBA_FULFILLMENT_LONGTERM_STORAGE_FEE_CHARGES_DATA "
        "(Seller Central Aged Inventory Surcharge report): sum of "
        "amount-charged per seller SKU for Event Months overlapping the window "
        "(same as the CSV download).",
        "Removal / Disposal uses GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA "
        "(Seller Central Removal Order Detail): BLENDED account total of "
        "removal-fee by request-date (Event Date) for Return + Disposal. "
        f"Months in filter: {', '.join(removal_months) or 'none'}. "
        "Never uses Finances ServiceFee spread.",
        "Reimbursements use GET_FBA_REIMBURSEMENTS_DATA (Seller Central "
        "Reimbursements report): BLENDED account income of amount-total by "
        f"approval-date. Months in filter: {', '.join(reimbursement_months) or 'none'}.",
        "Return processing and low-inventory fees come from the Finances API "
        "(45-day lookback before the window start) — recent orders may not have "
        "them yet.",
        "Ads are allocated uniformly per unit — per-SKU PPC attribution requires productAd joins.",
    ])
    if storage_report_total > 0:
        caveats.append(
            f"Storage report total for this window: ${storage_report_total:,.2f}"
            + (
                f" (${storage_unallocated:,.2f} on ASINs with no sales in the filter)."
                if storage_unallocated > 0
                else "."
            )
        )
    if aged_charges_trusted:
        caveats.append(
            f"Aged Inv report total for this window: ${aged_report_total:,.2f}"
            + (
                f" (${aged_unallocated:,.2f} on SKUs with no sales in the filter)."
                if aged_unallocated > 0
                else "."
            )
        )
    if removal_fees_trusted:
        caveats.append(
            f"Removal is BLENDED for this window: ${removal_report_total:,.2f} "
            f"(Removal Order Detail request-date total for "
            f"{', '.join(removal_months) or 'the filter'}). "
            "SKU rows show BLENDED; see Totals / Removal KPI."
        )
    elif removal_meta.get("access_denied"):
        caveats.append(
            "Removal: Removal Order Detail report is blocked (403) — "
            "showing $0 (Finances spread is disabled)."
        )
    elif removal_meta.get("source") == "unavailable":
        caveats.append(
            "Removal: Removal Order Detail unavailable for this window — "
            "showing $0 (retry; Finances spread is disabled so totals stay accurate)."
        )
    if reimbursement_trusted:
        caveats.append(
            f"Reimbursements are BLENDED for this window: "
            f"${reimbursement_report_total:,.2f} (FBA Reimbursements approval-date "
            f"total for {', '.join(reimbursement_months) or 'the filter'}). "
            "SKU rows show BLENDED; see Totals / Reimbursement."
        )
    elif reimbursement_meta.get("access_denied"):
        caveats.append(
            "Reimbursements: FBA Reimbursements report is blocked (403) — showing $0."
        )
    elif reimbursement_meta.get("source") == "unavailable":
        caveats.append(
            "Reimbursements: FBA Reimbursements report unavailable for this window — "
            "showing $0 (retry once Amazon finishes processing)."
        )
    if not storage_by_asin_month and not storage_from_finances:
        caveats.append("Storage fees: report unavailable for this window; values are 0.")
    elif storage_from_finances and float(storage_report_total or 0) <= 0:
        caveats.append(
            "Storage fees: Monthly Storage Fees report and Finances FBAStorageFee "
            "are both empty for this date filter."
        )
    if aged_charges_meta.get("access_denied"):
        caveats.append(
            "Aged Inv: the Aged Inventory Surcharge charges report is blocked "
            "(403) for this SP-API app — using Inventory Planning estimated-ais "
            "or Finances SKU amounts as fallback."
        )
    elif aged_charges_meta.get("source") == "planning_estimate_fallback":
        caveats.append(
            "Aged Inv: charges report failed (FATAL/unavailable); showing "
            "Inventory Planning estimated-ais monthly totals by SKU. Re-check "
            "against Seller Central's Aged Inventory Surcharge report once the "
            "charges report succeeds."
        )
    elif aged_charges_meta.get("source") == "finances_fallback":
        caveats.append(
            "Aged Inv: charges report unavailable and no planning estimates; "
            "using Finances API SKU-attributed aged-inventory amounts only."
        )
    placement_total = totals_out.get("inbound_placement_fee", 0)
    if placement_total == 0 and not placement_blended:
        src = placement_meta.get("source")
        if placement_meta.get("access_denied") and placement_meta.get("source") != "finances_shipment_join":
            caveats.append(
                "Inbound placement: the placement report is blocked (403) and no "
                "placement charges were found in Finances — either no placement was "
                "charged yet (~45 days after inbound) or your inbound split had "
                "$0 placement fees."
            )
        elif src in ("report_cache", "report_live") and placement_meta.get("sku_count", 0) == 0:
            caveats.append(
                "Inbound placement: the 12-month placement report returned no "
                "chargeable rows for your SKUs (common with Amazon-optimized "
                "inbound split or no minimal-split shipments)."
            )
        elif src == "finances_fallback":
            caveats.append(
                "Inbound placement: using Finances API for this sales window "
                "(placement report unavailable). Fees often post ~45 days after "
                "inbound receipt, so a short window may show $0."
            )
        elif unattributed_fees.get("inbound_placement", 0) == 0 and not fin_by_sku:
            caveats.append(
                "Inbound placement: no fee events found in Finances API for this "
                "date range. Amazon typically charges placement ~45 days after "
                "inventory is received, not when orders ship."
            )

    return {
        "days_back": round(window_days, 2),
        "window_days": round(window_days, 2),
        "start": display_start,
        "end": display_end,
        "timeZone": mp_tz,
        "created_after": created_after,
        "created_before": created_before,
        "orders_count": orders_count,
        "skus_count": len(skus),
        "rows": rows,
        "totals": totals_out,
        "missing_cogs": missing_cogs,
        "na_price_rows": na_price_rows,
        "fetch_errors": fetch_errors,
        "fee_errors": fee_errors,
        "ad_window": ad_window,
        "storage_cached_at": storage_cached_at,
        "storage_months_in_window": storage_months_in_window,
        "storage_blended": False,
        "storage_from_finances": bool(storage_from_finances),
        "storage_meta": storage_meta,
        "unattributed_fees": {k: round(v, 2) for k, v in unattributed_fees.items()},
        "placement_meta": placement_meta,
        "placement_blended": placement_blended,
        "removal_blended": bool(removal_blended),
        "removal_meta": removal_meta,
        "reimbursement_blended": bool(reimbursement_blended),
        "reimbursement_meta": reimbursement_meta,
        "caveats": caveats,
        "warnings": warnings,
        "detected_currencies": sorted(detected_currencies),
        "mixed_currency": len(detected_currencies) > 1,
        "marketplace_id": marketplace_id or None,
        "complete": partial_status["complete"],
        "partial_sections": partial_status["partial_sections"],
    }


async def _analyze_profitability(days_back: int | str = 7) -> str:
    try:
        days_back = int(days_back)
    except (TypeError, ValueError):
        days_back = 7

    data = await compute_profitability_data(days_back=days_back, paginate=False)
    if "error" in data:
        return data["error"]
    if data["orders_count"] == 0:
        return f"No orders in the last {days_back} days."
    if data["skus_count"] == 0:
        return f"Found {data['orders_count']} orders but no items with a valid price."

    table_rows = data["rows"]
    totals = data["totals"]
    missing_cogs = data["missing_cogs"]
    na_price_rows = data["na_price_rows"]
    fetch_errors = data["fetch_errors"]
    total_margin = totals["margin"]

    lines = [
        f"Profitability — last {days_back} days, {data['orders_count']} orders, {data['skus_count']} distinct SKUs.",
        "",
    ]
    if table_rows:
        # Main summary table — aggregated Amz Fees column to stay readable.
        lines.append("| SKU | Units | Revenue | Amz Fees | Storage | Ads | COGS+Inbound | Net | Margin % |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in table_rows:
            lines.append(
                f"| {r['sku']} | {r['units']} | ${r['revenue']:.2f} | "
                f"${r['amazon_fees']:.2f} | ${r['storage_fee']:.2f} | "
                f"${r['ad_cost']:.2f} | ${r['cogs_total']:.2f} | "
                f"${r['net']:.2f} | {r['margin']:.1f}% |"
            )
        lines.append(
            f"| **Totals** | **{totals['units']}** | **${totals['revenue']:.2f}** | "
            f"**${totals['amazon_fees']:.2f}** | **${totals['storage_fee']:.2f}** | "
            f"**${totals['ad_cost']:.2f}** | **${totals['cogs_total']:.2f}** | "
            f"**${totals['net']:.2f}** | **{total_margin:.1f}%** |"
        )
        lines.append("")

        # Per-SKU fee breakdown — referral / FBA / fuel split + COGS components
        lines.append("**Per-SKU fee breakdown** (Amazon fees split + COGS components):")
        lines.append("")
        lines.append("| SKU | ASIN | Avg price | Referral | FBA | Fuel surcharge | Product cost | Inbound shipping |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in table_rows:
            lines.append(
                f"| {r['sku']} | {r['asin'] or '—'} | ${r['avg_price']:.2f} | "
                f"${r['referral_fee']:.2f} | ${r['fba_fee']:.2f} | ${r['fuel_surcharge']:.2f} | "
                f"${r['product_cost']:.2f} | ${r['inbound_shipping']:.2f} |"
            )
        lines.append(
            f"| **Totals** | — | — | **${totals['referral_fee']:.2f}** | "
            f"**${totals['fba_fee']:.2f}** | **${totals['fuel_surcharge']:.2f}** | "
            f"**${totals['product_cost']:.2f}** | **${totals['inbound_shipping']:.2f}** |"
        )
        lines.append("")
        lines.append(
            f"Net profit ${totals['net']:.2f} on ${totals['revenue']:.2f} revenue across "
            f"{totals['units']} units ({total_margin:.1f}% margin)."
        )
        lines.append("")
        lines.append(
            "Formula: Revenue − Referral − FBA − Fuel (Product Fees API per ASIN at avg price) "
            "− Storage (FBA Storage Charges report, per-ASIN monthly × months in window) "
            "− Ads (Aurora total spend pro-rated to window, uniform per unit) "
            "− Product cost − Inbound shipping (your COGS CSV)."
        )
        ad_window = data.get("ad_window") or {}
        if ad_window.get("total_window"):
            lines.append(
                f"Ad allocation: ${ad_window['total_window']:.2f} spread across "
                f"{int(totals['units'])} units = ${ad_window['total_window'] / max(int(totals['units']), 1):.3f}/unit "
                f"(from {ad_window.get('campaign_count', 0)} Aurora campaigns)."
            )
        if data.get("storage_cached_at"):
            lines.append(f"Storage allocation from FBA report cached at {data['storage_cached_at']}.")
    else:
        lines.append("No rows produced — likely no orders with valid item prices in the window.")

    caveats = []
    for m in missing_cogs:
        caveats.append(
            f"- COGS missing for '{m['sku']}' ({m['units']} units, ${m['revenue']:.2f} revenue) — "
            "excluded from totals. Upload via the COGS panel."
        )
    for n in na_price_rows[:5]:
        caveats.append(f"- Order {n['order_id']} ('{n['sku']}' qty {n['qty']}) had price N/A — row excluded.")
    if len(na_price_rows) > 5:
        caveats.append(f"- …and {len(na_price_rows) - 5} more N/A-price rows excluded.")
    for err in fetch_errors[:3]:
        caveats.append(f"- Failed to fetch items for {err}")
    for err in (data.get("fee_errors") or [])[:3]:
        caveats.append(f"- Fees API failed for {err}")
    for w in data.get("warnings") or []:
        caveats.append(f"- {w}")
    for c in data.get("caveats") or []:
        caveats.append(f"- {c}")
    lines.append("")
    lines.append("Caveats:")
    lines.extend(caveats)
    return "\n".join(lines)


async def _get_report(report_type: str, days_back: int | str = 30) -> str:
    try:
        days_back = int(days_back)
    except (TypeError, ValueError):
        days_back = 30
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        result = await amazon_sp.create_report(report_type, start_date=start_str, end_date=end_str)
        report_id = result.get("reportId")
        if not report_id:
            return f"Failed to create report. Response: {json.dumps(result)}"
        return await amazon_sp.download_report(report_id)
    except Exception as e:
        return f"Error generating report: {e}"


# ── Forecasting tools (read from forecastCache, never refit on demand) ──

async def _forecast_sku(sku: str, horizon_days: int | str = 30) -> str:
    try:
        horizon_days = max(1, min(int(horizon_days), 90))
    except (TypeError, ValueError):
        horizon_days = 30
    cached = await get_forecast_cache(skus=[sku])
    if not cached:
        return (
            f"No cached forecast for **{sku}**. The forecast cache is refreshed by the "
            "nightly job at 03:00 UTC, or the user can trigger it manually via "
            "POST /forecasting/refresh."
        )
    c = cached[0]
    fc = c.get("forecast") or []
    window = fc[:horizon_days]
    total_p50 = sum(float(r.get("p50", 0)) for r in window)
    total_p90 = sum(float(r.get("p90", 0)) for r in window)
    daily_avg = total_p50 / max(1, len(window))
    drivers = c.get("drivers") or {}
    reorder = c.get("reorder") or {}
    growth_pct = float(drivers.get("growth_rate") or 0) * 100
    lines = [
        f"### Forecast for {sku} — next {horizon_days} days",
        f"- Method: **{c.get('method')}**",
        f"- Total demand p50: **{total_p50:.0f}** units (p90 upper bound: {total_p90:.0f})",
        f"- Avg daily demand: **{daily_avg:.1f}** units/day",
        f"- Recent 28-day average: {drivers.get('recent_avg', 0)}/day, growth {growth_pct:+.1f}%",
        "",
        "### Restock math",
        f"- On hand: **{reorder.get('on_hand', 0)}** | inbound: {reorder.get('inbound', 0)}",
        f"- Days of cover: **{reorder.get('days_of_cover', 'n/a')}**",
        f"- Reorder by: **{reorder.get('reorder_by_date') or 'n/a'}**",
        f"- Suggested PO qty: **{reorder.get('recommended_po_qty', 0)}** (MOQ {reorder.get('moq', 1)})",
        f"- Safety stock: {reorder.get('safety_stock', 0)}, reorder point: {reorder.get('reorder_point', 0)}",
    ]
    return "\n".join(lines)


async def _restock_recommendations(top_n: int | str = 10) -> str:
    try:
        top_n = max(1, min(int(top_n), 50))
    except (TypeError, ValueError):
        top_n = 10
    cached = await get_forecast_cache()
    if not cached:
        return "No forecasts cached yet. Run POST /forecasting/refresh to generate them."
    rows: list[dict] = []
    for c in cached:
        r = c.get("reorder") or {}
        rows.append({
            "sku": c["sku"],
            "on_hand": r.get("on_hand", 0),
            "inbound": r.get("inbound", 0),
            "days_of_cover": r.get("days_of_cover"),
            "reorder_by_date": r.get("reorder_by_date"),
            "recommended_po_qty": r.get("recommended_po_qty", 0),
        })
    # Stockouts (None / 0 cover) bubble to the top.
    rows.sort(key=lambda r: (1e9 if r["days_of_cover"] is None else r["days_of_cover"]))
    rows = rows[:top_n]

    header = "| SKU | On hand | Inbound | Days of cover | Reorder by | Suggested PO |"
    sep    = "| --- | --- | --- | --- | --- | --- |"
    out_lines = [header, sep]
    for r in rows:
        doc = r["days_of_cover"]
        if doc is None:
            cover_cell = "no demand"
        elif doc <= 7:
            cover_cell = f"⚠ {doc:.1f}"
        else:
            cover_cell = f"{doc:.1f}"
        out_lines.append(
            f"| {r['sku']} | {r['on_hand']} | {r['inbound']} | "
            f"{cover_cell} | {r['reorder_by_date'] or '—'} | {r['recommended_po_qty']} |"
        )
    return "### Top restock priorities\n" + "\n".join(out_lines)


async def _days_until_stockout(sku: str | None = None) -> str:
    skus = [sku] if sku else None
    cached = await get_forecast_cache(skus=skus)
    if not cached:
        scope = f"SKU {sku}" if sku else "any SKU"
        return f"No forecast data for {scope}. Run POST /forecasting/refresh first."
    if sku:
        r = (cached[0].get("reorder") or {})
        doc = r.get("days_of_cover")
        if doc is None:
            return f"**{sku}** has no recent demand signal — days of cover is undefined."
        return (
            f"**{sku}** has approximately **{doc:.1f} days** of cover remaining "
            f"({r.get('on_hand', 0)} on hand + {r.get('inbound', 0)} inbound)."
        )
    rows = []
    for c in cached:
        r = c.get("reorder") or {}
        rows.append((c["sku"], r.get("days_of_cover")))
    rows.sort(key=lambda r: (1e9 if r[1] is None else r[1]))
    top = rows[:15]
    body = "\n".join(
        f"- {s}: {'no demand signal' if d is None else f'{d:.1f} days'}"
        for s, d in top
    )
    return "### Days of cover (15 riskiest SKUs)\n" + body


TOOL_FUNCTIONS = {
    "get_campaigns_summary": _get_campaigns_summary,
    "get_keywords": _get_keywords,
    "get_negative_keywords": _get_negative_keywords,
    "analyze_campaign_performance": _analyze_campaign_performance,
    "create_campaign": _create_campaign,
    "search_meta_ads": _search_meta_ads,
    "get_orders": _get_orders,
    "get_order_items": _get_order_items,
    "get_inventory": _get_inventory,
    "get_cogs": _get_cogs,
    "analyze_profitability": _analyze_profitability,
    "get_report": _get_report,
    "get_marketplaces": lambda: _format_marketplaces(),
    "forecast_sku": _forecast_sku,
    "restock_recommendations": _restock_recommendations,
    "days_until_stockout": _days_until_stockout,
}


def _format_marketplaces() -> str:
    rows = data_resolver.list_marketplaces_resolved(require_user())
    if not rows:
        return "No marketplaces registered on this account."
    lines = ["The user is registered in the following marketplaces:"]
    for r in rows:
        flag = " (primary)" if r["is_primary"] else ""
        lines.append(f"- {r['id']} — {r['name']}{flag}")
    lines.append("\nWhen scoping a tool, pass marketplace=<id> or marketplace=<short name like 'US'>.")
    return "\n".join(lines)

# ── Helpers ─────────────────────────────────────────────────────────────────

async def _call_tool(fn, args: dict) -> str:
    """Call a tool function, handling both sync and async functions."""
    result = fn(**args) if not inspect.iscoroutinefunction(fn) else await fn(**args)
    return result if isinstance(result, str) else json.dumps(result)


# ── Main streaming entry point ──────────────────────────────────────────────

async def stream_response(user_message: str, *, session_id: str = "default") -> AsyncGenerator[dict, None]:
    """Stream a response as dict events, maintaining conversation history per session.

    Yields dicts with a "type" key:
      {"type": "token",       "content": "..."}
      {"type": "tool_start",  "name": "...", "args": {...}}
      {"type": "tool_result", "name": "...", "result": "..."}
      {"type": "error",       "content": "..."}
    The caller is responsible for sending a final {"type": "done"} after iteration.
    """
    # Ensure session exists in DB; auto-title from first user message
    await create_session(session_id)
    history = await get_messages(session_id)

    if not history:
        title = user_message[:50].strip()
        await update_session_title(session_id, title)

    await save_message(session_id, "user", user_message)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    max_tool_rounds = 25
    for round_num in range(max_tool_rounds):
        # ── Streaming call with tools ──────────────────────────────────────
        try:
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                stream=True,
                temperature=0.3,
                max_tokens=1024,
                # Discourage the model from emitting the same n-gram over and
                # over on vague prompts ("last 7 ?").
                frequency_penalty=0.5,
                presence_penalty=0.3,
            )
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            yield {"type": "error", "content": error_msg}
            await save_message(session_id, "assistant", error_msg)
            return

        # Accumulate the full response from streamed deltas
        text_chunks: list[str] = []
        # tool_calls_acc: dict of index -> {id, name, arguments_parts}
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None
        # Repetition guard: count how many times the most recent non-trivial
        # line has appeared. If it crosses a threshold, abort the stream —
        # the model is stuck in a loop and the rest is garbage.
        last_line = ""
        repeat_count = 0
        REPEAT_LIMIT = 6

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason or finish_reason

                # Text token
                if delta.content:
                    text_chunks.append(delta.content)
                    yield {"type": "token", "content": delta.content}

                    # Detect repetition on completed lines only — token-level
                    # comparison would false-positive on legitimate prose.
                    if "\n" in delta.content:
                        recent_text = "".join(text_chunks)
                        lines = [l.strip() for l in recent_text.splitlines() if l.strip()]
                        if lines:
                            current = lines[-1]
                            if current == last_line and len(current) >= 3:
                                repeat_count += 1
                                if repeat_count >= REPEAT_LIMIT:
                                    print(f"[agent] repetition loop detected on: {current!r} — aborting stream")
                                    await stream.close()
                                    break
                            else:
                                last_line = current
                                repeat_count = 0

                # Tool-call deltas arrive incrementally
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name or "" if tc_delta.function else "",
                                "arguments_parts": [],
                            }
                        acc = tool_calls_acc[idx]
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            acc["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            acc["arguments_parts"].append(tc_delta.function.arguments)
        except Exception as e:
            error_msg = f"Sorry, the LLM stream failed: {e}"
            print(f"[agent] stream error: {e}")
            yield {"type": "error", "content": error_msg}
            await save_message(session_id, "assistant", error_msg)
            return

        # ── Handle tool calls ──────────────────────────────────────────────
        if tool_calls_acc:
            # Build the assistant message for the conversation
            tool_calls_list = []
            for idx in sorted(tool_calls_acc):
                acc = tool_calls_acc[idx]
                tool_calls_list.append({
                    "id": acc["id"],
                    "type": "function",
                    "function": {
                        "name": acc["name"],
                        "arguments": "".join(acc["arguments_parts"]),
                    },
                })

            assistant_msg = {"role": "assistant", "tool_calls": tool_calls_list}
            await save_message(session_id, "tool_call", json.dumps(assistant_msg))
            messages.append(assistant_msg)

            # Execute each tool
            for tc in tool_calls_list:
                fn_name = tc["function"]["name"]
                fn = TOOL_FUNCTIONS.get(fn_name)
                raw_args = tc["function"]["arguments"]
                args = json.loads(raw_args or "{}") or {}

                yield {"type": "tool_start", "name": fn_name, "args": args}

                if fn:
                    try:
                        print(f"[agent] tool call: {fn_name} args={args}")
                        result = await _call_tool(fn, args)
                        print(f"[agent] tool done: {fn_name} -> {str(result)[:200]}")
                    except Exception as e:
                        print(f"[agent] tool error: {fn_name} -> {e}")
                        result = f"Tool error: {e}"
                else:
                    result = "Unknown tool"

                yield {"type": "tool_result", "name": fn_name, "result": result}

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
                await save_message(session_id, "tool", json.dumps(tool_msg))
                messages.append(tool_msg)

            # Continue loop — let the LLM decide if it needs more tools
            continue

        # ── No tool calls — this was the final text response ───────────────
        full_text = "".join(text_chunks)
        await save_message(session_id, "assistant", full_text)
        return

    # ── Exhausted tool rounds — force a text-only response ─────────────────
    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            temperature=0.3,
            max_tokens=1024,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        full_reply: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_reply.append(delta.content)
                yield {"type": "token", "content": delta.content}

        await save_message(session_id, "assistant", "".join(full_reply))
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {e}"
        yield {"type": "error", "content": error_msg}
        await save_message(session_id, "assistant", error_msg)
