import asyncio
import inspect
import json
import os
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

from groq import AsyncGroq

from campaigns import get_campaigns_summary, create_campaign, analyze_performance, search_campaigns
from keywords import fetch_amazon_keywords, suggest_negative_keywords
from meta_ads import search_meta_ads
from database import create_session, get_messages, save_message, update_session_title, get_cogs, get_forecast_cache
import amazon_sp

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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
    "GET_FBA_STORAGE_FEE_CHARGES_DATA (monthly storage), GET_FBA_INVENTORY_AGED_DATA (aged surcharge), "
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
        data = await amazon_sp.get_orders(
            created_after=created_after, statuses=statuses, marketplace=marketplace,
        )
    except Exception as e:
        return f"Error fetching orders: {e}"

    orders = data.get("payload", {}).get("Orders", [])
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
        data = await amazon_sp.get_inventory_summaries(skus=skus, marketplace=marketplace)
    except Exception as e:
        return f"Error fetching inventory: {e}"
    summaries = data.get("payload", {}).get("inventorySummaries", [])
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
        data = await amazon_sp.get_order_items(order_id)
    except Exception as e:
        return f"Error fetching order items: {e}"
    items = data.get("payload", {}).get("OrderItems", [])
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


async def compute_profitability_data(
    days_back: int = 7,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    paginate: bool = False,
) -> dict:
    """Per-SKU profitability following the FBA Profitability Calculator PDF:

      Net = Selling Price
            − Referral Fee
            − FBA Fulfilment Fee
            − Fuel Surcharge (3.5% of FBA)
            − Allocated Storage Fee
            − Product Cost (COGS)
            − Shipping Cost to Amazon (inbound)
            − Advertising Cost (allocated)

    Amazon fees (referral / FBA / fuel) come from Product Fees API per ASIN.
    Storage comes from GET_FBA_STORAGE_FEE_CHARGES_DATA (cached 24h — first
    cold call adds 30–120 s). Ads come from Aurora campaigns pro-rated to
    the window and allocated per SKU when a campaign lists its SKUs;
    otherwise spread uniformly across units sold. Returns / removals /
    low-inv / inbound-placement / aged-inv fees come from Finances API.

    Window is defined by `start`/`end` (YYYY-MM-DD strings or datetimes)
    when either is provided; falls back to `days_back` for legacy callers
    (LLM tool). End-exclusive vs Amazon docs is orders-`CreatedBefore`;
    for the FE picker that's inclusive-of-end-date so we push end to
    23:59:59Z."""
    from campaigns import get_ad_spend_for_range
    from database import get_storage_cache, put_storage_cache

    # Normalize window → (created_after ISO Z, created_before ISO Z or None,
    # window_days float).
    now = datetime.now(timezone.utc)
    def _parse_dt(v, end_of_day: bool = False) -> datetime | None:
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v.replace(tzinfo=v.tzinfo or timezone.utc)
        s = str(v).strip()
        # Accept 'YYYY-MM-DD' or full ISO.
        if len(s) == 10:
            d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if end_of_day:
                d = d.replace(hour=23, minute=59, second=59)
            return d
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end, end_of_day=True)
    if start_dt or end_dt:
        if not start_dt:
            start_dt = now - timedelta(days=days_back)
        if not end_dt:
            end_dt = now
    else:
        end_dt = now
        start_dt = end_dt - timedelta(days=days_back)

    # SP-API rejects CreatedBefore < 2 minutes before "now" — a 23:59Z
    # end-of-day for today's date is in the future. Clamp so `end_dt`
    # never exceeds now-3min. Preserves the user's intent (whole day
    # of history) while keeping Amazon happy.
    api_ceiling = now - timedelta(minutes=3)
    if end_dt > api_ceiling:
        end_dt = api_ceiling

    created_after = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    created_before = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    window_days = max((end_dt - start_dt).total_seconds() / 86400.0, 0.5)

    try:
        orders_resp = await amazon_sp.get_orders(
            created_after=created_after,
            created_before=created_before,
            paginate=paginate,
        )
    except Exception as e:
        return {"error": f"Error fetching orders: {e}"}
    orders = orders_resp.get("payload", {}).get("Orders", [])
    if not orders:
        return {
            "orders_count": 0, "skus_count": 0, "rows": [], "totals": None,
            "missing_cogs": [], "na_price_rows": [], "fetch_errors": [],
            "days_back": round(window_days, 2), "window_days": round(window_days, 2),
            "start": start_dt.date().isoformat(), "end": end_dt.date().isoformat(),
            "created_after": created_after, "created_before": created_before,
            "caveats": [], "warnings": [],
        }

    order_ids = [o.get("AmazonOrderId") for o in orders if o.get("AmazonOrderId")]

    # SP-API GetOrderItems is 0.5 req/s sustained — pace sequentially with
    # exponential backoff on 429 so a busy quota doesn't drop every order.
    items_results: list[tuple[str, dict]] = []
    pending = list(order_ids)
    attempt = 1
    while pending and attempt <= 4:
        base_sleep = 2.1 * attempt
        failed: list[str] = []
        for oid in pending:
            try:
                resp = await amazon_sp.get_order_items(oid)
                items_results.append((oid, resp))
            except Exception as e:
                msg = str(e)
                if "429" in msg or "QuotaExceeded" in msg:
                    failed.append(oid)
                    await asyncio.sleep(base_sleep * 2)
                else:
                    items_results.append((oid, {"_error": msg}))
                    continue
            await asyncio.sleep(base_sleep)
        pending = failed
        attempt += 1
        if pending:
            await asyncio.sleep(20)
    # Anything still pending after the retry loop = permanent failure
    for oid in pending:
        items_results.append((oid, {"_error": "rate limited after 4 retries"}))

    sku_data: dict[str, dict] = defaultdict(
        lambda: {"units": 0, "revenue": 0.0, "asin": None}
    )
    na_price_rows: list[dict] = []
    fetch_errors: list[str] = []

    for oid, items_resp in items_results:
        if "_error" in items_resp:
            fetch_errors.append(f"{oid}: {items_resp['_error']}")
            continue
        for it in items_resp.get("payload", {}).get("OrderItems", []):
            sku = it.get("SellerSKU")
            if not sku:
                continue
            qty = int(it.get("QuantityOrdered", 0) or 0)
            price = it.get("ItemPrice") or {}
            amount_raw = price.get("Amount")
            try:
                amount = float(amount_raw) if amount_raw is not None else None
            except (TypeError, ValueError):
                amount = None
            if amount is None:
                na_price_rows.append({"order_id": oid, "sku": sku, "qty": qty})
                continue
            sku_data[sku]["units"] += qty
            sku_data[sku]["revenue"] += amount
            if not sku_data[sku]["asin"] and it.get("ASIN"):
                sku_data[sku]["asin"] = it["ASIN"]

    skus = sorted(sku_data.keys())
    cogs_map: dict = {}
    if skus:
        try:
            cogs_rows = await get_cogs(skus)
            cogs_map = {r["sku"]: r for r in cogs_rows}
        except Exception as e:
            return {"error": f"Error reading COGS: {e}"}

    # ── Storage allocation (cached 24h) — keyed by ASIN, not SKU ──────────
    warnings: list[str] = []
    storage_per_asin_monthly: dict = {}
    storage_cached_at: str | None = None
    storage_meta = await get_storage_cache(max_age_hours=24)
    if storage_meta:
        storage_per_asin_monthly = storage_meta.get("per_sku_monthly") or {}
        storage_cached_at = storage_meta.get("updated_at")
    else:
        try:
            storage_per_asin_monthly, months = await amazon_sp.fetch_storage_fees_per_sku(months_back=2)
            await put_storage_cache(storage_per_asin_monthly, months)
            storage_cached_at = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            warnings.append(f"Storage report fetch failed ({e}); storage fees set to 0.")

    # ── Ad allocation ────────────────────────────────────────────────────
    # Per-SKU when a campaign lists its SKUs (Aurora Ad.skus[]); campaigns
    # without a SKU list fall into `unattributed` and get spread uniformly
    # per unit (legacy behavior).
    ad_window = {
        "total_window": 0.0, "campaign_count": 0,
        "by_sku": {}, "unattributed": 0.0,
    }
    try:
        ad_window = await get_ad_spend_for_range(start=start_dt, end=end_dt)
    except Exception as e:
        warnings.append(f"Ad spend fetch failed ({e}); ad cost set to 0.")
    ad_by_sku: dict[str, float] = ad_window.get("by_sku") or {}
    ad_unattributed: float = float(ad_window.get("unattributed") or 0.0)
    total_units_window = sum(d["units"] for d in sku_data.values())
    ad_unattr_per_unit = (
        ad_unattributed / total_units_window if total_units_window > 0 else 0.0
    )

    # ── Finances API — return processing, low inventory, inbound placement,
    # aged inventory surcharge, removal fees. Per-SKU where Amazon
    # attributes; otherwise the totals land in `unattributed_fees` and
    # get spread uniformly per unit (same treatment as ads).
    fin_by_sku: dict[str, dict] = {}
    unattributed_fees = amazon_sp._empty_fee_bucket()
    try:
        fin = await amazon_sp.get_financial_events(
            posted_after=created_after, paginate=True,
        )
        fin_by_sku = fin.get("by_sku") or {}
        unattributed_fees = fin.get("unattributed") or unattributed_fees
    except Exception as e:
        warnings.append(
            f"Finances API fetch failed ({e}); return / low-inv / placement "
            "/ aged / removal fees set to 0."
        )

    # Fees Amazon posted without a SellerSKU get spread across units — same
    # allocation model we use for ads. Removals often lack SKU attribution
    # (aggregated adjustments), so this matters.
    unattr_per_unit = {
        k: (unattributed_fees.get(k, 0.0) / total_units_window
            if total_units_window > 0 else 0.0)
        for k in ("return_processing", "low_inventory", "inbound_placement",
                  "aged_inventory", "removal")
    }

    # ── Per-SKU enrichment: Fees API → referral, FBA, fuel ────────────────
    months_in_window = max(window_days / 30.0, 0.01)
    rows: list[dict] = []
    totals = defaultdict(float)
    missing_cogs: list[dict] = []
    fee_errors: list[str] = []

    for sku in skus:
        d = sku_data[sku]
        units = d["units"]
        revenue = round(d["revenue"], 2)
        avg_price = revenue / units if units else 0.0
        asin = d["asin"]

        # Amazon fees per unit at the SKU's average selling price
        referral = fba = fuel = 0.0
        if asin and avg_price > 0:
            try:
                est = await amazon_sp.get_fees_estimate(asin, avg_price, is_fba=True)
                referral = est["referral"]
                fba = est["fba"]
                fuel = est["fuel_surcharge"]
            except Exception as e:
                fee_errors.append(f"{sku} ({asin}): {str(e)[:120]}")
        elif not asin:
            fee_errors.append(f"{sku}: no ASIN found in order items")

        referral_total = round(referral * units, 2)
        fba_total = round(fba * units, 2)
        fuel_total = round(fuel * units, 2)
        amazon_fees = round(referral_total + fba_total + fuel_total, 2)

        # Storage allocation: monthly per-ASIN × months in window
        storage = round(
            float(storage_per_asin_monthly.get(asin or "", 0)) * months_in_window, 2,
        )

        # Ads: per-campaign attribution when the campaign lists this SKU,
        # plus a share of the unattributed pool spread per unit.
        ad_cost = round(
            float(ad_by_sku.get(sku, 0.0)) + ad_unattr_per_unit * units, 2,
        )

        # Finances API fees per SKU + share of unattributed pool
        fin_sku = fin_by_sku.get(sku, {})
        return_processing_fee = round(
            (fin_sku.get("return_processing", 0.0)
             + unattr_per_unit["return_processing"] * units), 2)
        low_inventory_fee = round(
            (fin_sku.get("low_inventory", 0.0)
             + unattr_per_unit["low_inventory"] * units), 2)
        inbound_placement_fee = round(
            (fin_sku.get("inbound_placement", 0.0)
             + unattr_per_unit["inbound_placement"] * units), 2)
        aged_inventory_fee = round(
            (fin_sku.get("aged_inventory", 0.0)
             + unattr_per_unit["aged_inventory"] * units), 2)
        removal_fee = round(
            (fin_sku.get("removal", 0.0)
             + unattr_per_unit["removal"] * units), 2)

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
            - aged_inventory_fee - removal_fee,
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
            "return_processing_fee": return_processing_fee,
            "low_inventory_fee": low_inventory_fee,
            "inbound_placement_fee": inbound_placement_fee,
            "aged_inventory_fee": aged_inventory_fee,
            "removal_fee": removal_fee,
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
        totals["net"] += net

    rev = totals["revenue"]
    totals_out = {k: round(v, 2) for k, v in totals.items()}
    totals_out["units"] = int(totals["units"])
    totals_out["margin"] = round((totals["net"] / rev * 100), 1) if rev > 0 else 0.0

    caveats = [
        "Return processing, low-inventory, inbound placement, aged-inventory, "
        "and removal fees come from the Finances API — Amazon posts these "
        "at settlement, so very recent orders may not have them yet.",
        "Ads are allocated uniformly per unit — per-SKU PPC attribution requires productAd joins.",
        "Fees posted without a SellerSKU (typically removals) are spread across units proportionally.",
    ]
    if not storage_per_asin_monthly:
        caveats.append("Storage fees: report unavailable for this window; values are 0.")

    return {
        "days_back": round(window_days, 2),
        "window_days": round(window_days, 2),
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "created_after": created_after,
        "created_before": created_before,
        "orders_count": len(orders),
        "skus_count": len(skus),
        "rows": rows,
        "totals": totals_out,
        "missing_cogs": missing_cogs,
        "na_price_rows": na_price_rows,
        "fetch_errors": fetch_errors,
        "fee_errors": fee_errors,
        "ad_window": ad_window,
        "storage_cached_at": storage_cached_at,
        "unattributed_fees": {k: round(v, 2) for k, v in unattributed_fees.items()},
        "caveats": caveats,
        "warnings": warnings,
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
    rows = amazon_sp.list_marketplaces()
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
    if not (GROQ_API_KEY or "").strip():
        yield {
            "type": "error",
            "content": (
                "GROQ_API_KEY is not configured. Add it to aiModel/.env, then restart "
                "the AI service (uvicorn on port 8000)."
            ),
        }
        return

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
                # over — llama-4-scout is prone to token-repetition loops on
                # vague prompts ("last 7 ?") and needs a nudge.
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
