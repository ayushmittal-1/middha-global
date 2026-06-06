import inspect
import json
import os
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from campaigns import get_campaigns_summary, create_campaign, analyze_performance, search_campaigns
from keywords import fetch_amazon_keywords, suggest_negative_keywords
from meta_ads import search_meta_ads
from database import create_session, get_messages, save_message, update_session_title
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
    "When users ask about campaign performance, health, or what to optimize, call analyze_campaign_performance.\n\n"
    "## Competitor research\n"
    "- **search_meta_ads**: Search the public Meta (Facebook) Ad Library for competitor ads. "
    "Use this when the user asks about competitor ads, wants to see what others are advertising, "
    "or wants ad copy inspiration. Pass a product/brand query and optionally a country code.\n\n"
    "## Selling Partner API (Orders, Inventory, Reports)\n"
    "- **get_orders**: Fetch recent Amazon orders. Use when the user asks about order status, recent sales, shipments, or order history. "
    "Pass days_back (default 7) and optionally a status filter (Unshipped, Shipped, PartiallyShipped, Canceled, Unfulfillable).\n"
    "- **get_inventory**: Check FBA inventory levels. Use when the user asks about stock, inventory, or supply levels. "
    "Optionally pass a list of SKUs to filter.\n"
    "- **get_report**: Generate and download an Amazon report. Use when the user asks for sales reports, settlement reports, "
    "or other data exports. Common report types: GET_FLAT_FILE_ALL_ORDERS_DATA_BY_ORDER_DATE_GENERAL, "
    "GET_FBA_MYI_UNSUPPRESSED_INVENTORY_DATA, GET_V2_SETTLEMENT_REPORT_DATA_FLAT_FILE.\n\n"
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
            "description": "Fetch recent Amazon orders from the Selling Partner API. Returns order IDs, dates, statuses, and totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to look back (default 7).",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by order status.",
                        "enum": ["Unshipped", "PartiallyShipped", "Shipped", "Canceled", "Unfulfillable"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory",
            "description": "Check FBA inventory levels via the Selling Partner API. Returns fulfillable, inbound, reserved, and unfulfillable quantities per SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of seller SKUs to filter. Omit for all inventory.",
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
                        ],
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to cover (default 30).",
                    },
                },
                "required": ["report_type"],
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


def _analyze_campaign_performance() -> str:
    return analyze_performance()


async def _search_meta_ads(query: str, country: str = "US", active_only: bool = True) -> str:
    return await search_meta_ads(query, country=country, active_only=active_only)


def _get_campaigns_summary(query: str = "") -> str:
    if query:
        return search_campaigns(query)
    return get_campaigns_summary()


async def _get_orders(days_back: int = 7, status: str | None = None) -> str:
    from datetime import datetime, timedelta, timezone
    created_after = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    statuses = [status] if status else None
    try:
        data = await amazon_sp.get_orders(created_after=created_after, statuses=statuses)
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


async def _get_inventory(skus: list[str] | None = None) -> str:
    try:
        data = await amazon_sp.get_inventory_summaries(skus=skus)
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


async def _get_report(report_type: str, days_back: int = 30) -> str:
    from datetime import datetime, timedelta, timezone
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


TOOL_FUNCTIONS = {
    "get_campaigns_summary": _get_campaigns_summary,
    "get_keywords": _get_keywords,
    "get_negative_keywords": _get_negative_keywords,
    "analyze_campaign_performance": _analyze_campaign_performance,
    "create_campaign": _create_campaign,
    "search_meta_ads": _search_meta_ads,
    "get_orders": _get_orders,
    "get_inventory": _get_inventory,
    "get_report": _get_report,
}

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

    max_tool_rounds = 5
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

        async for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason or finish_reason

            # Text token
            if delta.content:
                text_chunks.append(delta.content)
                yield {"type": "token", "content": delta.content}

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
