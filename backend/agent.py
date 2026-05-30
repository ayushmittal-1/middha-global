import inspect
import json
import os
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from campaigns import get_campaigns_summary, create_campaign, analyze_performance
from keywords import fetch_amazon_keywords, suggest_negative_keywords
from database import create_session, get_messages, save_message, update_session_title

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"

client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a campaign analyst and creation assistant for Amazon advertising. "
    "You can fetch campaign data, analyze performance, suggest keywords, and help users create new campaigns.\n\n"
    "## Tools at your disposal\n"
    "- **get_campaigns_summary**: Fetch all campaign data (pipe-delimited). Columns: "
    "nm=name|tp=type(SP/SB/SD)|st=status(E/P/A)|co=country|bgt=budget|spd=spend|sal=sales|dt=start date. Values in USD.\n"
    "- **get_keywords**: Fetch keyword suggestions from Amazon Autocomplete for a seed keyword.\n"
    "- **get_negative_keywords**: Get negative keyword suggestions to exclude wasteful/irrelevant terms from a campaign.\n"
    "- **analyze_campaign_performance**: Analyze campaign health — ACOS, ROI, and actionable recommendations.\n"
    "- **create_campaign**: Create a campaign (only after user approves keywords & details).\n\n"
    "## Campaign creation flow\n"
    "1. Ask about the product or seed keyword.\n"
    "2. Call get_keywords to fetch suggestions.\n"
    "3. Optionally call get_negative_keywords to suggest exclusions.\n"
    "4. Present keywords to user for approval.\n"
    "5. Collect campaign details (name, type, budget, country) if not provided.\n"
    "6. Call create_campaign only after user approval.\n\n"
    "## Performance analysis\n"
    "When users ask about campaign performance, health, or what to optimize, call analyze_campaign_performance.\n\n"
    "Be concise and actionable."
)

# ── Tool definitions ────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_campaigns_summary",
            "description": "Fetch a pipe-delimited summary of all advertising campaigns including name, type, status, country, budget, spend, sales, and start date.",
            "parameters": {"type": "object", "properties": {}},
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
                },
                "required": ["campaign_name", "campaign_type", "budget", "country", "keywords"],
            },
        },
    },
]

# ── Tool function wrappers ──────────────────────────────────────────────────

async def _get_keywords(seed_keyword: str) -> str:
    keywords = await fetch_amazon_keywords(seed_keyword)
    if not keywords:
        return "No keyword suggestions found. Try a different seed keyword."
    return json.dumps(keywords)


async def _get_negative_keywords(seed_keyword: str) -> str:
    negatives = await suggest_negative_keywords(seed_keyword)
    if not negatives:
        return "No negative keyword suggestions found."
    return json.dumps(negatives)


def _create_campaign(campaign_name: str, campaign_type: str, budget: float, country: str, keywords: list[str]) -> str:
    return create_campaign({
        "campaign_name": campaign_name,
        "campaign_type": campaign_type,
        "budget": budget,
        "country": country,
        "keywords": keywords,
    })


def _analyze_campaign_performance() -> str:
    return analyze_performance()


TOOL_FUNCTIONS = {
    "get_campaigns_summary": get_campaigns_summary,
    "get_keywords": _get_keywords,
    "get_negative_keywords": _get_negative_keywords,
    "analyze_campaign_performance": _analyze_campaign_performance,
    "create_campaign": _create_campaign,
}

# ── Helpers ─────────────────────────────────────────────────────────────────

async def _call_tool(fn, args: dict) -> str:
    """Call a tool function, handling both sync and async functions."""
    result = fn(**args) if not inspect.iscoroutinefunction(fn) else await fn(**args)
    return result if isinstance(result, str) else json.dumps(result)


# ── Main streaming entry point ──────────────────────────────────────────────

async def stream_response(user_message: str, *, session_id: str = "default") -> AsyncGenerator[str, None]:
    """Stream a response, maintaining conversation history per session."""
    # Ensure session exists in DB; auto-title from first user message
    await create_session(session_id)
    history = await get_messages(session_id)

    # Auto-set title from first user message (if this is the first message)
    if not history:
        title = user_message[:50].strip()
        await update_session_title(session_id, title)

    # Save the incoming user message
    await save_message(session_id, "user", user_message)
    history.append({"role": "user", "content": user_message})

    # Build messages: system + history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # First call — the LLM may decide to call a tool
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        temperature=0.3,
        max_tokens=1024,
    )

    choice = response.choices[0]

    # If the model wants to call tool(s), execute them and make a follow-up request
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        # Persist assistant tool-call message
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ],
        }
        await save_message(session_id, "tool_call", json.dumps(assistant_msg))
        messages.append(assistant_msg)

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                result = await _call_tool(fn, args)
            else:
                result = "Unknown tool"
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            await save_message(session_id, "tool", json.dumps(tool_msg))
            messages.append(tool_msg)

        # Second call — stream the final answer with tool results
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=True,
            temperature=0.3,
            max_tokens=1024,
        )

        full_reply = []
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_reply.append(delta.content)
                yield delta.content

        # Persist assistant reply
        await save_message(session_id, "assistant", "".join(full_reply))
    else:
        # No tool call — just return the content directly
        content = choice.message.content or ""
        if content:
            yield content
        await save_message(session_id, "assistant", content)
