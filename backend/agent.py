import json
import os
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from campaigns import get_campaigns_summary

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"

client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a campaign analyst. Use the get_campaigns_summary tool to fetch campaign data "
    "when the user asks about campaigns. Values are in USD. "
    "The data uses pipe-delimited columns: nm=name|tp=type(SP=Sponsored Products,"
    "SB=Sponsored Brands,SD=Sponsored Display)|st=status(E=Enabled,P=Paused,"
    "A=Archived)|co=country|bgt=budget|spd=spend|sal=sales|dt=start date. Be concise."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_campaigns_summary",
            "description": "Fetch a pipe-delimited summary of all advertising campaigns including name, type, status, country, budget, spend, sales, and start date.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
]

TOOL_FUNCTIONS = {
    "get_campaigns_summary": get_campaigns_summary,
}


async def stream_response(user_message: str) -> AsyncGenerator[str, None]:
    """Stream a response from Groq LLM based on the user's question about campaigns."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

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
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn = TOOL_FUNCTIONS.get(fn_name)
            result = fn() if fn else "Unknown tool"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result if isinstance(result, str) else json.dumps(result),
                }
            )

        # Second call — stream the final answer with tool results
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=True,
            temperature=0.3,
            max_tokens=1024,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    else:
        # No tool call — just return the content directly
        if choice.message.content:
            yield choice.message.content
