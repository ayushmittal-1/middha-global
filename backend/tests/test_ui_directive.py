"""Tests for the `@@AURORA_UI@@` tool-result marker.

The marker is a contract with the chat client: a tool result may open with
one line of JSON asking the client to render an interactive component instead
of plain text. It rides inside the result string rather than arriving as a
new SSE event type so the transcript stays a list of strings — Mongo history,
replay, and any client that predates the marker all keep working.

The parser on the other side lives in
`aurorafrontend/src/pages/ai/tabs/ChatTab.tsx` (`parseUiDirective`). If the
shape here changes, that changes too.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

import agent  # noqa: E402


def _split(result: str):
    """Mirror of the client's parser, so both sides are tested against the
    same assumption: one JSON line, then human-readable text."""
    assert result.startswith(agent.UI_DIRECTIVE_PREFIX)
    head, _, tail = result.partition("\n")
    return json.loads(head[len(agent.UI_DIRECTIVE_PREFIX):]), tail


def test_directive_is_one_json_line_followed_by_readable_text():
    out = agent._ui_directive("demo", {"prefill": {"sku": "X"}}, "Fallback text.")
    directive, fallback = _split(out)
    assert directive == {"component": "demo", "prefill": {"sku": "X"}}
    assert fallback == "Fallback text."


def test_the_json_line_never_contains_a_newline():
    """A multi-line header would split into the fallback and corrupt both."""
    out = agent._ui_directive(
        "demo", {"prefill": {"campaign_name": "line one\nline two"}}, "Fallback."
    )
    head = out.split("\n", 1)[0]
    json.loads(head[len(agent.UI_DIRECTIVE_PREFIX):])  # parses on its own


def test_a_client_without_the_marker_still_sees_a_complete_answer():
    """Older clients render the raw string. The fallback has to stand alone,
    so it must not be empty or a stub like 'see above'."""
    out = agent._ui_directive("demo", {}, "Here is what happened and what to do.")
    _directive, fallback = _split(out)
    assert len(fallback.split()) >= 5


@pytest.mark.asyncio
async def test_picker_tool_emits_a_campaign_keyword_picker_directive():
    out = await agent._open_campaign_keyword_picker()
    directive, fallback = _split(out)
    assert directive["component"] == "campaign_keyword_picker"
    assert fallback


@pytest.mark.asyncio
async def test_picker_passes_collected_details_through_as_prefill():
    """Whatever the assistant already asked for must not be asked again."""
    out = await agent._open_campaign_keyword_picker(
        campaign_name="Incense — Manual",
        budget=25,
        country="US",
        sku="FG-2F35-W25S",
        asins=["b07shh2rjx", "  "],
    )
    directive, _ = _split(out)
    prefill = directive["prefill"]
    assert prefill["campaign_name"] == "Incense — Manual"
    assert prefill["budget"] == 25
    assert prefill["sku"] == "FG-2F35-W25S"
    # ASINs are normalized here so the picker's input box is ready to submit.
    assert prefill["asins"] == ["B07SHH2RJX"]


@pytest.mark.asyncio
async def test_picker_defaults_are_safe_when_the_assistant_knows_nothing():
    out = await agent._open_campaign_keyword_picker()
    directive, _ = _split(out)
    prefill = directive["prefill"]
    assert prefill["asins"] == []
    assert prefill["budget"] is None
    assert prefill["country"] == "US"


def test_picker_is_registered_as_a_tool_the_model_can_call():
    assert "open_campaign_keyword_picker" in agent.TOOL_FUNCTIONS
    names = {
        t["function"]["name"] for t in agent.TOOLS if t.get("type") == "function"
    }
    assert "open_campaign_keyword_picker" in names


def test_manual_flow_points_the_model_at_the_picker():
    """The system prompt has to stop the model listing keywords in text for a
    MANUAL campaign, or the user gets both the picker and a wall of text."""
    prompt = agent.SYSTEM_PROMPT
    assert "open_campaign_keyword_picker" in prompt
    manual = prompt[prompt.index("3. If **MANUAL**"):]
    assert "open_campaign_keyword_picker" in manual.split("If **AUTO**")[0]
