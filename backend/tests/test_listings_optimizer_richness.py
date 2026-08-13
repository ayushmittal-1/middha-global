"""Listing-optimizer richness gate + expansion retry.

Client shared a side-by-side comparison (listing optimizer.xlsx) of
Aurora's output vs a competing tool ("Seller Assistant") on two live
ASINs. Aurora's rewrites were visibly thinner across every field:
short titles, one-line bullets, one-paragraph descriptions, sparse
backend keywords. These tests lock in the fix:

  1. The system prompt explicitly instructs SA-parity depth.
  2. `check_rewrite_richness` measures per-field length and flags any
     field that fell short of SA-parity minimums.
  3. `rewrite_listing` retries once with an "expand these fields"
     instruction when the first pass is thin — closing the gap without
     regressing on rewrites that are already rich enough.
"""

import json
import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import listings_optimizer as lo


# ── System prompt content ────────────────────────────────────────────────


def test_system_prompt_targets_sa_parity_length():
    """The prompt must ask for the top of each range, not the minimum —
    the whole reason the original output was thin was that the prompt
    said 'under 200 chars' with no lower bound."""
    p = lo._REWRITE_SYSTEM_PROMPT
    assert "150-200" in p, "title length target missing from prompt"
    assert "250-500" in p, "bullet length target missing from prompt"
    assert "1200-1900" in p, "description length target missing from prompt"
    assert "200-249" in p, "backend keyword byte target missing from prompt"


def test_system_prompt_gives_bullet_structural_template():
    """Old prompt asked for '2-4 word capitalized hook, then a colon,
    then the sentence' — produced short one-liners. New prompt must ask
    for ALL CAPS HEADER + multi-sentence body."""
    p = lo._REWRITE_SYSTEM_PROMPT
    assert "ALL CAPS HEADER" in p
    assert "2-4 sentences" in p, "bullet body-length guidance missing"


def test_system_prompt_calls_out_thin_output_as_failure():
    """The most important addition — an explicit anti-pattern signal to
    the model so it stops defaulting to the safe/short baseline."""
    p = lo._REWRITE_SYSTEM_PROMPT
    assert "thin" in p.lower() or "under-length is a failure" in p.lower()


def test_system_prompt_keeps_all_compliance_rules():
    """Regression guard: the compliance section from the original prompt
    (no promotional terms, no emojis, no URLs, no health claims) must
    NOT be lost when tightening the length guidance."""
    p = lo._REWRITE_SYSTEM_PROMPT
    for phrase in (
        "promotional terms",
        "No emojis",
        "URLs",
        "medical",
        "brand name MUST appear",
    ):
        assert phrase in p, f"compliance rule dropped from prompt: {phrase!r}"


# ── check_rewrite_richness ────────────────────────────────────────────────


def _make_rewrite(
    title_chars: int = 180,
    n_bullets: int = 5,
    bullet_chars: int = 350,
    description_chars: int = 1500,
    backend_bytes: int = 240,
) -> dict:
    return {
        "title": "T" * title_chars,
        "bullets": ["B" * bullet_chars] * n_bullets,
        "description": "D" * description_chars,
        "backend_keywords": "k" * backend_bytes,
    }


def test_richness_all_above_minimums_returns_empty_needs():
    r = lo.check_rewrite_richness(_make_rewrite())
    assert r["needs_expansion"] == []
    assert r["title_chars"] == 180
    assert r["bullet_count"] == 5
    assert r["bullet_avg_chars"] == 350
    assert r["description_chars"] == 1500
    assert r["backend_bytes"] == 240


def test_richness_flags_short_title():
    r = lo.check_rewrite_richness(_make_rewrite(title_chars=50))
    assert "title" in r["needs_expansion"]
    assert "bullets" not in r["needs_expansion"]


def test_richness_flags_thin_bullets_by_average_length():
    r = lo.check_rewrite_richness(_make_rewrite(bullet_chars=80))
    assert "bullets" in r["needs_expansion"]
    assert r["bullet_avg_chars"] == 80


def test_richness_flags_too_few_bullets_even_if_long():
    """3 rich bullets is still a fail — Amazon renders 5 slots."""
    r = lo.check_rewrite_richness(_make_rewrite(n_bullets=3, bullet_chars=500))
    assert "bullets" in r["needs_expansion"]
    assert r["bullet_count"] == 3


def test_richness_flags_short_description():
    r = lo.check_rewrite_richness(_make_rewrite(description_chars=400))
    assert "description" in r["needs_expansion"]


def test_richness_flags_short_backend_keywords_by_byte_count():
    r = lo.check_rewrite_richness(_make_rewrite(backend_bytes=80))
    assert "backend_keywords" in r["needs_expansion"]


def test_richness_uses_utf8_byte_length_not_char_count_for_backend():
    """Amazon's 250-byte limit is bytes, not characters — a UTF-8
    multi-byte char takes 2-3 bytes."""
    rewrite = _make_rewrite()
    # 100 emoji chars = ~400 UTF-8 bytes (well past the byte minimum)
    rewrite["backend_keywords"] = "é" * 100  # é is 2 bytes in UTF-8 = 200
    r = lo.check_rewrite_richness(rewrite)
    assert r["backend_bytes"] == 200
    assert "backend_keywords" not in r["needs_expansion"]


def test_richness_handles_missing_or_bad_fields_gracefully():
    """A model that returns a partial JSON object shouldn't crash the
    richness gate — every missing field just reads as 0."""
    r = lo.check_rewrite_richness({})
    assert r["needs_expansion"] == [
        "title", "bullets", "description", "backend_keywords",
    ]
    assert r["title_chars"] == 0
    assert r["bullet_count"] == 0

    r2 = lo.check_rewrite_richness({"bullets": None, "title": None})
    assert r2["title_chars"] == 0
    assert r2["bullet_count"] == 0


def test_richness_ignores_blank_bullet_strings():
    """A common LLM failure mode is returning 5 bullets with 2 blank
    strings — those shouldn't inflate the count."""
    r = lo.check_rewrite_richness({
        "title": "T" * 180,
        "bullets": ["real bullet content " * 20, "", "  ", "another real bullet " * 20, ""],
        "description": "D" * 1500,
        "backend_keywords": "k" * 240,
    })
    assert r["bullet_count"] == 2  # only the two non-blank
    assert "bullets" in r["needs_expansion"]


# ── expansion instruction builder ─────────────────────────────────────────


def test_expansion_instruction_names_each_thin_field():
    inst = lo._build_expansion_instruction(["title", "backend_keywords"])
    assert "TITLE" in inst
    assert "BACKEND KEYWORDS" in inst
    # Fields not listed should not appear
    assert "BULLETS" not in inst
    assert "DESCRIPTION" not in inst


def test_expansion_instruction_includes_target_minimum():
    inst = lo._build_expansion_instruction(["description"])
    assert str(lo.RICHNESS_MIN_DESCRIPTION_CHARS) in inst


def test_expansion_instruction_ignores_unknown_field_names():
    """Defensive — if a future refactor adds a new field to the richness
    check but forgets to add a label here, the instruction should skip
    it rather than crash."""
    inst = lo._build_expansion_instruction(["title", "unknown_field"])
    assert "TITLE" in inst
    assert "unknown_field" not in inst


# ── rewrite_listing retry behavior ────────────────────────────────────────


class _FakeGroqResponse:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeGroqClient:
    """Records every chat.completions.create call and returns the next
    canned response from `responses` (a list)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = self  # so client.chat.completions.create... works
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("no more canned responses")
        return _FakeGroqResponse(self._responses.pop(0))


@pytest.mark.asyncio
async def test_rewrite_listing_calls_llm_once_when_first_pass_is_rich(monkeypatch):
    rich = json.dumps(_make_rewrite() | {"notes": "n"})
    fake = _FakeGroqClient([rich])
    monkeypatch.setattr(lo, "client", fake)

    result = await lo.rewrite_listing(
        title="old", bullets=["old"], description="old",
        backend_keywords="old", brand="NAQSH", category="Home",
    )
    assert len(fake.calls) == 1, "rich first pass should NOT trigger retry"
    assert result["richness"]["needs_expansion"] == []
    assert result["title"] == "T" * 180


@pytest.mark.asyncio
async def test_rewrite_listing_retries_when_first_pass_is_thin(monkeypatch):
    thin = json.dumps({
        "title": "short title",
        "bullets": ["HIGH QUALITY: short"] * 5,
        "description": "one paragraph.",
        "backend_keywords": "a b c",
        "notes": "n",
    })
    rich = json.dumps(_make_rewrite() | {"notes": "n"})
    fake = _FakeGroqClient([thin, rich])
    monkeypatch.setattr(lo, "client", fake)

    result = await lo.rewrite_listing(
        title="old", bullets=["old"], description="old",
        backend_keywords="old", brand="NAQSH", category="Home",
    )
    assert len(fake.calls) == 2, "thin first pass MUST trigger expansion retry"
    # Retry payload must include the expansion instruction with field names
    retry_msgs = fake.calls[1]["messages"]
    assistant_turn = next(m for m in retry_msgs if m["role"] == "assistant")
    assert json.loads(assistant_turn["content"])["title"] == "short title"
    final_user_msg = retry_msgs[-1]["content"]
    for f in ("TITLE", "BULLETS", "DESCRIPTION", "BACKEND KEYWORDS"):
        assert f in final_user_msg
    # Retry actually improved the output
    assert result["title"] == "T" * 180
    assert result["richness"]["retried"] is True
    assert result["richness"]["needs_expansion"] == []


@pytest.mark.asyncio
async def test_retry_never_regresses_shorter_content(monkeypatch):
    """If the expansion pass returns SHORTER content than the first pass
    (buggy model), keep the first pass rather than regress."""
    okay_ish = json.dumps({
        "title": "T" * 100,  # 100 < 130 min, flags 'title'
        "bullets": ["B" * 400] * 5,
        "description": "D" * 1500,
        "backend_keywords": "k" * 240,
        "notes": "n",
    })
    # Retry regresses the title
    regressed = json.dumps({
        "title": "T" * 40,  # even shorter!
        "bullets": ["B" * 400] * 5,
        "description": "D" * 1500,
        "backend_keywords": "k" * 240,
    })
    fake = _FakeGroqClient([okay_ish, regressed])
    monkeypatch.setattr(lo, "client", fake)

    result = await lo.rewrite_listing(
        title="old", bullets=["old"], description="old",
        backend_keywords="old", brand=None, category=None,
    )
    assert len(fake.calls) == 2
    # Kept the first-pass 100-char title rather than the regressed 40-char one
    assert result["title"] == "T" * 100


@pytest.mark.asyncio
async def test_retry_failure_is_recorded_not_raised(monkeypatch):
    """A network / JSON failure on the retry must NOT throw — the
    first-pass rewrite is still worth returning to the user."""
    thin = json.dumps({
        "title": "short",
        "bullets": ["hi"] * 5,
        "description": "d",
        "backend_keywords": "k",
    })
    class _BrokenClient(_FakeGroqClient):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _FakeGroqResponse(thin)
            raise RuntimeError("groq timeout")

    fake = _BrokenClient([])
    monkeypatch.setattr(lo, "client", fake)

    result = await lo.rewrite_listing(
        title="old", bullets=["old"], description="old",
        backend_keywords="old", brand=None, category=None,
    )
    assert "error" not in result, "retry failure must not surface as top-level error"
    assert result["richness"]["retried"] is False
    assert "groq timeout" in result["richness"]["retry_error"]


@pytest.mark.asyncio
async def test_max_tokens_bumped_to_fit_full_rich_rewrite(monkeypatch):
    """Regression guard: max_tokens=2000 truncated the description on
    SA-parity outputs; the new floor is 4000."""
    rich = json.dumps(_make_rewrite() | {"notes": "n"})
    fake = _FakeGroqClient([rich])
    monkeypatch.setattr(lo, "client", fake)

    await lo.rewrite_listing(
        title="t", bullets=["b"], description="d",
        backend_keywords="k", brand=None, category=None,
    )
    assert fake.calls[0]["max_tokens"] >= 4000
