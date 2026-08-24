"""Amazon Listing Optimizer — score + rewrite listings against Amazon's
2026 style guidelines.

Two paths:

- `score_listing(...)` runs deterministic rule checks (char counts,
  ALL CAPS, banned promotional terms, byte length for backend
  keywords). Fast, exact, no LLM.
- `rewrite_listing(...)` sends the current fields + guidelines rubric
  to Groq's Llama-4-Scout in JSON mode. Returns AI-suggested rewrites
  for each field.

The public entrypoint `analyze_listing(...)` runs both and returns
a unified response the FE can render into a score-card + before/after
comparison view.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

from agent import client  # Reuse the existing Groq client instance.

# Model fallback chain for the listing rewriter.
#
# Live inventory verified 2026-08-22 against this account via GET
# /openai/v1/models — the llama family (3.1/3.3-*) is gone, so the
# chain sticks to gpt-oss + qwen + allam + compound-router, which are
# the models actually available on this Groq key.
#
# Failure mode we're guarding against: Groq's `json_validate_failed`
# (400) hits both gpt-oss-120b AND gpt-oss-20b under json_object mode.
# That's a shared server-side validator bug in the gpt-oss family on
# this account. To escape it, the chain jumps out of the family after
# one gpt-oss attempt and tries a Qwen model (independent validator
# path), then falls through to smaller/simpler models.
#
# Order rationale:
#   1. openai/gpt-oss-120b — flagship, richest rewrites when it works
#   2. qwen/qwen3.6-27b — different family, Qwen has solid json_object
#      support and avoids the gpt-oss validator issue
#   3. groq/compound — Groq's router picks the best-available model
#      per request; useful last-resort because it dodges specific-
#      model outages
#   4. allam-2-7b — small, simple, most-forgiving JSON output
MODEL = "openai/gpt-oss-120b"
# allam-2-7b sits second because a live test showed it as the model
# actually completing the rewrite when gpt-oss-120b returns an empty
# response (a Groq server-side bug we can't work around from here).
# qwen/qwen3.6-27b was removed — its <think>...</think> reasoning mode
# consumes ~2 seconds per attempt without producing a JSON block.
_MODEL_FALLBACK_CHAIN = (
    "openai/gpt-oss-120b",
    "allam-2-7b",
    "openai/gpt-oss-safeguard-20b",
    "groq/compound",
)

# Vision model for image compliance checks. Groq's llama-4-scout is
# vision-capable but locked behind their Dev tier — free-tier accounts
# 404. We try Gemini first (see GEMINI_VISION_MODEL) when GEMINI_API_KEY
# is set; only fall back to Groq for accounts that have paid access.
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Google Gemini vision model — free tier gives 15 req/min, 1500/day.
# `gemini-flash-latest` is an alias that always points to the current
# stable Flash release, so we don't have to bump this every time Google
# deprecates a version (2.0-flash and 2.5-flash both got retired for
# new users mid-2026).
GEMINI_VISION_MODEL = "gemini-flash-latest"

log = logging.getLogger("listings_optimizer")

# ── Amazon 2026 style-guide constants ──────────────────────────────────────
# Values sourced from Amazon_Listing_Guidelines.docx (Aug 2026).

TITLE_MAX_CHARS_DEFAULT = 200        # US default (200 chars for most categories)
TITLE_MAX_CHARS_INDIA = 75           # India / many categories in the 2026 update
BULLET_MAX_CHARS = 500               # per bullet, US
BULLET_TARGET_COUNT = 5              # Amazon renders 5 by default
DESCRIPTION_MAX_CHARS = 2000
BACKEND_KEYWORDS_MAX_BYTES = 250     # UTF-8 bytes, not chars

# Words that get flagged in titles and bullets — Amazon rejects listings
# containing overt promotional language ("free shipping", "best seller",
# money-back, guarantees, sale, discount, and their variants).
PROMOTIONAL_TERMS = {
    "free shipping", "best seller", "best-seller", "bestseller",
    "money back", "money-back", "guarantee", "guaranteed",
    "sale", "discount", "off", "hot", "new arrival",
    "limited time", "cheapest", "amazon's choice", "amazons choice",
    "top rated", "#1", "no.1", "no 1",
}

# Emoji regex — Amazon rejects titles with pictorial characters.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE,
)


@dataclass
class FieldScore:
    """One field's scorecard row — max 10 points, deducted per issue."""
    score: int = 10
    max_score: int = 10
    issues: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    def fail(self, msg: str, penalty: int = 2) -> None:
        self.issues.append(msg)
        self.score = max(0, self.score - penalty)

    def pass_(self, msg: str) -> None:
        self.passed.append(msg)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "max_score": self.max_score,
            "issues": self.issues,
            "passed": self.passed,
        }


# ── Deterministic scorecard ────────────────────────────────────────────────


def _check_all_caps(text: str) -> bool:
    """True if the text is majority-uppercase (excluding non-letters)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 10:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.6


def _find_promotional(text: str) -> list[str]:
    """Return any promotional terms present in the text (case-insensitive)."""
    lo = text.lower()
    return sorted({term for term in PROMOTIONAL_TERMS if term in lo})


def _find_urls(text: str) -> list[str]:
    return re.findall(r"https?://\S+|www\.\S+", text)


def _find_emails(text: str) -> list[str]:
    return re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)


def _detect_keyword_stuffing(text: str) -> list[str]:
    """Words repeated 3+ times (excluding stopwords) — a stuffing signal."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    stopwords = {
        "the", "and", "for", "with", "your", "you", "our", "are",
        "this", "that", "from", "has", "have", "all", "not", "but",
    }
    counts: dict[str, int] = {}
    for w in words:
        if w in stopwords:
            continue
        counts[w] = counts.get(w, 0) + 1
    return sorted([w for w, c in counts.items() if c >= 4])


def score_title(title: str, brand: str | None = None,
                max_chars: int = TITLE_MAX_CHARS_DEFAULT) -> FieldScore:
    fs = FieldScore()
    t = (title or "").strip()
    if not t:
        fs.fail("Title is empty", penalty=10)
        return fs
    if len(t) > max_chars:
        fs.fail(f"Title is {len(t)} chars — exceeds {max_chars} char limit")
    else:
        fs.pass_(f"Length OK ({len(t)}/{max_chars} chars)")
    if _check_all_caps(t):
        fs.fail("Title uses excessive ALL CAPS")
    else:
        fs.pass_("Not ALL CAPS")
    promos = _find_promotional(t)
    if promos:
        fs.fail(f"Promotional terms not allowed in title: {', '.join(promos)}")
    else:
        fs.pass_("No promotional language")
    if _EMOJI_RE.search(t):
        fs.fail("Title contains emojis — Amazon rejects these")
    else:
        fs.pass_("No emojis")
    stuffed = _detect_keyword_stuffing(t)
    if stuffed:
        fs.fail(f"Possible keyword stuffing: {', '.join(stuffed[:3])} repeated")
    if brand and brand.strip().lower() not in t.lower():
        fs.fail(f"Brand name '{brand}' should appear in the title")
    elif brand:
        fs.pass_(f"Brand '{brand}' present")
    return fs


def score_bullets(bullets: list[str]) -> FieldScore:
    fs = FieldScore()
    cleaned = [b.strip() for b in (bullets or []) if b and b.strip()]
    n = len(cleaned)
    if n == 0:
        fs.fail("No bullet points provided", penalty=10)
        return fs
    if n < BULLET_TARGET_COUNT:
        fs.fail(f"Only {n} bullets — Amazon renders {BULLET_TARGET_COUNT}, "
                "use all available slots")
    else:
        fs.pass_(f"Using {n} bullet(s)")
    over = [i + 1 for i, b in enumerate(cleaned) if len(b) > BULLET_MAX_CHARS]
    if over:
        fs.fail(f"Bullets exceed {BULLET_MAX_CHARS} chars: #{', #'.join(map(str, over))}")
    all_caps = [i + 1 for i, b in enumerate(cleaned) if _check_all_caps(b)]
    if all_caps:
        fs.fail(f"Bullets in ALL CAPS: #{', #'.join(map(str, all_caps))}")
    for i, b in enumerate(cleaned):
        promos = _find_promotional(b)
        if promos:
            fs.fail(f"Bullet #{i+1} has promotional terms: {', '.join(promos)}")
            break
    else:
        fs.pass_("No promotional language in bullets")
    return fs


def score_description(desc: str) -> FieldScore:
    fs = FieldScore()
    d = (desc or "").strip()
    if not d:
        fs.fail("Description is empty", penalty=6)
        return fs
    if len(d) > DESCRIPTION_MAX_CHARS:
        fs.fail(f"Description is {len(d)} chars — exceeds {DESCRIPTION_MAX_CHARS} limit")
    else:
        fs.pass_(f"Length OK ({len(d)}/{DESCRIPTION_MAX_CHARS} chars)")
    urls = _find_urls(d)
    if urls:
        fs.fail(f"Contains URL(s) — not allowed: {', '.join(urls[:2])}")
    else:
        fs.pass_("No external URLs")
    emails = _find_emails(d)
    if emails:
        fs.fail("Contains email address — not allowed")
    promos = _find_promotional(d)
    if promos:
        fs.fail(f"Promotional language: {', '.join(promos[:3])}")
    return fs


def score_backend_keywords(keywords: str) -> FieldScore:
    """Backend search terms compliance — the tightest field to get right."""
    fs = FieldScore()
    k = (keywords or "").strip()
    if not k:
        fs.fail("No backend keywords provided", penalty=6)
        return fs
    byte_len = len(k.encode("utf-8"))
    if byte_len > BACKEND_KEYWORDS_MAX_BYTES:
        fs.fail(f"Byte length {byte_len} exceeds {BACKEND_KEYWORDS_MAX_BYTES} — "
                "Amazon silently truncates the excess")
    else:
        fs.pass_(f"Byte length OK ({byte_len}/{BACKEND_KEYWORDS_MAX_BYTES})")
    if "," in k:
        fs.fail("Backend keywords must be space-separated, not comma-separated")
    else:
        fs.pass_("Space-separated (no commas)")
    tokens = [t.lower() for t in k.split() if t]
    dup = sorted({t for t in tokens if tokens.count(t) > 1})
    if dup:
        fs.fail(f"Duplicate keywords waste bytes: {', '.join(dup[:5])}")
    else:
        fs.pass_("No duplicate keywords")
    return fs


# Amazon image guideline targets (2026).
IMAGE_MIN_LONG_EDGE = 1000       # min longest edge for zoom eligibility
IMAGE_RECOMMENDED_EDGE = 1600    # recommended for full zoom quality
IMAGE_TARGET_COUNT_MIN = 7       # 7–9 images is the recommended range
IMAGE_TARGET_COUNT_MAX = 9

# Manual-verify items we can't inspect without a vision model. Rendered
# as a separate self-check checklist under the auto-scored bits.
IMAGES_MANUAL_CHECKLIST = [
    "Main image on pure white background (RGB 255,255,255)",
    "Product occupies at least 85% of the frame",
    "Includes a lifestyle / in-use shot",
    "Includes an infographic with dimensions or key features",
    "No text/watermarks/logos overlaid on the main image",
]


def _image_long_edge(img: dict) -> int:
    """Best-effort longest-edge px for an image dict.

    Aurora stores images as {url, height, width}. When dims are missing
    we try to parse the Amazon CDN size hint from the URL — filenames
    like `_SL1500_.jpg` or `_SX679_SY522_.jpg` encode the rendered size.
    Returns 0 when we can't determine dims — treated as unknown, not zero.
    """
    if not isinstance(img, dict):
        return 0
    h = int(img.get("height") or 0)
    w = int(img.get("width") or 0)
    if h or w:
        return max(h, w)
    url = str(img.get("url") or "")
    # _SL1500_, _SX679_, _SY522_, _AC_SL1500_, etc. — the digits are px.
    m = re.findall(r"_S[LXY](\d{2,5})_", url)
    if m:
        return max(int(x) for x in m)
    return 0


def score_images(images: list[dict] | None) -> FieldScore:
    """Auto-score whatever we can from the image list — count + dims.

    Colour/composition checks stay in the manual checklist; we can't
    inspect pixels here without a vision model. Missing dims are treated
    as unknown (soft warning), not a hard fail — Aurora doesn't populate
    dims for every image and we don't want to punish that.
    """
    fs = FieldScore()
    imgs = [i for i in (images or []) if isinstance(i, dict) and i.get("url")]
    n = len(imgs)
    if n == 0:
        fs.fail("No product images uploaded", penalty=10)
        return fs

    if n < IMAGE_TARGET_COUNT_MIN:
        fs.fail(
            f"Only {n} image(s) — Amazon recommends "
            f"{IMAGE_TARGET_COUNT_MIN}–{IMAGE_TARGET_COUNT_MAX} "
            "(main + variants + lifestyle + infographic + size chart)",
            penalty=3,
        )
    elif n > IMAGE_TARGET_COUNT_MAX:
        # Not a hard fail — extras don't render but don't hurt either.
        fs.pass_(f"{n} images (Amazon renders the first {IMAGE_TARGET_COUNT_MAX})")
    else:
        fs.pass_(f"{n} images in the recommended {IMAGE_TARGET_COUNT_MIN}–{IMAGE_TARGET_COUNT_MAX} range")

    dims = [_image_long_edge(i) for i in imgs]
    known = [d for d in dims if d > 0]
    if known:
        too_small = [d for d in known if d < IMAGE_MIN_LONG_EDGE]
        if too_small:
            fs.fail(
                f"{len(too_small)}/{len(known)} image(s) below {IMAGE_MIN_LONG_EDGE}px "
                "on the long edge — zoom won't activate",
                penalty=3,
            )
        else:
            below_recommended = [d for d in known if d < IMAGE_RECOMMENDED_EDGE]
            if below_recommended:
                fs.fail(
                    f"{len(below_recommended)}/{len(known)} image(s) below "
                    f"{IMAGE_RECOMMENDED_EDGE}px — zoom works but quality is limited",
                    penalty=1,
                )
            else:
                fs.pass_(f"All measured images ≥{IMAGE_RECOMMENDED_EDGE}px on the long edge")
    if len(known) < n:
        # Not a scoring hit — just surface it so the seller knows why.
        fs.passed.append(
            f"{n - len(known)} image(s) missing dimensions — can't verify zoom eligibility"
        )
    return fs


def score_listing(
    title: str,
    bullets: list[str],
    description: str,
    backend_keywords: str,
    brand: str | None = None,
    images: list[dict] | None = None,
    max_title_chars: int = TITLE_MAX_CHARS_DEFAULT,
) -> dict:
    """Run all deterministic checks. LLM-free. Returns a unified scorecard
    dict that the FE renders directly."""
    scorecard = {
        "title": score_title(title, brand=brand, max_chars=max_title_chars).to_dict(),
        "bullets": score_bullets(bullets).to_dict(),
        "description": score_description(description).to_dict(),
        "backend_keywords": score_backend_keywords(backend_keywords).to_dict(),
        "images": score_images(images).to_dict(),
    }
    total = sum(s["score"] for s in scorecard.values())
    max_total = sum(s["max_score"] for s in scorecard.values())
    return {
        "scorecard": scorecard,
        "total_score": total,
        "max_score": max_total,
        "images_manual_checklist": IMAGES_MANUAL_CHECKLIST,
    }


# ── LLM-driven rewrite ────────────────────────────────────────────────────


_REWRITE_SYSTEM_PROMPT = """You are a senior Amazon listing optimization specialist. Your rewrites must match or exceed the depth and keyword coverage of top-tier tools (Helium 10, Jungle Scout, Seller Assistant) — not the minimum-viable output of a generic template.

The seller is paying for a rich, conversion-optimized rewrite. Thin, one-line-per-bullet output is a failure — expand every field to the top of what Amazon allows, keeping full compliance.

# Length + richness targets (aim for the TOP of each range — under-length is a failure)
- Title: 150-200 characters. Pack in brand, product type, material, key spec/pack size, primary use case, and 1-2 differentiators. Front-load the most-searched terms.
- Bullets: EXACTLY 5 bullets. Each bullet 250-500 characters (aim ~350). Format:
    ALL CAPS HEADER (5-8 words, hyphen-separated is fine) - detailed body sentence(s) covering the feature, concrete specs (dimensions/materials/quantity), the buyer benefit, and the use case or context. 2-4 sentences of body per bullet.
    Example bullet:
      COMPLETE 20-PIECE SET WITH HARDWARE - Includes 20 round ceramic knobs (1.6" diameter x 2" length) plus matching screws, nuts, and washers for easy installation. Universal fit works with standard cabinet and drawer holes. Excess screw length can be trimmed to fit your specific furniture thickness.
- Description: 1200-1900 characters. Structure it as an opening hook (1-2 sentences), then 3-5 short section headers (each on its own line, followed by 2-4 sentences of body). Cover: what it is, what's included / key specs, use cases, quality / material, care, and a closing brand promise. Do NOT use HTML tags — Amazon strips them. Use blank lines between sections.
- Backend search terms: single space-separated string, 200-249 UTF-8 bytes (fill the budget). Include 30-50 focused tokens covering: synonyms, alternate spellings, common misspellings, material variants, use-case terms, related-product terms buyers might type. NO commas, NO duplicates, NO competitor brand names, and do NOT repeat words already in the title (that wastes indexing budget).

# Compliance rules (MUST follow — Amazon rejects listings that violate these)
- No promotional terms anywhere: no "free shipping", "best seller", "money back", "guarantee", "sale", "discount", "#1", "top rated", "Amazon's Choice", "limited time", "cheapest", "hot", "new arrival".
- No emojis, no URLs, no email addresses, no phone numbers.
- No unsupported medical or health claims ("cures", "treats", "FDA approved" without proof, "clinically proven" without a citation).
- Title: not fully ALL CAPS (bullet headers are fine — those are prefixes, not full-caps bullets).
- If the seller provided a brand, the brand name MUST appear in the title.

# Return format
Return VALID JSON only — no prose outside the object — with EXACTLY this shape:
{
  "title": "150-200 char rewritten title",
  "bullets": ["bullet 1 (250-500 chars, ALL CAPS HEADER - body)", "bullet 2", "bullet 3", "bullet 4", "bullet 5"],
  "description": "1200-1900 char rewritten description with section headers on their own lines",
  "backend_keywords": "space separated keyword string, 200-249 UTF-8 bytes, 30-50 tokens",
  "notes": "2-3 sentence summary of the biggest improvements you made — call out specific keyword additions and structural changes."
}"""


# SA-parity minimums used to decide whether a first-pass rewrite is rich
# enough to ship or needs an expansion retry. Under these values the
# output looks like the thin, one-line-per-bullet baseline the client
# flagged as inferior to Seller Assistant. Above them, it matches the
# depth of top-tier tools while staying inside Amazon's hard limits.
RICHNESS_MIN_TITLE_CHARS = 130
RICHNESS_MIN_BULLET_AVG_CHARS = 220
RICHNESS_MIN_BULLET_COUNT = 5
RICHNESS_MIN_DESCRIPTION_CHARS = 1000
RICHNESS_MIN_BACKEND_BYTES = 180


def check_rewrite_richness(rewrite: dict) -> dict:
    """Measure per-field length + flag which fields fell short of the
    SA-parity minimums. Pure function — safe to unit test directly.

    Returns:
        {
          "title_chars": int,
          "bullet_count": int,
          "bullet_avg_chars": int,
          "description_chars": int,
          "backend_bytes": int,
          "needs_expansion": ["title", "bullets", ...]  # names of thin fields
        }
    """
    title = str(rewrite.get("title") or "")
    bullets = [str(b) for b in (rewrite.get("bullets") or []) if str(b).strip()]
    description = str(rewrite.get("description") or "")
    backend = str(rewrite.get("backend_keywords") or "")

    bullet_avg = (
        sum(len(b) for b in bullets) // len(bullets) if bullets else 0
    )

    needs: list[str] = []
    if len(title) < RICHNESS_MIN_TITLE_CHARS:
        needs.append("title")
    if len(bullets) < RICHNESS_MIN_BULLET_COUNT or bullet_avg < RICHNESS_MIN_BULLET_AVG_CHARS:
        needs.append("bullets")
    if len(description) < RICHNESS_MIN_DESCRIPTION_CHARS:
        needs.append("description")
    if len(backend.encode("utf-8")) < RICHNESS_MIN_BACKEND_BYTES:
        needs.append("backend_keywords")

    return {
        "title_chars": len(title),
        "bullet_count": len(bullets),
        "bullet_avg_chars": bullet_avg,
        "description_chars": len(description),
        "backend_bytes": len(backend.encode("utf-8")),
        "needs_expansion": needs,
    }


def _build_expansion_instruction(thin_fields: list[str]) -> str:
    """Second-pass user instruction: name the fields that fell short and
    the exact minimum each must clear. Precise beats polite here — the
    first-pass tendency is to under-deliver on length."""
    labels = {
        "title": f"TITLE — expand to at least {RICHNESS_MIN_TITLE_CHARS} chars (aim 180). Add material, spec/pack size, use case, differentiators.",
        "bullets": f"BULLETS — expand each bullet to at least {RICHNESS_MIN_BULLET_AVG_CHARS} chars on average (aim 350). Keep the ALL CAPS HEADER but add 2-4 sentences of detail per bullet: specs, dimensions, materials, buyer benefit, use case.",
        "description": f"DESCRIPTION — expand to at least {RICHNESS_MIN_DESCRIPTION_CHARS} chars (aim 1500). Add 3-5 section headers on their own lines, each with 2-4 sentences of body.",
        "backend_keywords": f"BACKEND KEYWORDS — expand to at least {RICHNESS_MIN_BACKEND_BYTES} UTF-8 bytes (aim 240). Add synonyms, alternate spellings, misspellings, material variants, use-case terms. Still space-separated, no commas, no duplicates.",
    }
    return (
        "The previous rewrite was too thin in the following fields. "
        "Return the same JSON shape but expand these fields to the "
        "specified minimums (keep the other fields as-is):\n\n"
        + "\n".join(f"- {labels[f]}" for f in thin_fields if f in labels)
    )


def _extract_json_block(text: str) -> dict:
    """Pull the first top-level `{...}` JSON object out of an LLM
    response. Handles both raw-JSON replies and replies wrapped in
    ```json``` fences or preambles like "Here is your rewrite: {...}".
    Raises json.JSONDecodeError if no parseable object is found.
    """
    if not text:
        raise json.JSONDecodeError("empty response", "", 0)
    # First try the whole thing (fast path when the model already returned raw JSON).
    stripped = text.strip()
    for candidate in (stripped, stripped.strip("`").removeprefix("json").strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # Walk the string, balance braces, extract the first complete {...} block.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = text[start:i + 1]
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        break  # try starting from next `{`
        start = text.find("{", start + 1)
    raise json.JSONDecodeError("no JSON object found", text, 0)


async def _rewrite_with_fallback(
    messages: list[dict],
) -> tuple[dict, str]:
    """Call Groq's chat/completions and parse a JSON object out of the
    text response, walking the _MODEL_FALLBACK_CHAIN on error.

    Deliberately does NOT use Groq's `response_format={"type":
    "json_object"}` — every model on this account currently fails
    Groq's server-side JSON validator with 400 json_validate_failed
    (even qwen and allam, which are outside the gpt-oss family).
    Instead the system prompt tells the model to emit JSON directly,
    and `_extract_json_block` recovers the object from whatever prose
    or code fence the model wrapped around it. This is the widely-used
    workaround when a provider's JSON mode is unreliable.

    Returns (parsed_json, model_name_that_succeeded). Raises the last
    exception when every model in the chain fails.
    """
    last_err: Exception | None = None
    for model_name in _MODEL_FALLBACK_CHAIN:
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
            )
            raw = resp.choices[0].message.content or ""
            try:
                return _extract_json_block(raw), model_name
            except json.JSONDecodeError as je:
                # Model produced text with no parseable JSON block.
                # Log a snippet + try the next model — different models
                # respond differently to the same prompt.
                log.warning(
                    "rewrite_listing: model %s returned unparseable "
                    "response (len=%d, snippet=%r) — trying next: %s",
                    model_name, len(raw), raw[:200], je,
                )
                last_err = je
                continue
        except json.JSONDecodeError:
            # Model returned non-JSON text despite json_object mode.
            # Bubble up — this is a prompt/parsing issue, not a model
            # capability issue, so trying the next model won't help.
            raise
        except Exception as e:
            msg = str(e)
            # Retryable: 400 (json_validate_failed / prompt issues that
            # a different model might tolerate), 404 (model decommissioned
            # — try the next), 429 (rate limit — burn the rest of the
            # chain since they're independent quotas), 5xx (transient
            # server). Non-retryable: 401/403 (auth) — that's the same
            # api key against every model, so short-circuit.
            retryable_markers = (
                "json_validate_failed", "model_not_found",
                "400", "404", "429", "500", "502", "503", "504",
            )
            auth_markers = ("401", "403", "invalid_api_key", "authentication")
            if any(m in msg for m in auth_markers):
                raise
            if any(m in msg for m in retryable_markers):
                log.warning(
                    "rewrite_listing: model %s failed (%s) — trying next: %s",
                    model_name, type(e).__name__, msg[:200],
                )
                last_err = e
                continue
            # Unknown error class — try next anyway rather than crashing
            # a user-visible feature. Log at warning so it shows up.
            log.warning(
                "rewrite_listing: model %s failed with unclassified "
                "error (%s) — trying next: %s",
                model_name, type(e).__name__, msg[:200],
            )
            last_err = e
            continue
    # Exhausted the chain — surface the last error to the caller.
    assert last_err is not None
    raise last_err


async def rewrite_listing(
    title: str,
    bullets: list[str],
    description: str,
    backend_keywords: str,
    brand: str | None,
    category: str | None,
    focus_keywords: list[str] | None = None,
) -> dict:
    """Ask Groq to rewrite each field, aiming for competitor-parity depth.

    Two-pass:
      1. First call generates a full rewrite with the SA-parity prompt.
      2. If check_rewrite_richness flags any field as thin, a second
         call re-runs with an explicit expansion instruction naming the
         under-length fields. This closes the gap that showed up when
         Aurora's output was compared side-by-side with Seller Assistant
         and looked visibly thinner.

    Returns a dict with the rewritten values plus a `richness` block so
    the FE (or tests) can see how the output stacks up against the
    minimums. Falls back to a minimal error payload if the model returns
    something that isn't valid JSON."""
    user_prompt_parts = []
    if brand:
        user_prompt_parts.append(f"BRAND: {brand}")
    if category:
        user_prompt_parts.append(f"CATEGORY: {category}")
    if focus_keywords:
        user_prompt_parts.append(
            "FOCUS KEYWORDS (weave these in naturally): "
            + ", ".join(focus_keywords[:10])
        )
    user_prompt_parts.append(f"\nCURRENT TITLE:\n{title or '(empty)'}")
    user_prompt_parts.append(
        "\nCURRENT BULLETS:\n"
        + ("\n".join(f"- {b}" for b in (bullets or [])) or "(none)")
    )
    user_prompt_parts.append(
        f"\nCURRENT DESCRIPTION:\n{description or '(empty)'}"
    )
    user_prompt_parts.append(
        f"\nCURRENT BACKEND KEYWORDS:\n{backend_keywords or '(empty)'}"
    )
    user_prompt_parts.append(
        "\nRewrite every field to be fully compliant AND to match the "
        "length + richness targets in the system prompt. If a field is "
        "empty, create one from scratch based on the other fields and "
        "the brand / category."
    )
    user_prompt = "\n".join(user_prompt_parts)

    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        rewrite, model_used = await _rewrite_with_fallback(messages)
        if model_used != MODEL:
            log.info("rewrite_listing: primary %s failed, %s succeeded",
                     MODEL, model_used)
    except json.JSONDecodeError as e:
        log.warning("rewrite_listing: LLM returned invalid JSON: %s", e)
        return {"error": "Model returned malformed JSON — try again"}
    except Exception as e:
        log.exception("rewrite_listing failed across all fallback models")
        return {"error": (
            "AI rewriter is temporarily unavailable "
            "(all Groq models returned errors). Try again in a minute."
        )}

    richness = check_rewrite_richness(rewrite)
    if richness["needs_expansion"]:
        try:
            expansion = _build_expansion_instruction(richness["needs_expansion"])
            expanded, _ = await _rewrite_with_fallback([
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(rewrite)},
                {"role": "user", "content": expansion},
            ])
            # Merge: prefer expanded fields ONLY if they're actually longer
            # than the first pass (a bad model can return shorter content
            # on the retry — don't regress).
            for field_name in richness["needs_expansion"]:
                if field_name == "bullets":
                    old_avg = richness["bullet_avg_chars"]
                    new_bullets = expanded.get("bullets") or []
                    new_avg = (
                        sum(len(str(b)) for b in new_bullets) // len(new_bullets)
                        if new_bullets else 0
                    )
                    if new_avg > old_avg:
                        rewrite["bullets"] = new_bullets
                    continue
                new_val = expanded.get(field_name)
                if new_val and len(str(new_val)) > len(str(rewrite.get(field_name) or "")):
                    rewrite[field_name] = new_val
            richness = check_rewrite_richness(rewrite)
            richness["retried"] = True
        except Exception as e:
            log.warning("rewrite_listing expansion retry failed: %s", e)
            richness["retried"] = False
            richness["retry_error"] = str(e)

    rewrite["richness"] = richness
    return rewrite


# ── Vision analysis for the image checklist ───────────────────────────────


_VISION_ITEMS = [
    # (id, question). Each becomes one JSON key in the model response.
    ("white_background", "Is the MAIN image (first one) on a pure white background (RGB 255,255,255, no shadows, gradients, or colored floor)?"),
    ("product_frame", "Does the product in the MAIN image occupy at least 85% of the frame?"),
    ("lifestyle_shot", "Does the image set include at least one lifestyle / in-use / in-context shot showing the product being used?"),
    ("infographic", "Does the image set include at least one infographic-style image (dimensions callouts, feature labels, comparison chart, or similar)?"),
    ("no_watermarks", "Is the MAIN image free of text, watermarks, logos, or promotional badges overlaid on the product?"),
]

_VISION_SYSTEM_PROMPT = (
    "You are an Amazon listing compliance reviewer. You will look at a "
    "seller's product images and grade them against Amazon's 2026 image "
    "guidelines. Be strict but fair — 'pass' means the image clearly meets "
    "the rule, 'fail' means it clearly violates it, 'unsure' when the "
    "image is too small/low-res to judge. The FIRST image is always the "
    "listing's main image; the rest are gallery/variants. Return VALID "
    "JSON only, no prose."
)


def _build_vision_user_prompt() -> str:
    lines = ["For each of the following checks, respond with pass / fail / unsure and one short reason."]
    for i, (key, question) in enumerate(_VISION_ITEMS, 1):
        lines.append(f"{i}. {key}: {question}")
    lines.append("")
    lines.append("Return JSON of the exact shape:")
    schema = {k: {"verdict": "pass|fail|unsure", "reason": "one short sentence"} for k, _ in _VISION_ITEMS}
    lines.append(json.dumps(schema, indent=2))
    return "\n".join(lines)


def _normalise_vision_response(parsed: dict, urls: list[str], provider: str) -> dict:
    """Filter a model's raw JSON down to the known checklist keys +
    coerce verdicts. Shared by both provider paths so the FE sees the
    same shape regardless of who scored the images."""
    out: dict = {}
    for key, question in _VISION_ITEMS:
        row = parsed.get(key) if isinstance(parsed, dict) else None
        if not isinstance(row, dict):
            continue
        verdict = str(row.get("verdict") or "").lower().strip()
        if verdict not in ("pass", "fail", "unsure"):
            verdict = "unsure"
        out[key] = {
            "question": question,
            "verdict": verdict,
            "reason": str(row.get("reason") or "").strip(),
        }
    return {"items": out, "images_reviewed": urls, "provider": provider}


async def _fetch_image_bytes(url: str, timeout: float = 15.0) -> tuple[bytes, str] | None:
    """Download an image and return (bytes, mime_type). Gemini needs
    inline_data (base64) since it doesn't fetch arbitrary URLs like
    OpenAI-compatible vision endpoints do."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            log.warning("vision: image fetch HTTP %s for %s", r.status_code, url)
            return None
        mime = r.headers.get("content-type", "").split(";")[0].strip() or "image/jpeg"
        return r.content, mime
    except Exception as e:
        log.warning("vision: image fetch failed for %s: %s", url, e)
        return None


async def _analyze_images_gemini(urls: list[str], api_key: str) -> dict:
    """Grade `urls` via Gemini Flash. Fetches each image (Gemini's
    generateContent takes inline base64, not URLs) and asks it to
    return a JSON object matching the _VISION_ITEMS schema."""
    fetched = await asyncio.gather(*[_fetch_image_bytes(u) for u in urls])
    parts: list[dict] = [{"text": _build_vision_user_prompt()}]
    successful_urls: list[str] = []
    for url, res in zip(urls, fetched):
        if not res:
            continue
        blob, mime = res
        parts.append({
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(blob).decode("ascii"),
            },
        })
        successful_urls.append(url)

    if not successful_urls:
        return {"error": "Could not fetch any image for vision analysis"}

    payload = {
        "system_instruction": {"parts": [{"text": _VISION_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800,
            "responseMimeType": "application/json",
        },
    }
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_VISION_MODEL}:generateContent?key={api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(endpoint, json=payload)
    except Exception as e:
        log.exception("gemini vision request failed")
        return {"error": f"Gemini transport error: {e}"}
    if r.status_code != 200:
        # Gemini surfaces model / auth errors as {"error": {"message": ...}}.
        err = "unknown"
        try:
            err = (r.json().get("error") or {}).get("message") or r.text[:200]
        except Exception:
            err = r.text[:200]
        return {"error": f"Gemini HTTP {r.status_code}: {err}"}
    try:
        body = r.json()
        raw = (
            (body.get("candidates") or [{}])[0]
            .get("content", {}).get("parts", [{}])[0]
            .get("text", "{}")
        )
        parsed = json.loads(raw)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        log.warning("gemini vision: bad response shape: %s", e)
        return {"error": "Gemini returned malformed JSON"}
    return _normalise_vision_response(parsed, successful_urls, provider="gemini")


async def _analyze_images_groq(urls: list[str]) -> dict:
    """Grade `urls` via Groq's vision model. Uses OpenAI-compatible
    image_url content parts — no image fetch needed. 404s on free-tier
    Groq accounts (the vision Llama-4 models are Dev-tier only)."""
    content: list[dict] = [{"type": "text", "text": _build_vision_user_prompt()}]
    for u in urls:
        content.append({"type": "image_url", "image_url": {"url": u}})
    try:
        resp = await client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("groq vision: bad JSON: %s", e)
        return {"error": "Groq vision returned malformed JSON"}
    except Exception as e:
        log.exception("groq vision failed")
        return {"error": str(e)}
    return _normalise_vision_response(parsed, urls, provider="groq")


async def analyze_images_vision(images: list[dict] | None) -> dict:
    """Grade the manual-checklist items (white background, lifestyle,
    watermarks, etc.) against real image content.

    Provider order:
      1. Gemini (when GEMINI_API_KEY is set) — free tier works.
      2. Groq llama-4-scout — requires Groq Dev tier; 404s on free.

    Both providers return the same {items, images_reviewed, provider}
    shape. Errors bubble up as {"error": "..."} so the FE can show a
    fallback checklist instead of failing silently.
    """
    urls = [
        str(i.get("url"))
        for i in (images or [])
        if isinstance(i, dict) and i.get("url")
    ]
    # Cap at 6 images — main + first 5 gallery is enough to judge the
    # set, keeps request payload small, avoids Gemini's per-request cap.
    urls = urls[:6]
    if not urls:
        return {}

    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if gemini_key:
        return await _analyze_images_gemini(urls, gemini_key)
    return await _analyze_images_groq(urls)


def _apply_vision_verdicts_to_score(image_score: dict, vision: dict) -> None:
    """Fold vision verdicts back into the image scorecard in place. Each
    `fail` verdict deducts 1 pt from images (min 0) so a listing that
    aces count+dims but flunks white-background/watermarks doesn't get
    a fake 10/10."""
    items = (vision or {}).get("items") or {}
    if not items:
        return
    fails = [k for k, v in items.items() if v.get("verdict") == "fail"]
    if not fails:
        image_score.setdefault("passed", []).append(
            "Vision review: no issues found on white-background, lifestyle, infographic, watermarks."
        )
        return
    label = {
        "white_background": "Main image background isn't pure white",
        "product_frame": "Product fills less than 85% of the main image",
        "lifestyle_shot": "No lifestyle / in-use shot detected",
        "infographic": "No infographic image detected",
        "no_watermarks": "Text/watermark/logo detected on the main image",
    }
    penalty = min(len(fails), image_score.get("score", 0))
    image_score["score"] = max(0, image_score.get("score", 0) - penalty)
    for k in fails:
        reason = items[k].get("reason", "")
        msg = label.get(k, k)
        image_score.setdefault("issues", []).append(
            f"AI vision: {msg}{f' — {reason}' if reason else ''}"
        )


# ── Unified entrypoint ────────────────────────────────────────────────────


async def analyze_listing(
    asin: str,
    title: str,
    bullets: list[str],
    description: str,
    backend_keywords: str,
    brand: str | None = None,
    category: str | None = None,
    focus_keywords: list[str] | None = None,
    marketplace: str = "US",
    images: list[dict] | None = None,
) -> dict:
    """Run scorecard + AI rewrite + vision review in one call. Returns
    the shape the FE expects — deterministic checks, LLM rewrites, and
    vision verdicts side by side."""
    max_title = (
        TITLE_MAX_CHARS_INDIA if marketplace.upper() == "IN"
        else TITLE_MAX_CHARS_DEFAULT
    )
    scored = score_listing(
        title=title, bullets=bullets, description=description,
        backend_keywords=backend_keywords, brand=brand, images=images,
        max_title_chars=max_title,
    )
    rewrites = await rewrite_listing(
        title=title, bullets=bullets, description=description,
        backend_keywords=backend_keywords, brand=brand, category=category,
        focus_keywords=focus_keywords,
    )
    vision = await analyze_images_vision(images)
    if vision and not vision.get("error"):
        _apply_vision_verdicts_to_score(scored["scorecard"]["images"], vision)
        # Re-sum totals since we may have deducted from the images row.
        scored["total_score"] = sum(
            s["score"] for s in scored["scorecard"].values()
        )
    return {
        "asin": asin,
        "brand": brand,
        "category": category,
        "marketplace": marketplace,
        **scored,
        "rewrites": rewrites,
        "images_vision": vision,
        "images": images or [],
    }
