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

import json
import logging
import re
from dataclasses import dataclass, field

from agent import client  # Reuse the existing Groq client instance.

# Override the model rather than reusing agent.MODEL. The listing rewrite
# leans on JSON mode + strict guideline-following, which needs the 70B
# tier; agent.py's default model may have drifted or been deprecated by
# Groq. Keep this feature independent so it stays working even if the
# chat path's model config changes.
MODEL = "llama-3.3-70b-versatile"

# Vision model for image compliance checks. Llama-4 Scout on Groq
# accepts image URLs via OpenAI-style content parts. Keep this separate
# from the text MODEL so a Groq deprecation on one side doesn't break
# the other.
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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


_REWRITE_SYSTEM_PROMPT = """You are an Amazon listing optimization expert following Amazon's 2026 US style guidelines.

Guidelines you MUST follow when rewriting:
- Title: include Brand + Product Type + 2-3 key attributes (size/color/quantity/material). Under 200 chars. No ALL CAPS. No promotional terms (free shipping, best seller, guarantee, sale, discount, etc.). No emojis. Use primary keywords naturally, never stuff.
- Bullets: exactly 5 bullets. Each under 500 chars. Structure: FEATURE → BENEFIT → USE CASE. Scannable — lead with a 2-4 word capitalized hook, then a colon, then the sentence. No promotional language.
- Description: 200-2000 chars. Include overview, benefits, materials/ingredients, usage, care. NO URLs, emails, phone numbers, or promotional/pricing claims. NO unsupported medical/health claims.
- Backend search terms: single space-separated string, under 250 UTF-8 bytes. Include synonyms, alternate spellings, common misspellings. No duplicates. No commas. No competitor brand names. Do not repeat words already in the title.

Return VALID JSON with exactly this shape:
{
  "title": "rewritten title string",
  "bullets": ["bullet 1", "bullet 2", "bullet 3", "bullet 4", "bullet 5"],
  "description": "rewritten description string",
  "backend_keywords": "space separated keyword string",
  "notes": "2-3 sentence summary of the biggest improvements you made"
}

Do not include any text outside the JSON object."""


async def rewrite_listing(
    title: str,
    bullets: list[str],
    description: str,
    backend_keywords: str,
    brand: str | None,
    category: str | None,
    focus_keywords: list[str] | None = None,
) -> dict:
    """Ask Groq to rewrite each field. Returns a dict with the rewritten
    values. Falls back to a minimal error payload if the model returns
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
        "\nRewrite every field to be fully compliant. If a field is empty, "
        "create one from scratch based on the other fields and the brand/category."
    )
    user_prompt = "\n".join(user_prompt_parts)

    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("rewrite_listing: LLM returned invalid JSON: %s", e)
        return {"error": "Model returned malformed JSON — try again"}
    except Exception as e:
        log.exception("rewrite_listing failed")
        return {"error": str(e)}


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


async def analyze_images_vision(images: list[dict] | None) -> dict:
    """Send image URLs to Groq's vision model and grade against the manual
    checklist. Best-effort — returns {} silently when there are no images
    or the model errors; the FE falls back to the human checklist in
    either case."""
    urls = [
        str(i.get("url"))
        for i in (images or [])
        if isinstance(i, dict) and i.get("url")
    ]
    # Cap at 6 images to keep the request small and the latency bounded.
    # Amazon shows main + up to 8 gallery — 6 is enough to judge the set.
    urls = urls[:6]
    if not urls:
        return {}

    content: list[dict] = [
        {"type": "text", "text": _build_vision_user_prompt()},
    ]
    for u in urls:
        content.append({"type": "image_url", "image_url": {"url": u}})

    try:
        resp = await client.chat.completions.create(
            model=VISION_MODEL,
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
        log.warning("analyze_images_vision: bad JSON: %s", e)
        return {"error": "Vision model returned malformed JSON"}
    except Exception as e:
        log.exception("analyze_images_vision failed")
        return {"error": str(e)}

    # Normalise: only keep the known keys, coerce verdict to lowercase.
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
    return {"items": out, "images_reviewed": urls}


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
