"""Product-aware relevance for the keyword matrix.

The matrix used to source keywords by chopping a product title into fragments
and feeding them to Amazon autocomplete, then rank whatever came back purely
by Brand Analytics demand. Both halves of that are relevance-blind, and the
failure mode is spectacular rather than subtle: an incense-cones listing
("NAQSH Hand Rolled Incense Cones ... Back Flow Masala Cones") produced a
"top" row of `artificial flowers for outdoors`, `garam masala` and
`patricia nash handbags` — high-volume search terms that happen to share a
word with a *fragment* of the title. Demand ranking without a relevance gate
reliably surfaces the loudest off-topic term in the pool.

This module supplies the missing half:

  * `build_profile` turns an SP-API catalog item into a `ProductProfile` —
    what the product actually *is*, per Amazon's own taxonomy, rather than
    per its marketing copy.
  * `build_vocabulary` weights every term that describes the product, with
    Amazon's own keyword recommendations for the ASIN folded in as evidence
    (they are Amazon's relevance judgement, so a term that recurs across them
    is a term the category really uses).
  * `score` grades one keyword against that vocabulary, and `RELEVANCE_FLOOR`
    is the cut-off below which a keyword is about something else.
  * `seed_phrases` derives autocomplete seeds anchored on the product's head
    noun, so the long-tail we pull back is on-topic by construction instead of
    being filtered clean afterwards.

Everything here is pure and synchronous — the callers own the API traffic.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# ── Tokenizing ──────────────────────────────────────────────────────────────

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_HAS_DIGIT_RE = re.compile(r"\d")

# Words that describe *how a listing is sold*, not what it is. They are the
# main source of false relevance: "pack", "premium" and "set" appear in half
# of all Amazon titles, so letting them into the vocabulary makes every
# keyword look related to every product.
_GENERIC_TOKENS = {
    "a", "about", "adjustable", "all", "and", "any", "are", "as", "assorted",
    "authentic", "back", "best", "big", "blend", "bulk", "bundle", "by",
    "classic", "combo", "count", "crafted", "cm", "designer", "durable",
    "each", "easy", "eco", "elegant", "extra", "for", "free", "friendly",
    "from", "genuine", "gift", "gifts", "good", "grade", "great", "hand",
    "handcrafted", "handmade", "handpainted", "heavy", "high", "in", "inch",
    "inches", "included", "includes",
    "including", "is", "it", "kit", "large", "layer", "layered", "long",
    "lasting", "long-lasting", "luxury", "made", "medium", "mesmerizing",
    "mini", "modern", "multi", "multipack", "natural", "new", "of", "on",
    "or", "original", "oz", "pack", "packs", "pcs", "perfect", "piece",
    "pieces", "plus", "premium", "pro", "professional", "pure", "quality",
    "painted", "printed", "reusable", "rolled", "rustic", "set", "sets",
    "size", "sized", "small", "soft",
    "special", "standard", "strong", "stylish", "super", "the", "to", "top",
    "traditional", "ultra", "unique", "use", "value", "variety", "vintage",
    "washable", "with", "woven", "women", "womens", "men", "mens", "your",
}

# Colours. They are in almost every Amazon title and are pure ambiguity as a
# standalone Meta seed — "white" returns `white bread` and `White House Black
# Market`, "navy" returns the armed forces. They stay out of the seed lists
# but remain in the vocabulary, where a colour is a perfectly good signal
# inside a longer keyword ("white cabinet knobs").
_COLOR_TOKENS = {
    "beige", "black", "blue", "bronze", "brown", "charcoal", "copper",
    "cream", "gold", "golden", "gray", "green", "grey", "ivory", "khaki",
    "maroon", "navy", "olive", "orange", "pink", "purple", "red", "rose",
    "silver", "tan", "teal", "turquoise", "violet", "white", "yellow",
}

# Classification levels that describe the whole storefront rather than the
# product. They sit at the top of every browse tree and would anchor a
# keyword like "kitchen sink" to an incense listing.
_GENERIC_CLASSIFICATIONS = {
    "categories", "home", "kitchen", "home kitchen", "home decor products",
    "home décor products", "tools home improvement", "hardware",
    "clothing shoes jewelry", "sports outdoors", "health household",
    "beauty personal care", "office products", "electronics", "toys games",
    "arts crafts sewing", "patio lawn garden", "grocery gourmet food",
    "industrial scientific", "pet supplies", "baby products", "automotive",
    "cell phones accessories", "musical instruments", "appliances",
}


def _stem(token: str) -> str:
    """Crude singularizer, enough to make `rug`/`rugs` and `knob`/`knobs` one
    term. Deliberately not a real stemmer: aggressive stemming collapses
    distinct product nouns ("cover"/"covering") and costs more relevance than
    it buys."""
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ches", "shes", "sses", "xes", "zes"):
        if token.endswith(suffix):
            return token[: -len(suffix) + 1]
    if token.endswith("s") and not token.endswith("ss") and not token.endswith("us"):
        return token[:-1]
    return token


def tokens(text: str, *, keep_generic: bool = False) -> list[str]:
    """Stemmed content tokens of a phrase, in order, deduped.

    Numeric tokens ("6x20", "15g", "8x10") are dropped — they are sizes and
    pack counts, and matching on them relates a product to every other
    product that happens to ship in the same quantity.
    """
    out: list[str] = []
    for raw in _TOKEN_SPLIT_RE.split((text or "").lower()):
        if not raw or len(raw) < 3 or _HAS_DIGIT_RE.search(raw):
            continue
        if not keep_generic and raw in _GENERIC_TOKENS:
            continue
        stemmed = _stem(raw)
        if stemmed and stemmed not in out:
            out.append(stemmed)
    return out


# ── Product profile ─────────────────────────────────────────────────────────


@dataclass
class ProductProfile:
    """What one ASIN is, assembled from the catalog rather than the title."""

    asin: str
    title: str = ""
    brand: str = ""
    # `item_type_keyword` — Amazon's own slug for the product type, e.g.
    # "area-rugs", "cabinet-and-furniture-knobs", "incense". The single most
    # reliable relevance anchor the catalog exposes.
    item_type: str = ""
    # Leaf browse node display name, e.g. "Area Rugs", "Knobs".
    browse_node: str = ""
    # Browse ancestors, leaf-first, minus storefront-level nodes.
    browse_ancestors: list[str] = field(default_factory=list)
    # Product type enum, e.g. "RUG", "HARDWARE_HANDLE".
    product_type: str = ""
    # Free-text attribute values worth matching on (material, scent, style).
    attribute_terms: list[str] = field(default_factory=list)

    @property
    def head_noun(self) -> str:
        """The one word that best names this product.

        Taken from the most specific source available: Amazon's item-type slug,
        then the leaf browse node, then the product type. Each is a noun phrase
        whose *last* token is its head ("area-rugs" -> rug, "Knobs" -> knob).
        Empty when the catalog gave us nothing, in which case
        `build_vocabulary` falls back to evidence from the suggested keywords.
        """
        for source in (self.item_type, self.browse_node, self.product_type):
            toks = tokens(source.replace("_", " ").replace("-", " "))
            if toks:
                return toks[-1]
        return ""


def _attr_values(attributes: dict, key: str) -> list[str]:
    """Pull the display values out of one SP-API attribute list.

    Attributes come back as `[{"value": ..., "language_tag": ...}, ...]`;
    a few are plain scalars. Both shapes appear on real listings.
    """
    raw = (attributes or {}).get(key)
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        return [str(raw)]
    out: list[str] = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, dict):
            val = entry.get("value")
            if isinstance(val, (str, int, float)) and str(val).strip():
                out.append(str(val).strip())
        elif isinstance(entry, str) and entry.strip():
            out.append(entry.strip())
    return out


# Attributes whose values genuinely describe the product. Deliberately short:
# every extra attribute widens the vocabulary, and a wide vocabulary is how an
# off-topic keyword sneaks back in.
_PROFILE_ATTRIBUTES = ("item_type_keyword", "material", "scent", "style", "item_shape")


def build_profile(asin: str, item: dict) -> ProductProfile:
    """Build a `ProductProfile` from a `/catalog/2022-04-01/items/{asin}`
    response fetched with `includedData=summaries,productTypes,classifications,attributes`.

    Missing sections are normal — not every listing carries classifications or
    a populated attribute set — so every lookup here degrades to "".
    """
    summaries = item.get("summaries") or []
    summary = summaries[0] if summaries else {}
    attributes = item.get("attributes") or {}

    product_types = item.get("productTypes") or []
    product_type = ""
    if product_types and isinstance(product_types[0], dict):
        product_type = product_types[0].get("productType") or ""

    item_type = ""
    itk = _attr_values(attributes, "item_type_keyword")
    if itk:
        item_type = itk[0]

    # Walk the classification chain leaf -> root, dropping storefront nodes.
    browse_node = ""
    ancestors: list[str] = []
    classifications = item.get("classifications") or []
    chain = []
    if classifications and isinstance(classifications[0], dict):
        chain = classifications[0].get("classifications") or []
    node = chain[0] if chain else None
    while isinstance(node, dict):
        name = (node.get("displayName") or "").strip()
        if name:
            normalized = " ".join(tokens(name, keep_generic=True))
            if normalized not in _GENERIC_CLASSIFICATIONS:
                ancestors.append(name)
        node = node.get("parent")
    if ancestors:
        browse_node = ancestors[0]
    else:
        # `summaries[].browseClassification` carries the leaf node even when
        # the full `classifications` block was not requested or is empty.
        bc = summary.get("browseClassification") or {}
        browse_node = (bc.get("displayName") or "").strip()

    attribute_terms: list[str] = []
    for key in _PROFILE_ATTRIBUTES:
        attribute_terms.extend(_attr_values(attributes, key))

    return ProductProfile(
        asin=asin,
        title=(summary.get("itemName") or "").strip(),
        brand=(summary.get("brand") or summary.get("manufacturer") or "").strip(),
        item_type=item_type,
        browse_node=browse_node,
        browse_ancestors=ancestors,
        product_type=product_type,
        attribute_terms=attribute_terms,
    )


# ── Vocabulary ──────────────────────────────────────────────────────────────

# Term weights. A keyword's relevance is the weight of the strongest product
# term it contains, so these are "how much does this term alone prove the
# keyword is about this product".
W_ANCHOR = 1.00   # the product's head noun — "incense", "rug", "knob"
W_CORE = 0.75     # other words from Amazon's own taxonomy for this ASIN
W_STRONG = 0.70   # recurs across Amazon's keyword recommendations
W_WEAK = 0.35     # appears once in the title or an attribute

# A keyword must clear this to enter the matrix. Sits above W_WEAK on purpose:
# a single incidental title word ("masala" in a colour name, "flow" in
# "Back Flow") is not evidence of relevance, and those were exactly the
# matches that let `garam masala` and `asus rog flow z13` into a top row.
RELEVANCE_FLOOR = 0.5

# Share of Amazon's suggested keywords a term must appear in to count as a
# category term rather than an incidental one. At 25%, "cone" qualifies for an
# incense-cones listing (so "dhoop cones" and "palo santo cones" survive) while
# "masala" — one suggestion in ninety-nine — does not.
_STRONG_TERM_SHARE = 0.25

# Matching several product terms is better evidence than matching one, but the
# strongest term still dominates; this only breaks ties between keywords that
# already passed the floor.
_MULTI_MATCH_BONUS = 0.06
_MAX_BONUS_MATCHES = 3


def build_vocabulary(
    profiles: list[ProductProfile],
    suggested_keywords: list[str] | None = None,
) -> dict[str, float]:
    """Weighted {term: weight} describing the product(s) under analysis.

    `suggested_keywords` are Amazon's own recommendations for the ASINs (the
    `amazon_asin` source). They matter more than the title does: the title is
    written by the seller, while the recommendations are Amazon telling us
    which words this category's shoppers actually use. A term that recurs
    across them is promoted to `W_STRONG` even when the title never mentions
    it, which is what keeps genuine category synonyms ("dhoop", "backflow")
    inside the matrix.

    When the catalog gave us no head noun at all, the most frequent
    recommendation term stands in for one — a product with no taxonomy is
    still anchored to whatever Amazon keeps suggesting for it.
    """
    vocab: dict[str, float] = {}

    def add(term: str, weight: float) -> None:
        if term and vocab.get(term, 0.0) < weight:
            vocab[term] = weight

    brand_tokens: set[str] = set()
    for profile in profiles:
        # Brand tokens are excluded outright. A brand word matches every
        # unrelated famous thing that shares it — "NAQSH" fuzzy-matches into
        # Nash, which is how `patricia nash handbags` and
        # `crosby stills nash and young cd` reached a top row.
        brand_tokens.update(tokens(profile.brand, keep_generic=True))

    for profile in profiles:
        anchor = profile.head_noun
        if anchor and anchor not in brand_tokens:
            add(anchor, W_ANCHOR)
        for source in (profile.item_type, profile.browse_node, profile.product_type):
            for term in tokens(source.replace("_", " ").replace("-", " ")):
                if term not in brand_tokens:
                    add(term, W_CORE)
        # Ancestors past the leaf describe a wider shelf ("Rugs, Pads &
        # Protectors"), so they are evidence but not an anchor.
        for ancestor in profile.browse_ancestors[1:]:
            for term in tokens(ancestor):
                if term not in brand_tokens:
                    add(term, W_WEAK)
        for value in profile.attribute_terms:
            for term in tokens(value):
                if term not in brand_tokens:
                    add(term, W_WEAK)
        for term in tokens(profile.title):
            if term not in brand_tokens:
                add(term, W_WEAK)

    suggested = [k for k in (suggested_keywords or []) if k]
    if suggested:
        counts: Counter[str] = Counter()
        for keyword in suggested:
            for term in tokens(keyword):
                counts[term] += 1
        threshold = max(2, int(len(suggested) * _STRONG_TERM_SHARE))
        for term, count in counts.items():
            if term in brand_tokens:
                continue
            add(term, W_STRONG if count >= threshold else W_WEAK)
        if not any(w >= W_ANCHOR for w in vocab.values()):
            # No catalog anchor — promote the single most common suggestion
            # term so the pool still has something to be relevant *to*.
            for term, _count in counts.most_common():
                if term not in brand_tokens:
                    add(term, W_ANCHOR)
                    break

    return vocab


# ── Scoring ─────────────────────────────────────────────────────────────────


def score(keyword: str, vocabulary: dict[str, float]) -> float:
    """Relevance of one keyword to the product, in 0..1.

    The strongest matched term sets the score; additional matches add a small
    bonus, so "cone burner" ranks above a bare "cone". A keyword that already
    contains the head noun is capped at 1.0 — it is about this product, and
    nothing further distinguishes it on relevance grounds; demand is what
    separates "incense" from "incense cone burner". A keyword that matches
    nothing scores 0 and is about a different product.
    """
    if not vocabulary:
        # No product signal at all — refuse to rank rather than pretend every
        # keyword is equally good. Callers treat 0 as "filtered out", and a
        # profile-less job is better off empty than confidently wrong.
        return 0.0
    matched = sorted(
        (vocabulary[t] for t in tokens(keyword) if t in vocabulary), reverse=True
    )
    if not matched:
        return 0.0
    bonus = _MULTI_MATCH_BONUS * min(len(matched), _MAX_BONUS_MATCHES + 1) - _MULTI_MATCH_BONUS
    return min(1.0, matched[0] + bonus)


def is_relevant(keyword: str, vocabulary: dict[str, float]) -> bool:
    return score(keyword, vocabulary) >= RELEVANCE_FLOOR


def all_terms_known(phrase: str, vocabulary: dict[str, float]) -> bool:
    """True when *every* content word of the phrase describes the product.

    A stricter, conjunctive counterpart to `score`, for sources that answer a
    correct product word with the wrong sense of it. Meta's ad-interest search
    is the case this exists for: seeded with "kitchen" (from a cabinet-knobs
    title) it returns `California Pizza Kitchen`, `Popeyes Louisiana Kitchen`
    and `America's Test Kitchen`, and seeded with "protection" (from a face
    mask) it returns `Symantec Endpoint Protection` and the `Consumer
    Financial Protection Bureau`. Every one of those contains a genuine
    product term, so `score` admits them all.

    What separates them is the *other* words: "pizza", "louisiana",
    "symantec", "financial" are nouns this product has nothing to do with.
    Requiring every word to be known keeps `kitchen cabinet` and
    `living room` while dropping all of the above.

    The trade is recall — `hatha yoga` goes too, since "hatha" is not a
    product word. For a column that is otherwise mostly noise, a small clean
    set is worth more than a full dirty one.
    """
    if not vocabulary:
        return True
    # Split unicode-aware rather than reusing `tokens`, which splits on
    # non-ASCII and so reduces "rāja" to the sub-3-character fragments "r" and
    # "ja" — both dropped as too short, leaving `rāja yoga` looking like the
    # known word "yoga" alone. Here an unrecognized word must survive as a
    # word so it can be rejected as one.
    words = [w for w in re.split(r"[^\w]+", (phrase or "").lower(), flags=re.UNICODE) if w]
    checked = 0
    for word in words:
        if len(word) < 3 or word in _GENERIC_TOKENS or _HAS_DIGIT_RE.search(word):
            continue
        checked += 1
        if _stem(word) not in vocabulary:
            return False
    return checked > 0


# ── Seeding ─────────────────────────────────────────────────────────────────

# Autocomplete expands a seed rather than correcting it, so seed quality caps
# result quality. Cap the count so a multi-ASIN job stays inside a sane number
# of requests (each seed costs 27 autocomplete calls).
_MAX_SEEDS_PER_PRODUCT = 6


def seed_phrases(profile: ProductProfile, suggested_keywords: list[str] | None = None) -> list[str]:
    """Autocomplete seeds for one product, best first.

    Every seed contains the product's head noun, which is the whole point:
    the previous approach fed autocomplete title fragments ("Back Flow Masala
    Cones" -> "back flow") and got back plumbing parts and gaming laptops.
    Anchoring each seed means the long-tail that comes back is on-topic before
    any filtering happens.

    Order matters — callers truncate. Amazon's own top recommendation leads
    (it is the term the category actually converts on), then the taxonomy
    name, then title phrases containing the head noun, then the bare noun as
    a catch-all for the broadest tail.
    """
    head = profile.head_noun
    if not head:
        # Fall back to Amazon's most-suggested term as the anchor.
        counts: Counter[str] = Counter()
        for keyword in suggested_keywords or []:
            for term in tokens(keyword):
                counts[term] += 1
        if not counts:
            return []
        head = counts.most_common(1)[0][0]

    seeds: list[str] = []

    def add(phrase: str) -> None:
        phrase = " ".join((phrase or "").split()).lower()
        if not phrase or phrase in seeds:
            return
        if head not in tokens(phrase, keep_generic=True) and head not in phrase:
            return
        seeds.append(phrase)

    # Amazon's highest-ranked recommendations, when we have them.
    for keyword in (suggested_keywords or [])[:3]:
        add(keyword)

    # Taxonomy names: "area rugs", "cabinet and furniture knobs", "incense".
    add(profile.item_type.replace("-", " ").replace("_", " "))
    add(profile.browse_node)

    # Title phrases anchored on the head noun — "incense cones",
    # "jute blend rug", "ceramic cabinet knobs". Two- and three-word windows
    # ending at (or containing) the head noun read as real search queries;
    # arbitrary title fragments do not.
    title_tokens = tokens(profile.title, keep_generic=True)
    brand_tokens = set(tokens(profile.brand, keep_generic=True))
    content = [t for t in title_tokens if t not in brand_tokens and t not in _GENERIC_TOKENS]
    for i, term in enumerate(content):
        if term != head:
            continue
        for width in (3, 2):
            start = max(0, i - width + 1)
            phrase = " ".join(content[start : i + 1])
            if len(phrase.split()) >= 2:
                add(phrase)

    # The bare noun last — broadest, and the only seed guaranteed to exist.
    add(head)

    return seeds[:_MAX_SEEDS_PER_PRODUCT]


# Meta's taxonomy is audience interests, not search queries, so it is seeded
# with single words. These are the product-adjacent *concepts* worth asking
# about — the head noun plus title words that name a use case or an audience.
#
# Wider than the autocomplete cap because Meta's value is precisely in the
# use-case words that sit late in a title ("... for Positivity & Pure Air
# Meditation, Yoga, Aromatherapy"). Those are the audiences worth targeting,
# and a tight cap spends every slot on product nouns before reaching them.
_MAX_META_SEEDS = 10


def meta_seeds(profile: ProductProfile) -> list[str]:
    """Single-word Meta ad-interest seeds for one product.

    Meta matches a bare word against its interest taxonomy with no sense of
    the product context, so an ambiguous seed returns interests about the
    other meaning of the word and nothing flags them as wrong. Two classes of
    word account for nearly all of it, and both are excluded here:

      * Taxonomy *modifiers* — every token of "area-rugs" or "safety-masks"
        except the head noun. "area" returned `Schengen Area`, `rural area`
        and `Harrisburg Area Community College`; "rug" returns rugs.
      * Colours — "White & Beige" in a knobs title seeded "white", which
        returned `white bread`, `white house` and `White House Black Market`.

    What survives is the head noun plus the title's own concept words, which
    is where Meta actually earns its column: "yoga", "meditation" and
    "aromatherapy" are real audiences for an incense listing even though no
    keyword source would ever suggest them.
    """
    brand_tokens = set(tokens(profile.brand, keep_generic=True))
    # Taxonomy tokens other than the head noun qualify the product ("area",
    # "cabinet", "disposable", "cup", "dust"); alone they mean something else.
    head = profile.head_noun
    modifiers = set()
    for source in (profile.item_type, profile.browse_node, profile.product_type):
        for term in tokens(source.replace("_", " ").replace("-", " ")):
            if term != head:
                modifiers.add(term)

    seeds: list[str] = []
    if head and head not in brand_tokens:
        seeds.append(head)
    for term in tokens(profile.title):
        if (
            term in brand_tokens
            or term in modifiers
            or term in _COLOR_TOKENS
            or term in seeds
            or len(term) < 4
        ):
            continue
        seeds.append(term)
        if len(seeds) >= _MAX_META_SEEDS:
            break
    return seeds
