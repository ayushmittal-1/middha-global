"""Tests for the keyword matrix's relevance model.

Every case here is a keyword the live matrix actually produced for a real
NAQSH listing before this model existed. The ranking was demand-only, so the
"top" row of each column filled with whatever high-volume search term happened
to share a word with a fragment of the product title — `garam masala` and
`asus rog flow z13` for incense cones, `king size bed frame` for a leather
journal, `crosby stills nash and young cd` for a face mask (the brand "NAQSH"
trimmed to "naqsh", which Amazon autocorrects to "nash").
"""

import keyword_relevance as kr


# Catalog data as SP-API actually returns it for these ASINs.
INCENSE = kr.ProductProfile(
    asin="B07SHH2RJX",
    title=(
        "NAQSH Hand Rolled Incense Cones Scented Long Lasting Aroma for "
        "Positivity & Pure Air Meditation, Yoga, Aromatherapy, Relaxing & "
        "Refreshing (10, Back Flow Masala Cones)"
    ),
    brand="NAQSH",
    item_type="incense",
    browse_node="Incense",
    browse_ancestors=["Incense", "Incense & Incense Holders", "Home Fragrance"],
    product_type="INCENSE",
    attribute_terms=["Herbs, Resin, Flowers", "Natural"],
)

# Amazon's own keyword recommendations for the incense ASIN, trimmed.
INCENSE_SUGGESTIONS = [
    "incense", "incense cone", "incense burner", "incense stick holder",
    "backflow incense burner", "incense holder cone", "cone incense burner",
    "dhoop cones", "encens cone", "incense waterfall", "backflow incense holder",
    "natural incense", "back flow incense cone", "dragon incense burner",
    "masala incense", "palo santo cones", "chakra cones", "frankincense cones",
]

RUG = kr.ProductProfile(
    asin="B0FK2QX26P",
    title=(
        "NAQSH Handmade Jute Blend Rug – Rustic Boho Area Mat for Living Room, "
        "Bedroom & Entryway – Durable Woven Floor Carpet with Polyester & "
        "Cotton – Eco-Conscious Home Decor"
    ),
    brand="NAQSH",
    item_type="area-rugs",
    browse_node="Area Rugs",
    browse_ancestors=["Area Rugs", "Rugs, Pads & Protectors"],
    product_type="RUG",
)

JOURNAL = kr.ProductProfile(
    asin="B0BJVTGWH4",
    title=(
        "NAQSH Handmade Unlined Leather Journal Diary - 4 Pack Vintage Writing "
        "Journal, Gift For Him Her, Travel Diary With Blank Pages, Large "
        "Diary(Size 10x7, Pack of 4, Tan Brown Dekcled Pages)"
    ),
    brand="NAQSH",
    item_type="journals",
    browse_node="Journals",
    product_type="NOTEBOOK",
)


# ── Head noun ───────────────────────────────────────────────────────────────


def test_head_noun_comes_from_amazons_item_type_not_the_title():
    # The title's last word is "Multi"/"Decor"/"Pages"; the item-type slug is
    # what actually names the product.
    assert INCENSE.head_noun == "incense"
    assert RUG.head_noun == "rug"
    assert JOURNAL.head_noun == "journal"


def test_head_noun_falls_back_through_browse_node_then_product_type():
    no_item_type = kr.ProductProfile(asin="X", browse_node="Area Rugs", product_type="RUG")
    assert no_item_type.head_noun == "rug"
    only_product_type = kr.ProductProfile(asin="X", product_type="HARDWARE_HANDLE")
    assert only_product_type.head_noun == "handle"
    assert kr.ProductProfile(asin="X").head_noun == ""


# ── Vocabulary ──────────────────────────────────────────────────────────────


def test_brand_tokens_never_enter_the_vocabulary():
    """The brand is why `patricia nash handbags` outranked every real keyword."""
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    assert "naqsh" not in vocab


def test_recurring_suggestion_terms_outweigh_incidental_title_words():
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    # "cone" recurs across Amazon's suggestions; "masala" appears in one of
    # them and in the product's colour name.
    assert vocab["cone"] >= kr.W_STRONG
    assert vocab.get("masala", 0.0) < kr.RELEVANCE_FLOOR


def test_vocabulary_anchors_on_suggestions_when_the_catalog_is_empty():
    bare = kr.ProductProfile(asin="B07SHH2RJX", title="", brand="")
    vocab = kr.build_vocabulary([bare], INCENSE_SUGGESTIONS)
    assert vocab["incense"] == kr.W_ANCHOR


def test_no_signal_at_all_yields_no_vocabulary():
    assert kr.build_vocabulary([kr.ProductProfile(asin="X")], []) == {}


# ── Scoring ─────────────────────────────────────────────────────────────────


def test_off_topic_high_volume_terms_are_rejected():
    """Each of these reached a live "top" row purely on search volume."""
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    for keyword in (
        "garam masala",
        "artificial flowers for outdoors",
        "patricia nash handbags",
        "asus rog flow z13",
        "blood flow restriction bands",
        "crosby stills nash and young cd",
        "long island",
        "dr browns preemie flow nipple",
    ):
        assert not kr.is_relevant(keyword, vocab), keyword


def test_category_synonyms_survive_even_when_the_title_never_says_them():
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    for keyword in (
        "dhoop cones",          # Hindi for incense
        "palo santo cones",
        "chakra cones",
        "backflow incense cones lavender",
        "incense waterfall",
    ):
        assert kr.is_relevant(keyword, vocab), keyword


def test_matching_more_product_terms_scores_higher():
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    assert kr.score("cone burner", vocab) > kr.score("cone", vocab)


def test_head_noun_matches_saturate_at_one():
    """Both keywords are unambiguously about the product; demand, not
    relevance, is what should separate them from there."""
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    assert kr.score("incense", vocab) == 1.0
    assert kr.score("incense cone burner", vocab) == 1.0


def test_score_is_zero_without_a_vocabulary():
    assert kr.score("incense", {}) == 0.0


def test_relevance_is_per_product_not_global():
    """The same keyword must pass for one ASIN and fail for another."""
    incense_vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    rug_vocab = kr.build_vocabulary([RUG], [])
    assert kr.is_relevant("jute area rug", rug_vocab)
    assert not kr.is_relevant("jute area rug", incense_vocab)
    assert kr.is_relevant("backflow incense cones", incense_vocab)
    assert not kr.is_relevant("backflow incense cones", rug_vocab)


def test_plural_and_singular_are_the_same_term():
    vocab = kr.build_vocabulary([RUG], [])
    assert kr.is_relevant("area rug", vocab)
    assert kr.is_relevant("area rugs", vocab)


# ── Seeds ───────────────────────────────────────────────────────────────────


def test_every_autocomplete_seed_contains_the_head_noun():
    """Seed quality caps result quality — autocomplete expands a seed, it never
    corrects one. An unanchored seed ("back flow", "size") is what returned
    plumbing valves and `king size bed frame`."""
    for profile, suggestions in (
        (INCENSE, INCENSE_SUGGESTIONS),
        (RUG, []),
        (JOURNAL, []),
    ):
        seeds = kr.seed_phrases(profile, suggestions)
        assert seeds, profile.asin
        for seed in seeds:
            assert profile.head_noun in seed, (profile.asin, seed)


def test_journal_seeds_never_reduce_to_a_packaging_word():
    """The old seeds took the title's tail — "Size 10x7, Pack of 4" — and
    autocompleted "size", returning mattresses, bed frames and ziploc bags."""
    seeds = kr.seed_phrases(JOURNAL, [])
    assert "size" not in seeds
    assert "pack of 4" not in seeds


def test_seeds_fall_back_to_amazons_top_term_without_a_catalog():
    bare = kr.ProductProfile(asin="X", title="", brand="")
    assert kr.seed_phrases(bare, INCENSE_SUGGESTIONS)
    assert kr.seed_phrases(bare, []) == []


def test_meta_seeds_skip_brand_and_filler_words():
    """"Long Lasting" seeded Meta with "long", which returned `long island`
    and `long hair (haircare)`."""
    seeds = kr.meta_seeds(INCENSE)
    assert "long" not in seeds
    assert "naqsh" not in seeds
    assert seeds[0] == "incense"


def test_meta_seeds_keep_the_use_case_words_meta_is_good_at():
    seeds = kr.meta_seeds(INCENSE)
    assert {"meditation", "yoga", "aromatherapy"} <= set(seeds)


# ── Meta's conjunctive gate ─────────────────────────────────────────────────


def test_meta_gate_rejects_the_wrong_sense_of_a_correct_product_word():
    """Every one of these contains a genuine product term, which is why the
    ordinary relevance score admits them all. The conjunctive gate looks at
    the other words instead."""
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    knob_vocab = kr.build_vocabulary(
        [kr.ProductProfile(
            asin="B08CF159WP",
            title="NAQSH Ceramic Cabinet Knobs Kitchen Drawer Knob Handpainted "
                  "Cupboard Door & Drawers Pulls (White & Beige, 12)",
            brand="NAQSH",
            item_type="cabinet-and-furniture-knobs",
            browse_node="Knobs",
            product_type="HARDWARE_HANDLE",
        )],
        [],
    )
    for phrase, vocabulary in (
        ("california pizza kitchen", knob_vocab),
        ("popeyes louisiana kitchen", knob_vocab),
        ("america's test kitchen", knob_vocab),
        ("knob creek", knob_vocab),
        ("symantec endpoint protection", vocab),
        ("king of mask singer", vocab),
    ):
        assert kr.score(phrase, vocabulary) >= 0.0
        assert not kr.all_terms_known(phrase, vocabulary), phrase


def test_meta_gate_keeps_interests_built_only_from_product_words():
    knob_vocab = kr.build_vocabulary(
        [kr.ProductProfile(
            asin="B08CF159WP",
            title="NAQSH Ceramic Cabinet Knobs Kitchen Drawer Knob Handpainted "
                  "Cupboard Door & Drawers Pulls (White & Beige, 12)",
            brand="NAQSH",
            item_type="cabinet-and-furniture-knobs",
            browse_node="Knobs",
            product_type="HARDWARE_HANDLE",
        )],
        [],
    )
    for phrase in ("kitchen cabinet", "ceramic", "door", "drawer knob"):
        assert kr.all_terms_known(phrase, knob_vocab), phrase


def test_meta_gate_is_not_fooled_by_non_ascii_words():
    """"rāja" splits into sub-3-character ASCII fragments, which made
    `rāja yoga` look like the known word "yoga" on its own."""
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    assert kr.all_terms_known("yoga", vocab)
    assert not kr.all_terms_known("rāja yoga", vocab)


def test_meta_gate_passes_everything_when_there_is_no_vocabulary():
    assert kr.all_terms_known("anything", {})


def test_meta_gate_rejects_a_phrase_with_nothing_to_check():
    vocab = kr.build_vocabulary([INCENSE], INCENSE_SUGGESTIONS)
    assert not kr.all_terms_known("the of and", vocab)
    assert not kr.all_terms_known("", vocab)
