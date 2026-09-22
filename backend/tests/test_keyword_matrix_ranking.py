"""Tests for how the keyword matrix turns sourced keywords into a 3x3 grid.

The ranking used to be demand-only, which is exactly backwards for what the
matrix is for: search volume is a property of the keyword, not of its fit to
the product, so ranking a mixed pool by volume promotes whatever off-topic
term is most popular. These cover the gate, the composite and the layout.
"""

import keyword_matrix as km
import keyword_relevance as kr


VOCAB = {"incense": kr.W_ANCHOR, "cone": kr.W_STRONG, "masala": 0.35}


# ── Relevance gate ──────────────────────────────────────────────────────────


def test_gated_source_drops_keywords_about_another_product():
    kept, relevance, dropped = km._filter_by_relevance(
        ["incense cones", "garam masala", "asus rog flow z13"], VOCAB, gated=True
    )
    assert kept == ["incense cones"]
    assert dropped == ["garam masala", "asus rog flow z13"]
    assert relevance["incense cones"] >= kr.RELEVANCE_FLOOR


def test_ungated_source_keeps_everything_and_floors_relevance():
    """Amazon's own suggestions and Meta's interests are vouched for by their
    source; scoring them as off-topic would bury "dhoop" and "Aromatherapy"."""
    kept, relevance, dropped = km._filter_by_relevance(
        ["dhoop", "aromatherapy"], VOCAB, gated=False
    )
    assert kept == ["dhoop", "aromatherapy"]
    assert dropped == []
    assert all(v >= kr.RELEVANCE_FLOOR for v in relevance.values())


def test_no_vocabulary_keeps_everything_rather_than_emptying_the_matrix():
    kept, relevance, dropped = km._filter_by_relevance(
        ["anything at all"], {}, gated=True
    )
    assert kept == ["anything at all"]
    assert dropped == []
    assert relevance == {"anything at all": 0.0}


def test_the_amazon_column_is_not_gated_against_its_own_vocabulary():
    assert "amazon_asin" not in km._RELEVANCE_GATED_SOURCES
    assert "meta" not in km._RELEVANCE_GATED_SOURCES
    assert "amazon_searchbar" in km._RELEVANCE_GATED_SOURCES


# ── Composite score ─────────────────────────────────────────────────────────


def _entry(keyword, relevance, sfr=None, click=None, conv=None):
    return {
        "keyword": keyword,
        "relevance": relevance,
        "sfr": sfr,
        "click_share": click,
        "conversion_share": conv,
        "cpc": None,
        "in_report": sfr is not None,
    }


def test_relevance_breaks_ties_between_comparable_keywords():
    """Relevance is a *tiebreaker* here, not the filter. Keywords that are
    flatly about another product never reach `_score` in a gated column — the
    gate drops them (see `test_gated_source_drops_keywords_about_another_
    product`). What the composite does is stop a marginally more popular but
    less on-topic keyword from taking a top slot."""
    scored = km._score([
        _entry("cone mold", 0.5, sfr=50_000, click=0.20, conv=0.20),
        _entry("incense holder", 1.0, sfr=50_000, click=0.20, conv=0.20),
    ])
    assert scored[0]["keyword"] == "incense holder"


def test_relevance_flips_a_modest_demand_gap_in_a_realistic_pool():
    """Demand is min-max normalized across the whole source pool, so a small
    absolute gap between two mid-pool keywords is a small normalized gap —
    small enough for relevance to overturn."""
    pool = [
        _entry(f"filler {i}", 0.7, sfr=1 + i * 100_000, click=0.05, conv=0.05)
        for i in range(10)
    ]
    pool.append(_entry("cone mold", 0.5, sfr=480_000, click=0.05, conv=0.05))
    pool.append(_entry("incense holder", 1.0, sfr=500_000, click=0.05, conv=0.05))
    ranked = [e["keyword"] for e in km._score(pool)]
    assert ranked.index("incense holder") < ranked.index("cone mold")


def test_relevance_cannot_override_a_large_demand_gap():
    """The weighting is deliberate: between two keywords that both passed the
    gate, a much more searched one still wins. Relevance decides *whether* a
    keyword belongs in the matrix; demand decides where."""
    scored = km._score([
        _entry("incense", 1.0, sfr=900_000, click=0.01, conv=0.01),
        _entry("cone", 0.5, sfr=100, click=0.90, conv=0.90),
    ])
    assert scored[0]["keyword"] == "cone"


def test_demand_still_separates_two_equally_relevant_keywords():
    scored = km._score([
        _entry("incense quiet", 1.0, sfr=900_000, click=0.01, conv=0.01),
        _entry("incense popular", 1.0, sfr=1_000, click=0.5, conv=0.5),
    ])
    assert scored[0]["keyword"] == "incense popular"


def test_keywords_without_brand_analytics_rank_after_every_keyword_with_it():
    scored = km._score([
        _entry("no demand data", 1.0),
        _entry("weakest with data", 0.0, sfr=900_000, click=0.0, conv=0.0),
    ])
    assert [e["keyword"] for e in scored] == ["weakest with data", "no demand data"]
    assert scored[0]["demand"] is not None
    assert scored[1]["demand"] is None


def test_keywords_without_brand_analytics_still_get_a_score():
    """They fill the tail of a thin column instead of leaving it empty."""
    scored = km._score([_entry("relevant but unmeasured", 1.0)])
    assert scored[0]["composite"] is not None
    assert scored[0]["composite"] > 0


def test_lower_search_frequency_rank_means_more_demand():
    scored = km._score([
        _entry("rare", 1.0, sfr=500_000, click=0.1, conv=0.1),
        _entry("common", 1.0, sfr=100, click=0.1, conv=0.1),
    ])
    assert scored[0]["keyword"] == "common"


# ── Layout ──────────────────────────────────────────────────────────────────


def test_tiers_split_the_top_fifteen_into_three_fives():
    scored = km._score([
        _entry(f"incense {i}", 1.0, sfr=i + 1, click=0.1, conv=0.1) for i in range(20)
    ])
    tiers = km._tier_split(scored)
    assert [len(tiers[t]) for t in ("top", "medium", "low")] == [5, 5, 5]
    assert tiers["top"][0]["composite"] >= tiers["low"][-1]["composite"]


def test_a_thin_source_pads_short_rather_than_borrowing_from_another():
    tiers = km._tier_split(km._score([_entry("incense", 1.0, sfr=10, click=0.1, conv=0.1)]))
    assert len(tiers["top"]) == 1
    assert tiers["medium"] == [] and tiers["low"] == []


# ── Interleaving ────────────────────────────────────────────────────────────


def test_interleave_gives_every_seed_a_place_before_any_seed_repeats():
    """One broad seed answers with twenty near-identical results; in seed
    order those take all fifteen slots and bury every other seed."""
    out = km._interleave({
        "yoga": ["hatha yoga", "bikram yoga", "yin yoga"],
        "incense": ["incense"],
        "aromatherapy": ["aromatherapy", "essential oils"],
    })
    assert out[:3] == ["hatha yoga", "incense", "aromatherapy"]
    assert set(out) == {
        "hatha yoga", "bikram yoga", "yin yoga", "incense",
        "aromatherapy", "essential oils",
    }


def test_interleave_handles_empty_input():
    assert km._interleave({}) == []
    assert km._interleave({"seed": []}) == []
