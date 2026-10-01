"""Memory bounds on report bodies.

Guards the OOM that killed the keyword-matrix picker: the Brand Analytics
search-terms week (millions of rows, hundreds of MB of text) was pinned in
`amazon_sp._REPORT_TEXT_CACHE` for the life of the process, because that
cache was bounded by ENTRY COUNT only. The instance died mid-job and the
browser surfaced the dropped poll as "Failed to fetch".

Two halves, both covered here:
  1. the caches refuse a body too large to hold, instead of holding it;
  2. parsing a BA body allocates nothing beyond the rows it returns.
"""

import json
import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import amazon_sp
import database


@pytest.fixture(autouse=True)
def _clear_cache():
    amazon_sp._REPORT_TEXT_CACHE.clear()
    yield
    amazon_sp._REPORT_TEXT_CACHE.clear()


# ── 1. cache size caps ──────────────────────────────────────────────────────


def test_oversized_body_is_not_memoized():
    """The BA-week case. Refusing to cache costs a re-fetch; caching it
    costs the process."""
    huge = "x" * (amazon_sp._REPORT_TEXT_CACHE_MAX_ENTRY_CHARS + 1)
    amazon_sp._cache_report_text("ba-week", huge)
    assert "ba-week" not in amazon_sp._REPORT_TEXT_CACHE


def test_normal_sized_body_is_memoized():
    """The cap must not break the reports the cache exists for — a
    settlement or fee body is a few MB at most."""
    body = "y" * 1024
    amazon_sp._cache_report_text("fees-1", body)
    assert amazon_sp._REPORT_TEXT_CACHE["fees-1"] == body


def test_total_bytes_stay_under_the_cap():
    """Many mid-sized bodies must not add up past the budget, which the
    old entry-count cap allowed."""
    entry = amazon_sp._REPORT_TEXT_CACHE_MAX_ENTRY_CHARS
    needed = (amazon_sp._REPORT_TEXT_CACHE_MAX_TOTAL_CHARS // entry) + 3
    for i in range(needed):
        amazon_sp._cache_report_text(f"r{i}", "z" * entry)
    total = sum(len(v) for v in amazon_sp._REPORT_TEXT_CACHE.values())
    assert total <= amazon_sp._REPORT_TEXT_CACHE_MAX_TOTAL_CHARS
    # Eviction is oldest-first, so the most recent body is the one kept.
    assert f"r{needed - 1}" in amazon_sp._REPORT_TEXT_CACHE
    assert "r0" not in amazon_sp._REPORT_TEXT_CACHE


def test_entry_count_cap_still_holds():
    for i in range(amazon_sp._REPORT_TEXT_CACHE_MAX + 10):
        amazon_sp._cache_report_text(f"small{i}", "q" * 16)
    assert len(amazon_sp._REPORT_TEXT_CACHE) <= amazon_sp._REPORT_TEXT_CACHE_MAX


def test_recaching_same_id_does_not_double_count():
    """Re-caching one report must replace it, not leak the old copy into
    the running total and trigger phantom evictions."""
    amazon_sp._cache_report_text("dup", "a" * 4096)
    amazon_sp._cache_report_text("dup", "b" * 4096)
    assert len(amazon_sp._REPORT_TEXT_CACHE) == 1
    assert amazon_sp._REPORT_TEXT_CACHE["dup"] == "b" * 4096


@pytest.mark.asyncio
async def test_durable_cache_skips_compressing_a_huge_body(monkeypatch):
    """`put_report_body_cache` used to measure a body by compressing it —
    two more full copies at the exact moment the parsed rows are resident.
    Length is checked first now, so nothing is allocated."""
    compressed = []

    def _boom(*a, **kw):
        compressed.append(1)
        raise AssertionError("compressed an oversized body")

    monkeypatch.setattr(database.gzip, "compress", _boom)
    await database.put_report_body_cache(
        "ba-week", "x" * (database._REPORT_BODY_MAX_CHARS + 1)
    )
    assert not compressed


# ── 2. BA parse holds nothing extra ─────────────────────────────────────────


def test_parses_json_envelope_and_returns_the_rows():
    rows = [{"searchTerm": "incense holder"}, {"searchTerm": "brass burner"}]
    body = json.dumps({
        "reportSpecification": {"reportType": "GET_BRAND_ANALYTICS_SEARCH_TERMS_REPORT"},
        "dataByDepartmentAndSearchTerm": rows,
    })
    assert amazon_sp._parse_brand_analytics_report(body) == rows


def test_parses_a_bare_json_list():
    rows = [{"searchTerm": "grey knob"}]
    assert amazon_sp._parse_brand_analytics_report(json.dumps(rows)) == rows


def test_tsv_fallback_when_the_body_is_not_json():
    body = "searchTerm\tsearchFrequencyRank\nincense holder\t1200\nbrass burner\t9\n"
    rows = amazon_sp._parse_brand_analytics_report(body)
    assert rows == [
        {"searchTerm": "incense holder", "searchFrequencyRank": "1200"},
        {"searchTerm": "brass burner", "searchFrequencyRank": "9"},
    ]


def test_csv_fallback_when_the_body_is_not_json():
    body = "searchTerm,searchFrequencyRank\nincense holder,1200\n"
    rows = amazon_sp._parse_brand_analytics_report(body)
    assert rows == [{"searchTerm": "incense holder", "searchFrequencyRank": "1200"}]


def test_line_iterator_matches_splitlines_without_materializing():
    """The delimited path feeds csv a line generator instead of a StringIO
    copy of the whole body. It has to agree with the obvious version."""
    for body in (
        "a\nb\nc",
        "a\nb\nc\n",
        "",
        "\n",
        "one line only",
        "trailing\n\nblank\n",
    ):
        assert (
            "".join(amazon_sp._iter_str_lines(body)) == body
        ), f"lossy for {body!r}"

    gen = amazon_sp._iter_str_lines("a\nb\nc\n")
    assert hasattr(gen, "__next__"), "must stay lazy, not build a list"
