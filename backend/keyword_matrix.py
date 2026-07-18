"""
Keyword sourcing → scoring → 3x3 matrix journey.

Given a list of ASINs, this module:
  1. Sources keywords from three paths (per source, per ASIN, deduped):
       - Amazon suggested keywords (Ads API, seeded by ASIN)
       - Meta ad interests (Marketing API, seeded by product title) [TODO token]
       - Amazon autocomplete (public endpoint, seeded by product title)
  2. Enriches every keyword with Brand Analytics (SFR, click share,
     conversion share), skipping any keyword the report doesn't contain.
  3. Enriches with Amazon Ads bid recommendations (CPC). Requires an
     ad_group_id — if none is supplied, this step is skipped and CPC stays
     null in the final matrix.
  4. Computes a composite score per keyword — equal-weighted normalized
     inverted-SFR + click share + conversion share (each rescaled to 0..1
     within the source pool). Every cell in the matrix keeps the raw
     SFR / click share / conversion share alongside the composite so the
     user can see what's driving the ranking.
  5. Lays out a 3x3 matrix: rows = Top / Medium / Low, cols = Amazon (ASIN
     suggestions) / Meta / Amazon Searchbar. Within a source, we take the
     top-15 by composite and slice into three tiers of five.

The flow is a job, not a single request. `start_job` kicks off the pipeline
as a background task and returns a job_id; `get_job` returns the current
step and any partial results collected so far. Storage is an in-memory
dict — dies with the process. Fine for now; move to Mongo later if we need
persistence across restarts.

Edge cases (title missing, autocomplete returns nothing, Brand Analytics
misses a keyword, source has fewer than 15 scorable keywords) are noted in
per-source metadata rather than aborting the job.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

import amazon_sp
from amazon_ads import (
    fetch_keyword_bid_recommendations,
    fetch_suggested_keywords,
    find_default_ad_group,
)
from auth import _db, require_user
from keywords import fetch_amazon_keywords

# ── Job store ───────────────────────────────────────────────────────────────
# Single-process, in-memory. Keys: job_id -> job dict. Enough for the
# dashboard's polling UX; nothing here needs to survive a restart.
_JOBS: dict[str, dict[str, Any]] = {}

SOURCES = ("amazon_asin", "meta", "amazon_searchbar")
_SOURCE_LABELS = {
    "amazon_asin": "Amazon (ASIN)",
    "meta": "Meta",
    "amazon_searchbar": "Amazon Searchbar",
}
STEPS = ("sourcing", "brand_analytics", "cpc", "scoring", "done")


# ── Public API ──────────────────────────────────────────────────────────────


def start_job(
    asins: list[str],
    ad_group_id: str | None = None,
    campaign_id: str | None = None,
) -> str:
    """Kick off a matrix job and return its id.

    Runs in the background so the endpoint can return immediately. The user's
    auth ContextVar is copied into the task automatically by asyncio.

    CPC enrichment needs BOTH `ad_group_id` AND `campaign_id` — Amazon's v4
    bid-recommendations endpoint rejects a request that's missing either.
    Passing only one is treated as "no CPC" and noted in the source metadata.
    """
    asins = [a.strip().upper() for a in asins if a and a.strip()]
    if not asins:
        raise ValueError("At least one ASIN is required.")
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "step": "sourcing",
        "asins": asins,
        "ad_group_id": ad_group_id,
        "campaign_id": campaign_id,
        "ad_group_source": "user" if (ad_group_id and campaign_id) else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sources": {s: {"keywords": [], "notes": []} for s in SOURCES},
        "titles": {},
        "matrix": None,
        "error": None,
    }
    asyncio.create_task(_run_job(job_id))
    return job_id


def get_job(job_id: str) -> dict | None:
    return _JOBS.get(job_id)


# ── Orchestrator ────────────────────────────────────────────────────────────


async def _run_job(job_id: str) -> None:
    job = _JOBS[job_id]
    try:
        # Titles are needed by both Meta and Amazon Searchbar sourcing, so
        # resolve them once upfront rather than twice inside those functions.
        job["step"] = "sourcing"
        job["titles"] = await _fetch_titles_for_asins(job["asins"])

        # Fan out the three sourcing paths concurrently — none of them share
        # state and each hits a different API.
        asin_res, meta_res, sb_res = await asyncio.gather(
            _source_amazon_asin(job["asins"]),
            _source_meta(list(job["titles"].values())),
            _source_amazon_searchbar(list(job["titles"].values())),
            return_exceptions=True,
        )
        for name, res in (
            ("amazon_asin", asin_res),
            ("meta", meta_res),
            ("amazon_searchbar", sb_res),
        ):
            if isinstance(res, Exception):
                job["sources"][name]["notes"].append(f"error: {res}")
                job["sources"][name]["keywords"] = []
            else:
                job["sources"][name]["keywords"] = res["keywords"]
                job["sources"][name]["notes"].extend(res.get("notes", []))

        job["step"] = "brand_analytics"
        ba_index = await _fetch_brand_analytics_index()
        for source in SOURCES:
            src = job["sources"][source]
            enriched, coverage = _enrich_with_brand_analytics(src["keywords"], ba_index)
            src["enriched"] = enriched
            src["ba_coverage"] = coverage

        job["step"] = "cpc"
        ad_group_id = job.get("ad_group_id")
        campaign_id = job.get("campaign_id")
        # Auto-discover an ad group when the caller didn't supply one. One call
        # is enough — Amazon's productAds/list returns any ad group in the
        # account (the asinFilter is currently ignored server-side, so we can't
        # rely on a per-ASIN match). Any valid (campaign_id, ad_group_id) pair
        # unlocks bid-recommendations in the account's marketplace context.
        if not (ad_group_id and campaign_id):
            try:
                found = await find_default_ad_group(preferred_asin=job["asins"][0])
            except Exception as e:
                print(f"[keyword_matrix] ad group lookup failed: {e}")
                found = None
            if found:
                ad_group_id = found["adGroupId"]
                campaign_id = found["campaignId"]
                job["ad_group_id"] = ad_group_id
                job["campaign_id"] = campaign_id
                job["ad_group_source"] = (
                    f"auto ({found['state']}, "
                    f"{'ASIN-matched' if found['matched_asin'] else 'account default'})"
                )
                print(
                    f"[keyword_matrix] auto-discovered ad group {ad_group_id} "
                    f"(campaign {campaign_id}) matched_asin={found['matched_asin']}"
                )
        if ad_group_id and campaign_id:
            for source in SOURCES:
                src = job["sources"][source]
                await _enrich_with_cpc(src["enriched"], ad_group_id, campaign_id)
        else:
            for source in SOURCES:
                job["sources"][source]["notes"].append(
                    "cpc skipped — no ad group targeting these ASINs (create one, "
                    "or pass ad_group_id + campaign_id explicitly)"
                )

        job["step"] = "scoring"
        for source in SOURCES:
            src = job["sources"][source]
            src["scored"] = _score(src["enriched"])

        job["matrix"] = _build_matrix(job["sources"])
        job["step"] = "done"
        job["status"] = "done"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


# ── Title resolution (SP-API catalog) ───────────────────────────────────────


async def _fetch_titles_for_asins(asins: list[str]) -> dict[str, str]:
    """Return {asin: title}. ASINs the catalog can't resolve map to an empty
    string — downstream sourcing degrades gracefully when the title is blank.

    Uses the user's PRIMARY marketplace only. Passing the full marketplace
    list causes a 400 ("operation not supported for fulfillment only
    marketplaces") when the account has remote-fulfillment marketplaces
    attached — the catalog endpoint scopes to a single storefront and
    fulfillment-only IDs are not valid there.
    """
    user = require_user()
    primary = amazon_sp._user_primary_marketplace_id(user)
    titles: dict[str, str] = {}

    async def _one(asin: str) -> None:
        try:
            data = await amazon_sp._sp_request(
                "GET",
                f"/catalog/2022-04-01/items/{asin}",
                params={
                    "marketplaceIds": primary,
                    "includedData": "summaries",
                },
            )
            summaries = data.get("summaries") if isinstance(data, dict) else None
            title = ""
            if summaries:
                title = summaries[0].get("itemName") or ""
            titles[asin] = title
        except Exception as e:
            print(f"[keyword_matrix] title lookup failed for {asin}: {e}")
            titles[asin] = ""

    # Sequential is fine — catalog is a rare call and 429 backoff is built
    # into `_sp_request`. Parallelizing risks tripping the 2 req/s bucket.
    for asin in asins:
        await _one(asin)
    return titles


# ── Source 1: Amazon suggested keywords (Ads API) ───────────────────────────


async def _source_amazon_asin(asins: list[str]) -> dict:
    """Union of `fetch_suggested_keywords` results across all ASINs."""
    seen: set[str] = set()
    keywords: list[str] = []
    notes: list[str] = []
    for asin in asins:
        try:
            resp = await fetch_suggested_keywords(asin, max_suggestions=100)
            raw = resp.get("suggestedKeywords") if isinstance(resp, dict) else []
            n_before = len(seen)
            for entry in raw or []:
                kw = (
                    entry.get("keywordText")
                    if isinstance(entry, dict)
                    else str(entry)
                )
                if not kw:
                    continue
                kw_norm = kw.strip().lower()
                if kw_norm and kw_norm not in seen:
                    seen.add(kw_norm)
                    keywords.append(kw_norm)
            if len(seen) == n_before:
                notes.append(f"{asin}: no suggested keywords (new ASIN?)")
        except Exception as e:
            notes.append(f"{asin}: error {e}")
    return {"keywords": keywords, "notes": notes}


# ── Source 2: Meta ad interests ─────────────────────────────────────────────

_META_GRAPH_URL = "https://graph.facebook.com/v19.0/search"


async def _source_meta(titles: list[str]) -> dict:
    """Ad-interest suggestions seeded by each ASIN's product title.

    TODO: fill META_ACCESS_TOKEN. Until then this returns an empty set with a
    note so the pipeline can still complete and the UI can show the empty
    column with an explanation.
    """
    token = os.getenv("META_ACCESS_TOKEN", "").strip()
    if not token:
        return {
            "keywords": [],
            "notes": ["META_ACCESS_TOKEN not set — Meta interest sourcing skipped"],
        }
    seen: set[str] = set()
    keywords: list[str] = []
    notes: list[str] = []
    async with httpx.AsyncClient(timeout=15) as client:
        for title in titles:
            if not title:
                notes.append("empty title — skipped Meta lookup for one ASIN")
                continue
            query = title.split(",")[0].strip()[:100]
            try:
                resp = await client.get(
                    _META_GRAPH_URL,
                    params={
                        "type": "adinterest",
                        "q": query,
                        "limit": 25,
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                notes.append(f"meta lookup failed for '{query}': {e}")
                continue
            for item in data.get("data", []):
                name = (item.get("name") or "").strip().lower()
                if name and name not in seen:
                    seen.add(name)
                    keywords.append(name)
    return {"keywords": keywords, "notes": notes}


# ── Source 3: Amazon autocomplete ───────────────────────────────────────────


_TITLE_SPLIT_RE = re.compile(r"[|,\-–—\(\)\[\]/]+")


def _seed_candidates(title: str) -> list[str]:
    """Derive multiple autocomplete seeds from a product title.

    `fetch_amazon_keywords` trims from the right until a prefix returns
    results — great for titles like "Kiwi Shoe Polish Black" (head term first),
    terrible for "Premium Cotton 5 Pack — Face Mask" (head term at the end).
    We extract a handful of candidate seeds so both structures work.
    """
    seeds: list[str] = []
    title = title.strip()
    if not title:
        return seeds
    # Full title (existing behaviour).
    seeds.append(title)
    # Split on Amazon's title separators — "|", commas, dashes, parens — and
    # feed each fragment to autocomplete. This surfaces the head term when it
    # lives in the last segment (e.g. "... | Medium Size Face Mask").
    for frag in _TITLE_SPLIT_RE.split(title):
        frag = frag.strip()
        if len(frag.split()) >= 2 and frag not in seeds:
            seeds.append(frag)
    # Also try the last 2 / 3 words of the raw title as a fallback for titles
    # with no separators (e.g. "Cotton Linen Unisex Face Masks").
    words = title.split()
    for n in (3, 2):
        if len(words) > n:
            tail = " ".join(words[-n:])
            if tail not in seeds:
                seeds.append(tail)
    return seeds


async def _source_amazon_searchbar(titles: list[str]) -> dict:
    """Autocomplete keywords seeded by each ASIN's product title.

    We fan out a small set of seeds per title (full, title fragments, tail
    words) rather than a single seed, so head terms that live at the end of a
    marketing-style title still make it into the pool.
    """
    seen: set[str] = set()
    keywords: list[str] = []
    notes: list[str] = []
    for title in titles:
        if not title:
            notes.append("empty title — skipped autocomplete for one ASIN")
            continue
        found_any = False
        for seed in _seed_candidates(title):
            try:
                results = await fetch_amazon_keywords(seed)
            except Exception as e:
                notes.append(f"autocomplete failed for '{seed[:40]}...': {e}")
                continue
            if not results:
                continue
            found_any = True
            for kw in results:
                norm = kw.strip().lower()
                if norm and norm not in seen:
                    seen.add(norm)
                    keywords.append(norm)
        if not found_any:
            notes.append(f"no autocomplete for '{title[:40]}...'")
    return {"keywords": keywords, "notes": notes}


# ── Brand Analytics enrichment ──────────────────────────────────────────────


_BA_MAX_WEEKS_BACK = 3

# BA reports are cached in Mongo (GridFS) as one gzipped JSON blob per
# (user, week). We keep only the trimmed 3-signal shape — SFR, total click
# share, total conversion share — because that's all the scorer needs; the
# raw report is ~500 MB uncompressed / ~7 GB as Python dicts, which is not
# something we want in-process memory. Trimmed shape shrinks to ~150 MB
# uncompressed / ~15-25 MB gzipped — a healthy Mongo doc size.
#
# Nothing is cached in-process across jobs — every job hits Mongo once, loads
# the ~20-40 MB blob, decodes it into a local dict for that request's
# enrichment, then lets it get garbage-collected when the job finishes. This
# is deliberate: keeping the index in RAM is where the memory blow-up hides.
_BA_GRIDFS_BUCKET = "brand_analytics_cache"
# Per-user lock so a second job kicked off while the first is still parsing
# waits on the same fetch instead of triggering a duplicate 500 MB download.
_BA_LOCKS: dict[str, asyncio.Lock] = {}


def _ba_user_key() -> str:
    user = require_user()
    return str(user.get("_id") or user.get("email") or "anon")


def _ba_cache_filename(user_key: str, week_start: str, week_end: str) -> str:
    return f"{user_key}:{week_start}:{week_end}"


def _ba_gridfs() -> AsyncIOMotorGridFSBucket:
    return AsyncIOMotorGridFSBucket(_db(), bucket_name=_BA_GRIDFS_BUCKET)


def _f(row: dict, *names: str) -> float | None:
    """Pull the first numeric value present under any of the given field names."""
    for n in names:
        v = row.get(n)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _trim_row_for_cache(row: dict) -> dict:
    """Collapse a raw BA report row down to just SFR + total click + total conv.

    The report ships each row with the top-3 clicked ASINs' click/conversion
    shares broken out separately (either as flat suffixed fields or as a
    nested list). We sum them into a single click/conv total here so the
    stored blob doesn't drag ASIN-level data we never look at. Saves ~90% of
    the row footprint vs. storing the raw report.
    """
    sfr = _f(row, "searchFrequencyRank", "search_frequency_rank")
    click_share = 0.0
    conv_share = 0.0
    for i in (1, 2, 3):
        cs = _f(row, f"clickShare_{i}", f"clickShare{i}", f"click_share_{i}")
        cv = _f(row, f"conversionShare_{i}", f"conversionShare{i}", f"conversion_share_{i}")
        if cs:
            click_share += cs
        if cv:
            conv_share += cv
    for top in row.get("topClickedAsins") or row.get("top_clicked_asins") or []:
        if isinstance(top, dict):
            cs = _f(top, "clickShare", "click_share")
            cv = _f(top, "conversionShare", "conversion_share")
            if cs:
                click_share += cs
            if cv:
                conv_share += cv
    return {
        "sfr": sfr,
        "click_share": click_share or None,
        "conversion_share": conv_share or None,
    }


async def _ba_mongo_read(user_key: str, week_start: str, week_end: str) -> dict[str, dict] | None:
    """Load a cached trimmed BA index from Mongo GridFS. Returns None on miss."""
    filename = _ba_cache_filename(user_key, week_start, week_end)
    try:
        stream = await _ba_gridfs().open_download_stream_by_name(filename)
    except Exception:
        return None
    # Motor's GridOut.read() is async; close() is sync. Don't await close().
    raw = await stream.read()
    try:
        payload = gzip.decompress(raw)
        return json.loads(payload)
    except Exception as e:
        print(f"[keyword_matrix] BA mongo decode failed for {filename}: {e}")
        return None


async def _ba_mongo_write(user_key: str, week_start: str, week_end: str, index: dict[str, dict]) -> None:
    """Persist the trimmed BA index to Mongo GridFS, replacing any older copy."""
    filename = _ba_cache_filename(user_key, week_start, week_end)
    payload = gzip.compress(
        json.dumps(index, separators=(",", ":")).encode("utf-8")
    )
    # Wipe stale copies of the same (user, week) so GridFS doesn't accumulate
    # duplicates on every re-fetch.
    fs = _ba_gridfs()
    async for f in fs.find({"filename": filename}):
        try:
            await fs.delete(f._id)
        except Exception as e:
            print(f"[keyword_matrix] BA mongo delete stale copy failed: {e}")
    await fs.upload_from_stream(filename, payload)
    print(
        f"[keyword_matrix] BA cache WROTE mongo {filename} "
        f"({len(index)} terms, {len(payload) / 1e6:.1f} MB gzipped)"
    )


async def _fetch_brand_analytics_index() -> dict[str, dict]:
    """Return the trimmed BA index for the most recent successful week.

    Amazon publishes weekly reports for full ISO weeks. The most recent one
    frequently FATALs because Amazon hasn't finished processing it yet — we
    walk back a week at a time and retry, up to `_BA_MAX_WEEKS_BACK` weeks,
    before giving up.

    Cache flow: check Mongo (per-user, per-week GridFS blob) newest → oldest,
    return the first hit. Miss → run the fetch loop, trim rows, store the
    trimmed dict back to Mongo, return it. Downstream code treats an empty
    index as "no keyword matches" rather than an error.
    """
    user_key = _ba_user_key()
    today = date.today()
    # Anchor on the last completed Sunday–Saturday window.
    days_since_sunday = (today.weekday() + 1) % 7
    last_sat = today - timedelta(days=days_since_sunday + 1)

    # Mongo cache probe (newest → oldest). First hit wins.
    for attempt in range(_BA_MAX_WEEKS_BACK):
        end = last_sat - timedelta(days=7 * attempt)
        start = end - timedelta(days=6)
        cached = await _ba_mongo_read(user_key, start.isoformat(), end.isoformat())
        if cached is not None:
            print(
                f"[keyword_matrix] BA cache HIT mongo user={user_key} "
                f"week {start}..{end} ({len(cached)} terms)"
            )
            return cached

    lock = _BA_LOCKS.setdefault(user_key, asyncio.Lock())
    async with lock:
        # Re-check under the lock — another concurrent job may have populated
        # Mongo while we were queued behind it.
        for attempt in range(_BA_MAX_WEEKS_BACK):
            end = last_sat - timedelta(days=7 * attempt)
            start = end - timedelta(days=6)
            cached = await _ba_mongo_read(user_key, start.isoformat(), end.isoformat())
            if cached is not None:
                print(
                    f"[keyword_matrix] BA cache HIT mongo (post-lock) "
                    f"user={user_key} week {start}..{end}"
                )
                return cached

        for attempt in range(_BA_MAX_WEEKS_BACK):
            end = last_sat - timedelta(days=7 * attempt)
            start = end - timedelta(days=6)
            try:
                rows = await amazon_sp.fetch_brand_analytics_search_terms(
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    period="WEEK",
                )
            except Exception as e:
                msg = str(e)
                # Retry an older week for: FATAL / not-yet-available / poll
                # timeouts. Older reports are usually pre-generated by Amazon
                # and return instantly, so walking back often succeeds even
                # when the freshest week is still cooking. Anything else
                # (auth, quota, network) will fail the same way on any date,
                # so bail.
                retryable = (
                    "FATAL" in msg
                    or "not available" in msg.lower()
                    or "did not finish" in msg.lower()
                )
                if not retryable:
                    print(f"[keyword_matrix] brand analytics fetch failed hard: {e}")
                    return {}
                print(
                    f"[keyword_matrix] BA week {start}..{end} unavailable "
                    f"({msg[:80]}) — attempt {attempt + 1}/{_BA_MAX_WEEKS_BACK}, "
                    f"retrying older week"
                )
                continue
            index: dict[str, dict] = {}
            for row in rows:
                term = (row.get("searchTerm") or row.get("search_term") or "").strip().lower()
                if term:
                    index[term] = _trim_row_for_cache(row)
            # Drop the raw rows list so the parsed BA report can be freed
            # while we're serializing to Mongo — otherwise we double the
            # memory footprint during the write.
            del rows
            try:
                await _ba_mongo_write(user_key, start.isoformat(), end.isoformat(), index)
            except Exception as e:
                print(f"[keyword_matrix] BA mongo write failed (continuing): {e}")
            print(
                f"[keyword_matrix] BA week {start}..{end} OK — indexed "
                f"{len(index)} terms"
            )
            return index

        print(
            f"[keyword_matrix] brand analytics FATAL for last "
            f"{_BA_MAX_WEEKS_BACK} weeks — giving up"
        )
        return {}


def _enrich_with_brand_analytics(
    keywords: list[str], index: dict[str, dict]
) -> tuple[list[dict], dict]:
    """Attach SFR + click share + conversion share to each keyword.

    Reads directly from the trimmed cache shape written by
    `_trim_row_for_cache` — {sfr, click_share, conversion_share} per term.
    Keywords not in the report are still returned but with all metrics None
    — the scorer drops them from ranking, and the notes surface coverage so
    the user sees the drop.
    """
    enriched: list[dict] = []
    hits = 0
    for kw in keywords:
        row = index.get(kw)
        if not row:
            enriched.append(
                {
                    "keyword": kw,
                    "sfr": None,
                    "click_share": None,
                    "conversion_share": None,
                    "cpc": None,
                    "in_report": False,
                }
            )
            continue
        hits += 1
        enriched.append(
            {
                "keyword": kw,
                "sfr": row.get("sfr"),
                "click_share": row.get("click_share"),
                "conversion_share": row.get("conversion_share"),
                "cpc": None,
                "in_report": True,
            }
        )
    coverage = {
        "total": len(keywords),
        "in_report": hits,
        "missing": len(keywords) - hits,
    }
    return enriched, coverage


# ── CPC enrichment ──────────────────────────────────────────────────────────


async def _enrich_with_cpc(
    enriched: list[dict],
    ad_group_id: str,
    campaign_id: str,
) -> None:
    """Mutate `enriched` in place, attaching CPC to any keyword the Ads API
    returns a suggested bid for. Silent on unrecognized keywords."""
    kws = [e["keyword"] for e in enriched if e.get("in_report")]
    if not kws:
        return
    try:
        data = await fetch_keyword_bid_recommendations(
            kws, ad_group_id=ad_group_id, campaign_id=campaign_id
        )
    except Exception as e:
        print(f"[keyword_matrix] bid recommendations failed: {e}")
        return
    by_kw: dict[str, float] = {}
    for rec in data.get("recommendations", []):
        kw = (rec.get("keyword") or "").strip().lower()
        bid = rec.get("suggestedBid") or {}
        suggested = bid.get("suggested") if isinstance(bid, dict) else None
        if kw and suggested is not None:
            by_kw[kw] = float(suggested)
    for e in enriched:
        cpc = by_kw.get(e["keyword"])
        if cpc is not None:
            e["cpc"] = cpc


# ── Scoring ─────────────────────────────────────────────────────────────────


def _norm(values: list[float | None]) -> list[float | None]:
    """Min-max normalize to 0..1. None passes through as None."""
    present = [v for v in values if v is not None]
    if not present:
        return values
    lo, hi = min(present), max(present)
    if hi == lo:
        return [1.0 if v is not None else None for v in values]
    return [None if v is None else (v - lo) / (hi - lo) for v in values]


def _score(enriched: list[dict]) -> list[dict]:
    """Attach a composite score to every scorable keyword and sort desc.

    Composite = mean of the three normalized signals (inverted SFR, click
    share, conversion share). We normalize *within the source pool* — the
    tiers are relative rankings within a source, not absolute quality
    scores. That matches how the matrix is used: "top-5 keywords from Meta"
    is a comparison inside Meta, not vs. Amazon.

    Keywords missing all three signals are excluded from the ranking but
    kept in the returned list so the caller can display the drop count.
    """
    # Invert SFR (lower rank = more search volume = better) by negating.
    sfr_raw = [(-e["sfr"] if e.get("sfr") is not None else None) for e in enriched]
    click_raw = [e.get("click_share") for e in enriched]
    conv_raw = [e.get("conversion_share") for e in enriched]

    sfr_n = _norm(sfr_raw)
    click_n = _norm(click_raw)
    conv_n = _norm(conv_raw)

    for i, e in enumerate(enriched):
        parts = [v for v in (sfr_n[i], click_n[i], conv_n[i]) if v is not None]
        e["composite"] = sum(parts) / len(parts) if parts else None

    # Sort scorable keywords desc by composite; unscored ones drop to the end.
    return sorted(
        enriched,
        key=lambda e: (e["composite"] is None, -(e["composite"] or 0.0)),
    )


# ── Matrix layout ───────────────────────────────────────────────────────────


def _tier_split(scored: list[dict], per_tier: int = 5) -> dict[str, list[dict]]:
    """Take the top 15 scorable keywords and split into three tiers of 5.

    Anything below 15 pads short — we return empty slots rather than
    pulling from the next source, so each column shows what its source
    actually produced.
    """
    ranked = [e for e in scored if e.get("composite") is not None]
    top15 = ranked[: per_tier * 3]
    return {
        "top": top15[:per_tier],
        "medium": top15[per_tier : per_tier * 2],
        "low": top15[per_tier * 2 : per_tier * 3],
    }


def _build_matrix(sources: dict[str, dict]) -> dict:
    """Assemble the final 3x3 matrix payload the frontend will render."""
    matrix: dict[str, dict[str, list[dict]]] = {
        "top": {},
        "medium": {},
        "low": {},
    }
    per_source_summary: dict[str, dict] = {}
    for source in SOURCES:
        src = sources[source]
        scored = src.get("scored") or []
        tiers = _tier_split(scored)
        for tier in ("top", "medium", "low"):
            matrix[tier][source] = tiers[tier]
        per_source_summary[source] = {
            "label": _SOURCE_LABELS[source],
            "raw_count": len(src.get("keywords") or []),
            "scorable_count": len(
                [e for e in scored if e.get("composite") is not None]
            ),
            "ba_coverage": src.get("ba_coverage"),
            "notes": src.get("notes") or [],
        }
    return {
        "rows": ["top", "medium", "low"],
        "cols": list(SOURCES),
        "col_labels": _SOURCE_LABELS,
        "cells": matrix,
        "sources": per_source_summary,
    }
