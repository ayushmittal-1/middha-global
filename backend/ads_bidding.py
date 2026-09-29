"""Bid strategy and keyword-performance judgement for manual campaigns.

Two decisions live here, both kept pure so they can be reasoned about and
tested without touching Amazon:

  * `bidding_strategy_for_budget` — whether a campaign bids a fixed amount or
    lets Amazon flex it, chosen from the daily budget.
  * `classify_keyword_performance` — which of the keywords a seller already
    runs are burning money, so the picker can warn before they bid on the
    same term again.

Neither talks to the network. `ads_keyword_performance` fetches the rows;
this module decides what they mean.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Bid strategy ────────────────────────────────────────────────────────────

# Amazon's three Sponsored Products bidding strategies, under the names the
# API uses rather than the ones the console shows:
#   MANUAL            -> console "Fixed bids"
#   LEGACY_FOR_SALES  -> console "Dynamic bids - down only"
#   AUTO_FOR_SALES    -> console "Dynamic bids - up and down"
STRATEGY_FIXED = "MANUAL"
STRATEGY_DYNAMIC_DOWN = "LEGACY_FOR_SALES"
STRATEGY_DYNAMIC_UP_DOWN = "AUTO_FOR_SALES"

# Daily-budget band that gets fixed bids, inclusive at both ends.
#
# The reasoning is about predictability at small scale. Inside this band a
# day's budget buys only a handful of clicks, so letting Amazon raise or lower
# each bid makes the day's spend hard to attribute — you cannot tell a bad
# keyword from an unlucky auction. A fixed bid makes every click cost what you
# said it would, which is what makes the first week of data readable.
#
# Below the band the budget is so small that down-only bidding stretches it
# further; above it there is enough volume for Amazon's optimisation to have
# something to learn from.
FIXED_BID_BUDGET_MIN = 10.0
FIXED_BID_BUDGET_MAX = 25.0


def bidding_strategy_for_budget(budget: float | int | None) -> str:
    """Return the Amazon bidding strategy for a daily budget.

    $10-25 inclusive gets fixed bids; anything outside gets dynamic down-only.
    A missing or unparseable budget falls back to dynamic down-only, which is
    the safer default — Amazon can only ever bid *less* than the stated amount
    under that strategy, never more.
    """
    try:
        value = float(budget)
    except (TypeError, ValueError):
        return STRATEGY_DYNAMIC_DOWN
    if FIXED_BID_BUDGET_MIN <= value <= FIXED_BID_BUDGET_MAX:
        return STRATEGY_FIXED
    return STRATEGY_DYNAMIC_DOWN


def describe_bidding_strategy(strategy: str) -> str:
    """Console-facing name, for telling the user what they are getting."""
    return {
        STRATEGY_FIXED: "Fixed bids",
        STRATEGY_DYNAMIC_DOWN: "Dynamic bids — down only",
        STRATEGY_DYNAMIC_UP_DOWN: "Dynamic bids — up and down",
    }.get(strategy, strategy)


# ── Keyword performance ─────────────────────────────────────────────────────

# A keyword is only judged once it has spent enough for zero sales to mean
# something. Below this it may simply not have had its chance yet, and warning
# on it would train the user to ignore the warnings.
#
# Expressed as a multiple of the keyword's own cost-per-click rather than a
# flat dollar figure, because "enough to judge" scales with the category: two
# clicks on a $0.20 keyword proves much less than two on a $3 one. The floor
# below catches keywords whose CPC is tiny or unknown.
MIN_CLICKS_TO_JUDGE = 3
MIN_SPEND_TO_JUDGE = 1.0

VERDICT_WASTING = "wasting"
VERDICT_OK = "ok"
VERDICT_UNPROVEN = "unproven"


@dataclass
class KeywordStats:
    """One keyword's performance over the report window."""

    keyword: str
    match_type: str = ""
    campaign_name: str = ""
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    orders: int = 0
    sales: float = 0.0

    @property
    def cpc(self) -> float | None:
        return round(self.spend / self.clicks, 2) if self.clicks else None

    @property
    def acos(self) -> float | None:
        """Spend as a share of the sales it produced. None when no sales —
        ACOS is undefined there, not infinite, and rendering it as a number
        would imply a measurement nobody made."""
        return round(self.spend / self.sales, 4) if self.sales else None


@dataclass
class KeywordVerdict:
    stats: KeywordStats
    verdict: str
    reason: str


def classify_keyword_performance(stats: KeywordStats) -> KeywordVerdict:
    """Judge one keyword the seller is already running.

    Only one thing counts as wasting: real spend that produced no orders at
    all. That is the signal a seller can act on without knowing their target
    margin — money left with Amazon in exchange for nothing.

    Deliberately NOT flagged: a keyword with a high ACOS. High relative to
    what? Acceptable ACOS depends on margin, which this system does not know,
    and a keyword at 45% ACOS may be the best one in a category that converts
    at 60%. Calling that "underperforming" would be a guess dressed as advice.
    """
    if stats.clicks < MIN_CLICKS_TO_JUDGE or stats.spend < MIN_SPEND_TO_JUDGE:
        return KeywordVerdict(
            stats,
            VERDICT_UNPROVEN,
            f"only {stats.clicks} click(s) and ${stats.spend:.2f} spent — "
            "too early to judge",
        )
    if stats.orders == 0:
        return KeywordVerdict(
            stats,
            VERDICT_WASTING,
            f"${stats.spend:.2f} spent over {stats.clicks} clicks with no orders",
        )
    return KeywordVerdict(
        stats,
        VERDICT_OK,
        f"{stats.orders} order(s) from ${stats.spend:.2f} spend",
    )


def summarize_verdicts(verdicts: list[KeywordVerdict]) -> dict:
    """Roll a set of verdicts into something the chat can say in one line."""
    wasting = [v for v in verdicts if v.verdict == VERDICT_WASTING]
    wasted_spend = round(sum(v.stats.spend for v in wasting), 2)
    return {
        "evaluated": len(verdicts),
        "wasting": len(wasting),
        "ok": len([v for v in verdicts if v.verdict == VERDICT_OK]),
        "unproven": len([v for v in verdicts if v.verdict == VERDICT_UNPROVEN]),
        "wasted_spend": wasted_spend,
    }
