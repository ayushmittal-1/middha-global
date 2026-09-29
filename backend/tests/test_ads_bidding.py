"""Tests for bid strategy selection and keyword-performance judgement.

Both decisions spend real money if they are wrong: the strategy decides
whether Amazon may move a bid the seller set, and the verdict decides which
keywords a seller is told to stop paying for.
"""

import ads_bidding as ab
import pytest


# ── Bid strategy from budget ────────────────────────────────────────────────


@pytest.mark.parametrize("budget", [10, 10.0, 15, 25, 25.0, 17.5])
def test_budgets_inside_the_band_get_fixed_bids(budget):
    """$10-25 inclusive. At this scale a day buys a handful of clicks, and a
    fixed bid is what makes the first week's data readable."""
    assert ab.bidding_strategy_for_budget(budget) == ab.STRATEGY_FIXED


@pytest.mark.parametrize("budget", [1, 9.99, 25.01, 50, 500])
def test_budgets_outside_the_band_get_dynamic_bids(budget):
    assert ab.bidding_strategy_for_budget(budget) == ab.STRATEGY_DYNAMIC_DOWN


def test_the_band_is_inclusive_at_both_edges():
    assert ab.bidding_strategy_for_budget(10) == ab.STRATEGY_FIXED
    assert ab.bidding_strategy_for_budget(25) == ab.STRATEGY_FIXED
    assert ab.bidding_strategy_for_budget(9.99) != ab.STRATEGY_FIXED
    assert ab.bidding_strategy_for_budget(25.01) != ab.STRATEGY_FIXED


@pytest.mark.parametrize("budget", [None, "", "abc", float("nan")])
def test_an_unusable_budget_falls_back_to_the_safer_strategy(budget):
    """Down-only means Amazon may bid less than asked but never more — the
    right default when we cannot tell what the seller intended."""
    if budget != budget:  # NaN compares unequal to itself
        assert ab.bidding_strategy_for_budget(budget) == ab.STRATEGY_DYNAMIC_DOWN
        return
    assert ab.bidding_strategy_for_budget(budget) == ab.STRATEGY_DYNAMIC_DOWN


def test_dynamic_default_never_lets_amazon_raise_a_bid():
    """AUTO_FOR_SALES can bid above what the seller set. Nothing in this
    rule should select it without an explicit decision to."""
    for budget in (1, 5, 9, 26, 100, 10_000):
        assert ab.bidding_strategy_for_budget(budget) != ab.STRATEGY_DYNAMIC_UP_DOWN


def test_strategies_are_described_in_the_words_the_console_uses():
    assert ab.describe_bidding_strategy(ab.STRATEGY_FIXED) == "Fixed bids"
    assert "down only" in ab.describe_bidding_strategy(ab.STRATEGY_DYNAMIC_DOWN)


# ── Keyword verdicts ────────────────────────────────────────────────────────


def _stats(**kw):
    base = dict(keyword="incense", clicks=10, spend=5.0, orders=0, sales=0.0)
    base.update(kw)
    return ab.KeywordStats(**base)


def test_spend_with_no_orders_is_flagged_as_wasting():
    v = ab.classify_keyword_performance(_stats(clicks=12, spend=7.40, orders=0))
    assert v.verdict == ab.VERDICT_WASTING
    assert "7.40" in v.reason and "no orders" in v.reason


def test_a_keyword_that_sells_is_not_flagged():
    v = ab.classify_keyword_performance(_stats(clicks=12, spend=7.40, orders=3, sales=60.0))
    assert v.verdict == ab.VERDICT_OK


def test_a_barely_used_keyword_is_unproven_not_wasting():
    """Warning on a keyword that has had two clicks trains the seller to
    ignore the warnings."""
    v = ab.classify_keyword_performance(_stats(clicks=2, spend=0.80, orders=0))
    assert v.verdict == ab.VERDICT_UNPROVEN


def test_many_clicks_but_trivial_spend_is_still_unproven():
    v = ab.classify_keyword_performance(_stats(clicks=20, spend=0.40, orders=0))
    assert v.verdict == ab.VERDICT_UNPROVEN


def test_expensive_but_converting_keywords_are_not_called_underperforming():
    """A 75% ACOS may be terrible or may be the category norm — the system
    does not know the seller's margin, so it must not pass judgement."""
    v = ab.classify_keyword_performance(
        _stats(clicks=40, spend=75.0, orders=2, sales=100.0)
    )
    assert v.verdict == ab.VERDICT_OK


def test_acos_is_none_rather_than_infinite_when_nothing_sold():
    assert _stats(spend=9.0, sales=0.0).acos is None
    assert _stats(spend=25.0, sales=100.0).acos == 0.25


def test_cpc_is_none_rather_than_a_divide_by_zero_when_unclicked():
    assert _stats(clicks=0, spend=0.0).cpc is None
    assert _stats(clicks=4, spend=2.0).cpc == 0.5


# ── Summary ─────────────────────────────────────────────────────────────────


def test_summary_counts_each_verdict_and_totals_the_waste():
    verdicts = [
        ab.classify_keyword_performance(_stats(keyword="a", clicks=10, spend=6.0, orders=0)),
        ab.classify_keyword_performance(_stats(keyword="b", clicks=10, spend=4.0, orders=0)),
        ab.classify_keyword_performance(_stats(keyword="c", clicks=10, spend=5.0, orders=2, sales=40.0)),
        ab.classify_keyword_performance(_stats(keyword="d", clicks=1, spend=0.2, orders=0)),
    ]
    s = ab.summarize_verdicts(verdicts)
    assert s == {
        "evaluated": 4, "wasting": 2, "ok": 1, "unproven": 1, "wasted_spend": 10.0,
    }


def test_summary_of_nothing_is_all_zeros_not_an_error():
    assert ab.summarize_verdicts([])["evaluated"] == 0
    assert ab.summarize_verdicts([])["wasted_spend"] == 0
