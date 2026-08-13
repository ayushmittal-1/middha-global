"""Docstring must match the implemented storage-allocation formula.

Review issue #6: the docstring for compute_profitability_data claimed
storage was allocated as `monthly storage ÷ avg units on hand × units
sold`, but the code actually uses a sales-share proxy
(`asin_fee × sku_units / asin_total_units`) because no on-hand data is
loaded. Anyone reading the docstring was misled about how the Storage
Fee line was derived. This test locks the docstring to the sales-share
wording."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from agent import compute_profitability_data


def test_storage_docstring_describes_sales_share_not_on_hand():
    doc = compute_profitability_data.__doc__ or ""
    # The old (misleading) phrase must not appear.
    assert "avg units on hand" not in doc, (
        "docstring still claims an on-hand-based storage allocation that "
        "the code doesn't implement — see review issue #6"
    )
    # The new phrase should describe the actual formula.
    assert "sales-share" in doc.lower() or "units sold for" in doc.lower(), (
        "docstring should describe the sales-share proxy formula that "
        "the code actually implements"
    )
