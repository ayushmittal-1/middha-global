"""Restock 'Reserved' column bifurcation.

Amazon's reservedQuantity is the sum of three sub-buckets — pending
customer orders, FC processing, and pending transshipment. Aurora Node
flattens these into separate fields on `products.inventory`. The
Restock tab now shows the customer-order vs FC-processing split so a
seller can answer "why is my inventory locked up?" instead of just
seeing a single total.

These tests lock in the FE bifurcation rendering (via HTML string
inspection so a refactor that drops the wire fails CI)."""

import os
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest


@pytest.fixture(scope="module")
def frontend_html() -> str:
    return (
        Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")


def test_reserved_header_tooltip_documents_the_three_sub_buckets(frontend_html):
    """The <th> tooltip must explain the bifurcation so a user hovering
    the header understands what 'cust' and 'fc' in the cell mean."""
    # Find the Reserved column header block.
    idx = frontend_html.find(">Reserved</th>")
    assert idx > 0, "Reserved column header missing"
    # Look backward from the header to find its <th ... title="...">.
    header_start = frontend_html.rfind("<th", 0, idx)
    header_html = frontend_html[header_start:idx + len(">Reserved</th>")]
    assert "pendingCustomerOrder" in header_html
    assert "fcProcessing" in header_html
    assert "pendingTransshipment" in header_html or "transit between FCs" in header_html


def test_reserved_cell_reads_both_sub_fields(frontend_html):
    """The rendered row must consume the two new backend fields."""
    assert "row.reserved_customer_order" in frontend_html
    assert "row.reserved_fc_processing" in frontend_html


def test_reserved_cell_preserves_total_as_primary_number(frontend_html):
    """The total must remain the leading number in the cell so scanning
    + sort behavior don't shift for users who don't care about the
    breakdown."""
    # A comment on the rendering explaining the total-first design.
    assert "primary number" in frontend_html.lower() or "sort order" in frontend_html.lower()


def test_reserved_cell_falls_back_to_plain_total_when_subfields_missing(frontend_html):
    """Older cached responses may lack the bifurcation fields — the
    cell must render cleanly showing just the total instead of blank
    or '0+0'."""
    # Guard on: when cust == 0 && fc == 0, return plain total.
    assert "return fmtRestockNum(total)" in frontend_html


def test_reserved_cell_has_dedicated_css_class(frontend_html):
    """Small CSS class controls the sub-text styling (muted + smaller)
    so the bifurcation doesn't visually shout over the total."""
    assert ".reserved-brk" in frontend_html
    assert "reserved-brk" in frontend_html


# ── Backend field mapping ────────────────────────────────────────────────


def test_latest_inventory_maps_customer_order_and_fc_processing_fields():
    """Guard the field-name mapping between Aurora's Mongo fields and
    the shape latest_inventory_for_user emits. If Aurora Node ever
    renames the source fields this test won't catch it (that's an
    integration concern) — but a Python-side rename WILL break here."""
    import database
    import inspect
    src = inspect.getsource(database.latest_inventory_for_user)
    # Source Aurora fields the sync writes:
    assert "reservedPendingCustomerOrder" in src
    assert "reservedFcProcessing" in src
    # Names we expose to callers:
    assert "reserved_customer_order" in src
    assert "reserved_fc_processing" in src


def test_restock_endpoint_row_includes_both_bifurcation_fields():
    import inspect
    import main
    src = inspect.getsource(main.forecasting_restock)
    assert '"reserved_customer_order"' in src
    assert '"reserved_fc_processing"' in src
