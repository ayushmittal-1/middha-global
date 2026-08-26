"""Frontend wiring locks for the profitability + campaigns tabs.

The verification review flagged that several backend fixes had never
been wired into `frontend/index.html`, so end users still saw the
original (mislabelled / mixed-currency / silently-incomplete) numbers
even though the API contract was correct. These tests parse the actual
frontend file and grep for the required strings, so a regression that
un-wires them shows up as a failing test rather than as a shipped
regression that has to be caught in a manual QA pass."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def frontend_html() -> str:
    path = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    assert path.exists(), f"frontend/index.html not found at {path}"
    return path.read_text(encoding="utf-8")


# ── Issue #1: marketplaceId is passed on /profitability calls ─────────────


def test_profitability_toolbar_has_marketplace_select(frontend_html):
    """The Profitability tab must expose a marketplace selector so
    users can scope totals to a single currency."""
    assert 'id="profit-marketplace"' in frontend_html, (
        "profit-marketplace select missing — user has no way to filter "
        "to a single marketplace (review issue #1 wiring)"
    )


def test_profitability_fetch_passes_marketplace_id(frontend_html):
    """When the user picks a marketplace, the /profitability request
    MUST include `marketplaceId=…` on the query string. Regression
    guard: the pre-fix code fetched only `?start=…&end=…`."""
    assert "params.set('marketplaceId'" in frontend_html or \
           'params.set("marketplaceId"' in frontend_html, (
        "marketplaceId not appended to /profitability fetch URL"
    )


def test_profitability_defaults_to_primary_marketplace(frontend_html):
    """First paint should be single-currency by default (seller's
    primary marketplace) instead of blended-across-markets."""
    assert "profitMarketplaceEl.value = primaryId" in frontend_html


def test_mixed_currency_banner_is_rendered(frontend_html):
    """When conversion cannot finish, mixed_currency still surfaces a banner."""
    assert "data.mixed_currency" in frontend_html
    assert "Mixed currencies" in frontend_html


def test_converted_to_usd_banner_is_rendered(frontend_html):
    """CAD/MXN (etc.) sales converted to USD must be labelled, not silent."""
    assert "data.converted_to_usd" in frontend_html
    assert "Converted to USD" in frontend_html


# ── Issue #4: partial-status contract is read + surfaced ──────────────────


def test_auto_reload_reads_structured_complete_flag(frontend_html):
    """scheduleProfitAutoReload must prefer the structured
    `data.complete === false` contract over parsing free-text
    warnings (the free-text check stays as a fallback)."""
    assert "data.complete === false" in frontend_html


def test_partial_data_banner_uses_partial_sections(frontend_html):
    """The FE must render the list of `partial_sections` from the
    structured contract, not just echo the warnings string."""
    assert "data.partial_sections" in frontend_html
    assert "Partial data" in frontend_html


# ── Issue #5: ROI column relabelled to ROAS and reads roas_pct ────────────


def test_campaigns_column_header_is_roas_not_roi(frontend_html):
    """The <th> label was 'ROI %' but the underlying value was really
    ROAS-100. Header must now say ROAS to match what's shown."""
    assert ">ROAS %<" in frontend_html
    # Regression guard — no lingering "ROI %" header
    assert ">ROI %<" not in frontend_html


def test_campaigns_row_reads_roas_pct_field(frontend_html):
    """The row-render code must read `c.roas_pct` (the correctly-named
    field the backend now emits), with a fallback for older responses."""
    assert "c.roas_pct" in frontend_html


def test_campaigns_row_still_has_roi_pct_fallback(frontend_html):
    """Backward-compat: if the backend hasn't been redeployed yet and
    still returns only roi_pct, the FE should synthesise a ROAS value
    (= roi_pct + 100) rather than showing '—'."""
    assert "c.roi_pct + 100" in frontend_html


# ── Anti-regression: banned old patterns ──────────────────────────────────


def test_no_orphan_roi_pct_in_render_output(frontend_html):
    """The rendered cell used to interpolate `${roi}%` — that variable
    is gone, replaced by `${roas}%`. Guard against a partial revert
    that would ship the wrong-labelled value again."""
    # `${roi}` inside a template literal, ending with %
    assert "${roi}%" not in frontend_html, (
        "residual ${roi}% template — the ROI→ROAS rename is incomplete"
    )
