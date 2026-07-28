"""Aged Inventory Surcharge report parsing / allocation tests."""

from pathlib import Path

from amazon_sp import parse_aged_surcharge_charges_report


def _load_sample(name: str) -> str:
    path = Path(__file__).resolve().parents[3] / "Downloads" / name
    if not path.exists():
        path = Path(rf"C:\Users\Desktop\Downloads\{name}")
    return path.read_text(encoding="utf-8-sig")


def test_may_csv_parses_amount_charged_total():
    text = _load_sample("401659020661 (1).csv")
    per_sku, months = parse_aged_surcharge_charges_report(text)
    assert months == ["2026-05"]
    assert len(per_sku) == 29
    total = sum(b["charged_total"] for b in per_sku.values())
    # Seller Central May Aged Inventory Surcharge CSV total.
    assert abs(total - 262.12) < 0.05


def test_june_csv_parses_amount_charged_total():
    text = _load_sample("401660020661 (1).csv")
    per_sku, months = parse_aged_surcharge_charges_report(text)
    assert months == ["2026-06"]
    assert len(per_sku) == 32
    total = sum(b["charged_total"] for b in per_sku.values())
    assert abs(total - 263.63) < 0.05


def test_event_month_filter_excludes_other_months():
    may = _load_sample("401659020661 (1).csv")
    june = _load_sample("401660020661 (1).csv")
    combined = may.rstrip() + "\n" + "\n".join(june.splitlines()[1:])
    only_may, months = parse_aged_surcharge_charges_report(
        combined, months_filter=["2026-05"],
    )
    assert months == ["2026-05", "2026-06"] or "2026-05" in months
    total = sum(b["charged_total"] for b in only_may.values())
    assert abs(total - 262.12) < 0.05
    # June-only SKU rows must not inflate May.
    only_june, _ = parse_aged_surcharge_charges_report(
        combined, months_filter=["2026-06"],
    )
    june_total = sum(b["charged_total"] for b in only_june.values())
    assert abs(june_total - 263.63) < 0.05


def test_tsv_aged_report_still_parses():
    """SP-API often emits TSV — must keep working alongside SC CSV."""
    lines = [
        "snapshot-date\tsku\tfnsku\tasin\tamount-charged\tqty-charged",
        "2026-05-15T08:00:00+00:00\tTEST-SKU\tX1\tB000000000\t1.25\t2",
    ]
    per_sku, months = parse_aged_surcharge_charges_report("\n".join(lines))
    assert months == ["2026-05"]
    assert per_sku["TEST-SKU"]["charged_total"] == 1.25
    assert per_sku["TEST-SKU"]["qty_charged"] == 2


def test_wrong_month_report_does_not_count_as_match():
    """Reused June file must not satisfy a March Event Month request."""
    june = _load_sample("401660020661 (1).csv")
    parsed, seen = parse_aged_surcharge_charges_report(june, months_filter=["2026-03"])
    assert parsed == {}
    assert "2026-03" not in seen
    assert "2026-06" in seen


def test_march_window_months_are_marketplace_not_utc_spill():
    """March 31 PDT end-of-day is April 1 UTC — must still be only 2026-03."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from amazon_sp import calendar_months_in_window

    tz = "America/Los_Angeles"
    z = ZoneInfo(tz)
    start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=z).astimezone(timezone.utc)
    end = datetime(2026, 3, 31, 23, 59, 59, 999999, tzinfo=z).astimezone(timezone.utc)
    assert end.month == 4  # UTC spill
    assert calendar_months_in_window(start, end, tz) == ["2026-03"]
