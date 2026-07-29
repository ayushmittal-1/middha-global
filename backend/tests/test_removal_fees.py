"""Removal Order Detail report parsing / window filter tests."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from amazon_sp import parse_removal_order_detail_report


def _load_sample(name: str) -> str:
    path = Path(__file__).resolve().parents[3] / "Downloads" / name
    if not path.exists():
        path = Path(rf"C:\Users\Desktop\Downloads\{name}")
    return path.read_text(encoding="utf-8-sig")


def _month_window(year: int, month: int, tz_name: str = "America/Los_Angeles"):
    tz = ZoneInfo(tz_name)
    start = datetime(year, month, 1, tzinfo=tz)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=tz)
    else:
        end = datetime(year, month + 1, 1, tzinfo=tz)
    return start, end


def test_allmart_csv_total_matches_seller_central():
    text = _load_sample("removal all mart248691020662.csv")
    per_sku, total = parse_removal_order_detail_report(text)
    assert abs(total - 27.36) < 0.05
    assert abs(sum(b["removal_fee"] for b in per_sku.values()) - 27.36) < 0.05
    assert per_sku["AM SKIP BO CARD GAME"]["removal_fee"] == 6.12


def test_request_date_window_july_only():
    text = _load_sample("removal all mart248691020662.csv")
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 1, tzinfo=timezone.utc)
    per_sku, total = parse_removal_order_detail_report(text, start, end)
    assert abs(total - 20.25) < 0.05
    assert abs(sum(b["removal_fee"] for b in per_sku.values()) - 20.25) < 0.05


def test_request_date_window_june_only():
    text = _load_sample("removal all mart248691020662.csv")
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 1, tzinfo=timezone.utc)
    per_sku, total = parse_removal_order_detail_report(text, start, end)
    assert abs(total - 7.11) < 0.05
    assert abs(sum(b["removal_fee"] for b in per_sku.values()) - 7.11) < 0.05


def test_window_excludes_outside_request_dates():
    text = _load_sample("removal all mart248691020662.csv")
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = datetime(2026, 6, 1, tzinfo=timezone.utc)
    per_sku, total = parse_removal_order_detail_report(text, start, end)
    assert total == 0.0
    assert per_sku == {}


def test_tsv_removal_report_parses():
    lines = [
        "request-date\torder-id\tsku\tremoval-fee\trequested-quantity",
        "2026-07-10T12:00:00-07:00\tabc123\tTEST-SKU\t1.50\t2",
        "2026-06-01T12:00:00-07:00\tdef456\tTEST-SKU\t0.75\t1",
    ]
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 1, tzinfo=timezone.utc)
    per_sku, total = parse_removal_order_detail_report(
        "\n".join(lines), start, end,
    )
    assert abs(total - 1.50) < 0.001
    assert per_sku["TEST-SKU"]["removal_fee"] == 1.50
    assert "abc123" in per_sku["TEST-SKU"]["order_ids"]


def test_empty_removal_fee_rows_ignored():
    lines = [
        "request-date,order-id,sku,removal-fee",
        "2026-07-10T12:00:00-07:00,a,SKU-A,0.84",
        "2026-07-10T12:00:00-07:00,a,SKU-B,",
    ]
    per_sku, total = parse_removal_order_detail_report("\n".join(lines))
    assert abs(total - 0.84) < 0.001
    assert "SKU-A" in per_sku
    assert "SKU-B" not in per_sku


def test_rawyal_june_csv_matches_event_date_total():
    """SC Removal Order Detail Event Date June 2026 (Rawyal/AE sample)."""
    text = _load_sample("650046020663.csv")
    start, end = _month_window(2026, 6)
    per_sku, total = parse_removal_order_detail_report(text, start, end)
    assert abs(total - 248.12) < 0.02
    assert abs(sum(b["removal_fee"] for b in per_sku.values()) - 248.12) < 0.02
    assert abs(per_sku["AE-LOCTITE-243"]["removal_fee"] - 204.96) < 0.02
    # Disposal + Return both counted
    assert len(per_sku) >= 10


def test_rawyal_july_csv_includes_pending_fees():
    """July request-date fees include Pending rows that already have removal-fee."""
    text = _load_sample("650047020663.csv")
    start, end = _month_window(2026, 7)
    per_sku, total = parse_removal_order_detail_report(text, start, end)
    assert abs(total - 27.56) < 0.02
    assert abs(sum(b["removal_fee"] for b in per_sku.values()) - 27.56) < 0.02
    assert "AE-LOCTITE-243" in per_sku


def test_june_window_excludes_july_request_dates():
    text = _load_sample("650047020663.csv")
    start, end = _month_window(2026, 6)
    _per_sku, total = parse_removal_order_detail_report(text, start, end)
    assert total == 0.0
