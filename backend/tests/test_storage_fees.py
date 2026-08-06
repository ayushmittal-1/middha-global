"""Storage fee parsing and window allocation tests."""

from pathlib import Path

from amazon_sp import (
    parse_storage_fee_report,
    storage_per_unit_for_window,
)


def _load_sample(name: str) -> str:
    path = Path(__file__).resolve().parents[3] / "Downloads" / name
    if not path.exists():
        path = Path(rf"C:\Users\Desktop\Downloads\{name}")
    return path.read_text(encoding="utf-8-sig")


def test_april_storage_matches_seller_central_total():
    text = _load_sample("april storage.csv")
    per_asin, months = parse_storage_fee_report(text)
    assert "2026-04" in months
    total = sum(
        by_month["2026-04"]["monthly_fee"]
        for by_month in per_asin.values()
        if "2026-04" in by_month
    )
    # Seller Central April report total ≈ $112.71 for Rawyal/NAQSH catalog.
    assert 110 < total < 115

    asin = "B07Y3DPHP5"
    april = per_asin[asin]["2026-04"]
    assert abs(april["monthly_fee"] - 16.4167) < 0.01
    per_unit = storage_per_unit_for_window(per_asin[asin], ["2026-04"])
    assert abs(per_unit - (16.4167 / 204.39)) < 0.001


def test_window_uses_april_not_blended_average():
    text = _load_sample("april storage.csv")
    per_asin, _ = parse_storage_fee_report(text)
    asin = "B07Y3DPHP5"

    april_only = storage_per_unit_for_window(per_asin[asin], ["2026-04"])

    may_text = _load_sample("401630020661.csv")
    per_asin_may, _ = parse_storage_fee_report(may_text)
    merged = {**per_asin[asin], **per_asin_may.get(asin, {})}
    blended = storage_per_unit_for_window(merged, ["2026-04", "2026-05"])

    assert april_only > blended  # old bug averaged months downward for April filter


def test_january_window_requests_single_month():
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from amazon_sp import calendar_months_in_window, storage_report_range_for_months

    tz = "America/Los_Angeles"
    z = ZoneInfo(tz)
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=z).astimezone(timezone.utc)
    end = datetime(2026, 1, 31, 23, 59, 59, tzinfo=z).astimezone(timezone.utc)
    months = calendar_months_in_window(start, end, tz)
    assert months == ["2026-01"]

    rs, re = storage_report_range_for_months(months, tz)
    assert rs.year == 2026 and rs.month == 1
    # End is last instant of January in marketplace TZ (may be Feb 1 UTC).
    assert re >= datetime(2026, 1, 31, 0, 0, 0, tzinfo=timezone.utc)
    assert re < datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_merge_storage_preserves_other_months():
    from amazon_sp import merge_storage_by_asin_month

    cache = {
        "B07Y3DPHP5": {
            "2026-06": {"monthly_fee": 10.0, "avg_quantity_on_hand": 100.0},
            "2026-07": {"monthly_fee": 12.0, "avg_quantity_on_hand": 110.0},
        }
    }
    fetched = {
        "B07Y3DPHP5": {
            "2026-01": {"monthly_fee": 8.0, "avg_quantity_on_hand": 90.0},
        }
    }
    merged = merge_storage_by_asin_month(cache, fetched)
    assert set(merged["B07Y3DPHP5"].keys()) == {"2026-01", "2026-06", "2026-07"}


def test_string_fees_do_not_crash_allocation():
    """Mongo/cache may hand back string numerics — must not TypeError on +."""
    from amazon_sp import (
        normalize_storage_fee_map,
        storage_per_unit_for_window,
    )

    raw = {
        "B07Y3DPHP5": {
            "2026-04": {
                "monthly_fee": "16.4167",
                "avg_quantity_on_hand": "204.39",
            }
        }
    }
    normalized = normalize_storage_fee_map(raw)
    per_unit = storage_per_unit_for_window(
        normalized["B07Y3DPHP5"], ["2026-04"],
    )
    assert abs(per_unit - (16.4167 / 204.39)) < 0.001


def test_parse_may_quoted_csv():
    text = _load_sample("401630020661.csv")
    from amazon_sp import parse_storage_fee_report

    per_asin, months = parse_storage_fee_report(text)
    assert "2026-05" in months
    assert len(per_asin) > 0
    total = sum(
        by_month["2026-05"]["monthly_fee"]
        for by_month in per_asin.values()
        if "2026-05" in by_month
    )
    assert total > 0


def test_months_covered_union_not_concat():
    """Regression: set + list used to TypeError and zero out Storage KPI."""
    old = ["2026-06", "2026-07"]
    new = ["2026-04"]
    merged = sorted(set(old) | set(new))
    assert merged == ["2026-04", "2026-06", "2026-07"]
    try:
        sorted(set(old) + list(new))
        assert False, "expected TypeError"
    except TypeError as e:
        assert "set" in str(e) and "list" in str(e)


def test_storage_months_with_fees_ignores_empty_buckets():
    from amazon_sp import storage_months_with_fees

    per = {
        "B0001": {
            "2026-06": {"monthly_fee": 10.0, "avg_quantity_on_hand": 5},
            "2026-07": {"monthly_fee": 0.0, "avg_quantity_on_hand": 1},
        },
        "B0002": {
            "2026-06": {"monthly_fee": 1.5, "avg_quantity_on_hand": 2},
        },
    }
    assert storage_months_with_fees(per) == ["2026-06"]


def test_fresh_empty_checked_months_ttl():
    from datetime import datetime, timedelta, timezone

    from database import _fresh_empty_checked_months

    now = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)
    empty = {
        "2026-07": now - timedelta(hours=2),   # fresh
        "2026-08": now - timedelta(hours=7),   # expired
        "bad": now,
    }
    assert _fresh_empty_checked_months(empty, now=now) == {"2026-07"}


def test_neighbor_month_must_not_poison_requested_month():
    """Amazon returns June while July was requested → July stays uncovered
    after empty-check expires. Applies to every seller cache doc."""
    from datetime import datetime, timedelta, timezone

    from amazon_sp import storage_months_with_fees
    from database import _fresh_empty_checked_months

    # Cache only has June fees but monthsCovered falsely listed July (old bug).
    per = {
        "B0001": {"2026-06": {"monthly_fee": 9.1, "avg_quantity_on_hand": 3}},
    }
    fee_months = set(storage_months_with_fees(per))
    claimed = {"2026-06", "2026-07", "2026-08"}
    phantoms = claimed - fee_months
    assert phantoms == {"2026-07", "2026-08"}

    now = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)
    # Immediately after a neighbor-month fetch we empty-check July for 6h.
    empty_at = {"2026-07": now}
    covered_now = fee_months | _fresh_empty_checked_months(empty_at, now=now)
    assert "2026-07" in covered_now
    # After TTL, July must be missing again so the next load refetches.
    later = now + timedelta(hours=7)
    covered_later = fee_months | _fresh_empty_checked_months(empty_at, now=later)
    assert "2026-07" not in covered_later
    assert "2026-06" in covered_later


def test_june_report_total_matches_csv_allocation():
    """Full June filter Storage KPI must equal Seller Central CSV total (~$39.44)."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from amazon_sp import (
        parse_storage_fee_report,
        storage_fees_by_asin_for_window,
    )

    text = _load_sample("401643020661.csv")
    per_asin, months = parse_storage_fee_report(text)
    assert months == ["2026-06"]

    tz = "America/Los_Angeles"
    z = ZoneInfo(tz)
    start = datetime(2026, 6, 1, 0, 0, 0, tzinfo=z).astimezone(timezone.utc)
    end = datetime(2026, 6, 30, 23, 59, 59, 999999, tzinfo=z).astimezone(timezone.utc)
    fees, total, fractions = storage_fees_by_asin_for_window(
        per_asin, ["2026-06"], start, end, tz,
    )
    assert abs(fractions["2026-06"] - 1.0) < 1e-6
    assert abs(total - 39.44) < 0.02
    assert abs(sum(fees.values()) - total) < 0.02

    # rate × units sold would NOT match (user's mismatch).
    asin = "B0CJFNHJRX"
    report_fee = fees[asin]
    wrong = per_asin[asin]["2026-06"]["storage_per_unit"] * 500
    assert wrong > report_fee * 2


def test_asin_fee_split_across_skus_sums_to_report():
    from amazon_sp import parse_storage_fee_report, storage_fees_by_asin_for_window

    text = _load_sample("401643020661.csv")
    per_asin, _ = parse_storage_fee_report(text)
    fees, total, _ = storage_fees_by_asin_for_window(per_asin, ["2026-06"])
    asin = "B0CJFNHJRX"
    asin_fee = fees[asin]
    # two SKUs share ASIN 60/40
    s1 = round(asin_fee * (60 / 100), 2)
    s2 = round(asin_fee * (40 / 100), 2)
    assert abs((s1 + s2) - asin_fee) <= 0.01
    assert abs(total - 39.44) < 0.02
