"""FBA Reimbursements report parsing / window filter tests."""

from datetime import datetime, timezone

from amazon_sp import parse_reimbursements_report


def test_tsv_reimbursements_parses_and_sums():
    lines = [
        "approval-date\treimbursement-id\treason\tsku\tamount-total\tquantity-reimbursed-total",
        "2026-07-10T12:00:00-07:00\tr1\tDamaged_Warehouse\tSKU-A\t12.50\t1",
        "2026-06-01T12:00:00-07:00\tr2\tLost_Warehouse\tSKU-A\t3.25\t1",
        "2026-07-15T12:00:00Z\tr3\tCustomerReturn\tSKU-B\t5.00\t2",
    ]
    per_sku, total = parse_reimbursements_report("\n".join(lines))
    assert abs(total - 20.75) < 0.001
    assert abs(per_sku["SKU-A"]["reimbursement"] - 15.75) < 0.001
    assert abs(per_sku["SKU-B"]["reimbursement"] - 5.00) < 0.001


def test_approval_date_window_july_only():
    lines = [
        "approval-date,reimbursement-id,reason,sku,amount-total",
        "2026-07-10T12:00:00Z,r1,Damaged_Warehouse,SKU-A,12.50",
        "2026-06-01T12:00:00Z,r2,Lost_Warehouse,SKU-A,3.25",
        "2026-07-15T12:00:00Z,r3,CustomerReturn,SKU-B,5.00",
    ]
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 1, tzinfo=timezone.utc)
    per_sku, total = parse_reimbursements_report("\n".join(lines), start, end)
    assert abs(total - 17.50) < 0.001
    assert "SKU-A" in per_sku
    assert "SKU-B" in per_sku
    assert abs(per_sku["SKU-A"]["reimbursement"] - 12.50) < 0.001


def test_zero_amount_rows_ignored():
    lines = [
        "approval-date,sku,amount-total,quantity-reimbursed-inventory",
        "2026-07-10T12:00:00Z,SKU-A,0,2",
        "2026-07-11T12:00:00Z,SKU-B,4.00,1",
    ]
    per_sku, total = parse_reimbursements_report("\n".join(lines))
    assert abs(total - 4.00) < 0.001
    assert "SKU-A" not in per_sku
    assert per_sku["SKU-B"]["reimbursement"] == 4.00


def test_rows_without_sku_still_count_in_total():
    lines = [
        "approval-date,sku,amount-total",
        "2026-07-10T12:00:00Z,,7.25",
        "2026-07-11T12:00:00Z,SKU-A,2.00",
    ]
    per_sku, total = parse_reimbursements_report("\n".join(lines))
    assert abs(total - 9.25) < 0.001
    assert abs(per_sku["SKU-A"]["reimbursement"] - 2.00) < 0.001


def test_window_excludes_outside_approval_dates():
    lines = [
        "approval-date,sku,amount-total",
        "2026-05-10T12:00:00Z,SKU-A,9.00",
    ]
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 1, tzinfo=timezone.utc)
    per_sku, total = parse_reimbursements_report("\n".join(lines), start, end)
    assert total == 0.0
    assert per_sku == {}


def test_empty_report():
    per_sku, total = parse_reimbursements_report("")
    assert per_sku == {}
    assert total == 0.0


def test_seller_central_quoted_csv_sample():
    """Matches Downloads/408040020672.csv shape (quoted SC CSV)."""
    from pathlib import Path

    path = Path(r"C:\Users\Desktop\Downloads\408040020672.csv")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8-sig")
    per_sku, total = parse_reimbursements_report(text)
    # Net of credits (+120.55) and reversals (-205.51)
    assert abs(total - (-84.96)) < 0.02
    assert abs(per_sku["EP - Multi Knob 1 Set of 20"]["reimbursement"] - 54.57) < 0.02
