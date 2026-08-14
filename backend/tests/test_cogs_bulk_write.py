"""Bulk COGS upsert — audit H3.

Pre-fix, each row triggered its own `update_one` round-trip. Post-fix,
valid rows shape into `pymongo.UpdateOne` and dispatch via
`bulk_write(ordered=False)`. Test the shaping helper in isolation."""

import os
from datetime import datetime, timezone

os.environ.setdefault("GROQ_API_KEY", "test-stub")

from bson import ObjectId
from pymongo import UpdateOne

from database import _build_cogs_bulk_ops


USER_ID = ObjectId("507f1f77bcf86cd799439011")
NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)


def test_valid_rows_shape_into_update_one_ops():
    rows = [
        {"sku": "A-1", "unit_cost": 3.50, "inbound_shipping_per_unit": 0.42},
        {"sku": "B-2", "unit_cost": 10.00},
    ]
    ops = _build_cogs_bulk_ops(rows, USER_ID, NOW)
    assert len(ops) == 2
    for op in ops:
        assert isinstance(op, UpdateOne)


def test_op_targets_correct_user_sku_filter():
    rows = [{"sku": "A-1", "unit_cost": 3.50}]
    ops = _build_cogs_bulk_ops(rows, USER_ID, NOW)
    op = ops[0]
    filter_ = op._filter  # pymongo internal but stable
    assert filter_["userId"] == USER_ID
    assert filter_["sku"] == "A-1"
    assert op._upsert is True


def test_missing_or_blank_sku_is_dropped():
    rows = [
        {"sku": "", "unit_cost": 5.0},
        {"sku": "   ", "unit_cost": 5.0},
        {"unit_cost": 5.0},  # no sku key at all
        {"sku": "keep-me", "unit_cost": 5.0},
    ]
    ops = _build_cogs_bulk_ops(rows, USER_ID, NOW)
    assert len(ops) == 1


def test_zero_or_negative_unit_cost_is_dropped():
    rows = [
        {"sku": "A", "unit_cost": 0},
        {"sku": "B", "unit_cost": -1.50},
        {"sku": "C", "unit_cost": 5.00},
    ]
    ops = _build_cogs_bulk_ops(rows, USER_ID, NOW)
    assert len(ops) == 1


def test_non_numeric_unit_cost_is_dropped():
    rows = [
        {"sku": "A", "unit_cost": "not-a-number"},
        {"sku": "B", "unit_cost": None},
        {"sku": "C", "unit_cost": 5.00},
    ]
    ops = _build_cogs_bulk_ops(rows, USER_ID, NOW)
    assert len(ops) == 1


def test_missing_shipping_defaults_to_zero():
    rows = [{"sku": "A", "unit_cost": 5.0}]
    ops = _build_cogs_bulk_ops(rows, USER_ID, NOW)
    update = ops[0]._doc
    assert update["$set"]["inboundShippingPerUnit"] == 0.0


def test_empty_input_returns_empty_list():
    assert _build_cogs_bulk_ops([], USER_ID, NOW) == []
    assert _build_cogs_bulk_ops(None, USER_ID, NOW) == []  # type: ignore[arg-type]


def test_sku_is_stripped_of_whitespace():
    rows = [{"sku": "  padded  ", "unit_cost": 5.0}]
    ops = _build_cogs_bulk_ops(rows, USER_ID, NOW)
    assert ops[0]._filter["sku"] == "padded"
