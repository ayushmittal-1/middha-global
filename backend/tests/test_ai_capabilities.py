"""Tests for AI Assistant environment / tab capabilities."""
import os

import pytest

from ai_capabilities import (
    ALL_TABS,
    PRODUCTION_TABS,
    get_ai_capabilities,
    get_enabled_tabs,
    is_feature_enabled,
    normalize_node_env,
)


@pytest.fixture(autouse=True)
def _restore_env():
    prev = os.environ.get("NODE_ENV")
    prev_all = os.environ.get("SHOW_AI_ASSISTANT_ALL_TABS")
    yield
    if prev is None:
        os.environ.pop("NODE_ENV", None)
    else:
        os.environ["NODE_ENV"] = prev
    if prev_all is None:
        os.environ.pop("SHOW_AI_ASSISTANT_ALL_TABS", None)
    else:
        os.environ["SHOW_AI_ASSISTANT_ALL_TABS"] = prev_all


def test_normalize_fail_closed():
    assert normalize_node_env("development") == "development"
    assert normalize_node_env("dev") == "development"
    assert normalize_node_env("production") == "production"
    assert normalize_node_env("") == "production"
    assert normalize_node_env("staging") == "production"
    assert normalize_node_env(None) == "production"


def test_development_all_nine_tabs():
    os.environ["NODE_ENV"] = "development"
    assert get_enabled_tabs() == ALL_TABS
    assert len(get_enabled_tabs()) == 9
    assert get_ai_capabilities()["environment"] == "development"
    assert is_feature_enabled("chat") is True
    assert is_feature_enabled("orders") is True


def test_production_four_tabs_only():
    os.environ["NODE_ENV"] = "production"
    os.environ.pop("SHOW_AI_ASSISTANT_ALL_TABS", None)
    assert get_enabled_tabs() == PRODUCTION_TABS
    assert get_enabled_tabs() == ["restock", "profit", "keywords", "listings"]
    assert is_feature_enabled("chat") is False
    assert is_feature_enabled("cogs") is False
    assert is_feature_enabled("core") is True


def test_show_all_tabs_flag_in_production():
    os.environ["NODE_ENV"] = "production"
    os.environ["SHOW_AI_ASSISTANT_ALL_TABS"] = "true"
    assert get_enabled_tabs() == ALL_TABS
    assert len(get_enabled_tabs()) == 9
    assert is_feature_enabled("chat") is True
    assert is_feature_enabled("deals") is True


def test_missing_env_fail_closed():
    os.environ.pop("SHOW_AI_ASSISTANT_ALL_TABS", None)
    os.environ.pop("NODE_ENV", None)
    assert get_enabled_tabs() == PRODUCTION_TABS
