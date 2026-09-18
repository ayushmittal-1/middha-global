"""AI Assistant feature visibility — mirrors Node aiCapabilities.js.

Fail closed: missing/invalid NODE_ENV → production (core tabs only),
unless SHOW_AI_ASSISTANT_ALL_TABS is true (all tabs in any environment).
"""
from __future__ import annotations

import os
from typing import FrozenSet, List

ALL_TABS: List[str] = [
    "chat",
    "restock",
    "orders",
    "campaigns",
    "profit",
    "cogs",
    "keywords",
    "listings",
    "deals",
]

PRODUCTION_TABS: List[str] = [
    "restock",
    "profit",
    "keywords",
    "listings",
]

EXTENDED_TABS: FrozenSet[str] = frozenset(
    {"chat", "orders", "campaigns", "cogs", "deals"}
)

# Map HTTP path prefixes / exact paths to feature keys.
# Core features are always allowed.
FEATURE_PATH_PREFIXES = (
    ("/chat", "chat"),
    ("/ws/chat", "chat"),
    ("/sessions", "chat"),
    ("/amazon/orders", "orders"),
    ("/campaigns/performance", "campaigns"),
    ("/cogs/upload", "cogs"),
    ("/deals", "deals"),
)


def normalize_node_env(raw: str | None) -> str:
    v = (raw or "").strip().lower()
    if v in ("development", "dev"):
        return "development"
    if v in ("production", "prod"):
        return "production"
    return "production"


def _env_flag_true(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("true", "1", "yes", "on")


def show_all_ai_tabs() -> bool:
    return _env_flag_true("SHOW_AI_ASSISTANT_ALL_TABS")


def get_ai_environment() -> str:
    return normalize_node_env(os.getenv("NODE_ENV"))


def is_ai_development() -> bool:
    return get_ai_environment() == "development"


def _extended_features_unlocked() -> bool:
    return show_all_ai_tabs() or is_ai_development()


def get_enabled_tabs() -> List[str]:
    if _extended_features_unlocked():
        return list(ALL_TABS)
    return list(PRODUCTION_TABS)


def is_tab_enabled(tab_id: str) -> bool:
    return tab_id in get_enabled_tabs()


def is_feature_enabled(feature: str) -> bool:
    if feature == "core":
        return True
    if not _extended_features_unlocked():
        return False
    return feature in EXTENDED_TABS


def get_ai_capabilities() -> dict:
    return {
        "environment": get_ai_environment(),
        "enabledTabs": get_enabled_tabs(),
    }


def feature_for_path(path: str) -> str | None:
    """Return extended feature key for a path, or None if core/unrelated."""
    p = path or "/"
    # Exact / preferential matches first
    if p == "/cogs/upload" or p.startswith("/cogs/upload"):
        return "cogs"
    # DELETE /cogs/{sku} is extended; GET/PUT remain core (Profitability).
    # Guarded separately in main.py via method check.
    for prefix, feature in FEATURE_PATH_PREFIXES:
        if p == prefix or p.startswith(prefix + "/") or p.startswith(prefix + "?"):
            return feature
        if p.startswith(prefix) and prefix != "/cogs/upload":
            # /sessions/foo, /amazon/orders/x/items, etc.
            if prefix in ("/sessions", "/amazon/orders", "/chat", "/ws/chat", "/deals", "/campaigns/performance"):
                if p == prefix or p.startswith(prefix + "/"):
                    return feature
    return None
