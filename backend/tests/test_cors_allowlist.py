"""CORS allowlist parser — audit C3.

Pre-fix code used allow_origins=[\"*\"], removing any defense-in-depth
against a future XSS or token-leak vector. The parser under test always
seeds Aurora's known production origins, unions with an env-configurable
extras list, and refuses wildcards even if smuggled in via env."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

from main import parse_allowed_origins, PRODUCTION_BASELINE_ORIGINS


def test_baseline_origins_include_the_two_render_and_auroratest_ones():
    """Regression guard against a partial revert. The two production
    origins the operator asked for MUST always be present."""
    assert "https://middha-global-1.onrender.com" in PRODUCTION_BASELINE_ORIGINS
    assert "https://middha-global.onrender.com" in PRODUCTION_BASELINE_ORIGINS
    assert "https://www.auroratest.in" in PRODUCTION_BASELINE_ORIGINS
    assert "https://auroratest.in" in PRODUCTION_BASELINE_ORIGINS
    assert "https://www.aurora-ai.io" in PRODUCTION_BASELINE_ORIGINS
    assert "https://aurora-ai.io" in PRODUCTION_BASELINE_ORIGINS


def test_empty_env_in_production_still_gets_baseline():
    """A prod deployment that forgot to set AURORA_ALLOWED_ORIGINS
    still comes up with the known-safe baseline, not a wildcard and
    not a startup crash."""
    origins = parse_allowed_origins("", production=True)
    for o in PRODUCTION_BASELINE_ORIGINS:
        assert o in origins


def test_empty_env_in_dev_adds_localhost_defaults():
    origins = parse_allowed_origins("", production=False)
    assert "http://localhost:3000" in origins
    assert "http://localhost:5173" in origins


def test_env_extras_are_unioned_with_baseline():
    origins = parse_allowed_origins(
        "https://staging.middha-global.onrender.com, https://preview.aurora.example",
        production=True,
    )
    # Baselines present
    assert "https://middha-global-1.onrender.com" in origins
    assert "https://middha-global.onrender.com" in origins
    # Extras present
    assert "https://staging.middha-global.onrender.com" in origins
    assert "https://preview.aurora.example" in origins


def test_wildcards_are_dropped_even_from_env():
    """Belt-and-braces — a well-intentioned env value can't downgrade
    to the pre-audit wildcard behavior."""
    origins = parse_allowed_origins("*, https://real.example", production=True)
    assert "*" not in origins
    assert "https://real.example" in origins


def test_trailing_slashes_are_stripped():
    origins = parse_allowed_origins("https://foo.example/, https://bar.example/",
                                    production=True)
    assert "https://foo.example" in origins
    assert "https://foo.example/" not in origins


def test_duplicates_are_deduped_preserving_order():
    """The baseline includes middha-global-1; adding it again via env
    shouldn't duplicate it."""
    origins = parse_allowed_origins(
        "https://middha-global-1.onrender.com, https://new.example",
        production=True,
    )
    assert origins.count("https://middha-global-1.onrender.com") == 1
    # Order: baselines first (as declared), then env extras.
    assert origins.index("https://middha-global-1.onrender.com") < origins.index(
        "https://new.example"
    )


def test_dev_mode_does_not_lose_baseline_origins():
    """Dev deployments still need the real production origins if
    someone runs a staging box in dev mode."""
    origins = parse_allowed_origins("", production=False)
    for o in PRODUCTION_BASELINE_ORIGINS:
        assert o in origins
