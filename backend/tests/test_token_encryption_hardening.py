"""Token-encryption key resolution — audit C1.

Prior code silently fell back to `sha256("aurora-token-enc:" +
JWT_SECRET)` outside production, and further to
`sha256("aurora-token-enc:aurora-dev-insecure-key")` when JWT_SECRET
was also unset — a key anyone reading the source could reconstruct.
These tests lock in the hardened resolution rules."""

import os

os.environ.setdefault("GROQ_API_KEY", "test-stub")

import pytest

import token_encryption as te


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "TOKEN_ENCRYPTION_KEY",
        "SELLER_APP_ENCRYPTION_KEY",
        "NODE_ENV",
        "JWT_SECRET",
        "AURORA_ALLOW_DEV_TOKEN_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


def test_hardcoded_fallback_string_is_gone():
    """Regression guard — the literal string that gated the pre-fix
    insecure key must no longer appear anywhere in the module."""
    src = open(te.__file__).read()
    assert "aurora-dev-insecure-key" not in src


def test_production_with_no_key_returns_none(monkeypatch):
    monkeypatch.setenv("NODE_ENV", "production")
    assert te._resolve_token_key() is None


def test_production_with_no_key_startup_check_raises(monkeypatch):
    monkeypatch.setenv("NODE_ENV", "production")
    with pytest.raises(RuntimeError) as exc:
        te.assert_token_key_configured()
    assert "TOKEN_ENCRYPTION_KEY" in str(exc.value)


def test_dev_without_opt_in_returns_none(monkeypatch):
    """Local dev with a JWT_SECRET set is NOT enough — the operator
    must explicitly opt into the JWT-derived dev key."""
    monkeypatch.setenv("NODE_ENV", "development")
    monkeypatch.setenv("JWT_SECRET", "some-dev-secret")
    assert te._resolve_token_key() is None


def test_dev_with_opt_in_and_jwt_secret_returns_a_key(monkeypatch):
    monkeypatch.setenv("NODE_ENV", "development")
    monkeypatch.setenv("JWT_SECRET", "some-dev-secret")
    monkeypatch.setenv("AURORA_ALLOW_DEV_TOKEN_KEY", "1")
    key = te._resolve_token_key()
    assert isinstance(key, bytes)
    assert len(key) == 32


def test_dev_with_opt_in_but_empty_jwt_secret_returns_none(monkeypatch):
    """The opt-in flag alone is not enough — a blank JWT_SECRET must
    NOT resolve to a hardcoded fallback."""
    monkeypatch.setenv("NODE_ENV", "development")
    monkeypatch.setenv("JWT_SECRET", "")
    monkeypatch.setenv("AURORA_ALLOW_DEV_TOKEN_KEY", "1")
    assert te._resolve_token_key() is None


def test_explicit_token_encryption_key_takes_precedence(monkeypatch):
    import base64
    real_key = b"a" * 32
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", base64.b64encode(real_key).decode())
    monkeypatch.setenv("NODE_ENV", "production")
    resolved = te._resolve_token_key()
    assert resolved == real_key


def test_seller_app_encryption_key_used_when_primary_absent(monkeypatch):
    import base64
    real_key = b"b" * 32
    monkeypatch.setenv("SELLER_APP_ENCRYPTION_KEY", base64.b64encode(real_key).decode())
    monkeypatch.setenv("NODE_ENV", "production")
    resolved = te._resolve_token_key()
    assert resolved == real_key
