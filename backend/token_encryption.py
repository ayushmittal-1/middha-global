"""
Decrypt Aurora Node backend's `enc:v1:` refresh tokens.

The Node backend stores `amazonRefreshToken` and `amazonAdsRefreshToken`
AES-256-GCM encrypted (see auroraBackend/src/utils/fieldEncryption.js), and
we share the same Mongo. The Python backend must decrypt them before sending
to LWA — passing the ciphertext through unchanged is what produces the
`invalid_grant` we saw on the profitability tab.

Format: `enc:v1:{iv_hex}:{authTag_hex}:{ciphertext_hex}`
Key priority:
  1. TOKEN_ENCRYPTION_KEY  (base64, 32 bytes) — REQUIRED in production
  2. SELLER_APP_ENCRYPTION_KEY  (base64, 32 bytes) — alternative for Node parity
  3. Local-dev only, and only when AURORA_ALLOW_DEV_TOKEN_KEY=1 is
     explicitly set: sha256("aurora-token-enc:" + JWT_SECRET). Never
     falls through to any hardcoded string (audit C1).
"""

import base64
import hashlib
import os
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ENCRYPTED_PREFIX = "enc:v1:"


def _decode_configured_key(raw: Optional[str]) -> Optional[bytes]:
    if not raw:
        return None
    trimmed = raw.strip()
    if not trimmed:
        return None
    try:
        b = base64.b64decode(trimmed, validate=False)
        if len(b) == 32:
            return b
    except Exception:
        pass
    # Legacy passphrase form — matches Node's sha256 fallback.
    return hashlib.sha256(f"aurora-env-key:{trimmed}".encode("utf-8")).digest()


def _is_production() -> bool:
    """Production if NODE_ENV=production. Kept as a helper so the check
    is centralized — any hardening added later (e.g. the CORS allowlist
    guard in main.py) reads the same flag."""
    return os.getenv("NODE_ENV", "").strip().lower() == "production"


def _resolve_token_key() -> Optional[bytes]:
    """Return the AES-256 key used to decrypt Amazon refresh tokens.

    Audit C1 hardened the resolution rules: no hardcoded fallback string
    is ever returned, and the JWT_SECRET-derived key is ONLY allowed when
    the operator has opted in explicitly (AURORA_ALLOW_DEV_TOKEN_KEY=1)
    outside of production. Missing configuration returns None so
    `decrypt_token` can raise a loud RuntimeError at call time, and the
    startup check (`assert_token_key_configured`) fails fast on boot."""
    for env_name in ("TOKEN_ENCRYPTION_KEY", "SELLER_APP_ENCRYPTION_KEY"):
        k = _decode_configured_key(os.getenv(env_name))
        if k:
            return k
    if _is_production():
        return None
    # Local-dev opt-in only: an operator has to acknowledge they're
    # deriving the key from JWT_SECRET (which itself must exist and be
    # non-empty — no hardcoded fallback string).
    if os.getenv("AURORA_ALLOW_DEV_TOKEN_KEY", "").strip() != "1":
        return None
    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        return None
    return hashlib.sha256(f"aurora-token-enc:{secret}".encode("utf-8")).digest()


def assert_token_key_configured() -> None:
    """Fail-fast startup check for audit C1.

    Called from main.py's lifespan handler so a misconfigured deployment
    dies at boot instead of silently running with a resolvable-from-code
    key that anyone reading the source could reconstruct."""
    if _resolve_token_key() is not None:
        return
    if _is_production():
        raise RuntimeError(
            "TOKEN_ENCRYPTION_KEY (or SELLER_APP_ENCRYPTION_KEY) is not "
            "set. Amazon refresh tokens cannot be decrypted and no "
            "insecure fallback is available in production. Set the key "
            "and restart."
        )
    raise RuntimeError(
        "No token-decryption key configured. For local dev, set "
        "TOKEN_ENCRYPTION_KEY explicitly, or set "
        "AURORA_ALLOW_DEV_TOKEN_KEY=1 together with a non-empty "
        "JWT_SECRET to opt into the JWT-derived dev key (never for prod)."
    )


def is_encrypted(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)


def decrypt_token(value: Optional[str]) -> Optional[str]:
    """Decrypt an `enc:v1:` token. Passes plain strings through unchanged so
    callers can call this unconditionally on whatever Mongo returned."""
    if not is_encrypted(value):
        return value
    payload = value[len(ENCRYPTED_PREFIX):]
    parts = payload.split(":")
    if len(parts) != 3:
        raise ValueError("Encrypted token has invalid format (expected iv:tag:ct)")
    iv_hex, tag_hex, ct_hex = parts
    key = _resolve_token_key()
    if not key:
        raise RuntimeError(
            "Cannot decrypt Amazon token: no TOKEN_ENCRYPTION_KEY or "
            "SELLER_APP_ENCRYPTION_KEY configured (and NODE_ENV=production "
            "disables the JWT_SECRET fallback)."
        )
    iv = bytes.fromhex(iv_hex)
    tag = bytes.fromhex(tag_hex)
    ct = bytes.fromhex(ct_hex)
    # Python's AESGCM expects ciphertext || tag concatenated.
    try:
        plaintext = AESGCM(key).decrypt(iv, ct + tag, None)
    except Exception as e:
        raise ValueError(
            f"decryption failed ({type(e).__name__}): "
            "TOKEN_ENCRYPTION_KEY does not match the Node backend"
        ) from e
    return plaintext.decode("utf-8")


TOKEN_FIELDS = ("amazonRefreshToken", "amazonAdsRefreshToken")


def hydrate_user_tokens(user: dict) -> dict:
    """Decrypt encrypted Amazon tokens on a user document in-place."""
    for field in TOKEN_FIELDS:
        value = user.get(field)
        if value and is_encrypted(value):
            user[field] = decrypt_token(value)
    return user
