"""
Decrypt Aurora Node backend's `enc:v1:` refresh tokens.

The Node backend stores `amazonRefreshToken` and `amazonAdsRefreshToken`
AES-256-GCM encrypted (see auroraBackend/src/utils/fieldEncryption.js), and
we share the same Mongo. The Python backend must decrypt them before sending
to LWA — passing the ciphertext through unchanged is what produces the
`invalid_grant` we saw on the profitability tab.

Format: `enc:v1:{iv_hex}:{authTag_hex}:{ciphertext_hex}`
Key priority (matches Node's `resolveEncryptionKey('token')`):
  1. TOKEN_ENCRYPTION_KEY  (base64, 32 bytes)
  2. SELLER_APP_ENCRYPTION_KEY  (base64, 32 bytes)
  3. non-prod fallback: sha256("aurora-token-enc:" + JWT_SECRET)
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


def _resolve_token_key() -> Optional[bytes]:
    for env_name in ("TOKEN_ENCRYPTION_KEY", "SELLER_APP_ENCRYPTION_KEY"):
        k = _decode_configured_key(os.getenv(env_name))
        if k:
            return k
    if os.getenv("NODE_ENV") != "production":
        secret = os.getenv("JWT_SECRET") or "aurora-dev-insecure-key"
        return hashlib.sha256(f"aurora-token-enc:{secret}".encode("utf-8")).digest()
    return None


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
    plaintext = AESGCM(key).decrypt(iv, ct + tag, None)
    return plaintext.decode("utf-8")
