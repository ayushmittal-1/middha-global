"""
Per-user Amazon SP-API LWA credential lookup — mirrors Aurora's
`sellerAppHelper.getSellerAppCredentials`.

Aurora stores per-organization SP-API credentials in a `sellerapplications`
collection (mongoose model: SellerApplication), with the LWA client id/secret
AES-256-GCM encrypted (see auroraBackend/src/utils/sellerAppEncryption.js).
Legacy AES-256-CBC values (`iv_hex:ciphertext_hex`) are still decrypted.
Each User doc may carry `sellerApplicationId` pointing to one. If the user
has one, SP-API token refresh uses those creds; otherwise we fall back to env.

The Ads API uses a single shared LWA app (env-only) — Aurora's
SellerApplication has no ads-LWA fields, so neither do we.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from typing import Optional

from bson import ObjectId
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from auth import _db

ENCRYPTED_PREFIX = "enc:v1:"
_LEGACY_CBC_PATTERN = re.compile(r"^[0-9a-f]{32}:[0-9a-f]+$", re.IGNORECASE)


def _decode_configured_key(raw: Optional[str]) -> Optional[bytes]:
    if not raw:
        return None
    trimmed = raw.strip()
    if not trimmed:
        return None
    try:
        decoded = base64.b64decode(trimmed, validate=False)
        if len(decoded) == 32:
            return decoded
    except Exception:
        pass
    # Legacy passphrase form — matches Node's sha256 fallback.
    return hashlib.sha256(f"aurora-env-key:{trimmed}".encode("utf-8")).digest()


def _encryption_key() -> bytes:
    for env_name in ("SELLER_APP_ENCRYPTION_KEY", "TOKEN_ENCRYPTION_KEY"):
        key = _decode_configured_key(os.getenv(env_name))
        if key:
            return key
    raise RuntimeError(
        "SELLER_APP_ENCRYPTION_KEY (or TOKEN_ENCRYPTION_KEY) is not configured"
    )


def _decrypt_gcm(text: str) -> str:
    payload = text[len(ENCRYPTED_PREFIX) :]
    parts = payload.split(":")
    if len(parts) != 3:
        raise ValueError("Encrypted seller credential has an invalid format")
    iv_hex, tag_hex, ct_hex = parts
    iv = bytes.fromhex(iv_hex)
    tag = bytes.fromhex(tag_hex)
    ct = bytes.fromhex(ct_hex)
    # Python AESGCM expects ciphertext || tag concatenated.
    return AESGCM(_encryption_key()).decrypt(iv, ct + tag, None).decode("utf-8")


def _decrypt_legacy_cbc(text: str) -> str:
    iv_hex, ct_hex = text.split(":", 1)
    iv = bytes.fromhex(iv_hex)
    ct = bytes.fromhex(ct_hex)
    cipher = Cipher(algorithms.AES(_encryption_key()), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ct) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plain = unpadder.update(padded) + unpadder.finalize()
    return plain.decode("utf-8")


def _decrypt(text: Optional[str]) -> Optional[str]:
    """Decrypt Node-encoded seller-app credentials (GCM or legacy CBC).

    Returns the input unchanged if it doesn't look encrypted.
    """
    if not text or not isinstance(text, str):
        return text
    if text.startswith(ENCRYPTED_PREFIX):
        return _decrypt_gcm(text)
    if _LEGACY_CBC_PATTERN.match(text):
        try:
            return _decrypt_legacy_cbc(text)
        except Exception:
            return text  # not actually encrypted / wrong key → leave for caller
    return text


async def get_seller_app_credentials(user: dict) -> dict:
    """Look up a user's SellerApplication and decrypt its LWA creds. Falls
    back to env (AMAZON_LWA_CLIENT_ID / AMAZON_LWA_CLIENT_SECRET) if the
    user has no seller app, the app is inactive, or decryption fails."""
    env_fallback = {
        "amazonLwaClientId": os.getenv("AMAZON_LWA_CLIENT_ID", ""),
        "amazonLwaClientSecret": os.getenv("AMAZON_LWA_CLIENT_SECRET", ""),
        "source": "environment",
    }

    seller_app_id = user.get("sellerApplicationId")
    if not seller_app_id:
        return env_fallback

    try:
        sa = await _db().sellerapplications.find_one(
            {"_id": ObjectId(str(seller_app_id)), "isActive": True}
        )
        if not sa:
            return env_fallback
        return {
            "amazonLwaClientId": _decrypt(sa.get("amazonLwaClientId")) or env_fallback["amazonLwaClientId"],
            "amazonLwaClientSecret": _decrypt(sa.get("amazonLwaClientSecret")) or env_fallback["amazonLwaClientSecret"],
            "sellerAppId": str(sa["_id"]),
            "source": "seller_app",
        }
    except Exception as e:
        print(f"[seller_app] decrypt/load failed for user {user.get('email')}: {e}")
        return env_fallback
