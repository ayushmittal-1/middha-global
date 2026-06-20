"""
Per-user Amazon SP-API LWA credential lookup — mirrors Aurora's
`sellerAppHelper.getSellerAppCredentials`.

Aurora stores per-organization SP-API credentials in a `sellerapplications`
collection (mongoose model: SellerApplication), with the LWA client id/secret
AES-256-CBC encrypted using a key from SELLER_APP_ENCRYPTION_KEY. Each User
doc may carry `sellerApplicationId` pointing to one. If the user has one,
SP-API token refresh uses those creds; otherwise we fall back to env.

The Ads API uses a single shared LWA app (env-only) — Aurora's
SellerApplication has no ads-LWA fields, so neither do we.
"""

import base64
import os
from typing import Optional

from bson import ObjectId
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from auth import _db


def _encryption_key() -> bytes:
    raw = os.getenv("SELLER_APP_ENCRYPTION_KEY", "")
    if not raw:
        raise RuntimeError("SELLER_APP_ENCRYPTION_KEY is not configured")
    return base64.b64decode(raw)


def _decrypt(text: Optional[str]) -> Optional[str]:
    """Decrypt a Node-encoded `iv_hex:ciphertext_hex` AES-256-CBC string.
    Returns the input unchanged if it doesn't look encrypted."""
    if not text or not isinstance(text, str) or ":" not in text:
        return text
    iv_hex, ct_hex = text.split(":", 1)
    try:
        iv = bytes.fromhex(iv_hex)
        ct = bytes.fromhex(ct_hex)
    except ValueError:
        return text  # not actually encrypted
    cipher = Cipher(algorithms.AES(_encryption_key()), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ct) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plain = unpadder.update(padded) + unpadder.finalize()
    return plain.decode("utf-8")


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
