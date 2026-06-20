"""
Chat history + COGS persistence — Mongo-backed, per-user-scoped.

One `conversations` document per chat with its messages embedded as an array.
Shape:
    {
      _id: ObjectId,
      convId: str,              # client-supplied UUID, matches the frontend's session_id
      userId: ObjectId,         # owning user
      title: str,
      messages: [
        { role: "user"|"assistant"|"tool_call"|"tool", content: str, createdAt: Date }
      ],
      createdAt: Date,
      updatedAt: Date,
    }

COGS lives in its own `userCogs` collection (one doc per user-SKU pair).

Caveat: Mongo caps documents at 16 MB. With the current policy of persisting
full tool results, a very long conversation that includes large inventory /
report dumps can approach that limit. We refuse the write past 15 MB so the
caller sees a clear error instead of a silent corruption.
"""

import json
from datetime import datetime, timezone

from bson import ObjectId
from pymongo.errors import DocumentTooLarge

from auth import _db, require_user

MAX_HISTORY = 40  # max messages loaded per session to avoid token overflow
MAX_DOC_BYTES = 15 * 1024 * 1024  # leave headroom under Mongo's 16 MB cap


def _conversations():
    return _db().conversations


def _cogs():
    return _db().userCogs


def _user_oid() -> ObjectId:
    user = require_user()
    return ObjectId(str(user["_id"]))


async def init_db():
    """Create indexes if they don't exist. Safe to run repeatedly."""
    await _conversations().create_index([("userId", 1), ("updatedAt", -1)])
    await _conversations().create_index(
        [("convId", 1), ("userId", 1)], unique=True
    )
    await _cogs().create_index([("userId", 1), ("sku", 1)], unique=True)


# ── Conversations ────────────────────────────────────────────────────────

async def create_session(session_id: str, title: str = "New Chat"):
    """Insert a conversation doc if one with this (convId, userId) doesn't
    already exist. The function name keeps the old signature so callers
    don't need to change."""
    user_id = _user_oid()
    now = datetime.now(timezone.utc)
    await _conversations().update_one(
        {"convId": session_id, "userId": user_id},
        {
            "$setOnInsert": {
                "convId": session_id,
                "userId": user_id,
                "title": title,
                "messages": [],
                "createdAt": now,
                "updatedAt": now,
            }
        },
        upsert=True,
    )


async def list_sessions() -> list[dict]:
    """Return the current user's conversations, newest-updated first."""
    user_id = _user_oid()
    cursor = (
        _conversations()
        .find(
            {"userId": user_id},
            {"_id": 0, "convId": 1, "title": 1, "createdAt": 1},
        )
        .sort("updatedAt", -1)
    )
    rows = await cursor.to_list(length=500)
    return [
        {
            "id": r["convId"],
            "title": r.get("title", "New Chat"),
            "created_at": r["createdAt"].isoformat() if r.get("createdAt") else None,
        }
        for r in rows
    ]


async def get_messages(session_id: str) -> list[dict]:
    """Return the last MAX_HISTORY messages for the user's conversation,
    oldest first. Uses $slice so we don't drag the whole array off the
    wire when only the tail matters."""
    user_id = _user_oid()
    doc = await _conversations().find_one(
        {"convId": session_id, "userId": user_id},
        {"messages": {"$slice": -MAX_HISTORY}},
    )
    if not doc:
        return []

    results: list[dict] = []
    for m in doc.get("messages", []):
        role = m.get("role")
        raw = m.get("content")
        if role in ("tool_call", "tool"):
            results.append(json.loads(raw))
        else:
            results.append({"role": role, "content": raw})
    return results


async def save_message(session_id: str, role: str, content: str):
    """Append a message to the conversation. Auto-creates the conversation if
    it doesn't exist yet (matches the old behavior; `create_session` is
    idempotent anyway)."""
    user_id = _user_oid()
    now = datetime.now(timezone.utc)
    msg = {"role": role, "content": content, "createdAt": now}
    try:
        await _conversations().update_one(
            {"convId": session_id, "userId": user_id},
            {
                "$push": {"messages": msg},
                "$set": {"updatedAt": now},
                "$setOnInsert": {
                    "convId": session_id,
                    "userId": user_id,
                    "title": "New Chat",
                    "createdAt": now,
                },
            },
            upsert=True,
        )
    except DocumentTooLarge:
        # Conversation hit Mongo's 16 MB cap — almost always because tool
        # results were persisted verbatim. Surface the cause; the caller
        # can decide whether to truncate or start a fresh conversation.
        raise RuntimeError(
            f"Conversation {session_id} exceeds {MAX_DOC_BYTES // (1024*1024)} MB. "
            "Tool results are being stored in full — consider trimming or starting "
            "a new chat."
        )


async def update_session_title(session_id: str, title: str):
    user_id = _user_oid()
    await _conversations().update_one(
        {"convId": session_id, "userId": user_id},
        {"$set": {"title": title, "updatedAt": datetime.now(timezone.utc)}},
    )


async def delete_session(session_id: str):
    user_id = _user_oid()
    await _conversations().delete_one(
        {"convId": session_id, "userId": user_id}
    )


# ── COGS (Cost of Goods Sold) ─────────────────────────────────────────────

async def upsert_cogs(rows: list[dict]) -> int:
    """Insert or update COGS rows for the current user. Each row needs sku and
    unit_cost; inbound_shipping_per_unit is optional. Returns the count actually
    written.
    """
    user_id = _user_oid()
    written = 0
    now = datetime.now(timezone.utc)
    for r in rows:
        sku = (r.get("sku") or "").strip()
        if not sku:
            continue
        try:
            unit_cost = float(r.get("unit_cost") or 0)
        except (TypeError, ValueError):
            continue
        if unit_cost <= 0:
            continue
        try:
            shipping = float(r.get("inbound_shipping_per_unit") or 0)
        except (TypeError, ValueError):
            shipping = 0.0
        await _cogs().update_one(
            {"userId": user_id, "sku": sku},
            {
                "$set": {
                    "unitCost": unit_cost,
                    "inboundShippingPerUnit": shipping,
                    "updatedAt": now,
                },
                "$setOnInsert": {"userId": user_id, "sku": sku},
            },
            upsert=True,
        )
        written += 1
    return written


async def get_cogs(skus: list[str] | None = None) -> list[dict]:
    user_id = _user_oid()
    query: dict = {"userId": user_id}
    if skus:
        query["sku"] = {"$in": skus}
    cursor = _cogs().find(query, {"_id": 0, "userId": 0}).sort("sku", 1)
    rows = await cursor.to_list(length=5000)
    return [
        {
            "sku": r["sku"],
            "unit_cost": r.get("unitCost"),
            "inbound_shipping_per_unit": r.get("inboundShippingPerUnit", 0),
            "updated_at": r["updatedAt"].isoformat() if r.get("updatedAt") else None,
        }
        for r in rows
    ]
