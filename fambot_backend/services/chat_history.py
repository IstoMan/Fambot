from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from firebase_admin import firestore

from fambot_backend.core.firebase_init import init_firebase


def _db():
    init_firebase()
    return firestore.client()


def _chat_ref(uid: str, chat_id: str):
    return _db().collection("users").document(uid).collection("chats").document(chat_id)


def create_chat(uid: str, chat_id: str, title: str | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "id": chat_id,
        "title": title or "New chat",
        "created_at": now,
        "last_updated": now,
        "user_id": uid,
    }
    _chat_ref(uid, chat_id).set(payload)
    return payload


def get_chat(uid: str, chat_id: str) -> dict[str, Any] | None:
    snap = _chat_ref(uid, chat_id).get()
    if not snap.exists:
        return None
    data = snap.to_dict() or {}
    if "id" not in data:
        data["id"] = chat_id
    return data


def list_chats(uid: str) -> list[dict[str, Any]]:
    rows = (
        _db()
        .collection("users")
        .document(uid)
        .collection("chats")
        .order_by("last_updated", direction=firestore.Query.DESCENDING)
        .stream()
    )
    items: list[dict[str, Any]] = []
    for row in rows:
        data = row.to_dict() or {}
        if "id" not in data:
            data["id"] = row.id
        items.append(data)
    return items


def list_chat_messages(uid: str, chat_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    chat = get_chat(uid, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    query = _chat_ref(uid, chat_id).collection("messages").order_by("created_at")
    if limit is not None and limit > 0:
        query = query.limit(limit)
    rows = query.stream()
    return [(row.to_dict() or {}) for row in rows]


def append_chat_message(
    uid: str,
    chat_id: str,
    *,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    has_file: bool | None = None,
) -> dict[str, Any]:
    chat = get_chat(uid, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "role": role,
        "content": content,
        "created_at": now,
    }
    if citations is not None:
        payload["citations"] = citations
    if has_file is not None:
        payload["has_file"] = has_file
    _chat_ref(uid, chat_id).collection("messages").add(payload)
    return payload


def update_chat_metadata(uid: str, chat_id: str, *, title: str | None = None) -> None:
    chat = get_chat(uid, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    update: dict[str, Any] = {"last_updated": datetime.now(timezone.utc)}
    if title:
        update["title"] = title
    _chat_ref(uid, chat_id).set(update, merge=True)
