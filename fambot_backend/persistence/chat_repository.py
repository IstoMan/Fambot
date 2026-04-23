from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from fambot_backend.core.firebase_init import init_firebase
from fambot_backend.services.chat_history import (
    append_chat_message,
    get_chat,
    list_chat_messages,
    update_chat_metadata,
)

try:
    from firebase_admin import firestore
except Exception:  # pragma: no cover - import-time fallback for tests
    firestore = None

_SKIP_TURNS: dict[tuple[str, str, str], dict[str, Any]] = {}


class ChatRepository:
    def require_chat(self, *, uid: str, chat_id: str) -> dict[str, Any]:
        chat = get_chat(uid, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat

    def list_recent_messages(self, *, uid: str, chat_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return list_chat_messages(uid, chat_id, limit=limit)

    def create_turn(
        self,
        *,
        uid: str,
        chat_id: str,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        turn_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "id": turn_id,
            "state": "queued",
            "created_at": now,
            "updated_at": now,
            "idempotency_key": idempotency_key,
        }
        if idempotency_key:
            previous = self.find_turn_by_idempotency_key(
                uid=uid,
                chat_id=chat_id,
                idempotency_key=idempotency_key,
            )
            if previous:
                return previous
        self._set_turn(uid=uid, chat_id=chat_id, turn_id=turn_id, payload=payload)
        return payload

    def find_turn_by_idempotency_key(
        self,
        *,
        uid: str,
        chat_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        if _skip_firestore():
            return _SKIP_TURNS.get((uid, chat_id, idempotency_key))
        turns = (
            self._chat_ref(uid, chat_id)
            .collection("turns")
            .where("idempotency_key", "==", idempotency_key)
            .limit(1)
            .stream()
        )
        for row in turns:
            data = row.to_dict() or {}
            data.setdefault("id", row.id)
            return data
        return None

    def update_turn_state(
        self,
        *,
        uid: str,
        chat_id: str,
        turn_id: str,
        state: str,
        detail: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "state": state,
            "updated_at": datetime.now(timezone.utc),
        }
        if detail:
            payload["detail"] = detail
        self._set_turn(uid=uid, chat_id=chat_id, turn_id=turn_id, payload=payload, merge=True)

    def finalize_turn(
        self,
        *,
        uid: str,
        chat_id: str,
        turn_id: str,
        user_message: str,
        model_content: str,
        citations: list[dict[str, Any]] | None,
        new_title: str | None,
        has_file: bool,
    ) -> None:
        append_chat_message(
            uid,
            chat_id,
            role="user",
            content=user_message,
            has_file=has_file,
        )
        append_chat_message(
            uid,
            chat_id,
            role="model",
            content=model_content,
            citations=citations,
        )
        update_chat_metadata(uid, chat_id, title=new_title)
        self._set_turn(
            uid=uid,
            chat_id=chat_id,
            turn_id=turn_id,
            payload={
                "state": "completed",
                "updated_at": datetime.now(timezone.utc),
                "content": model_content,
                "citations": citations,
                "new_title": new_title,
            },
            merge=True,
        )

    def _set_turn(
        self,
        *,
        uid: str,
        chat_id: str,
        turn_id: str,
        payload: dict[str, Any],
        merge: bool = False,
    ) -> None:
        if _skip_firestore():
            existing = _SKIP_TURNS.get((uid, chat_id, str(payload.get("idempotency_key") or "")), {})
            merged = {**existing, **payload} if merge else payload
            key = (
                uid,
                chat_id,
                str(merged.get("idempotency_key") or turn_id),
            )
            _SKIP_TURNS[key] = merged
            return
        self._chat_ref(uid, chat_id).collection("turns").document(turn_id).set(payload, merge=merge)

    def _chat_ref(self, uid: str, chat_id: str):
        if firestore is None:
            raise HTTPException(status_code=500, detail="Firestore client unavailable")
        init_firebase()
        db = firestore.client()
        return db.collection("users").document(uid).collection("chats").document(chat_id)


def _skip_firestore() -> bool:
    return os.environ.get("FAMBOT_SKIP_FIRESTORE", "").strip() == "1"
