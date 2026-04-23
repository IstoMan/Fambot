from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile

from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import (
    ChatCreateRequest,
    ChatInteractionResponse,
    ChatResponse,
    MessageResponse,
)
from fambot_backend.services.chat_history import (
    append_chat_message,
    create_chat,
    get_chat,
    list_chat_messages,
    list_chats as list_chat_sessions,
    update_chat_metadata,
)
from fambot_backend.services.gemini_document_analysis import generate_chat_turn

router = APIRouter(tags=["chats"])


@router.post("/chat/new", response_model=ChatResponse)
def create_chat_session(
    body: ChatCreateRequest,
    uid: str = Depends(firebase_uid),
) -> ChatResponse:
    chat_id = body.chat_id or str(uuid.uuid4())
    payload = create_chat(uid, chat_id=chat_id, title=body.title)
    return ChatResponse(
        id=chat_id,
        title=str(payload.get("title") or "New Chat"),
        created_at=_as_dt(payload.get("created_at")),
        last_updated=_as_dt(payload.get("last_updated")),
    )


@router.get("/chats", response_model=list[ChatResponse])
def list_chats(uid: str = Depends(firebase_uid)) -> list[ChatResponse]:
    payloads = list_chat_sessions(uid)
    out: list[ChatResponse] = []
    now = datetime.now(timezone.utc)
    for item in payloads:
        out.append(
            ChatResponse(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or "New Chat"),
                created_at=_as_dt(item.get("created_at"), now),
                last_updated=_as_dt(item.get("last_updated"), now),
            )
        )
    return out


@router.post("/chat/{chat_id}", response_model=ChatInteractionResponse)
def chat_interaction(
    chat_id: str,
    message: str = Form(...),
    file: UploadFile | None = File(None),
    uid: str = Depends(firebase_uid),
) -> ChatInteractionResponse:
    get_chat_or_404 = get_chat(uid, chat_id)
    if not get_chat_or_404:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Chat not found")

    file_payload: bytes | None = None
    file_name: str | None = None
    file_content_type: str | None = None
    if file is not None:
        file_name = file.filename
        file_content_type = file.content_type or "application/octet-stream"
        file_payload = file.file.read()

    history = list_chat_messages(uid, chat_id, limit=20)
    ai_result = generate_chat_turn(
        uid=uid,
        user_message=message,
        history=history,
        upload_name=file_name,
        upload_content_type=file_content_type,
        upload_payload=file_payload,
    )
    ai_text = str(ai_result.get("content") or "").strip()
    citations = ai_result.get("citations")
    new_title = ai_result.get("new_title")

    append_chat_message(
        uid,
        chat_id,
        role="user",
        content=message,
        has_file=bool(file_payload),
    )
    append_chat_message(
        uid,
        chat_id,
        role="model",
        content=ai_text,
        citations=citations if isinstance(citations, list) else None,
    )
    update_chat_metadata(
        uid,
        chat_id,
        title=new_title if isinstance(new_title, str) else None,
    )
    return ChatInteractionResponse(
        role="model",
        content=ai_text,
        citations=citations if isinstance(citations, list) else None,
        new_title=new_title if isinstance(new_title, str) else None,
    )


@router.get("/chat/{chat_id}/history", response_model=list[MessageResponse])
def get_history(chat_id: str, uid: str = Depends(firebase_uid)) -> list[MessageResponse]:
    payloads = list_chat_messages(uid, chat_id)
    now = datetime.now(timezone.utc)
    out: list[MessageResponse] = []
    for item in payloads:
        citations = item.get("citations")
        out.append(
            MessageResponse(
                role=str(item.get("role") or "unknown"),
                content=str(item.get("content") or ""),
                created_at=_as_dt(item.get("created_at"), now),
                citations=citations if isinstance(citations, list) else None,
                has_file=item.get("has_file") if isinstance(item.get("has_file"), bool) else None,
            )
        )
    return out


def _as_dt(raw: object, fallback: datetime | None = None) -> datetime:
    if isinstance(raw, datetime):
        return raw
    return fallback or datetime.now(timezone.utc)
