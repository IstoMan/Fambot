from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from fambot_backend.core.context_builder import build_chat_history_context
from fambot_backend.core.tool_runtime import dispatch_tool
from fambot_backend.persistence.chat_repository import ChatRepository
from fambot_backend.providers.gemini_provider import GeminiProvider
from fambot_backend.providers.model_provider import ProviderContext, ProviderEvent
from fambot_backend.schemas import ChatMessageResponse, ChatTurnState
from fambot_backend.services.gemini_document_analysis import (
    CHAT_ASSISTANT_FALLBACK,
    maybe_new_chat_title,
)
from fambot_backend.telemetry.chat_telemetry import finish_turn_trace, start_turn_trace

_MAX_MESSAGE_CHARS = 12000


@dataclass
class StreamEvent:
    payload: dict[str, Any]
    terminal: bool = False


class ChatOrchestrator:
    def __init__(
        self,
        *,
        repository: ChatRepository | None = None,
        provider: GeminiProvider | None = None,
    ) -> None:
        self.repository = repository or ChatRepository()
        self.provider = provider or GeminiProvider()

    def run_buffered(
        self,
        *,
        uid: str,
        chat_id: str,
        user_message: str,
        upload_name: str | None,
        upload_content_type: str | None,
        upload_payload: bytes | None,
        idempotency_key: str | None,
    ) -> ChatMessageResponse:
        self.repository.require_chat(uid=uid, chat_id=chat_id)
        turn = self.repository.create_turn(
            uid=uid,
            chat_id=chat_id,
            idempotency_key=idempotency_key,
        )
        turn_id = str(turn.get("id") or uuid.uuid4())
        existing = self._existing_response(uid=uid, chat_id=chat_id, turn=turn)
        if existing is not None:
            return existing
        started_at = start_turn_trace(chat_id=chat_id, turn_id=turn_id, uid=uid)
        history = build_chat_history_context(
            messages=self.repository.list_recent_messages(uid=uid, chat_id=chat_id, limit=20),
            max_messages=20,
        )
        self.repository.update_turn_state(
            uid=uid,
            chat_id=chat_id,
            turn_id=turn_id,
            state=ChatTurnState.STREAMING.value,
        )
        text_parts: list[str] = []
        citations: list[dict[str, Any]] | None = None
        try:
            for event in self.provider.stream_turn(
                context=ProviderContext(
                    uid=uid,
                    user_message=_validate_message(user_message),
                    history=history,
                    upload_name=upload_name,
                    upload_content_type=upload_content_type,
                    upload_payload=upload_payload,
                ),
                tool_dispatch=dispatch_tool,
            ):
                if event.kind == "token" and event.text:
                    text_parts.append(event.text)
                if event.kind == "done":
                    citations = event.citations
            full_text = "".join(text_parts).strip() or CHAT_ASSISTANT_FALLBACK
            new_title = maybe_new_chat_title(user_message=user_message, history=history)
            self.repository.finalize_turn(
                uid=uid,
                chat_id=chat_id,
                turn_id=turn_id,
                user_message=user_message,
                model_content=full_text,
                citations=citations if isinstance(citations, list) else None,
                new_title=new_title if isinstance(new_title, str) else None,
                has_file=bool(upload_payload),
            )
            finish_turn_trace(
                chat_id=chat_id,
                turn_id=turn_id,
                uid=uid,
                state=ChatTurnState.COMPLETED.value,
                started_at=started_at,
            )
            return ChatMessageResponse(
                chat_id=chat_id,
                turn_id=turn_id,
                content=full_text,
                citations=citations if isinstance(citations, list) else None,
                new_title=new_title if isinstance(new_title, str) else None,
                state=ChatTurnState.COMPLETED,
            )
        except Exception as exc:
            self.repository.update_turn_state(
                uid=uid,
                chat_id=chat_id,
                turn_id=turn_id,
                state=ChatTurnState.FAILED.value,
                detail=str(exc)[:240],
            )
            finish_turn_trace(
                chat_id=chat_id,
                turn_id=turn_id,
                uid=uid,
                state=ChatTurnState.FAILED.value,
                started_at=started_at,
                extra={"detail": str(exc)[:240]},
            )
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=502, detail=f"Chat analysis failed: {exc}") from exc

    def run_stream(
        self,
        *,
        uid: str,
        chat_id: str,
        user_message: str,
        upload_name: str | None,
        upload_content_type: str | None,
        upload_payload: bytes | None,
        idempotency_key: str | None,
    ) -> Iterator[StreamEvent]:
        self.repository.require_chat(uid=uid, chat_id=chat_id)
        turn = self.repository.create_turn(
            uid=uid,
            chat_id=chat_id,
            idempotency_key=idempotency_key,
        )
        turn_id = str(turn.get("id") or uuid.uuid4())
        existing = self._existing_response(uid=uid, chat_id=chat_id, turn=turn)
        if existing is not None:
            yield StreamEvent(payload=self._event_payload("message_start", chat_id, turn_id, 1))
            yield StreamEvent(
                payload=self._event_payload(
                    "token",
                    chat_id,
                    turn_id,
                    2,
                    {"text": existing.content},
                )
            )
            yield StreamEvent(
                payload=self._event_payload(
                    "message_end",
                    chat_id,
                    turn_id,
                    3,
                    {
                        "state": existing.state.value,
                        "new_title": existing.new_title,
                        "citations": existing.citations,
                    },
                ),
                terminal=True,
            )
            return

        started_at = start_turn_trace(chat_id=chat_id, turn_id=turn_id, uid=uid)
        history = build_chat_history_context(
            messages=self.repository.list_recent_messages(uid=uid, chat_id=chat_id, limit=20),
            max_messages=20,
        )
        self.repository.update_turn_state(
            uid=uid,
            chat_id=chat_id,
            turn_id=turn_id,
            state=ChatTurnState.STREAMING.value,
        )
        sequence = 0
        emitted = False
        text_parts: list[str] = []
        citations: list[dict[str, Any]] | None = None
        yield StreamEvent(payload=self._event_payload("message_start", chat_id, turn_id, sequence := sequence + 1))
        try:
            for event in self.provider.stream_turn(
                context=ProviderContext(
                    uid=uid,
                    user_message=_validate_message(user_message),
                    history=history,
                    upload_name=upload_name,
                    upload_content_type=upload_content_type,
                    upload_payload=upload_payload,
                ),
                tool_dispatch=dispatch_tool,
            ):
                emitted = True
                sequence += 1
                if event.kind == "token" and event.text:
                    text_parts.append(event.text)
                    yield StreamEvent(
                        payload=self._event_payload(
                            "token",
                            chat_id,
                            turn_id,
                            sequence,
                            {"text": event.text},
                        )
                    )
                    continue
                if event.kind == "tool_call":
                    yield StreamEvent(
                        payload=self._event_payload(
                            "tool_call",
                            chat_id,
                            turn_id,
                            sequence,
                            {
                                "name": event.tool_name,
                                "args": event.tool_args,
                                "tool_call_id": event.tool_call_id,
                            },
                        )
                    )
                    continue
                if event.kind == "tool_result":
                    yield StreamEvent(
                        payload=self._event_payload(
                            "tool_result",
                            chat_id,
                            turn_id,
                            sequence,
                            {
                                "name": event.tool_name,
                                "result": event.tool_args,
                                "tool_call_id": event.tool_call_id,
                            },
                        )
                    )
                    continue
                if event.kind == "done":
                    citations = event.citations if isinstance(event.citations, list) else None
                    if citations:
                        for citation in citations:
                            sequence += 1
                            yield StreamEvent(
                                payload=self._event_payload(
                                    "citation",
                                    chat_id,
                                    turn_id,
                                    sequence,
                                    {"citation": citation},
                                )
                            )

            full_text = "".join(text_parts).strip() or CHAT_ASSISTANT_FALLBACK
            new_title = maybe_new_chat_title(user_message=user_message, history=history)
            self.repository.finalize_turn(
                uid=uid,
                chat_id=chat_id,
                turn_id=turn_id,
                user_message=user_message,
                model_content=full_text,
                citations=citations if isinstance(citations, list) else None,
                new_title=new_title if isinstance(new_title, str) else None,
                has_file=bool(upload_payload),
            )
            finish_turn_trace(
                chat_id=chat_id,
                turn_id=turn_id,
                uid=uid,
                state=ChatTurnState.COMPLETED.value,
                started_at=started_at,
            )
            yield StreamEvent(
                payload=self._event_payload(
                    "message_end",
                    chat_id,
                    turn_id,
                    sequence + 1,
                    {
                        "state": ChatTurnState.COMPLETED.value,
                        "new_title": new_title if isinstance(new_title, str) else None,
                        "citations": citations if isinstance(citations, list) else None,
                    },
                ),
                terminal=True,
            )
        except GeneratorExit:
            self.repository.update_turn_state(
                uid=uid,
                chat_id=chat_id,
                turn_id=turn_id,
                state=ChatTurnState.CANCELLED.value,
                detail="Client disconnected",
            )
            finish_turn_trace(
                chat_id=chat_id,
                turn_id=turn_id,
                uid=uid,
                state=ChatTurnState.CANCELLED.value,
                started_at=started_at,
            )
            raise
        except Exception as exc:
            self.repository.update_turn_state(
                uid=uid,
                chat_id=chat_id,
                turn_id=turn_id,
                state=ChatTurnState.FAILED.value,
                detail=str(exc)[:240],
            )
            finish_turn_trace(
                chat_id=chat_id,
                turn_id=turn_id,
                uid=uid,
                state=ChatTurnState.FAILED.value,
                started_at=started_at,
                extra={"detail": str(exc)[:240]},
            )
            if not emitted and isinstance(exc, HTTPException):
                raise
            yield StreamEvent(
                payload=self._event_payload(
                    "error",
                    chat_id,
                    turn_id,
                    sequence + 1,
                    {"detail": _error_detail(exc)},
                ),
                terminal=True,
            )

    def _existing_response(
        self,
        *,
        uid: str,
        chat_id: str,
        turn: dict[str, Any],
    ) -> ChatMessageResponse | None:
        state = str(turn.get("state") or "")
        content = turn.get("content")
        if state != ChatTurnState.COMPLETED.value or not isinstance(content, str):
            return None
        turn_id = str(turn.get("id") or "")
        return ChatMessageResponse(
            chat_id=chat_id,
            turn_id=turn_id,
            content=content,
            citations=turn.get("citations") if isinstance(turn.get("citations"), list) else None,
            new_title=turn.get("new_title") if isinstance(turn.get("new_title"), str) else None,
            state=ChatTurnState.COMPLETED,
        )

    def _event_payload(
        self,
        event_type: str,
        chat_id: str,
        turn_id: str,
        sequence: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": event_type,
            "chatId": chat_id,
            "turnId": turn_id,
            "sequence": sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        return payload


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, HTTPException) and isinstance(exc.detail, str):
        return exc.detail
    return str(exc)


def _validate_message(message: str) -> str:
    cleaned = message.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="message cannot be empty")
    if len(cleaned) > _MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=422, detail="message too long")
    return cleaned
