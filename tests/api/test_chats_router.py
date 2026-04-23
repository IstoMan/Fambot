"""Chat routes with chat_history and Gemini helpers mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fambot_backend.schemas import ChatMessageResponse, ChatTurnState


@pytest.mark.api
@patch("fambot_backend.api.routers.chats.create_chat")
def test_create_chat_session(
    create_chat: MagicMock,
    client: TestClient,
    dry_api_env: None,
) -> None:
    now = datetime.now(timezone.utc)
    create_chat.return_value = {
        "id": "chat-1",
        "title": "My Chat",
        "created_at": now,
        "last_updated": now,
    }
    r = client.post("/chat/new", json={"chat_id": "chat-1", "title": "My Chat"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["id"] == "chat-1"
    assert data["title"] == "My Chat"
    create_chat.assert_called_once()


@pytest.mark.api
@patch("fambot_backend.api.routers.chats.list_chat_sessions")
def test_list_chats(
    list_sessions: MagicMock,
    client: TestClient,
    dry_api_env: None,
) -> None:
    now = datetime.now(timezone.utc)
    list_sessions.return_value = [
        {"id": "a", "title": "One", "created_at": now, "last_updated": now},
    ]
    r = client.get("/chats")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["id"] == "a"
    assert items[0]["title"] == "One"


@pytest.mark.api
def test_chat_interaction_not_found(
    client: TestClient,
    dry_api_env: None,
) -> None:
    with patch(
        "fambot_backend.api.routers.chats._orchestrator.run_buffered",
        side_effect=HTTPException(status_code=404, detail="Chat not found"),
    ):
        r = client.post(
            "/chat/missing",
            data={"message": "hello"},
        )
    assert r.status_code == 404


@pytest.mark.api
def test_chat_interaction_success(
    client: TestClient,
    dry_api_env: None,
) -> None:
    with patch(
        "fambot_backend.api.routers.chats._orchestrator.run_buffered",
        return_value=ChatMessageResponse(
            chat_id="c1",
            turn_id="t1",
            content="Hello back",
            citations=[{"source": "x"}],
            new_title="Greeting",
            state=ChatTurnState.COMPLETED,
        ),
    ) as run_buffered:
        r = client.post(
            "/chat/c1",
            data={"message": "hello there"},
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["role"] == "model"
    assert data["content"] == "Hello back"
    assert data["citations"] == [{"source": "x"}]
    assert data["new_title"] == "Greeting"
    run_buffered.assert_called_once()


@pytest.mark.api
def test_chat_interaction_with_file(
    client: TestClient,
    dry_api_env: None,
) -> None:
    with patch(
        "fambot_backend.api.routers.chats._orchestrator.run_buffered",
        return_value=ChatMessageResponse(
            chat_id="c2",
            turn_id="t2",
            content="ok",
            citations=None,
            new_title=None,
            state=ChatTurnState.COMPLETED,
        ),
    ) as run_buffered:
        r = client.post(
            "/chat/c2",
            data={"message": "see file"},
            files={"file": ("note.txt", BytesIO(b"data"), "text/plain")},
        )
    assert r.status_code == 200, r.text
    kw = run_buffered.call_args.kwargs
    assert kw["upload_name"] == "note.txt"
    assert kw["upload_content_type"] == "text/plain"
    assert kw["upload_payload"] == b"data"


@pytest.mark.api
def test_chat_message_v1_json(
    client: TestClient,
    dry_api_env: None,
) -> None:
    with patch(
        "fambot_backend.api.routers.chats._orchestrator.run_buffered",
        return_value=ChatMessageResponse(
            chat_id="c9",
            turn_id="turn-9",
            content="stream compatible",
            citations=None,
            new_title=None,
            state=ChatTurnState.COMPLETED,
        ),
    ):
        r = client.post("/v1/chats/c9/messages", data={"message": "hi"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["chat_id"] == "c9"
    assert payload["turn_id"] == "turn-9"
    assert payload["content"] == "stream compatible"


@pytest.mark.api
@patch("fambot_backend.api.routers.chats.list_chat_messages")
def test_get_chat_history(
    list_messages: MagicMock,
    client: TestClient,
    dry_api_env: None,
) -> None:
    now = datetime.now(timezone.utc)
    list_messages.return_value = [
        {"role": "user", "content": "a", "created_at": now, "has_file": False},
    ]
    r = client.get("/chat/c3/history")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "a"
