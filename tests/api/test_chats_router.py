"""Chat routes with chat_history and Gemini helpers mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


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
@patch("fambot_backend.api.routers.chats.get_chat")
def test_chat_interaction_not_found(
    get_chat: MagicMock,
    client: TestClient,
    dry_api_env: None,
) -> None:
    get_chat.return_value = None
    r = client.post(
        "/chat/missing",
        data={"message": "hello"},
    )
    assert r.status_code == 404


@pytest.mark.api
@patch("fambot_backend.api.routers.chats.update_chat_metadata")
@patch("fambot_backend.api.routers.chats.append_chat_message")
@patch("fambot_backend.api.routers.chats.generate_chat_turn")
@patch("fambot_backend.api.routers.chats.list_chat_messages")
@patch("fambot_backend.api.routers.chats.get_chat")
def test_chat_interaction_success(
    get_chat: MagicMock,
    list_messages: MagicMock,
    gen_turn: MagicMock,
    append_msg: MagicMock,
    update_meta: MagicMock,
    client: TestClient,
    dry_api_env: None,
) -> None:
    get_chat.return_value = {"id": "c1", "title": "T"}
    list_messages.return_value = [{"role": "user", "content": "hi", "created_at": None}]
    gen_turn.return_value = {
        "content": "Hello back",
        "citations": [{"source": "x"}],
        "new_title": "Greeting",
    }

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

    gen_turn.assert_called_once()
    call_kw = gen_turn.call_args.kwargs
    assert call_kw["uid"] == "dev-user"
    assert call_kw["user_message"] == "hello there"
    assert call_kw["history"] == list_messages.return_value
    assert call_kw["upload_payload"] is None

    assert append_msg.call_count == 2
    update_meta.assert_called_once()


@pytest.mark.api
@patch("fambot_backend.api.routers.chats.update_chat_metadata")
@patch("fambot_backend.api.routers.chats.append_chat_message")
@patch("fambot_backend.api.routers.chats.generate_chat_turn")
@patch("fambot_backend.api.routers.chats.list_chat_messages")
@patch("fambot_backend.api.routers.chats.get_chat")
def test_chat_interaction_with_file(
    get_chat: MagicMock,
    list_messages: MagicMock,
    gen_turn: MagicMock,
    append_msg: MagicMock,
    update_meta: MagicMock,
    client: TestClient,
    dry_api_env: None,
) -> None:
    get_chat.return_value = {"id": "c2", "title": "T"}
    list_messages.return_value = []
    gen_turn.return_value = {"content": "ok", "citations": None, "new_title": None}

    r = client.post(
        "/chat/c2",
        data={"message": "see file"},
        files={"file": ("note.txt", BytesIO(b"data"), "text/plain")},
    )
    assert r.status_code == 200, r.text
    kw = gen_turn.call_args.kwargs
    assert kw["upload_name"] == "note.txt"
    assert kw["upload_content_type"] == "text/plain"
    assert kw["upload_payload"] == b"data"
    assert append_msg.call_args_list[0].kwargs["has_file"] is True


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
