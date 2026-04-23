"""API tests for SSE chat streaming."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fambot_backend.services.gemini_document_analysis import CHAT_ASSISTANT_FALLBACK


def _stream_hello(*args: object, **kwargs: object) -> object:
    yield "Hel"
    yield "lo"


def _stream_empty(*args: object, **kwargs: object) -> object:
    yield from ()


@pytest.mark.api
def test_chat_stream_returns_sse_text_and_done(
    client: TestClient,
    dry_api_env: None,
) -> None:
    with (
        patch(
            "fambot_backend.api.routers.chats.get_chat",
            return_value={"id": "c1", "title": "T"},
        ),
        patch("fambot_backend.api.routers.chats.list_chat_messages", return_value=[]),
        patch(
            "fambot_backend.api.routers.chats.generate_chat_turn_stream",
            side_effect=_stream_hello,
        ),
        patch("fambot_backend.api.routers.chats.maybe_new_chat_title", return_value=None),
        patch("fambot_backend.api.routers.chats.append_chat_message") as mock_append,
        patch("fambot_backend.api.routers.chats.update_chat_metadata") as mock_update,
    ):
        r = client.post("/chat/c1/stream", data={"message": "hi"})

    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    assert r.headers.get("cache-control") == "no-cache"
    body = r.text
    assert "data:" in body
    assert '"type"' in body and "text" in body
    assert "Hel" in body and "lo" in body
    assert "done" in body
    assert mock_append.call_count == 2
    mock_update.assert_called_once()


@pytest.mark.api
def test_chat_stream_no_chunks_uses_assistant_fallback(
    client: TestClient,
    dry_api_env: None,
) -> None:
    with (
        patch(
            "fambot_backend.api.routers.chats.get_chat",
            return_value={"id": "c1", "title": "T"},
        ),
        patch("fambot_backend.api.routers.chats.list_chat_messages", return_value=[]),
        patch(
            "fambot_backend.api.routers.chats.generate_chat_turn_stream",
            side_effect=_stream_empty,
        ),
        patch("fambot_backend.api.routers.chats.maybe_new_chat_title", return_value=None),
        patch("fambot_backend.api.routers.chats.append_chat_message"),
        patch("fambot_backend.api.routers.chats.update_chat_metadata"),
    ):
        r = client.post("/chat/c1/stream", data={"message": "hi"})

    assert r.status_code == 200
    assert CHAT_ASSISTANT_FALLBACK in r.text
