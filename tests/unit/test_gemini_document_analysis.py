"""Gemini orchestration with client and storage I/O mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from fambot_backend.schemas import UserProfileOut
from fambot_backend.services import gemini_document_analysis as gda


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis._get_client")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes", return_value="fake-file-ref")
def test_analyze_uploaded_document_returns_model_and_analysis(
    _upload: MagicMock,
    get_client: MagicMock,
    get_profile: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", onboarding_complete=False)

    gen = MagicMock()

    def fake_generate(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text="Eat well. Follow up with your clinician.")

    gen.models.generate_content.side_effect = fake_generate
    get_client.return_value = gen

    out = gda.analyze_uploaded_document(
        uid="u1",
        file_name="r.pdf",
        content_type="application/pdf",
        payload=b"%PDF",
    )
    assert out["model"]
    assert "Eat well" in out["analysis"]
    gen.models.generate_content.assert_called_once()


class _FakeModels:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self.calls = 0

    def generate_content(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


class _FakeGenaiClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.models = _FakeModels(responses)


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_file_search_store_name", return_value=None)
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis.list_user_documents", return_value=[])
@patch("fambot_backend.services.gemini_document_analysis._get_client")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes", return_value="fake")
def test_generate_chat_turn_returns_content_and_calls_model(
    _upload: MagicMock,
    get_client: MagicMock,
    _list_docs: MagicMock,
    get_profile: MagicMock,
    _fs_store: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", age=40, onboarding_complete=True)
    # First call: main reply; optional second call: title when no prior user in history
    get_client.return_value = _FakeGenaiClient([SimpleNamespace(text="  Here is help.  ")])

    out = gda.generate_chat_turn(
        uid="u1",
        user_message="What about BP?",
        history=[{"role": "user", "content": "Hi", "created_at": None}],
    )
    assert out["content"] == "Here is help."
    assert out.get("citations") is None


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_file_search_store_name", return_value=None)
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis.list_user_documents", return_value=[])
@patch("fambot_backend.services.gemini_document_analysis._get_client")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes", return_value="fake")
def test_generate_chat_turn_empty_response_uses_fallback(
    _upload: MagicMock,
    get_client: MagicMock,
    _list_docs: MagicMock,
    get_profile: MagicMock,
    _fs_store: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", onboarding_complete=False)
    get_client.return_value = _FakeGenaiClient(
        [SimpleNamespace(text="   "), SimpleNamespace(text="t")]
    )

    out = gda.generate_chat_turn(uid="u1", user_message="x", history=[])
    assert "try again" in out["content"].lower()


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes")
@patch("fambot_backend.services.gemini_document_analysis._get_client")
def test_analyze_uploaded_document_empty_gemini_raises(
    get_client: MagicMock,
    upload_bytes: MagicMock,
    get_profile: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", onboarding_complete=False)
    upload_bytes.return_value = "ref"
    gen = MagicMock()
    gen.models.generate_content.return_value = SimpleNamespace(text="")
    get_client.return_value = gen

    with pytest.raises(HTTPException) as ei:
        gda.analyze_uploaded_document(
            uid="u1",
            file_name="x.pdf",
            content_type="application/pdf",
            payload=b"x",
        )
    assert ei.value.status_code == 502
