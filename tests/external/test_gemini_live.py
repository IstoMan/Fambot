"""Live Gemini API smoke; opt-in only (cost + network)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gemini_live


def _gemini_live_enabled() -> bool:
    return os.environ.get("FAMBOT_RUN_GEMINI_LIVE") == "1"


@pytest.mark.skipif(not _gemini_live_enabled(), reason="Set FAMBOT_RUN_GEMINI_LIVE=1 to run live Gemini tests")
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY", "").strip(),
    reason="GEMINI_API_KEY is required for live Gemini tests",
)
def test_live_gemini_generate_content_minimal() -> None:
    """One short generation to verify API key and default model id work."""
    from fambot_backend.services.gemini_document_analysis import _get_client, _model_name

    client = _get_client()
    model_name = _model_name()
    response = client.models.generate_content(
        model=model_name,
        contents="Reply with exactly the word OK and nothing else.",
    )
    text = (response.text or "").strip()
    assert text, "Gemini returned empty text"
