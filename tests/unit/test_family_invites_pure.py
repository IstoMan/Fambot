"""Unit tests for invite URL, QR, and TTL helpers (no Firestore)."""

from __future__ import annotations

import base64

import pytest

from fambot_backend.services.family_invites import (
    _invite_ttl_seconds,
    build_invite_url,
    qr_png_base64_for_url,
)


@pytest.mark.unit
def test_build_invite_url_default_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAMBOT_INVITE_BASE_URL", raising=False)
    url = build_invite_url("tok_123")
    assert url == "fambot://family-invite?token=tok_123"


@pytest.mark.unit
def test_build_invite_url_with_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMBOT_INVITE_BASE_URL", "https://app.example.com/join")
    assert "token=tok" in build_invite_url("tok")
    assert build_invite_url("tok").startswith("https://app.example.com/join")


@pytest.mark.unit
def test_build_invite_url_base_has_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMBOT_INVITE_BASE_URL", "https://app.example.com/join?x=1")
    u = build_invite_url("abc")
    assert "&token=abc" in u or "?token=abc" in u


@pytest.mark.unit
def test_qr_png_base64_is_png() -> None:
    raw = base64.b64decode(qr_png_base64_for_url("https://example.com/invite?token=x"))
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_invite_ttl_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMBOT_FAMILY_INVITE_TTL_SECONDS", "30")
    assert _invite_ttl_seconds() == 60
    monkeypatch.setenv("FAMBOT_FAMILY_INVITE_TTL_SECONDS", "999999999")
    assert _invite_ttl_seconds() == 60 * 60 * 24 * 30
    monkeypatch.setenv("FAMBOT_FAMILY_INVITE_TTL_SECONDS", "not-a-number")
    assert _invite_ttl_seconds() == 86400
