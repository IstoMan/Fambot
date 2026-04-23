"""Unit tests for JWT minting and verification."""

from __future__ import annotations

import jwt
import pytest

from fambot_backend.core import jwt_tokens


@pytest.mark.unit
def test_mint_and_decode_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "unit-test-secret-value-32chars!!")
    monkeypatch.delenv("FAMBOT_JWT_EXPIRES_SECONDS", raising=False)
    token, exp = jwt_tokens.mint_access_token("uid-abc", "u@example.com")
    assert exp == jwt_tokens.expires_seconds()
    claims = jwt_tokens.decode_and_verify(token)
    assert claims["sub"] == "uid-abc"
    assert claims["email"] == "u@example.com"
    assert "exp" in claims and "iat" in claims


@pytest.mark.unit
def test_expires_seconds_clamp_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMBOT_JWT_EXPIRES_SECONDS", "30")
    assert jwt_tokens.expires_seconds() == 60


@pytest.mark.unit
def test_mint_without_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAMBOT_JWT_SECRET", raising=False)
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "")
    with pytest.raises(ValueError, match="FAMBOT_JWT_SECRET"):
        jwt_tokens.mint_access_token("x", None)


@pytest.mark.unit
def test_decode_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "same-secret-for-decode-test-32!")
    with pytest.raises(jwt.exceptions.InvalidTokenError):
        jwt_tokens.decode_and_verify("not-a-jwt")
