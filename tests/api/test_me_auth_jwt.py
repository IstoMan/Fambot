"""JWT Bearer auth path for protected routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fambot_backend.core.jwt_tokens import mint_access_token


@pytest.mark.api
def test_me_missing_token_returns_401(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    monkeypatch.delenv("FAMBOT_SKIP_AUTH", raising=False)
    r = client.get("/me")
    assert r.status_code == 401


@pytest.mark.api
def test_me_with_valid_jwt(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    monkeypatch.delenv("FAMBOT_SKIP_AUTH", raising=False)
    monkeypatch.setenv("FAMBOT_SKIP_FIRESTORE", "1")
    token, _ = mint_access_token("jwt-test-user", "jwt-test@example.com")
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["uid"] == "jwt-test-user"
