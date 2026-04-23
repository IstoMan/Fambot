"""Auth routes with Firebase / Identity Toolkit mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fambot_backend.services.identity_toolkit import IdentityToolkitError


@pytest.mark.api
@patch("fambot_backend.api.routers.auth.ensure_user_document")
@patch("fambot_backend.api.routers.auth.auth.create_user")
@patch("fambot_backend.api.routers.auth.init_firebase")
def test_signup_success(
    _init: object,
    create_user: object,
    _ensure: object,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    create_user.return_value = SimpleNamespace(uid="new-firebase-uid")
    r = client.post(
        "/auth/signup",
        json={"email": "new@example.com", "password": "longenough", "name": "Test User"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["uid"] == "new-firebase-uid"
    assert data["token_type"] == "bearer"
    assert data["access_token"]
    create_user.assert_called_once()


@pytest.mark.api
@patch("fambot_backend.api.routers.auth.sign_in_with_password")
@patch("fambot_backend.api.routers.auth.init_firebase")
def test_login_success(
    _init: object,
    sign_in: object,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    sign_in.return_value = {"localId": "login-uid", "email": "login@example.com"}
    r = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "secret"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["uid"] == "login-uid"


@pytest.mark.api
@patch("fambot_backend.api.routers.auth.sign_in_with_password")
@patch("fambot_backend.api.routers.auth.init_firebase")
def test_login_invalid_password_maps_to_401(
    _init: object,
    sign_in: object,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    sign_in.side_effect = IdentityToolkitError(400, "INVALID_PASSWORD")
    r = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "wrong"},
    )
    assert r.status_code == 401


@pytest.mark.api
@patch("fambot_backend.api.routers.auth.sign_in_with_password")
@patch("fambot_backend.api.routers.auth.init_firebase")
def test_login_identity_toolkit_502(
    _init: object,
    sign_in: object,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    sign_in.side_effect = IdentityToolkitError(503, "UNAVAILABLE")
    r = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "secret"},
    )
    assert r.status_code == 502
