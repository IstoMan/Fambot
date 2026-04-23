"""Family invite flow using in-memory state (FAMBOT_SKIP_FIRESTORE=1) and real JWTs."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fambot_backend.core.jwt_tokens import mint_access_token


@pytest.fixture(autouse=True)
def _family_skip_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    monkeypatch.setenv("FAMBOT_SKIP_FIRESTORE", "1")
    monkeypatch.delenv("FAMBOT_SKIP_AUTH", raising=False)


@pytest.mark.api
def test_create_invite_accept_and_list_family(
    client: TestClient,
    reset_family_invite_skip_state: None,
) -> None:
    owner_t, _ = mint_access_token("owner-user-1", "o@example.com")
    invitee_t, _ = mint_access_token("invitee-user-1", "i@example.com")

    cr = client.post(
        "/me/family/invitations",
        json={"target_role": "son"},
        headers={"Authorization": f"Bearer {owner_t}"},
    )
    assert cr.status_code == 200, cr.text
    token = cr.json()["token"]

    ar = client.post(
        "/me/family/invitations/accept",
        json={"token": token},
        headers={"Authorization": f"Bearer {invitee_t}"},
    )
    assert ar.status_code == 200, ar.text
    assert ar.json()["group_id"]

    gr = client.get(
        "/me/family",
        headers={"Authorization": f"Bearer {invitee_t}"},
    )
    assert gr.status_code == 200, gr.text
    fam = gr.json()
    assert fam["owner_uid"] == "owner-user-1"
    assert any(m["uid"] == "owner-user-1" for m in fam["members"])
