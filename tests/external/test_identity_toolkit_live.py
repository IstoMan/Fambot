"""Live Identity Toolkit calls; opt-in only."""

from __future__ import annotations

import os

import pytest

from fambot_backend.services.identity_toolkit import IdentityToolkitError, sign_in_with_password

pytestmark = pytest.mark.external


def _external_enabled() -> bool:
    return os.environ.get("FAMBOT_RUN_EXTERNAL_TESTS") == "1"


@pytest.mark.skipif(not _external_enabled(), reason="Set FAMBOT_RUN_EXTERNAL_TESTS=1 to run external tests")
@pytest.mark.skipif(
    not os.environ.get("FIREBASE_WEB_API_KEY", "").strip(),
    reason="FIREBASE_WEB_API_KEY is required for Identity Toolkit",
)
def test_sign_in_with_password_invalid_credentials_reachable() -> None:
    """Hits Google's API with fake credentials; expects a structured toolkit error."""
    with pytest.raises(IdentityToolkitError) as exc_info:
        sign_in_with_password("no-such-user@example.com", "wrong-password-not-real")
    assert exc_info.value.status_code == 400
    assert "EMAIL_NOT_FOUND" in exc_info.value.message or "INVALID" in exc_info.value.message


@pytest.mark.skipif(not _external_enabled(), reason="Set FAMBOT_RUN_EXTERNAL_TESTS=1 to run external tests")
@pytest.mark.skipif(
    not os.environ.get("FIREBASE_WEB_API_KEY", "").strip(),
    reason="FIREBASE_WEB_API_KEY is required for Identity Toolkit",
)
def test_sign_in_with_password_success_optional() -> None:
    """Optional happy path when disposable test credentials are provided."""
    email = os.environ.get("FAMBOT_EXTERNAL_TEST_EMAIL", "").strip()
    password = os.environ.get("FAMBOT_EXTERNAL_TEST_PASSWORD", "").strip()
    if not email or not password:
        pytest.skip("Set FAMBOT_EXTERNAL_TEST_EMAIL and FAMBOT_EXTERNAL_TEST_PASSWORD for live login")
    data = sign_in_with_password(email, password)
    assert isinstance(data.get("localId"), str) and data["localId"]
