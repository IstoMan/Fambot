"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fambot_backend.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def dry_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Firebase I/O; fixed JWT secret for tests that use skip auth."""
    monkeypatch.setenv("FAMBOT_JWT_SECRET", "test-jwt-secret-for-pytest-32bytes!")
    monkeypatch.setenv("FAMBOT_SKIP_AUTH", "1")
    monkeypatch.setenv("FAMBOT_SKIP_FIRESTORE", "1")


@pytest.fixture
def reset_family_invite_skip_state() -> None:
    """Clear in-memory family group state when FAMBOT_SKIP_FIRESTORE=1."""
    from fambot_backend.services import family_invites as fi

    fi._skip_groups.clear()
    fi._skip_user_to_group.clear()
    fi._skip_invites.clear()
    yield
    fi._skip_groups.clear()
    fi._skip_user_to_group.clear()
    fi._skip_invites.clear()


@pytest.fixture
def clear_model_cache() -> None:
    from fambot_backend.services import inference

    inference._load_model.cache_clear()
    yield
    inference._load_model.cache_clear()
