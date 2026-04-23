"""API smoke tests for public routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.api
def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
