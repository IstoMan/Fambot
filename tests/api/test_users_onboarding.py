"""Authenticated /me routes with skip flags (no Firebase)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fambot_backend.cardio_features import Gender


@pytest.mark.api
def test_put_onboarding_returns_profile_and_risk(
    client: TestClient,
    dry_api_env: None,
    clear_model_cache: None,
) -> None:
    body = {
        "age": 42,
        "height_cm": 172,
        "weight_kg": 73,
        "blood_pressure_systolic": 125,
        "blood_pressure_diastolic": 82,
        "gender": "female",
        "cholesterol": 2,
        "gluc_ordinal": 2,
        "smokes": False,
        "drinks_alcohol": None,
        "physically_active": True,
    }
    r = client.put("/me/onboarding", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "risk_score" in data and "risk_class" in data
    assert data["profile"]["uid"]
    assert data["profile"]["onboarding_complete"] is True
    assert data["profile"]["gender"] == Gender.female.value


@pytest.mark.api
def test_put_onboarding_invalid_bp(client: TestClient, dry_api_env: None) -> None:
    r = client.put(
        "/me/onboarding",
        json={
            "age": 40,
            "height_cm": 170,
            "weight_kg": 70,
            "blood_pressure_systolic": 100,
            "blood_pressure_diastolic": 110,
            "gender": "male",
            "cholesterol": 1,
            "gluc_ordinal": 1,
        },
    )
    assert r.status_code == 422
