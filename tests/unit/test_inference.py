"""Tests for inference helpers and predict_risk using the committed model artifact."""

from __future__ import annotations

import pytest

from fambot_backend.cardio_features import Gender
from fambot_backend.schemas import OnboardingIn
from fambot_backend.services.inference import compute_bmi, predict_risk


@pytest.mark.unit
def test_compute_bmi() -> None:
    bmi = compute_bmi(height_cm=180, weight_kg=81)
    assert abs(bmi - 25.0) < 0.01


@pytest.mark.unit
def test_predict_risk_uses_committed_model(
    clear_model_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    payload = OnboardingIn(
        age=50,
        height_cm=175,
        weight_kg=78,
        blood_pressure_systolic=140,
        blood_pressure_diastolic=90,
        gender=Gender.male,
        cholesterol=2,
        gluc_ordinal=2,
        smokes=False,
        drinks_alcohol=False,
        physically_active=True,
    )
    score, rclass = predict_risk(payload)
    assert 0.0 <= score <= 100.0
    assert rclass in ("low", "moderate", "high")
