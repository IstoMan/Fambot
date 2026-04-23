"""Unit tests for cardiovascular feature layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fambot_backend.cardio_features import (
    FEATURE_ORDER,
    Gender,
    build_feature_frame,
    gender_to_dataset_code,
)


@pytest.mark.unit
def test_gender_to_dataset_code_enum_and_str() -> None:
    assert gender_to_dataset_code(Gender.female) == 1
    assert gender_to_dataset_code(Gender.male) == 2
    assert gender_to_dataset_code("female") == 1
    assert gender_to_dataset_code("male") == 2


@pytest.mark.unit
def test_gender_to_dataset_code_invalid() -> None:
    with pytest.raises(ValueError):
        gender_to_dataset_code("other")  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_feature_frame_columns_and_bp_validation() -> None:
    with pytest.raises(ValueError, match="Systolic"):
        build_feature_frame(
            age=40,
            height_cm=170,
            weight_kg=75,
            blood_pressure_systolic=110,
            blood_pressure_diastolic=120,
            gender=Gender.male,
            cholesterol=1,
            gluc_ordinal=1,
        )

    df = build_feature_frame(
        age=45,
        height_cm=180,
        weight_kg=80,
        blood_pressure_systolic=130,
        blood_pressure_diastolic=85,
        gender="female",
        cholesterol=2,
        gluc_ordinal=2,
        smokes=True,
        drinks_alcohol=False,
        physically_active=None,
    )
    assert list(df.columns) == FEATURE_ORDER
    assert df.shape == (1, len(FEATURE_ORDER))
    assert df.loc[0, "age_years"] == 45.0
    assert df.loc[0, "gender"] == 1.0
    assert df.loc[0, "smoke"] == 1.0
    assert df.loc[0, "alco"] == 0.0
    assert pd.isna(df.loc[0, "active"])


@pytest.mark.unit
def test_build_feature_frame_optional_bools_nan() -> None:
    df = build_feature_frame(
        age=30,
        height_cm=175,
        weight_kg=70,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        gender=Gender.male,
        cholesterol=1,
        gluc_ordinal=1,
        smokes=None,
        drinks_alcohol=None,
        physically_active=None,
    )
    assert np.isnan(df.loc[0, "smoke"])
    assert np.isnan(df.loc[0, "alco"])
    assert np.isnan(df.loc[0, "active"])
