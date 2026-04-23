"""Regression check: committed pipeline vs cardio_train holdout (same split as model.py)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model import _clean_cardio_xy

# Measured locally with sklearn 1.8 / committed pkl: ~0.87 ROC-AUC on this holdout.
# Floor is conservative for dependency drift; raise only if the model is intentionally replaced.
_MIN_HOLDOUT_ROC_AUC = 0.74


@pytest.mark.unit
def test_committed_model_holdout_roc_auc() -> None:
    raw = pd.read_csv(Path("sources/cardio_train.csv"), sep=";")
    df = raw.drop(columns=["id"], errors="ignore")
    X, y = _clean_cardio_xy(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = joblib.load(Path("cardiovascular_model.pkl"))
    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    assert auc >= _MIN_HOLDOUT_ROC_AUC, f"holdout ROC-AUC {auc:.4f} below floor {_MIN_HOLDOUT_ROC_AUC}"
