# =========================
# 1. IMPORTS
# =========================
import os
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parent
_DATA_CSV = _ROOT / "sources" / "diabetes.csv"
_MODEL_PATH = _ROOT / "diabetes_model.pkl"
_PLOT_PATH = _ROOT / "feature_importance.png"


def main() -> None:
    # =========================
    # 2. LOAD DATA
    # =========================
    df = pd.read_csv(_DATA_CSV)

    print("Initial shape:", df.shape)
    print(df.head())

    # =========================
    # 3. CLEAN DATA (IMPORTANT)
    # =========================

    # Columns where 0 is invalid
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    print("\nAfter cleaning:")
    print(df.describe())

    # =========================
    # 4. FEATURE / LABEL SPLIT
    # =========================
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # =========================
    # 5. TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # 6. BASELINE MODEL
    # =========================
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    # =========================
    # 7. XGBOOST MODEL (BETTER)
    # =========================
    print("\n--- XGBoost ---")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
    )

    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))

    # =========================
    # 8. FEATURE IMPORTANCE
    # =========================
    importances = xgb_model.feature_importances_

    plt.figure()
    plt.barh(X.columns, importances)
    plt.title("Feature Importance (XGBoost)")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(_PLOT_PATH)
    plt.close()
    print(f"\nFeature importance plot saved to {_PLOT_PATH}")

    # =========================
    # 9. OPTIONAL: SAVE MODEL
    # =========================
    import joblib

    joblib.dump(xgb_model, _MODEL_PATH)

    print(f"\nModel saved as {_MODEL_PATH}")


if __name__ == "__main__":
    main()
