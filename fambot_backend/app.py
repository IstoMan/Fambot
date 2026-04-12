from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fambot_backend.deps import firebase_uid
from fambot_backend.firestore_users import get_user_profile, upsert_onboarding
from fambot_backend.inference import compute_bmi, predict_risk
from fambot_backend.schemas import OnboardingIn, OnboardingOut, UserProfileOut

app = FastAPI(title="Fambot API", version="0.2.0")

_origins = os.environ.get("FAMBOT_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/me", response_model=UserProfileOut)
def read_me(uid: str = Depends(firebase_uid)) -> UserProfileOut:
    return get_user_profile(uid)


@app.put("/v1/me/onboarding", response_model=OnboardingOut)
def complete_onboarding(
    body: OnboardingIn,
    uid: str = Depends(firebase_uid),
) -> OnboardingOut:
    score, rclass = predict_risk(body)
    bmi = compute_bmi(body.height_cm, body.weight_kg)
    profile = upsert_onboarding(uid, body, bmi, score, rclass)
    return OnboardingOut(profile=profile, risk_score=score, risk_class=rclass)


def run() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("fambot_backend.app:app", host=host, port=port, reload=False)
