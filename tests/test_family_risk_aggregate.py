from __future__ import annotations

import pytest

from fambot_backend.schemas import UserProfileOut
from fambot_backend.services.family_risk_aggregate import (
    RISK_SCORE_HIGH_MIN,
    compute_family_risk_feature_row,
    neutral_family_features,
)


@pytest.mark.unit
def test_neutral_when_no_peers(monkeypatch) -> None:
    monkeypatch.setattr(
        "fambot_backend.services.family_risk_aggregate.family_peers_for_scoring",
        lambda _uid: [],
    )
    row = compute_family_risk_feature_row("self")
    assert row == neutral_family_features()


@pytest.mark.unit
def test_weighted_aggregates_use_member_scores(monkeypatch) -> None:
    monkeypatch.setattr(
        "fambot_backend.services.family_risk_aggregate.family_peers_for_scoring",
        lambda _uid: [("m1", "mother"), ("m2", "nephew")],
    )

    def fake_profile(uid: str) -> UserProfileOut:
        if uid == "m1":
            return UserProfileOut(
                uid=uid,
                onboarding_complete=True,
                risk_score=60.0,
                risk_class="moderate",
            )
        if uid == "m2":
            return UserProfileOut(
                uid=uid,
                onboarding_complete=True,
                risk_score=40.0,
                risk_class="low",
            )
        return UserProfileOut(uid=uid)

    monkeypatch.setattr(
        "fambot_backend.services.family_risk_aggregate.get_user_profile",
        fake_profile,
    )

    row = compute_family_risk_feature_row("self")
    # mother weight 1.0, nephew 0.5 -> (60 + 0.5*40) / 1.5 = 53.333...
    assert row["fam_weighted_mean_risk"] is not None
    assert abs(row["fam_weighted_mean_risk"] - 160.0 / 3.0) < 1e-6
    assert row["fam_max_member_risk"] == 60.0
    assert row["fam_first_deg_mean_risk"] == 60.0
    assert row["fam_any_member_high_risk"] == 0.0


@pytest.mark.unit
def test_high_flag_when_member_above_threshold(monkeypatch) -> None:
    monkeypatch.setattr(
        "fambot_backend.services.family_risk_aggregate.family_peers_for_scoring",
        lambda _uid: [("m1", "sister")],
    )
    monkeypatch.setattr(
        "fambot_backend.services.family_risk_aggregate.get_user_profile",
        lambda uid: UserProfileOut(
            uid=uid,
            onboarding_complete=True,
            risk_score=RISK_SCORE_HIGH_MIN,
            risk_class="high",
        ),
    )
    row = compute_family_risk_feature_row("self")
    assert row["fam_any_member_high_risk"] == 1.0
