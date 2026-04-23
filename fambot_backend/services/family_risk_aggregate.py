"""Aggregate family-group members' stored risk scores for model features."""

from __future__ import annotations

import numpy as np

from fambot_backend.schemas import FamilyRole
from fambot_backend.services.family_invites import family_peers_for_scoring
from fambot_backend.services.firestore_users import get_user_profile

# Match `fambot_backend.services.inference._risk_class` high threshold (score >= 67).
RISK_SCORE_HIGH_MIN = 67.0

_FIRST_DEGREE_ROLES: frozenset[FamilyRole] = frozenset(
    {
        "mother",
        "father",
        "son",
        "daughter",
        "brother",
        "sister",
        "husband",
        "wife",
    }
)
_EXTENDED_ROLES: frozenset[FamilyRole] = frozenset({"uncle", "aunt", "nephew", "niece"})


def _role_weight(role: FamilyRole | None) -> float:
    if role in _FIRST_DEGREE_ROLES:
        return 1.0
    if role in _EXTENDED_ROLES:
        return 0.5
    if role is None:
        return 0.65
    return 0.65


def neutral_family_features() -> dict[str, float | None]:
    """Missing family signal; pipeline median-imputes these columns."""
    return {
        "fam_weighted_mean_risk": None,
        "fam_max_member_risk": None,
        "fam_first_deg_mean_risk": None,
        "fam_any_member_high_risk": None,
    }


def compute_family_risk_feature_row(subject_uid: str) -> dict[str, float | None]:
    """Weighted aggregates from peers' completed onboarding risk scores."""
    peers = family_peers_for_scoring(subject_uid)
    if not peers:
        return neutral_family_features()

    weighted_sum = 0.0
    weight_total = 0.0
    max_risk: float | None = None
    first_deg_scores: list[float] = []
    any_high = False
    any_score = False

    for member_uid, role in peers:
        profile = get_user_profile(member_uid)
        if not profile.onboarding_complete or profile.risk_score is None:
            continue
        any_score = True
        r = float(profile.risk_score)
        w = _role_weight(role)
        weighted_sum += w * r
        weight_total += w
        max_risk = r if max_risk is None else max(max_risk, r)
        if role in _FIRST_DEGREE_ROLES:
            first_deg_scores.append(r)
        if r >= RISK_SCORE_HIGH_MIN:
            any_high = True

    if not any_score or weight_total <= 0.0:
        return neutral_family_features()

    wmean = weighted_sum / weight_total
    fd_mean: float | None
    if first_deg_scores:
        fd_mean = float(np.mean(first_deg_scores))
    else:
        fd_mean = None

    return {
        "fam_weighted_mean_risk": float(np.clip(wmean, 0.0, 100.0)),
        "fam_max_member_risk": float(np.clip(max_risk, 0.0, 100.0)) if max_risk is not None else None,
        "fam_first_deg_mean_risk": float(np.clip(fd_mean, 0.0, 100.0)) if fd_mean is not None else None,
        "fam_any_member_high_risk": 1.0 if any_high else 0.0,
    }
