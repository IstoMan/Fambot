"""Unit tests for reciprocal family role mapping."""

from __future__ import annotations

import pytest

from fambot_backend.schemas import FamilyRole
from fambot_backend.services.family_roles import reciprocal_role


def _r(
    owner_to_invitee: FamilyRole,
    *,
    owner_gender: str | None = None,
    invitee_gender: str | None = None,
) -> FamilyRole:
    return reciprocal_role(
        owner_to_invitee,
        owner_gender=owner_gender,
        invitee_gender=invitee_gender,
    )


@pytest.mark.unit
def test_reciprocal_parent_child() -> None:
    assert _r("mother", owner_gender="female") == "daughter"
    assert _r("father", owner_gender="male") == "son"
    assert _r("son", owner_gender="male") == "father"
    assert _r("daughter", owner_gender="female") == "mother"


@pytest.mark.unit
def test_reciprocal_siblings() -> None:
    assert _r("brother", owner_gender="male") == "brother"
    assert _r("sister", owner_gender="female") == "sister"


@pytest.mark.unit
def test_reciprocal_uncle_aunt_nephew_niece() -> None:
    assert _r("uncle", owner_gender="male") == "nephew"
    assert _r("aunt", owner_gender="female") == "niece"
    assert _r("nephew", owner_gender="male") == "uncle"
    assert _r("nephew", owner_gender="female") == "aunt"
    assert _r("niece", owner_gender="male") == "uncle"
    assert _r("niece", owner_gender="female") == "aunt"


@pytest.mark.unit
def test_reciprocal_spouse() -> None:
    assert _r("husband") == "wife"
    assert _r("wife") == "husband"


@pytest.mark.unit
def test_reciprocal_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown family role"):
        reciprocal_role("cousin", owner_gender=None, invitee_gender=None)  # type: ignore[arg-type]
