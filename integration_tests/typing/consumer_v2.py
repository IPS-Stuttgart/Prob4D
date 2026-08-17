from __future__ import annotations

from pathlib import Path
from typing import assert_type

from prob4d.api.v2 import (
    Sim3,
    ValidatedClaimBearingObservation,
    load_claim_bearing_observation_belief,
)


def load_artifact(path: Path) -> str:
    """Exercise the concrete installed-wheel observation loader type."""

    validated = load_claim_bearing_observation_belief(path)
    assert_type(validated, ValidatedClaimBearingObservation)
    assert_type(validated.artifact_id, str)
    return validated.artifact_id


def compose_round_trip(transform: Sim3) -> Sim3:
    """Exercise method return types from the installed inline annotations."""

    inverse = transform.inverse()
    assert_type(inverse, Sim3)
    round_trip = inverse.compose(transform)
    assert_type(round_trip, Sim3)
    return round_trip


identity = Sim3.identity()
assert_type(identity, Sim3)
