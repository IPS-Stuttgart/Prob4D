from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUEST = (
    ROOT
    / "protocols/execution_requests/tracking_cloth_finite_orbit_hosted_recovery_v1.json"
)


def test_reviewed_control_plane_does_not_open_tracking_cloth_outcomes() -> None:
    assert not REQUEST.exists()
