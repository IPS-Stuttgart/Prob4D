from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "protocols/README-tracking-cloth-hosted-recovery-v1.md"


def test_recovery_index_names_the_separate_request_and_schema() -> None:
    text = INDEX.read_text(encoding="utf-8")
    assert "tracking_cloth_finite_orbit_hosted_recovery_v1.json" in text
    assert "tracking-cloth-finite-orbit-hosted-recovery-v1.schema.json" in text
    assert "does not change the scientific protocol" in text
