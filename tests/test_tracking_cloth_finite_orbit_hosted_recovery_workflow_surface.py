from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/tracking-cloth-finite-orbit-hosted-recovery-v1.yml"


def test_hosted_recovery_has_one_request_only_main_trigger() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    trigger = text.split("permissions:", maxsplit=1)[0]
    request = (
        "protocols/execution_requests/"
        "tracking_cloth_finite_orbit_hosted_recovery_v1.json"
    )
    assert request in trigger
    assert "branches: [main]" in trigger
    assert "workflow_dispatch" not in trigger
    assert "issue_comment" not in trigger
