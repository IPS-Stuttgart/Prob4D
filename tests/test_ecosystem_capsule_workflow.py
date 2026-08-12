from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ecosystem-release-capsule.yml"


def test_repository_owned_integration_tests_trigger_ecosystem_capsule() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    pull_request, remainder = text.split("  push:\n", maxsplit=1)
    push, _ = remainder.split("  workflow_dispatch:\n", maxsplit=1)

    path_filter = '      - "integration_tests/**"'
    assert path_filter in pull_request
    assert path_filter in push
    assert text.count(path_filter) == 2
