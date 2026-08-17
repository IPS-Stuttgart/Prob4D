from __future__ import annotations

from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
DISALLOWED_PREFIXES = (
    "_one_shot",
    "_temporary",
    "agent-",
    "temporary-",
)
DISALLOWED_SUFFIXES = (
    "-once.yml",
    "-once.yaml",
)


def test_default_branch_ships_no_temporary_or_one_shot_workflows() -> None:
    offenders = sorted(
        path.name
        for path in WORKFLOW_DIR.iterdir()
        if path.is_file()
        and path.suffix in {".yml", ".yaml"}
        and (
            path.name.startswith(DISALLOWED_PREFIXES)
            or path.name.endswith(DISALLOWED_SUFFIXES)
        )
    )

    assert offenders == []


def test_workflow_hygiene_policy_covers_historical_agent_patterns() -> None:
    assert "agent-" in DISALLOWED_PREFIXES
    assert "_temporary" in DISALLOWED_PREFIXES
    assert "_one_shot" in DISALLOWED_PREFIXES
    assert "-once.yml" in DISALLOWED_SUFFIXES
