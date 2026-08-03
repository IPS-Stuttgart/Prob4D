from __future__ import annotations

import json

import pytest

from prob4d.cli import main as cli_main
from prob4d.project_identity import (
    PROB4D_CANONICAL_REPOSITORY,
    PROB4D_FROZEN_ARTIFACT_REPOSITORY,
    PROB4D_GITHUB_REPOSITORY_ID,
    PROB4D_PROJECT_ID,
    canonical_prob4d_repository,
    is_prob4d_repository,
    main,
    prob4d_project_identity,
    validate_prob4d_project_identity,
)


def test_project_identity_is_transfer_resistant_and_preserves_frozen_alias() -> None:
    descriptor = prob4d_project_identity()

    assert descriptor["project_id"] == PROB4D_PROJECT_ID
    assert descriptor["github_repository_id"] == PROB4D_GITHUB_REPOSITORY_ID
    assert descriptor["canonical_repository"] == PROB4D_CANONICAL_REPOSITORY
    assert (
        descriptor["frozen_artifact_repository"]
        == PROB4D_FROZEN_ARTIFACT_REPOSITORY
    )
    assert descriptor["accepted_repository_aliases"] == [
        PROB4D_CANONICAL_REPOSITORY,
        PROB4D_FROZEN_ARTIFACT_REPOSITORY,
    ]
    assert validate_prob4d_project_identity(descriptor) == descriptor


def test_current_and_historical_repository_names_resolve_to_canonical() -> None:
    assert (
        canonical_prob4d_repository("ips-stuttgart/prob4d")
        == PROB4D_CANONICAL_REPOSITORY
    )
    assert (
        canonical_prob4d_repository("FlorianPfaff/Prob4D")
        == PROB4D_CANONICAL_REPOSITORY
    )
    assert is_prob4d_repository(PROB4D_CANONICAL_REPOSITORY)
    assert is_prob4d_repository(PROB4D_FROZEN_ARTIFACT_REPOSITORY)
    assert not is_prob4d_repository("other/Prob4D")
    with pytest.raises(ValueError, match="unrecognized Prob4D repository"):
        canonical_prob4d_repository("other/Prob4D")


def test_project_identity_rejects_modified_descriptors() -> None:
    descriptor = prob4d_project_identity()
    descriptor["canonical_repository"] = "other/Prob4D"
    with pytest.raises(ValueError, match="does not match"):
        validate_prob4d_project_identity(descriptor)


def test_project_identity_cli_emits_valid_json(capsys) -> None:
    assert main(["--compact"]) == 0
    assert json.loads(capsys.readouterr().out) == prob4d_project_identity()


def test_grouped_cli_exposes_project_identity(capsys) -> None:
    assert cli_main(["project", "identity", "--compact"]) == 0
    assert json.loads(capsys.readouterr().out) == prob4d_project_identity()
