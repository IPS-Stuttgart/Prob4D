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


class _RepositoryStringLike:
    def __str__(self) -> str:
        return PROB4D_CANONICAL_REPOSITORY


class _RepositoryStringSubclass(str):
    pass


def test_project_identity_is_transfer_resistant_and_preserves_frozen_alias() -> None:
    descriptor = prob4d_project_identity()

    assert descriptor["project_id"] == PROB4D_PROJECT_ID
    assert descriptor["github_repository_id"] == PROB4D_GITHUB_REPOSITORY_ID
    assert descriptor["canonical_repository"] == PROB4D_CANONICAL_REPOSITORY
    assert descriptor["frozen_artifact_repository"] == PROB4D_FROZEN_ARTIFACT_REPOSITORY
    assert descriptor["accepted_repository_aliases"] == [
        PROB4D_CANONICAL_REPOSITORY,
        PROB4D_FROZEN_ARTIFACT_REPOSITORY,
    ]
    assert validate_prob4d_project_identity(descriptor) == descriptor


def test_current_and_historical_repository_names_resolve_to_canonical() -> None:
    assert canonical_prob4d_repository("ips-stuttgart/prob4d") == PROB4D_CANONICAL_REPOSITORY
    assert canonical_prob4d_repository("FlorianPfaff/Prob4D") == PROB4D_CANONICAL_REPOSITORY
    assert is_prob4d_repository(PROB4D_CANONICAL_REPOSITORY)
    assert is_prob4d_repository(PROB4D_FROZEN_ARTIFACT_REPOSITORY)
    assert not is_prob4d_repository("other/Prob4D")
    with pytest.raises(ValueError, match="unrecognized Prob4D repository"):
        canonical_prob4d_repository("other/Prob4D")


def test_repository_aliases_reject_normalization_and_string_coercion() -> None:
    invalid_values = (
        f" {PROB4D_CANONICAL_REPOSITORY}",
        f"{PROB4D_CANONICAL_REPOSITORY} ",
        "",
        _RepositoryStringLike(),
        _RepositoryStringSubclass(PROB4D_CANONICAL_REPOSITORY),
        None,
    )
    for value in invalid_values:
        with pytest.raises(ValueError):
            canonical_prob4d_repository(value)
        assert not is_prob4d_repository(value)


def test_project_identity_rejects_modified_descriptors() -> None:
    descriptor = prob4d_project_identity()
    descriptor["canonical_repository"] = "other/Prob4D"
    with pytest.raises(ValueError, match="does not match"):
        validate_prob4d_project_identity(descriptor)


def test_project_identity_rejects_coercible_primitive_aliases() -> None:
    descriptor = prob4d_project_identity()
    descriptor["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version must be a genuine integer"):
        validate_prob4d_project_identity(descriptor)

    descriptor = prob4d_project_identity()
    descriptor["schema_name"] = _RepositoryStringSubclass(str(descriptor["schema_name"]))
    with pytest.raises(ValueError, match="schema_name must be a genuine string"):
        validate_prob4d_project_identity(descriptor)

    descriptor = prob4d_project_identity()
    descriptor["accepted_repository_aliases"] = tuple(descriptor["accepted_repository_aliases"])
    with pytest.raises(ValueError, match="must be a JSON array"):
        validate_prob4d_project_identity(descriptor)


def test_project_identity_rejects_non_string_field_names() -> None:
    descriptor = prob4d_project_identity()
    descriptor[1] = descriptor.pop("schema_name")  # type: ignore[index]
    with pytest.raises(ValueError, match="field names must be genuine strings"):
        validate_prob4d_project_identity(descriptor)  # type: ignore[arg-type]


def test_project_identity_cli_emits_valid_json(capsys) -> None:
    assert main(["--compact"]) == 0
    assert json.loads(capsys.readouterr().out) == prob4d_project_identity()


def test_grouped_cli_exposes_project_identity(capsys) -> None:
    assert cli_main(["project", "identity", "--compact"]) == 0
    assert json.loads(capsys.readouterr().out) == prob4d_project_identity()
