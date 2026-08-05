"""Transfer-resistant identity for the Prob4D project.

Historical observation artifacts deliberately retain the repository name that was
current when their schemas were frozen. Repository display names are therefore
not suitable as the long-lived identity of the project. This module publishes a
stable GitHub repository ID, the current canonical name, and the accepted
historical aliases without changing any frozen provider-v1 or causal-stream
payload.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from ._strict_json import require_exact_string

PROB4D_PROJECT_IDENTITY_SCHEMA: Final = "prob4d.project-identity"
PROB4D_PROJECT_IDENTITY_VERSION: Final = 1
PROB4D_GITHUB_REPOSITORY_ID: Final = 1_295_794_737
PROB4D_PROJECT_ID: Final = f"github-repository-id:{PROB4D_GITHUB_REPOSITORY_ID}"
PROB4D_CANONICAL_REPOSITORY: Final = "IPS-Stuttgart/Prob4D"
PROB4D_FROZEN_ARTIFACT_REPOSITORY: Final = "FlorianPfaff/Prob4D"
PROB4D_REPOSITORY_ALIASES: Final[tuple[str, ...]] = (
    PROB4D_CANONICAL_REPOSITORY,
    PROB4D_FROZEN_ARTIFACT_REPOSITORY,
)

_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "project_id",
        "github_repository_id",
        "canonical_repository",
        "frozen_artifact_repository",
        "accepted_repository_aliases",
    }
)
_STRING_IDENTITY_FIELDS: Final = (
    "schema_name",
    "project_id",
    "canonical_repository",
    "frozen_artifact_repository",
)
_INTEGER_IDENTITY_FIELDS: Final = (
    "schema_version",
    "github_repository_id",
)


def canonical_prob4d_repository(value: object) -> str:
    """Return the canonical repository name for a recognized Prob4D alias.

    GitHub repository names are case-insensitive. Whitespace, coercible objects,
    and unrelated repositories fail closed instead of being silently normalized.
    """

    repository = require_exact_string(value, name="Prob4D repository identity")
    for alias in PROB4D_REPOSITORY_ALIASES:
        if repository.casefold() == alias.casefold():
            return PROB4D_CANONICAL_REPOSITORY
    raise ValueError(f"unrecognized Prob4D repository identity: {repository!r}")


def is_prob4d_repository(value: object) -> bool:
    """Return whether *value* is a recognized current or historical alias."""

    try:
        canonical_prob4d_repository(value)
    except ValueError:
        return False
    return True


def prob4d_project_identity() -> dict[str, object]:
    """Return the machine-readable stable project identity descriptor."""

    return {
        "schema_name": PROB4D_PROJECT_IDENTITY_SCHEMA,
        "schema_version": PROB4D_PROJECT_IDENTITY_VERSION,
        "project_id": PROB4D_PROJECT_ID,
        "github_repository_id": PROB4D_GITHUB_REPOSITORY_ID,
        "canonical_repository": PROB4D_CANONICAL_REPOSITORY,
        "frozen_artifact_repository": PROB4D_FROZEN_ARTIFACT_REPOSITORY,
        "accepted_repository_aliases": list(PROB4D_REPOSITORY_ALIASES),
    }


def validate_prob4d_project_identity(value: Mapping[str, Any]) -> dict[str, object]:
    """Validate one descriptor without accepting coercible primitive aliases."""

    field_names = tuple(value)
    if any(type(field) is not str for field in field_names):
        raise ValueError("Prob4D project-identity field names must be genuine strings")
    fields = set(field_names)
    if fields != _IDENTITY_FIELDS:
        missing = sorted(_IDENTITY_FIELDS - fields)
        extra = sorted(fields - _IDENTITY_FIELDS)
        raise ValueError(
            "Prob4D project-identity fields changed; "
            f"missing={missing}, extra={extra}"
        )

    normalized = dict(value)
    for field in _STRING_IDENTITY_FIELDS:
        require_exact_string(
            normalized[field],
            name=f"Prob4D project-identity {field}",
        )
    for field in _INTEGER_IDENTITY_FIELDS:
        if type(normalized[field]) is not int:
            raise ValueError(
                f"Prob4D project-identity {field} must be a genuine integer"
            )

    aliases = normalized["accepted_repository_aliases"]
    if type(aliases) is not list:
        raise ValueError(
            "Prob4D project-identity accepted_repository_aliases must be a JSON array"
        )
    for index, alias in enumerate(aliases):
        require_exact_string(
            alias,
            name=f"Prob4D project-identity accepted_repository_aliases[{index}]",
        )

    expected = prob4d_project_identity()
    if normalized != expected:
        raise ValueError("Prob4D project-identity descriptor does not match this project")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    """Print the transfer-resistant identity as JSON."""

    parser = argparse.ArgumentParser(
        prog="prob4d project identity",
        description=(
            "Print Prob4D's stable project ID, canonical repository, and frozen "
            "artifact alias."
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit canonical compact JSON instead of indented JSON",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    separators = (",", ":") if arguments.compact else None
    print(
        json.dumps(
            prob4d_project_identity(),
            sort_keys=True,
            indent=None if arguments.compact else 2,
            separators=separators,
            allow_nan=False,
        )
    )
    return 0


__all__ = [
    "PROB4D_CANONICAL_REPOSITORY",
    "PROB4D_FROZEN_ARTIFACT_REPOSITORY",
    "PROB4D_GITHUB_REPOSITORY_ID",
    "PROB4D_PROJECT_ID",
    "PROB4D_PROJECT_IDENTITY_SCHEMA",
    "PROB4D_PROJECT_IDENTITY_VERSION",
    "PROB4D_REPOSITORY_ALIASES",
    "canonical_prob4d_repository",
    "is_prob4d_repository",
    "prob4d_project_identity",
    "validate_prob4d_project_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
