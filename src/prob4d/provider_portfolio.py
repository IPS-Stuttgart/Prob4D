"""Content-addressed evidence-first budgets for competing 4-D providers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._atomic_file import atomic_write_bytes
from ._immutable_json import plain_json
from ._provider_portfolio_model import (
    ACTIVE_PROVIDER_ROLES,
    MAX_ACTIVE_ALTERNATIVE,
    MAX_ACTIVE_PRIMARY,
    POLICY_FIELDS,
    PROVIDER_PORTFOLIO_CLAIM_BOUNDARY,
    PROVIDER_PORTFOLIO_SCHEMA,
    PROVIDER_PORTFOLIO_VERSION,
    PROVIDER_STAGES,
    PROVIDER_STAGES_V1,
    SUPPORTED_PROVIDER_PORTFOLIO_VERSIONS,
    canonical_policy,
    normalize_entries,
    provider_stages_for_version,
)
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)

_SPEC_FIELDS: Final = frozenset({"entries", "metadata"})
_PORTFOLIO_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "policy",
        "entries",
        "metadata",
        "claim_boundary",
        "portfolio_id",
    }
)


def _canonical_json(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            plain_json(value),
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + ("\n" if pretty else "")
    ).encode("utf-8")


def _portfolio_id(value: Mapping[str, Any]) -> str:
    unsigned = dict(plain_json(value))
    unsigned.pop("portfolio_id", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def build_provider_portfolio(specification: Mapping[str, Any]) -> dict[str, Any]:
    """Build one canonical content-addressed v2 portfolio from a strict specification."""

    spec = require_mapping(specification, name="provider portfolio specification")
    require_exact_fields(spec, _SPEC_FIELDS, name="provider portfolio specification")
    payload: dict[str, Any] = {
        "schema": PROVIDER_PORTFOLIO_SCHEMA,
        "schema_version": PROVIDER_PORTFOLIO_VERSION,
        "policy": canonical_policy(),
        "entries": normalize_entries(spec["entries"], stages=PROVIDER_STAGES),
        "metadata": plain_json(
            require_finite_json_mapping(spec["metadata"], name="metadata")
        ),
        "claim_boundary": PROVIDER_PORTFOLIO_CLAIM_BOUNDARY,
    }
    payload["portfolio_id"] = _portfolio_id(payload)
    return validate_provider_portfolio(payload)


def _validated_policy(value: object, *, version: int) -> dict[str, object]:
    policy = require_mapping(value, name="policy")
    require_exact_fields(policy, POLICY_FIELDS, name="policy")
    expected = canonical_policy(version=version)
    if plain_json(policy) != expected:
        raise ValueError("provider portfolio policy is not canonical")
    return expected


def validate_provider_portfolio(value: object) -> dict[str, Any]:
    """Validate a persisted v1 or v2 portfolio and return canonical plain data."""

    portfolio = require_mapping(value, name="provider portfolio")
    require_exact_fields(portfolio, _PORTFOLIO_FIELDS, name="provider portfolio")
    if require_exact_string(portfolio["schema"], name="schema") != PROVIDER_PORTFOLIO_SCHEMA:
        raise ValueError("unsupported provider portfolio schema")
    version = portfolio["schema_version"]
    if type(version) is not int or version not in SUPPORTED_PROVIDER_PORTFOLIO_VERSIONS:
        raise ValueError("unsupported provider portfolio schema version")
    stages = provider_stages_for_version(version)
    if (
        require_exact_string(portfolio["claim_boundary"], name="claim_boundary")
        != PROVIDER_PORTFOLIO_CLAIM_BOUNDARY
    ):
        raise ValueError("provider portfolio claim boundary is not canonical")

    normalized: dict[str, Any] = {
        "schema": PROVIDER_PORTFOLIO_SCHEMA,
        "schema_version": version,
        "policy": _validated_policy(portfolio["policy"], version=version),
        "entries": normalize_entries(portfolio["entries"], stages=stages),
        "metadata": plain_json(
            require_finite_json_mapping(portfolio["metadata"], name="metadata")
        ),
        "claim_boundary": PROVIDER_PORTFOLIO_CLAIM_BOUNDARY,
        "portfolio_id": require_sha256(portfolio["portfolio_id"], name="portfolio_id"),
    }
    if normalized["portfolio_id"] != _portfolio_id(normalized):
        raise ValueError("portfolio_id does not match the canonical portfolio content")
    return cast(dict[str, Any], json.loads(_canonical_json(normalized)))


def load_provider_portfolio(path: str | Path) -> dict[str, Any]:
    """Load and validate one strict finite-JSON provider portfolio."""

    return validate_provider_portfolio(
        load_json_object(path, name="provider portfolio artifact")
    )


def write_provider_portfolio(
    path: str | Path,
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a portfolio atomically without replacing different existing bytes."""

    validated = validate_provider_portfolio(portfolio)
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("provider portfolio destination must not be a symbolic link")
    encoded = _canonical_json(validated, pretty=True)
    try:
        atomic_write_bytes(destination, encoded, overwrite=False)
    except FileExistsError:
        existing = load_provider_portfolio(destination)
        if existing != validated:
            raise FileExistsError(
                f"refusing to replace a different provider portfolio: {destination}"
            ) from None
        return existing
    return validated


def provider_portfolio_summary(portfolio: Mapping[str, Any]) -> dict[str, object]:
    """Return an operational summary without changing the validated artifact."""

    validated = validate_provider_portfolio(portfolio)
    stages = provider_stages_for_version(cast(int, validated["schema_version"]))
    entries = cast(list[dict[str, object]], validated["entries"])
    counts = {
        status: sum(entry["status"] == status for entry in entries)
        for status in ("active", "parked", "promoted", "rejected", "archived")
    }
    active: list[dict[str, object]] = []
    for entry in entries:
        if entry["status"] != "active":
            continue
        gates = cast(Mapping[str, object], entry["gates"])
        stage = next(
            candidate
            for candidate in stages
            if cast(Mapping[str, object], gates[candidate])["decision"]
            == "in-progress"
        )
        active.append(
            {
                "provider_id": entry["provider_id"],
                "role": entry["role"],
                "stage": stage,
            }
        )
    return {
        "portfolio_id": validated["portfolio_id"],
        "schema_version": validated["schema_version"],
        "entry_count": len(entries),
        "status_counts": counts,
        "active": active,
        "claim_boundary": PROVIDER_PORTFOLIO_CLAIM_BOUNDARY,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a v2 portfolio from a JSON spec")
    build.add_argument("specification")
    build.add_argument("--output", required=True)
    build.set_defaults(handler=_build_command)

    verify = subparsers.add_parser("verify", help="verify a persisted v1 or v2 portfolio")
    verify.add_argument("portfolio")
    verify.set_defaults(handler=_verify_command)

    summarize = subparsers.add_parser("summarize", help="summarize a portfolio")
    summarize.add_argument("portfolio")
    summarize.set_defaults(handler=_summarize_command)
    return parser


def _build_command(arguments: argparse.Namespace) -> int:
    spec = load_json_object(arguments.specification, name="provider portfolio specification")
    portfolio = build_provider_portfolio(spec)
    write_provider_portfolio(arguments.output, portfolio)
    print(portfolio["portfolio_id"])
    return 0


def _verify_command(arguments: argparse.Namespace) -> int:
    portfolio = load_provider_portfolio(arguments.portfolio)
    print(portfolio["portfolio_id"])
    return 0


def _summarize_command(arguments: argparse.Namespace) -> int:
    summary = provider_portfolio_summary(load_provider_portfolio(arguments.portfolio))
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the provider-portfolio command-line interface."""

    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    return int(arguments.handler(arguments))


__all__ = [
    "ACTIVE_PROVIDER_ROLES",
    "MAX_ACTIVE_ALTERNATIVE",
    "MAX_ACTIVE_PRIMARY",
    "PROVIDER_PORTFOLIO_CLAIM_BOUNDARY",
    "PROVIDER_PORTFOLIO_SCHEMA",
    "PROVIDER_PORTFOLIO_VERSION",
    "PROVIDER_STAGES",
    "PROVIDER_STAGES_V1",
    "SUPPORTED_PROVIDER_PORTFOLIO_VERSIONS",
    "build_provider_portfolio",
    "load_provider_portfolio",
    "main",
    "provider_portfolio_summary",
    "validate_provider_portfolio",
    "write_provider_portfolio",
]


if __name__ == "__main__":
    raise SystemExit(main())
