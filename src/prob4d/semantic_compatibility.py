"""Build and compare additive semantic-compatibility manifests.

The semantic manifest is deliberately weaker than a claim-bearing evidence pin:
it records the API major, named mandatory conformance vectors, and required
capabilities without binding the complete corpus inventory or repository revision.
Frozen scientific runs must continue to bind exact distributions, revisions, and
content-addressed artifact bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._atomic_file import atomic_write_bytes
from ._immutable_json import plain_json
from ._provider_v2_contract_common import (
    PROVIDER_V2_CONTRACT_BUNDLE,
    PROVIDER_V2_CONTRACT_BUNDLE_VERSION,
    provider_v2_contract_vector,
)
from .api import v2 as api_v2
from .observation_contract_bundle import (
    OBSERVATION_BELIEF_CONTRACT_BUNDLE,
    OBSERVATION_BELIEF_CONTRACT_BUNDLE_VERSION,
    observation_contract_vector,
)
from .project_identity import PROB4D_PROJECT_ID

SEMANTIC_COMPATIBILITY_SCHEMA: Final = "prob4d.semantic-compatibility"
SEMANTIC_COMPATIBILITY_VERSION: Final = 1
SEMANTIC_COMPATIBILITY_CLAIM_BOUNDARY: Final = (
    "Semantic compatibility establishes only that named interfaces, contract "
    "vectors, and capabilities are available. Claim-bearing runs must additionally "
    "bind exact Prob4D, BayesianPhysTwin, and Causal4D distributions, revisions, "
    "complete contract-corpus identities, protocols, and input/output artifacts."
)
PROVIDED_CAPABILITIES: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "causal-source-lineage-v1",
            "content-addressed-artifacts-v1",
            "exact-invalid-fallback-v1",
            "joint-gauge-covariance-v1",
            "strict-artifact-loading-v1",
            "tree-sparse-observation-factors-v1",
        }
    )
)

_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "project_id",
        "python_api",
        "contracts",
        "capabilities",
        "claim_bearing_evidence_pin_required",
        "claim_boundary",
        "manifest_id",
    }
)
_API_FIELDS: Final = frozenset(
    {
        "module",
        "api_version",
        "provider_api_version",
        "provider_factor_api_version",
        "lifecycle",
    }
)
_CONTRACT_FIELDS: Final = frozenset(
    {"contract_id", "contract_version", "required_vectors"}
)
_CONTRACT_NAMES: Final = ("observation_belief", "provider_v2_factors")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _manifest_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], *, name: str) -> None:
    keys = set(value)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty exact string without surrounding whitespace"
        )
    return cast(str, value)


def _strict_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a genuine integer >= {minimum}")
    return cast(int, value)


def _sha256(value: Any, *, name: str) -> str:
    digest = _strict_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validate_vectors(value: Any, *, name: str) -> dict[str, str]:
    mapping = _strict_mapping(value, name=name)
    if not mapping:
        raise ValueError(f"{name} must contain at least one named vector")
    vectors: dict[str, str] = {}
    for vector_name, digest in sorted(mapping.items()):
        normalized_name = _strict_string(vector_name, name=f"{name} vector name")
        vectors[normalized_name] = _sha256(
            digest,
            name=f"{name}.{normalized_name}",
        )
    return vectors


def _validate_contract(
    value: Any,
    *,
    name: str,
    expected_id: str,
) -> dict[str, Any]:
    contract = _strict_mapping(value, name=name)
    _exact_keys(contract, _CONTRACT_FIELDS, name=name)
    contract_id = _strict_string(contract["contract_id"], name=f"{name}.contract_id")
    if contract_id != expected_id:
        raise ValueError(f"{name}.contract_id is not supported")
    contract_version = _strict_integer(
        contract["contract_version"],
        name=f"{name}.contract_version",
    )
    required_vectors = _validate_vectors(
        contract["required_vectors"],
        name=f"{name}.required_vectors",
    )
    return {
        "contract_id": contract_id,
        "contract_version": contract_version,
        "required_vectors": required_vectors,
    }


def _provider_vector_sha256(name: str) -> str:
    vector = provider_v2_contract_vector(name)
    return hashlib.sha256(_canonical_json(vector.payload)).hexdigest()


def build_semantic_compatibility_manifest() -> dict[str, Any]:
    """Build the additive compatibility descriptor for this installation."""

    observation_vectors = {
        name: observation_contract_vector(name).expected_artifact_id
        for name in ("minimal", "zero_rank")
    }
    provider_vectors = {"minimal": _provider_vector_sha256("minimal")}
    payload: dict[str, Any] = {
        "schema": SEMANTIC_COMPATIBILITY_SCHEMA,
        "schema_version": SEMANTIC_COMPATIBILITY_VERSION,
        "project_id": PROB4D_PROJECT_ID,
        "python_api": {
            "module": "prob4d.api.v2",
            "api_version": api_v2.API_VERSION,
            "provider_api_version": api_v2.PROVIDER_API_VERSION,
            "provider_factor_api_version": api_v2.PROVIDER_FACTOR_API_VERSION,
            "lifecycle": "current",
        },
        "contracts": {
            "observation_belief": {
                "contract_id": OBSERVATION_BELIEF_CONTRACT_BUNDLE,
                "contract_version": OBSERVATION_BELIEF_CONTRACT_BUNDLE_VERSION,
                "required_vectors": observation_vectors,
            },
            "provider_v2_factors": {
                "contract_id": PROVIDER_V2_CONTRACT_BUNDLE,
                "contract_version": PROVIDER_V2_CONTRACT_BUNDLE_VERSION,
                "required_vectors": provider_vectors,
            },
        },
        "capabilities": list(PROVIDED_CAPABILITIES),
        "claim_bearing_evidence_pin_required": True,
        "claim_boundary": SEMANTIC_COMPATIBILITY_CLAIM_BOUNDARY,
    }
    payload["manifest_id"] = _manifest_id(payload)
    return validate_semantic_compatibility_manifest(payload)


def validate_semantic_compatibility_manifest(value: Any) -> dict[str, Any]:
    """Strictly validate one semantic-compatibility manifest."""

    payload = _strict_mapping(value, name="semantic compatibility manifest")
    _exact_keys(payload, _MANIFEST_FIELDS, name="semantic compatibility manifest")
    if _strict_string(payload["schema"], name="schema") != SEMANTIC_COMPATIBILITY_SCHEMA:
        raise ValueError("unsupported semantic-compatibility schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version")
        != SEMANTIC_COMPATIBILITY_VERSION
    ):
        raise ValueError("unsupported semantic-compatibility schema version")
    if _strict_string(payload["project_id"], name="project_id") != PROB4D_PROJECT_ID:
        raise ValueError("semantic compatibility project identity is not Prob4D")

    raw_api = _strict_mapping(payload["python_api"], name="python_api")
    _exact_keys(raw_api, _API_FIELDS, name="python_api")
    module = _strict_string(raw_api["module"], name="python_api.module")
    if module != "prob4d.api.v2":
        raise ValueError("semantic compatibility requires prob4d.api.v2")
    python_api = {
        "module": module,
        "api_version": _strict_integer(raw_api["api_version"], name="python_api.api_version"),
        "provider_api_version": _strict_integer(
            raw_api["provider_api_version"],
            name="python_api.provider_api_version",
        ),
        "provider_factor_api_version": _strict_integer(
            raw_api["provider_factor_api_version"],
            name="python_api.provider_factor_api_version",
        ),
        "lifecycle": _strict_string(raw_api["lifecycle"], name="python_api.lifecycle"),
    }
    if python_api["lifecycle"] != "current":
        raise ValueError("semantic compatibility Python API must be current")

    raw_contracts = _strict_mapping(payload["contracts"], name="contracts")
    _exact_keys(raw_contracts, set(_CONTRACT_NAMES), name="contracts")
    contracts = {
        "observation_belief": _validate_contract(
            raw_contracts["observation_belief"],
            name="contracts.observation_belief",
            expected_id=OBSERVATION_BELIEF_CONTRACT_BUNDLE,
        ),
        "provider_v2_factors": _validate_contract(
            raw_contracts["provider_v2_factors"],
            name="contracts.provider_v2_factors",
            expected_id=PROVIDER_V2_CONTRACT_BUNDLE,
        ),
    }

    raw_capabilities = payload["capabilities"]
    if type(raw_capabilities) is not list:
        raise ValueError("capabilities must be a JSON array")
    capabilities = [
        _strict_string(item, name=f"capabilities[{index}]")
        for index, item in enumerate(raw_capabilities)
    ]
    if not capabilities or capabilities != sorted(capabilities):
        raise ValueError("capabilities must be nonempty and sorted canonically")
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("capabilities must be unique")

    if payload["claim_bearing_evidence_pin_required"] is not True:
        raise ValueError("semantic compatibility may not replace the evidence pin")
    claim_boundary = _strict_string(payload["claim_boundary"], name="claim_boundary")
    if claim_boundary != SEMANTIC_COMPATIBILITY_CLAIM_BOUNDARY:
        raise ValueError("semantic compatibility claim boundary is not canonical")

    manifest_id = _sha256(payload["manifest_id"], name="manifest_id")
    normalized: dict[str, Any] = {
        "schema": SEMANTIC_COMPATIBILITY_SCHEMA,
        "schema_version": SEMANTIC_COMPATIBILITY_VERSION,
        "project_id": PROB4D_PROJECT_ID,
        "python_api": python_api,
        "contracts": contracts,
        "capabilities": capabilities,
        "claim_bearing_evidence_pin_required": True,
        "claim_boundary": claim_boundary,
        "manifest_id": manifest_id,
    }
    unsigned = dict(normalized)
    unsigned.pop("manifest_id")
    if manifest_id != _manifest_id(unsigned):
        raise ValueError("manifest_id does not match the canonical manifest content")
    return cast(dict[str, Any], json.loads(_canonical_json(normalized)))


def semantic_compatibility_report(
    required: Any,
    provided: Any,
) -> dict[str, Any]:
    """Report whether *provided* satisfies all additive requirements."""

    requirement = validate_semantic_compatibility_manifest(required)
    implementation = validate_semantic_compatibility_manifest(provided)
    reasons: list[str] = []

    if requirement["python_api"] != implementation["python_api"]:
        reasons.append("python-api-mismatch")

    missing_or_changed_vectors: list[str] = []
    for contract_name in _CONTRACT_NAMES:
        required_contract = requirement["contracts"][contract_name]
        provided_contract = implementation["contracts"][contract_name]
        if (
            required_contract["contract_id"] != provided_contract["contract_id"]
            or required_contract["contract_version"]
            != provided_contract["contract_version"]
        ):
            reasons.append(f"{contract_name}-identity-mismatch")
            continue
        for vector_name, digest in required_contract["required_vectors"].items():
            if provided_contract["required_vectors"].get(vector_name) != digest:
                missing_or_changed_vectors.append(f"{contract_name}:{vector_name}")

    required_capabilities = set(requirement["capabilities"])
    provided_capabilities = set(implementation["capabilities"])
    missing_capabilities = sorted(required_capabilities - provided_capabilities)
    if missing_or_changed_vectors:
        reasons.append("mandatory-vector-mismatch")
    if missing_capabilities:
        reasons.append("missing-capability")

    return {
        "compatible": not reasons,
        "required_manifest_id": requirement["manifest_id"],
        "provided_manifest_id": implementation["manifest_id"],
        "missing_or_changed_vectors": sorted(missing_or_changed_vectors),
        "missing_capabilities": missing_capabilities,
        "reasons": reasons,
    }


def load_semantic_compatibility_manifest(path: str | Path) -> dict[str, Any]:
    """Load strict JSON and validate a semantic-compatibility manifest."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    return validate_semantic_compatibility_manifest(payload)


def write_semantic_compatibility_manifest(
    path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one no-clobber semantic manifest, allowing idempotent writes."""

    payload = (
        build_semantic_compatibility_manifest()
        if manifest is None
        else validate_semantic_compatibility_manifest(manifest)
    )
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("semantic compatibility destination must not be a symbolic link")
    encoded = _canonical_json(payload) + b"\n"
    try:
        atomic_write_bytes(destination, encoded, overwrite=False)
    except FileExistsError:
        existing = load_semantic_compatibility_manifest(destination)
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace a different semantic compatibility manifest: {destination}"
            ) from None
        return existing
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction compatibility",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("print", help="print the current semantic manifest")
    show.set_defaults(handler=_print_command)

    build = subparsers.add_parser("build", help="write the current semantic manifest")
    build.add_argument("--output", required=True)
    build.set_defaults(handler=_build_command)

    verify = subparsers.add_parser("verify", help="verify one semantic manifest")
    verify.add_argument("manifest")
    verify.add_argument(
        "--require-current",
        action="store_true",
        help="require the executing installation to satisfy the manifest",
    )
    verify.set_defaults(handler=_verify_command)

    check = subparsers.add_parser("check", help="compare required and provided manifests")
    check.add_argument("required")
    check.add_argument("provided")
    check.set_defaults(handler=_check_command)
    return parser


def _print_command(arguments: argparse.Namespace) -> int:
    del arguments
    print(
        json.dumps(
            build_semantic_compatibility_manifest(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def _build_command(arguments: argparse.Namespace) -> int:
    manifest = write_semantic_compatibility_manifest(arguments.output)
    print(manifest["manifest_id"])
    return 0


def _verify_command(arguments: argparse.Namespace) -> int:
    manifest = load_semantic_compatibility_manifest(arguments.manifest)
    if arguments.require_current:
        report = semantic_compatibility_report(
            manifest,
            build_semantic_compatibility_manifest(),
        )
        if not report["compatible"]:
            raise ValueError(f"current installation is incompatible: {report['reasons']}")
    print(manifest["manifest_id"])
    return 0


def _check_command(arguments: argparse.Namespace) -> int:
    report = semantic_compatibility_report(
        load_semantic_compatibility_manifest(arguments.required),
        load_semantic_compatibility_manifest(arguments.provided),
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["compatible"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the semantic-compatibility command."""

    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    return int(arguments.handler(arguments))


__all__ = [
    "PROVIDED_CAPABILITIES",
    "SEMANTIC_COMPATIBILITY_CLAIM_BOUNDARY",
    "SEMANTIC_COMPATIBILITY_SCHEMA",
    "SEMANTIC_COMPATIBILITY_VERSION",
    "build_semantic_compatibility_manifest",
    "load_semantic_compatibility_manifest",
    "main",
    "semantic_compatibility_report",
    "validate_semantic_compatibility_manifest",
    "write_semantic_compatibility_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
