"""Content-locked data access for the provider-v2 conformance corpus."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

PROVIDER_V2_CONTRACT_BUNDLE = "prob4d.provider_v2_factors.v1"
PROVIDER_V2_CONTRACT_BUNDLE_VERSION = 1
PROVIDER_V2_CONTRACT_BUNDLE_SHA256 = (
    "fe0374f46319287e3709497de9cbb73f7497286cf4f157f246096f2c352e4446"
)

_BUNDLE_DIRECTORY = ("contract_data", "provider_v2_factors_v1")
_VECTOR_NAMES = frozenset({"minimal"})


@dataclass(frozen=True, slots=True)
class ProviderV2ContractVector:
    """One verified neutral provider-v2 conformance vector."""

    name: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class InvalidProviderV2ContractVector:
    """One declaratively mutated invalid vector."""

    case_id: str
    stage: str
    expected_error: str
    payload: Mapping[str, Any]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _bundle_root():
    root = resources.files(__package__)
    for component in _BUNDLE_DIRECTORY:
        root = root.joinpath(component)
    return root


def _bundle_member(relative_path: str):
    member = _bundle_root()
    for component in relative_path.split("/"):
        member = member.joinpath(component)
    return member


def _read_json(relative_path: str) -> Any:
    try:
        return json.loads(_bundle_member(relative_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise ValueError(
            f"provider-v2 contract bundle member {relative_path!r} is invalid"
        ) from error


def provider_v2_contract_bundle_manifest() -> dict[str, Any]:
    """Load and verify the complete content-addressed corpus manifest."""

    payload = _read_json("manifest.json")
    if not isinstance(payload, dict):
        raise ValueError("provider-v2 contract manifest must be a JSON object")
    required = {
        "bundle_name",
        "bundle_version",
        "bundle_sha256",
        "canonical_repository",
        "files",
    }
    if set(payload) != required:
        raise ValueError("provider-v2 contract manifest fields changed")
    if payload["bundle_name"] != PROVIDER_V2_CONTRACT_BUNDLE:
        raise ValueError("unexpected provider-v2 contract bundle name")
    version = payload["bundle_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != PROVIDER_V2_CONTRACT_BUNDLE_VERSION
    ):
        raise ValueError("unsupported provider-v2 contract bundle version")
    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("provider-v2 contract manifest has no files")
    normalized: dict[str, str] = {}
    for relative_path, expected_digest in sorted(files.items()):
        if not isinstance(relative_path, str):
            raise TypeError("provider-v2 contract path must be a string")
        if (
            not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise ValueError("provider-v2 contract manifest contains an unsafe path")
        expected = _validate_sha256(
            expected_digest,
            name=f"bundle file digest for {relative_path}",
        )
        try:
            content = _bundle_member(relative_path).read_bytes()
        except FileNotFoundError as error:
            raise ValueError(
                f"provider-v2 contract bundle is missing {relative_path}"
            ) from error
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(
                f"provider-v2 contract member {relative_path} failed its content lock"
            )
        normalized[relative_path] = expected
    descriptor = {
        "bundle_name": payload["bundle_name"],
        "bundle_version": version,
        "canonical_repository": str(payload["canonical_repository"]),
        "files": normalized,
    }
    expected_bundle = _validate_sha256(
        payload["bundle_sha256"],
        name="bundle_sha256",
    )
    actual_bundle = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    if actual_bundle != expected_bundle:
        raise ValueError("provider-v2 contract bundle digest does not match manifest")
    if actual_bundle != PROVIDER_V2_CONTRACT_BUNDLE_SHA256:
        raise ValueError("installed provider-v2 contract bundle differs from code lock")
    return {**descriptor, "bundle_sha256": actual_bundle}


def provider_v2_contract_schema() -> dict[str, Any]:
    """Return the verified normative schema descriptor."""

    provider_v2_contract_bundle_manifest()
    payload = _read_json("schema.json")
    if not isinstance(payload, dict):
        raise ValueError("provider-v2 contract schema must be a JSON object")
    if payload.get("contract_id") != PROVIDER_V2_CONTRACT_BUNDLE:
        raise ValueError("provider-v2 contract schema identity changed")
    if payload.get("schema_version") != PROVIDER_V2_CONTRACT_BUNDLE_VERSION:
        raise ValueError("provider-v2 contract schema version changed")
    return payload


def provider_v2_contract_vector(name: str = "minimal") -> ProviderV2ContractVector:
    """Load one verified valid vector."""

    if name not in _VECTOR_NAMES:
        raise KeyError(f"unknown provider-v2 contract vector {name!r}")
    provider_v2_contract_bundle_manifest()
    payload = _read_json(f"vectors/{name}.json")
    if not isinstance(payload, dict) or set(payload) != {
        "vector_version",
        "bundle",
        "tree_prior",
        "expected",
    }:
        raise ValueError("provider-v2 contract vector fields changed")
    if payload["vector_version"] != 1:
        raise ValueError("unsupported provider-v2 contract vector version")
    return ProviderV2ContractVector(name=name, payload=copy.deepcopy(payload))


def _set_path(root: object, path: Sequence[object], value: object) -> None:
    if not path:
        raise ValueError("provider-v2 mutation path must not be empty")
    current = root
    for component in path[:-1]:
        if isinstance(component, int):
            if not isinstance(current, MutableSequence):
                raise ValueError("provider-v2 mutation integer path reached non-list")
            current = current[component]
        else:
            if not isinstance(component, str) or not isinstance(
                current,
                MutableMapping,
            ):
                raise ValueError("provider-v2 mutation key path reached non-mapping")
            current = current[component]
    final = path[-1]
    if isinstance(final, int):
        if not isinstance(current, MutableSequence):
            raise ValueError("provider-v2 mutation final integer reached non-list")
        current[final] = value
    else:
        if not isinstance(final, str) or not isinstance(current, MutableMapping):
            raise ValueError("provider-v2 mutation final key reached non-mapping")
        current[final] = value


def invalid_provider_v2_contract_vectors() -> tuple[
    InvalidProviderV2ContractVector,
    ...,
]:
    """Load every verified adversarial mutation of the minimal vector."""

    base = provider_v2_contract_vector("minimal")
    payload = _read_json("invalid_cases.json")
    if not isinstance(payload, Mapping) or set(payload) != {"base_vector", "cases"}:
        raise ValueError("provider-v2 invalid-case corpus fields changed")
    if payload["base_vector"] != base.name:
        raise ValueError("provider-v2 invalid-case base vector changed")
    cases = payload["cases"]
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise ValueError("provider-v2 invalid cases must be a sequence")
    result: list[InvalidProviderV2ContractVector] = []
    identifiers: set[str] = set()
    for raw_case in cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != {
            "id",
            "stage",
            "expected_error",
            "mutations",
        }:
            raise ValueError("provider-v2 invalid case fields changed")
        case_id = str(raw_case["id"])
        if not case_id or case_id in identifiers:
            raise ValueError("provider-v2 invalid case IDs must be unique and nonempty")
        identifiers.add(case_id)
        mutated = copy.deepcopy(base.payload)
        mutations = raw_case["mutations"]
        if isinstance(mutations, (str, bytes)) or not isinstance(
            mutations,
            Sequence,
        ):
            raise ValueError("provider-v2 invalid mutations must be a sequence")
        for mutation in mutations:
            if not isinstance(mutation, Mapping) or set(mutation) != {
                "path",
                "value",
            }:
                raise ValueError("provider-v2 invalid mutation fields changed")
            path = mutation["path"]
            if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
                raise ValueError("provider-v2 invalid mutation path must be a sequence")
            _set_path(mutated, path, copy.deepcopy(mutation["value"]))
        result.append(
            InvalidProviderV2ContractVector(
                case_id=case_id,
                stage=str(raw_case["stage"]),
                expected_error=str(raw_case["expected_error"]),
                payload=mutated,
            )
        )
    return tuple(result)


__all__ = [
    "InvalidProviderV2ContractVector",
    "PROVIDER_V2_CONTRACT_BUNDLE",
    "PROVIDER_V2_CONTRACT_BUNDLE_SHA256",
    "PROVIDER_V2_CONTRACT_BUNDLE_VERSION",
    "ProviderV2ContractVector",
    "invalid_provider_v2_contract_vectors",
    "provider_v2_contract_bundle_manifest",
    "provider_v2_contract_schema",
    "provider_v2_contract_vector",
]
