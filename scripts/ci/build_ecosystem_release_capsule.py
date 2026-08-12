#!/usr/bin/env python3
"""Build or verify a deterministic evidence-bound three-repository capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "prob4d.ecosystem-release-capsule"
SCHEMA_VERSION: Final = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_WHEEL_LINE = re.compile(r"^(?P<sha>[0-9a-f]{64})\s+(?P<path>\S+\.whl)\s*$")
_WHEEL_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("prob4d", "prob4d-"),
    ("bayesian_phystwin", "bayesian_phystwin-"),
    ("causal4d", "causal4d-"),
)
_COMPONENTS: Final[tuple[str, ...]] = tuple(item[0] for item in _WHEEL_PREFIXES)
_EVIDENCE_FILES: Final[tuple[str, ...]] = (
    "accepted-selection.json",
    "exact-prob4d-observation.npz",
    "golden-path-bundle.json",
    "lineage-bound-physical-posterior.npz",
    "lineage-bound-twin-belief.npz",
    "profile-bound.npz",
    "public-api-manifest.json",
    "rejected-selection.json",
    "run-manifest-v2.json",
)
_PUBLIC_API_PROJECT_ID: Final = "github-repository-id:1295794737"
_PUBLIC_API_SCHEMA: Final = "prob4d.public-api-manifest"
_PUBLIC_API_CLAIM_BOUNDARY: Final = (
    "This manifest records installed Python compatibility surfaces and their exact "
    "export inventories. It is interoperability evidence only; it does not establish "
    "provider accuracy, uncertainty calibration, physical-query improvement, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)
_SELECTION_SCHEMA: Final = "bayesian_phystwin.three_repository_golden_path_selection"
_BUNDLE_SCHEMA: Final = "bayesian_phystwin.three_repository_golden_path_bundle"
CLAIM_BOUNDARY: Final = (
    "This capsule is installed-wheel interoperability, decision-handoff, exact-fallback, "
    "and provenance evidence only. It does not establish provider accuracy, uncertainty "
    "calibration, physical-query improvement, intervention benefit, deployment safety, "
    "or state of the art."
)

_SELECTION_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "schema_name",
        "schema_version",
        "case_id",
        "protocol_id",
        "decision",
        "reason",
        "inference_admissible",
        "regret_guard_present",
        "regret_guard_accepted",
        "candidate_accepted",
        "observation_artifact_id",
        "twin_belief_id",
        "physical_posterior_id",
        "provider_manifest_id",
        "run_manifest_id",
        "evidence_fingerprint",
        "repository_revisions",
        "wheel_sha256",
        "package_versions",
        "baseline_identity",
        "candidate_identity",
        "selected_identity",
        "exact_fallback_identity",
        "metadata",
    }
)
_ARRAY_FIELDS: Final = frozenset({"array_id", "dtype", "shape", "nbytes", "payload_sha256"})
_BUNDLE_FIELDS: Final = frozenset(
    {"bundle_id", "schema_name", "schema_version", "accepted", "rejected"}
)
_EVIDENCE_FIELDS: Final = frozenset(
    {
        "golden_path_log_sha256",
        "public_api_manifest_id",
        "run_manifest_id",
        "evidence_fingerprint",
        "golden_path_bundle_id",
        "accepted_selection_artifact_id",
        "rejected_selection_artifact_id",
        "exact_fallback_identity",
        "observation_artifact_id",
        "twin_belief_id",
        "physical_posterior_id",
        "provider_manifest_id",
        "artifact_files",
    }
)


def _canonical_json(value: Mapping[str, Any], *, newline: bool) -> bytes:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if newline:
        serialized += "\n"
    return serialized.encode("utf-8")


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string without surrounding whitespace")
    return value


def _strict_digest(value: Any, *, name: str, git: bool = False) -> str:
    digest = _strict_string(value, name=name)
    pattern = _GIT_SHA if git else _SHA256
    if pattern.fullmatch(digest) is None:
        kind = "Git commit" if git else "SHA-256"
        raise ValueError(f"{name} must be a lowercase {kind} digest")
    return digest


def _strict_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return value


def _exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    keys = set(mapping)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _content_id(value: Mapping[str, Any], *, id_field: str, newline: bool = False) -> str:
    identifier = _strict_digest(value.get(id_field), name=id_field)
    unsigned = dict(value)
    unsigned.pop(id_field)
    observed = hashlib.sha256(_canonical_json(unsigned, newline=newline)).hexdigest()
    if identifier != observed:
        raise ValueError(f"{id_field} does not match the canonical content")
    return identifier


def _load_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def parse_wheel_hashes(log_text: str) -> dict[str, dict[str, str]]:
    """Extract exactly one wheel hash for each ecosystem package."""

    if type(log_text) is not str:
        raise TypeError("log_text must be an exact string")
    wheels: dict[str, dict[str, str]] = {}
    for raw_line in log_text.splitlines():
        match = _WHEEL_LINE.fullmatch(raw_line.strip())
        if match is None:
            continue
        digest = match.group("sha")
        filename = Path(match.group("path")).name
        lowered = filename.lower()
        for component, prefix in _WHEEL_PREFIXES:
            if not lowered.startswith(prefix):
                continue
            record = {"filename": filename, "sha256": digest}
            previous = wheels.get(component)
            if previous is not None and previous != record:
                raise ValueError(f"golden-path log contains conflicting {component} wheels")
            wheels[component] = record
            break

    missing = sorted(component for component in _COMPONENTS if component not in wheels)
    if missing:
        raise ValueError(f"golden-path log omitted wheel hashes for: {missing}")
    return {component: wheels[component] for component in _COMPONENTS}


def _validate_component_map(
    value: Any,
    *,
    name: str,
    revisions: bool,
) -> dict[str, str]:
    mapping = _strict_mapping(value, name=name)
    _exact_keys(mapping, set(_COMPONENTS), name=name)
    return {
        component: _strict_digest(
            mapping[component],
            name=f"{name}.{component}",
            git=revisions,
        )
        if revisions
        else _strict_string(mapping[component], name=f"{name}.{component}")
        for component in _COMPONENTS
    }


def _validate_array_identity(value: Any, *, name: str) -> dict[str, Any]:
    record = _strict_mapping(value, name=name)
    _exact_keys(record, _ARRAY_FIELDS, name=name)
    dtype = _strict_string(record["dtype"], name=f"{name}.dtype")
    shape = record["shape"]
    if type(shape) is not list or not shape:
        raise ValueError(f"{name}.shape must be a nonempty JSON array")
    normalized_shape = [
        _strict_integer(item, name=f"{name}.shape[{index}]", minimum=1)
        for index, item in enumerate(shape)
    ]
    normalized = {
        "array_id": record["array_id"],
        "dtype": dtype,
        "shape": normalized_shape,
        "nbytes": _strict_integer(record["nbytes"], name=f"{name}.nbytes", minimum=1),
        "payload_sha256": _strict_digest(
            record["payload_sha256"],
            name=f"{name}.payload_sha256",
        ),
    }
    _content_id(normalized, id_field="array_id")
    return normalized


def _validate_selection(value: Any, *, decision: str) -> dict[str, Any]:
    record = _strict_mapping(value, name=f"{decision} selection")
    _exact_keys(record, _SELECTION_FIELDS, name=f"{decision} selection")
    if _strict_string(record["schema_name"], name=f"{decision}.schema_name") != (_SELECTION_SCHEMA):
        raise ValueError(f"{decision} selection schema changed")
    if (
        _strict_integer(
            record["schema_version"],
            name=f"{decision}.schema_version",
            minimum=1,
        )
        != 1
    ):
        raise ValueError(f"{decision} selection schema changed")
    if record["decision"] != decision:
        raise ValueError(f"{decision} selection decision changed")
    artifact_id = _content_id(record, id_field="artifact_id")
    for field in (
        "observation_artifact_id",
        "twin_belief_id",
        "physical_posterior_id",
        "provider_manifest_id",
        "run_manifest_id",
        "evidence_fingerprint",
    ):
        _strict_digest(record[field], name=f"{decision}.{field}")
    revisions = _validate_component_map(
        record["repository_revisions"],
        name=f"{decision}.repository_revisions",
        revisions=True,
    )
    wheel_sha256 = _validate_component_map(
        record["wheel_sha256"],
        name=f"{decision}.wheel_sha256",
        revisions=False,
    )
    for component, digest in wheel_sha256.items():
        _strict_digest(digest, name=f"{decision}.wheel_sha256.{component}")
    package_versions = _validate_component_map(
        record["package_versions"],
        name=f"{decision}.package_versions",
        revisions=False,
    )
    baseline = _validate_array_identity(record["baseline_identity"], name=f"{decision}.baseline")
    candidate = _validate_array_identity(record["candidate_identity"], name=f"{decision}.candidate")
    selected = _validate_array_identity(record["selected_identity"], name=f"{decision}.selected")
    if baseline["array_id"] == candidate["array_id"]:
        raise ValueError("golden-path baseline and candidate must be distinct")
    inference_admissible = _strict_bool(
        record["inference_admissible"],
        name=f"{decision}.inference_admissible",
    )
    guard_present = _strict_bool(
        record["regret_guard_present"],
        name=f"{decision}.regret_guard_present",
    )
    guard_accepted = _strict_bool(
        record["regret_guard_accepted"],
        name=f"{decision}.regret_guard_accepted",
    )
    candidate_accepted = _strict_bool(
        record["candidate_accepted"],
        name=f"{decision}.candidate_accepted",
    )
    reason = _strict_string(record["reason"], name=f"{decision}.reason")
    if decision == "accepted":
        if reason != "candidate-accepted":
            raise ValueError("accepted selection reason changed")
        if not (inference_admissible and guard_present and guard_accepted and candidate_accepted):
            raise ValueError("accepted selection did not pass every admission gate")
        if selected["array_id"] != candidate["array_id"]:
            raise ValueError("accepted selection did not preserve exact candidate bytes")
        if record["exact_fallback_identity"] is not None:
            raise ValueError("accepted selection cannot claim an exact fallback")
    else:
        if not reason.endswith("exact-baseline-fallback"):
            raise ValueError("rejected selection reason changed")
        if candidate_accepted or guard_accepted:
            raise ValueError("rejected selection cannot accept the candidate")
        fallback = _strict_digest(
            record["exact_fallback_identity"],
            name="rejected.exact_fallback_identity",
        )
        if selected["array_id"] != baseline["array_id"] or fallback != baseline["array_id"]:
            raise ValueError("rejected selection did not preserve exact baseline fallback")
    normalized = dict(record)
    normalized["artifact_id"] = artifact_id
    normalized["repository_revisions"] = revisions
    normalized["wheel_sha256"] = wheel_sha256
    normalized["package_versions"] = package_versions
    normalized["baseline_identity"] = baseline
    normalized["candidate_identity"] = candidate
    normalized["selected_identity"] = selected
    return json.loads(_canonical_json(normalized, newline=False))


def _validate_public_api_manifest(value: Any) -> dict[str, Any]:
    manifest = _strict_mapping(value, name="public API manifest")
    _exact_keys(
        manifest,
        {"schema", "schema_version", "package", "surfaces", "claim_boundary", "manifest_id"},
        name="public API manifest",
    )
    if _strict_string(manifest["schema"], name="public API schema") != _PUBLIC_API_SCHEMA:
        raise ValueError("public API manifest schema changed")
    if (
        _strict_integer(
            manifest["schema_version"],
            name="public API schema_version",
            minimum=1,
        )
        != 2
    ):
        raise ValueError("public API manifest schema changed")
    if manifest["claim_boundary"] != _PUBLIC_API_CLAIM_BOUNDARY:
        raise ValueError("public API manifest claim boundary changed")

    package = _strict_mapping(manifest["package"], name="public API package")
    _exact_keys(package, {"name", "version", "project_id"}, name="public API package")
    if _strict_string(package["name"], name="public API package name") != "prob4d":
        raise ValueError("public API manifest package identity changed")
    if (
        _strict_string(
            package["project_id"],
            name="public API project identity",
        )
        != _PUBLIC_API_PROJECT_ID
    ):
        raise ValueError("public API manifest package identity changed")
    _strict_string(package["version"], name="public API package version")

    surfaces = _strict_mapping(manifest["surfaces"], name="public API surfaces")
    _exact_keys(surfaces, {"package_root", "api_v2"}, name="public API surfaces")
    package_root = _strict_mapping(surfaces["package_root"], name="package root")
    api2 = _strict_mapping(surfaces["api_v2"], name="api_v2")
    _exact_keys(
        package_root,
        {"module", "surface_version", "loading", "exports"},
        name="package root",
    )
    _exact_keys(
        api2,
        {
            "module",
            "api_version",
            "provider_api_version",
            "provider_factor_api_version",
            "lifecycle",
            "exports",
        },
        name="api_v2",
    )
    if (
        _strict_string(package_root.get("module"), name="package root module") != "prob4d"
        or _strict_integer(
            package_root.get("surface_version"),
            name="package root surface_version",
            minimum=1,
        )
        != 2
        or _strict_string(
            package_root.get("loading"),
            name="package root loading",
        )
        != "minimal-version-root-v1"
        or package_root.get("exports") != ["__version__"]
    ):
        raise ValueError("public API package-root semantics changed")
    if (
        _strict_string(api2.get("module"), name="api_v2 module") != "prob4d.api.v2"
        or _strict_integer(api2.get("api_version"), name="api_v2 version", minimum=1) != 2
        or _strict_string(api2.get("lifecycle"), name="api_v2 lifecycle") != "current"
    ):
        raise ValueError("public API v2 semantics changed")
    if (
        _strict_integer(
            api2.get("provider_api_version"),
            name="api_v2 provider version",
            minimum=1,
        )
        != 2
        or _strict_integer(
            api2.get("provider_factor_api_version"),
            name="api_v2 provider-factor version",
            minimum=1,
        )
        != 2
    ):
        raise ValueError("public API v2 provider versions changed")

    for surface_name, surface in surfaces.items():
        exports = surface.get("exports")
        if (
            type(exports) is not list
            or exports != sorted(exports)
            or len(exports) != len(set(exports))
        ):
            raise ValueError(f"{surface_name} export inventory is not canonical")
        if not exports or any(type(item) is not str or not item for item in exports):
            raise ValueError(f"{surface_name} export inventory is invalid")
    _content_id(manifest, id_field="manifest_id", newline=True)
    return json.loads(_canonical_json(manifest, newline=False))


def _validate_run_manifest(value: Any) -> dict[str, Any]:
    manifest = _strict_mapping(value, name="run manifest")
    if _strict_string(manifest.get("schema_name"), name="run-manifest schema") != (
        "bayesian_phystwin.run_manifest"
    ):
        raise ValueError("run-manifest schema changed")
    if (
        _strict_integer(
            manifest.get("schema_version"),
            name="run-manifest version",
            minimum=1,
        )
        != 2
    ):
        raise ValueError("run-manifest version changed")
    manifest_id = _content_id(manifest, id_field="manifest_id")
    evidence_fingerprint = _strict_digest(
        manifest.get("evidence_fingerprint"),
        name="run manifest evidence_fingerprint",
    )
    scientific_descriptor = {
        key: item
        for key, item in manifest.items()
        if key
        not in {
            "manifest_id",
            "evidence_fingerprint",
            "created_utc",
            "notes",
        }
    }
    observed_fingerprint = hashlib.sha256(
        _canonical_json(scientific_descriptor, newline=False)
    ).hexdigest()
    if evidence_fingerprint != observed_fingerprint:
        raise ValueError("run-manifest evidence fingerprint does not match its content")
    normalized = dict(manifest)
    normalized["manifest_id"] = manifest_id
    normalized["evidence_fingerprint"] = evidence_fingerprint
    return json.loads(_canonical_json(normalized, newline=False))


def _validate_bundle(value: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = _strict_mapping(value, name="golden-path bundle")
    _exact_keys(bundle, _BUNDLE_FIELDS, name="golden-path bundle")
    if _strict_string(bundle["schema_name"], name="bundle schema_name") != _BUNDLE_SCHEMA:
        raise ValueError("golden-path bundle schema changed")
    if (
        _strict_integer(
            bundle["schema_version"],
            name="bundle schema_version",
            minimum=1,
        )
        != 1
    ):
        raise ValueError("golden-path bundle schema changed")
    bundle_id = _content_id(bundle, id_field="bundle_id")
    accepted = _validate_selection(bundle["accepted"], decision="accepted")
    rejected = _validate_selection(bundle["rejected"], decision="rejected")
    normalized = dict(bundle)
    normalized["bundle_id"] = bundle_id
    normalized["accepted"] = accepted
    normalized["rejected"] = rejected
    return json.loads(_canonical_json(normalized, newline=False)), accepted, rejected


def _file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence entry is not a regular file: {path.name}")
    payload = path.read_bytes()
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def build_evidence_record(
    *,
    golden_path_log: str,
    evidence_root: str | Path,
    revisions: Mapping[str, str],
    wheels: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate the installed-wheel evidence roster and return its content identities."""

    root = Path(evidence_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("evidence_root must be a regular directory")
    actual = {path.name for path in root.iterdir()}
    expected = set(_EVIDENCE_FILES)
    if actual != expected:
        raise ValueError(
            "golden-path evidence roster changed; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    artifact_files = {name: _file_record(root / name) for name in _EVIDENCE_FILES}

    public_manifest = _validate_public_api_manifest(_load_json(root / "public-api-manifest.json"))
    run_manifest = _validate_run_manifest(_load_json(root / "run-manifest-v2.json"))
    accepted_file = _validate_selection(
        _load_json(root / "accepted-selection.json"),
        decision="accepted",
    )
    rejected_file = _validate_selection(
        _load_json(root / "rejected-selection.json"),
        decision="rejected",
    )
    bundle, accepted, rejected = _validate_bundle(_load_json(root / "golden-path-bundle.json"))
    if accepted != accepted_file or rejected != rejected_file:
        raise ValueError("golden-path bundle and selection files disagree")

    common_fields = (
        "observation_artifact_id",
        "twin_belief_id",
        "physical_posterior_id",
        "provider_manifest_id",
        "run_manifest_id",
        "evidence_fingerprint",
        "repository_revisions",
        "wheel_sha256",
        "package_versions",
    )
    for field in common_fields:
        if accepted[field] != rejected[field]:
            raise ValueError(f"accepted and rejected selections disagree on {field}")
    if accepted["run_manifest_id"] != run_manifest["manifest_id"]:
        raise ValueError("selection and run-manifest identities disagree")
    if accepted["evidence_fingerprint"] != run_manifest["evidence_fingerprint"]:
        raise ValueError("selection and run-manifest evidence fingerprints disagree")
    expected_revisions = {component: revisions[component] for component in _COMPONENTS}
    expected_wheels = {component: wheels[component]["sha256"] for component in _COMPONENTS}
    if accepted["repository_revisions"] != expected_revisions:
        raise ValueError("selection repository revisions differ from checked-out revisions")
    if accepted["wheel_sha256"] != expected_wheels:
        raise ValueError("selection wheel identities differ from golden-path wheel hashes")
    if public_manifest["package"]["version"] != accepted["package_versions"]["prob4d"]:
        raise ValueError("public API manifest and installed Prob4D package versions disagree")

    return {
        "golden_path_log_sha256": hashlib.sha256(golden_path_log.encode("utf-8")).hexdigest(),
        "public_api_manifest_id": public_manifest["manifest_id"],
        "run_manifest_id": run_manifest["manifest_id"],
        "evidence_fingerprint": run_manifest["evidence_fingerprint"],
        "golden_path_bundle_id": bundle["bundle_id"],
        "accepted_selection_artifact_id": accepted["artifact_id"],
        "rejected_selection_artifact_id": rejected["artifact_id"],
        "exact_fallback_identity": rejected["exact_fallback_identity"],
        "observation_artifact_id": accepted["observation_artifact_id"],
        "twin_belief_id": accepted["twin_belief_id"],
        "physical_posterior_id": accepted["physical_posterior_id"],
        "provider_manifest_id": accepted["provider_manifest_id"],
        "artifact_files": artifact_files,
    }


def _capsule_identity(payload_without_id: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload_without_id, newline=True)).hexdigest()


def build_capsule(
    *,
    golden_path_log: str,
    evidence_root: str | Path,
    prob4d_revision: str,
    bayesian_phystwin_revision: str,
    causal4d_revision: str,
    prob4d_repository: str = "IPS-Stuttgart/Prob4D",
    bayesian_phystwin_repository: str = "IPS-Stuttgart/BayesianPhysTwin",
    causal4d_repository: str = "IPS-Stuttgart/Causal4D",
    python_version: str,
    runner_os: str,
    run_id: int,
    run_attempt: int,
    run_url: str,
) -> dict[str, Any]:
    """Create a strict content-addressed capsule from exact installed-wheel evidence."""

    wheels = parse_wheel_hashes(golden_path_log)
    revisions = {
        "prob4d": _strict_digest(prob4d_revision, name="prob4d_revision", git=True),
        "bayesian_phystwin": _strict_digest(
            bayesian_phystwin_revision,
            name="bayesian_phystwin_revision",
            git=True,
        ),
        "causal4d": _strict_digest(causal4d_revision, name="causal4d_revision", git=True),
    }
    evidence = build_evidence_record(
        golden_path_log=golden_path_log,
        evidence_root=evidence_root,
        revisions=revisions,
        wheels=wheels,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "repositories": {
            "prob4d": {
                "repository": _strict_string(prob4d_repository, name="prob4d_repository"),
                "revision": revisions["prob4d"],
            },
            "bayesian_phystwin": {
                "repository": _strict_string(
                    bayesian_phystwin_repository,
                    name="bayesian_phystwin_repository",
                ),
                "revision": revisions["bayesian_phystwin"],
            },
            "causal4d": {
                "repository": _strict_string(causal4d_repository, name="causal4d_repository"),
                "revision": revisions["causal4d"],
            },
        },
        "wheels": wheels,
        "contracts": {
            "prob4d_provider_api": 2,
            "observation_belief": 1,
            "observation_factor_bundle": 4,
            "observation_factor_stream": 1,
            "public_api_manifest": 1,
            "golden_path_selection": 1,
            "golden_path_bundle": 1,
            "run_manifest": 2,
        },
        "evidence": evidence,
        "execution": {
            "python_version": _strict_string(python_version, name="python_version"),
            "runner_os": _strict_string(runner_os, name="runner_os"),
            "run_id": _strict_integer(run_id, name="run_id", minimum=1),
            "run_attempt": _strict_integer(run_attempt, name="run_attempt", minimum=1),
            "run_url": _strict_string(run_url, name="run_url"),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["capsule_id"] = _capsule_identity(payload)
    validate_capsule(payload)
    return payload


def _validate_file_records(value: Any) -> dict[str, dict[str, Any]]:
    records = _strict_mapping(value, name="evidence.artifact_files")
    _exact_keys(records, set(_EVIDENCE_FILES), name="evidence.artifact_files")
    normalized: dict[str, dict[str, Any]] = {}
    for name in _EVIDENCE_FILES:
        record = _strict_mapping(records[name], name=f"artifact_files.{name}")
        _exact_keys(record, {"size", "sha256"}, name=f"artifact_files.{name}")
        normalized[name] = {
            "size": _strict_integer(record["size"], name=f"artifact_files.{name}.size", minimum=1),
            "sha256": _strict_digest(record["sha256"], name=f"artifact_files.{name}.sha256"),
        }
    return normalized


def validate_capsule(value: Any) -> dict[str, Any]:
    """Deeply validate and return a plain canonical capsule mapping."""

    payload = _strict_mapping(value, name="capsule")
    _exact_keys(
        payload,
        {
            "schema",
            "schema_version",
            "status",
            "repositories",
            "wheels",
            "contracts",
            "evidence",
            "execution",
            "claim_boundary",
            "capsule_id",
        },
        name="capsule",
    )
    if _strict_string(payload["schema"], name="schema") != SCHEMA:
        raise ValueError("unsupported ecosystem release capsule schema")
    if _strict_integer(payload["schema_version"], name="schema_version", minimum=1) != 2:
        raise ValueError("unsupported ecosystem release capsule schema")
    if _strict_string(payload["status"], name="status") != "passed":
        raise ValueError("capsule status must be 'passed'")
    if _strict_string(payload["claim_boundary"], name="claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("capsule claim boundary is not canonical")

    repositories = _strict_mapping(payload["repositories"], name="repositories")
    _exact_keys(repositories, set(_COMPONENTS), name="repositories")
    for component, record_value in repositories.items():
        record = _strict_mapping(record_value, name=f"repositories.{component}")
        _exact_keys(record, {"repository", "revision"}, name=f"repositories.{component}")
        _strict_string(record["repository"], name=f"repositories.{component}.repository")
        _strict_digest(record["revision"], name=f"repositories.{component}.revision", git=True)

    wheels = _strict_mapping(payload["wheels"], name="wheels")
    _exact_keys(wheels, set(_COMPONENTS), name="wheels")
    for component, record_value in wheels.items():
        record = _strict_mapping(record_value, name=f"wheels.{component}")
        _exact_keys(record, {"filename", "sha256"}, name=f"wheels.{component}")
        filename = _strict_string(record["filename"], name=f"wheels.{component}.filename")
        expected_prefix = dict(_WHEEL_PREFIXES)[component]
        if not filename.lower().startswith(expected_prefix) or not filename.endswith(".whl"):
            raise ValueError(f"wheels.{component}.filename is not the expected wheel")
        _strict_digest(record["sha256"], name=f"wheels.{component}.sha256")

    expected_contracts = {
        "prob4d_provider_api": 2,
        "observation_belief": 1,
        "observation_factor_bundle": 4,
        "observation_factor_stream": 1,
        "public_api_manifest": 1,
        "golden_path_selection": 1,
        "golden_path_bundle": 1,
        "run_manifest": 2,
    }
    contracts = _strict_mapping(payload["contracts"], name="contracts")
    _exact_keys(contracts, set(expected_contracts), name="contracts")
    for key, expected in expected_contracts.items():
        if _strict_integer(contracts[key], name=f"contracts.{key}", minimum=1) != expected:
            raise ValueError("capsule contract versions are not canonical")

    evidence = _strict_mapping(payload["evidence"], name="evidence")
    _exact_keys(evidence, _EVIDENCE_FIELDS, name="evidence")
    for field in _EVIDENCE_FIELDS - {"artifact_files"}:
        _strict_digest(evidence[field], name=f"evidence.{field}")
    _validate_file_records(evidence["artifact_files"])

    execution = _strict_mapping(payload["execution"], name="execution")
    _exact_keys(
        execution,
        {"python_version", "runner_os", "run_id", "run_attempt", "run_url"},
        name="execution",
    )
    _strict_string(execution["python_version"], name="execution.python_version")
    _strict_string(execution["runner_os"], name="execution.runner_os")
    _strict_integer(execution["run_id"], name="execution.run_id", minimum=1)
    _strict_integer(execution["run_attempt"], name="execution.run_attempt", minimum=1)
    _strict_string(execution["run_url"], name="execution.run_url")

    capsule_id = _strict_digest(payload["capsule_id"], name="capsule_id")
    unsigned = dict(payload)
    unsigned.pop("capsule_id")
    if capsule_id != _capsule_identity(unsigned):
        raise ValueError("capsule_id does not match the canonical capsule content")
    return json.loads(_canonical_json(payload, newline=True))


def verify_capsule_evidence(
    capsule: Mapping[str, Any],
    *,
    golden_path_log: str,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Require the capsule to match the supplied log and evidence bytes exactly."""

    validated = validate_capsule(capsule)
    revisions = {
        component: validated["repositories"][component]["revision"] for component in _COMPONENTS
    }
    current = build_evidence_record(
        golden_path_log=golden_path_log,
        evidence_root=evidence_root,
        revisions=revisions,
        wheels=validated["wheels"],
    )
    if current != validated["evidence"]:
        raise ValueError("capsule evidence does not match the supplied execution bytes")
    return validated


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload, newline=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_command(arguments: argparse.Namespace) -> int:
    log_path = Path(arguments.golden_path_log)
    capsule = build_capsule(
        golden_path_log=log_path.read_text(encoding="utf-8"),
        evidence_root=arguments.evidence_root,
        prob4d_revision=arguments.prob4d_revision,
        bayesian_phystwin_revision=arguments.bayesian_phystwin_revision,
        causal4d_revision=arguments.causal4d_revision,
        prob4d_repository=arguments.prob4d_repository,
        bayesian_phystwin_repository=arguments.bayesian_phystwin_repository,
        causal4d_repository=arguments.causal4d_repository,
        python_version=arguments.python_version,
        runner_os=arguments.runner_os,
        run_id=arguments.run_id,
        run_attempt=arguments.run_attempt,
        run_url=arguments.run_url,
    )
    output = Path(arguments.output)
    if output.exists():
        existing = validate_capsule(_load_json(output))
        if existing != capsule:
            raise FileExistsError(f"refusing to replace a different capsule: {output}")
        return 0
    _atomic_write_json(output, capsule)
    print(capsule["capsule_id"])
    return 0


def _verify_command(arguments: argparse.Namespace) -> int:
    capsule = validate_capsule(_load_json(Path(arguments.capsule)))
    if (arguments.golden_path_log is None) != (arguments.evidence_root is None):
        raise ValueError("--golden-path-log and --evidence-root must be supplied together")
    if arguments.golden_path_log is not None:
        capsule = verify_capsule_evidence(
            capsule,
            golden_path_log=Path(arguments.golden_path_log).read_text(encoding="utf-8"),
            evidence_root=arguments.evidence_root,
        )
    print(capsule["capsule_id"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a capsule from installed-wheel evidence")
    build.add_argument("--golden-path-log", required=True)
    build.add_argument("--evidence-root", required=True)
    build.add_argument("--prob4d-revision", required=True)
    build.add_argument("--bayesian-phystwin-revision", required=True)
    build.add_argument("--causal4d-revision", required=True)
    build.add_argument("--prob4d-repository", default="IPS-Stuttgart/Prob4D")
    build.add_argument(
        "--bayesian-phystwin-repository",
        default="IPS-Stuttgart/BayesianPhysTwin",
    )
    build.add_argument("--causal4d-repository", default="IPS-Stuttgart/Causal4D")
    build.add_argument("--python-version", required=True)
    build.add_argument("--runner-os", required=True)
    build.add_argument("--run-id", required=True, type=int)
    build.add_argument("--run-attempt", required=True, type=int)
    build.add_argument("--run-url", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=_build_command)

    verify = subparsers.add_parser("verify", help="deeply verify an existing capsule")
    verify.add_argument("capsule")
    verify.add_argument("--golden-path-log")
    verify.add_argument("--evidence-root")
    verify.set_defaults(handler=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
