#!/usr/bin/env python3
"""Build or verify a deterministic three-repository release capsule."""

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
SCHEMA_VERSION: Final = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_WHEEL_LINE = re.compile(r"^(?P<sha>[0-9a-f]{64})\s+(?P<path>\S+\.whl)\s*$")
_WHEEL_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("prob4d", "prob4d-"),
    ("bayesian_phystwin", "bayesian_phystwin-"),
    ("causal4d", "causal4d-"),
)
CLAIM_BOUNDARY: Final = (
    "This capsule is installed-wheel interoperability and provenance evidence only. "
    "It does not establish provider accuracy, uncertainty calibration, physical-query "
    "improvement, intervention benefit, deployment safety, or state of the art."
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (serialized + "\n").encode("utf-8")


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

    missing = sorted(component for component, _ in _WHEEL_PREFIXES if component not in wheels)
    if missing:
        raise ValueError(f"golden-path log omitted wheel hashes for: {missing}")
    return {component: wheels[component] for component, _ in _WHEEL_PREFIXES}


def _capsule_identity(payload_without_id: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload_without_id)).hexdigest()


def build_capsule(
    *,
    golden_path_log: str,
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
    """Create a strict content-addressed capsule from a passed golden-path log."""

    wheels = parse_wheel_hashes(golden_path_log)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "repositories": {
            "prob4d": {
                "repository": _strict_string(prob4d_repository, name="prob4d_repository"),
                "revision": _strict_digest(prob4d_revision, name="prob4d_revision", git=True),
            },
            "bayesian_phystwin": {
                "repository": _strict_string(
                    bayesian_phystwin_repository,
                    name="bayesian_phystwin_repository",
                ),
                "revision": _strict_digest(
                    bayesian_phystwin_revision,
                    name="bayesian_phystwin_revision",
                    git=True,
                ),
            },
            "causal4d": {
                "repository": _strict_string(causal4d_repository, name="causal4d_repository"),
                "revision": _strict_digest(causal4d_revision, name="causal4d_revision", git=True),
            },
        },
        "wheels": wheels,
        "contracts": {
            "prob4d_provider_api": 2,
            "observation_belief": 1,
            "observation_factor_bundle": 4,
            "observation_factor_stream": 1,
        },
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


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    keys = set(mapping)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return value


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
            "execution",
            "claim_boundary",
            "capsule_id",
        },
        name="capsule",
    )
    if _strict_string(payload["schema"], name="schema") != SCHEMA:
        raise ValueError("unsupported ecosystem release capsule schema")
    schema_version = _strict_integer(
        payload["schema_version"],
        name="schema_version",
        minimum=1,
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported ecosystem release capsule schema")
    if _strict_string(payload["status"], name="status") != "passed":
        raise ValueError("capsule status must be 'passed'")
    if _strict_string(payload["claim_boundary"], name="claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("capsule claim boundary is not canonical")

    repositories = _strict_mapping(payload["repositories"], name="repositories")
    _exact_keys(repositories, {"prob4d", "bayesian_phystwin", "causal4d"}, name="repositories")
    for component, record_value in repositories.items():
        record = _strict_mapping(record_value, name=f"repositories.{component}")
        _exact_keys(record, {"repository", "revision"}, name=f"repositories.{component}")
        _strict_string(record["repository"], name=f"repositories.{component}.repository")
        _strict_digest(record["revision"], name=f"repositories.{component}.revision", git=True)

    wheels = _strict_mapping(payload["wheels"], name="wheels")
    _exact_keys(wheels, {"prob4d", "bayesian_phystwin", "causal4d"}, name="wheels")
    for component, record_value in wheels.items():
        record = _strict_mapping(record_value, name=f"wheels.{component}")
        _exact_keys(record, {"filename", "sha256"}, name=f"wheels.{component}")
        filename = _strict_string(record["filename"], name=f"wheels.{component}.filename")
        expected_prefix = dict(_WHEEL_PREFIXES)[component]
        if not filename.lower().startswith(expected_prefix) or not filename.endswith(".whl"):
            raise ValueError(f"wheels.{component}.filename is not the expected wheel")
        _strict_digest(record["sha256"], name=f"wheels.{component}.sha256")

    contracts = _strict_mapping(payload["contracts"], name="contracts")
    expected_contracts = {
        "prob4d_provider_api": 2,
        "observation_belief": 1,
        "observation_factor_bundle": 4,
        "observation_factor_stream": 1,
    }
    _exact_keys(contracts, set(expected_contracts), name="contracts")
    for key, expected in expected_contracts.items():
        if _strict_integer(contracts[key], name=f"contracts.{key}", minimum=1) != expected:
            raise ValueError("capsule contract versions are not canonical")

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
    return json.loads(_canonical_json(payload))


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
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_command(arguments: argparse.Namespace) -> int:
    log_path = Path(arguments.golden_path_log)
    capsule = build_capsule(
        golden_path_log=log_path.read_text(encoding="utf-8"),
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
    print(capsule["capsule_id"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a capsule from a passed golden-path log")
    build.add_argument("--golden-path-log", required=True)
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
    verify.set_defaults(handler=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
