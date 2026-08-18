"""Provider-neutral, content-addressed external execution attestations.

The contract binds how an external 4-D provider was executed without importing
that provider or its runtime stack. It records exact code and model identities,
command arguments, causal declarations, runtime fingerprints, and input/output
bytes. Validation establishes artifact integrity and declared execution lineage;
it does not establish provider accuracy, calibration, physical-query benefit,
or causal-intervention benefit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, cast

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_revision,
    require_sha256,
)

PROVIDER_EXECUTION_ATTESTATION_SCHEMA: Final = "prob4d.provider-execution-attestation"
PROVIDER_EXECUTION_ATTESTATION_VERSION: Final = 1
PROVIDER_EXECUTION_ATTESTATION_CLAIM_BOUNDARY: Final = (
    "This artifact binds declared external-provider execution identity, runtime "
    "fingerprints, causal settings, and exact input/output bytes. It does not by "
    "itself prove that an untrusted wrapper executed those settings, establish "
    "provider accuracy or uncertainty calibration, authorize protected target "
    "access, justify a BayesianPhysTwin update, establish Causal4D intervention "
    "benefit, deployment safety, or state of the art."
)

TerminalStatus = Literal["succeeded", "failed", "cancelled"]
ExecutionEvidenceMode = Literal["declarative-only-v1", "wrapper-observed-v1"]

_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UTC_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)

_SPEC_FIELDS = frozenset(
    {
        "provider_repository",
        "provider_revision",
        "provider_run_id",
        "model_set_id",
        "loader_id",
        "execution_mode",
        "command_argv",
        "causal_declarations",
        "runtime",
        "environment_variables",
        "input_artifacts",
        "output_artifacts",
        "execution_evidence_mode",
        "execution_evidence_sha256",
        "prediction_provider_manifest_id",
        "started_at_utc",
        "completed_at_utc",
        "terminal_status",
        "metadata",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        *_SPEC_FIELDS,
        "execution_evidence_complete",
        "claim_boundary",
        "provider_execution_attestation_id",
    }
)
_CAUSAL_FIELDS = frozenset(
    {
        "source_order_preserved",
        "online_prefix_only",
        "revisit_count",
        "global_alignment",
        "future_frame_postprocessing",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "python_version",
        "implementation",
        "platform",
        "container_image_digest",
        "environment_lock_sha256",
    }
)
_ENVIRONMENT_FIELDS = frozenset({"name", "value_sha256"})
_ARTIFACT_FIELDS = frozenset({"name", "sha256", "byte_count"})


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


def _attestation_id(value: Mapping[str, Any]) -> str:
    unsigned = dict(plain_json(value))
    unsigned.pop("provider_execution_attestation_id", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _nullable_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_sha256(value, name=name)


def _container_digest(value: object) -> str | None:
    if value is None:
        return None
    digest = require_exact_string(value, name="runtime.container_image_digest")
    if not digest.startswith("sha256:"):
        raise ValueError("runtime.container_image_digest must use the sha256:<digest> form")
    require_sha256(digest.removeprefix("sha256:"), name="runtime.container_image_digest")
    return digest


def _utc_timestamp(value: object, *, name: str) -> tuple[str, datetime]:
    text = require_exact_string(value, name=name)
    if _UTC_TIMESTAMP.fullmatch(text) is None:
        raise ValueError(f"{name} must be a canonical UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be a valid UTC timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{name} must be UTC")
    return text, parsed


def _repository(value: object) -> str:
    repository = require_exact_string(value, name="provider_repository")
    if _REPOSITORY.fullmatch(repository) is None:
        raise ValueError("provider_repository must use owner/name form")
    owner, name = repository.split("/", maxsplit=1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise ValueError("provider_repository must use a canonical owner/name")
    return repository


def _command_argv(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("command_argv must be a nonempty JSON array")
    result: list[str] = []
    for index, item in enumerate(value):
        argument = require_exact_string(item, name=f"command_argv[{index}]")
        if "\x00" in argument:
            raise ValueError("command_argv must not contain NUL characters")
        result.append(argument)
    return result


def _causal_declarations(value: object) -> dict[str, object]:
    mapping = require_mapping(value, name="causal_declarations")
    require_exact_fields(mapping, _CAUSAL_FIELDS, name="causal_declarations")
    return {
        "source_order_preserved": _exact_bool(
            mapping["source_order_preserved"],
            name="causal_declarations.source_order_preserved",
        ),
        "online_prefix_only": _exact_bool(
            mapping["online_prefix_only"],
            name="causal_declarations.online_prefix_only",
        ),
        "revisit_count": require_exact_integer(
            mapping["revisit_count"],
            name="causal_declarations.revisit_count",
            minimum=0,
        ),
        "global_alignment": _exact_bool(
            mapping["global_alignment"],
            name="causal_declarations.global_alignment",
        ),
        "future_frame_postprocessing": _exact_bool(
            mapping["future_frame_postprocessing"],
            name="causal_declarations.future_frame_postprocessing",
        ),
    }


def _runtime(value: object) -> dict[str, object]:
    mapping = require_mapping(value, name="runtime")
    require_exact_fields(mapping, _RUNTIME_FIELDS, name="runtime")
    return {
        "python_version": require_exact_string(
            mapping["python_version"],
            name="runtime.python_version",
        ),
        "implementation": require_exact_string(
            mapping["implementation"],
            name="runtime.implementation",
        ),
        "platform": require_exact_string(mapping["platform"], name="runtime.platform"),
        "container_image_digest": _container_digest(mapping["container_image_digest"]),
        "environment_lock_sha256": _nullable_sha256(
            mapping["environment_lock_sha256"],
            name="runtime.environment_lock_sha256",
        ),
    }


def _environment_variables(value: object) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("environment_variables must be a JSON array")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        mapping = require_mapping(item, name=f"environment_variables[{index}]")
        require_exact_fields(
            mapping,
            _ENVIRONMENT_FIELDS,
            name=f"environment_variables[{index}]",
        )
        name = require_exact_string(
            mapping["name"],
            name=f"environment_variables[{index}].name",
        )
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise ValueError("environment variable names must be portable identifiers")
        if name in names:
            raise ValueError(f"duplicate environment variable name: {name}")
        names.add(name)
        result.append(
            {
                "name": name,
                "value_sha256": require_sha256(
                    mapping["value_sha256"],
                    name=f"environment_variables[{index}].value_sha256",
                ),
            }
        )
    return sorted(result, key=lambda record: cast(str, record["name"]))


def _artifacts(value: object, *, name: str) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        mapping = require_mapping(item, name=f"{name}[{index}]")
        require_exact_fields(mapping, _ARTIFACT_FIELDS, name=f"{name}[{index}]")
        artifact_name = require_exact_string(
            mapping["name"],
            name=f"{name}[{index}].name",
        )
        if artifact_name in names:
            raise ValueError(f"duplicate {name} name: {artifact_name}")
        names.add(artifact_name)
        result.append(
            {
                "name": artifact_name,
                "sha256": require_sha256(
                    mapping["sha256"],
                    name=f"{name}[{index}].sha256",
                ),
                "byte_count": require_exact_integer(
                    mapping["byte_count"],
                    name=f"{name}[{index}].byte_count",
                    minimum=0,
                ),
            }
        )
    return sorted(result, key=lambda record: cast(str, record["name"]))


def _evidence_mode(value: object) -> ExecutionEvidenceMode:
    mode = require_exact_string(value, name="execution_evidence_mode")
    if mode not in {"declarative-only-v1", "wrapper-observed-v1"}:
        raise ValueError("unsupported execution_evidence_mode")
    return cast(ExecutionEvidenceMode, mode)


def _terminal_status(value: object) -> TerminalStatus:
    status = require_exact_string(value, name="terminal_status")
    if status not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("unsupported terminal_status")
    return cast(TerminalStatus, status)


def _normalize_spec(value: object) -> dict[str, Any]:
    mapping = require_mapping(value, name="provider execution attestation specification")
    require_exact_fields(
        mapping,
        _SPEC_FIELDS,
        name="provider execution attestation specification",
    )
    started_text, started = _utc_timestamp(mapping["started_at_utc"], name="started_at_utc")
    completed_text, completed = _utc_timestamp(
        mapping["completed_at_utc"],
        name="completed_at_utc",
    )
    if completed < started:
        raise ValueError("completed_at_utc must not precede started_at_utc")

    evidence_mode = _evidence_mode(mapping["execution_evidence_mode"])
    evidence_sha = _nullable_sha256(
        mapping["execution_evidence_sha256"],
        name="execution_evidence_sha256",
    )
    if evidence_mode == "wrapper-observed-v1" and evidence_sha is None:
        raise ValueError("wrapper-observed execution evidence requires its exact SHA-256")
    if evidence_mode == "declarative-only-v1" and evidence_sha is not None:
        raise ValueError("declarative-only execution evidence cannot claim wrapper bytes")

    status = _terminal_status(mapping["terminal_status"])
    outputs = _artifacts(mapping["output_artifacts"], name="output_artifacts")
    if status == "succeeded" and not outputs:
        raise ValueError("a successful provider execution must bind at least one output artifact")

    manifest_id = _nullable_sha256(
        mapping["prediction_provider_manifest_id"],
        name="prediction_provider_manifest_id",
    )
    metadata = require_finite_json_mapping(mapping["metadata"], name="metadata")
    return {
        "provider_repository": _repository(mapping["provider_repository"]),
        "provider_revision": require_revision(
            mapping["provider_revision"],
            name="provider_revision",
        ),
        "provider_run_id": require_sha256(
            mapping["provider_run_id"],
            name="provider_run_id",
        ),
        "model_set_id": require_sha256(mapping["model_set_id"], name="model_set_id"),
        "loader_id": require_sha256(mapping["loader_id"], name="loader_id"),
        "execution_mode": require_exact_string(
            mapping["execution_mode"],
            name="execution_mode",
        ),
        "command_argv": _command_argv(mapping["command_argv"]),
        "causal_declarations": _causal_declarations(mapping["causal_declarations"]),
        "runtime": _runtime(mapping["runtime"]),
        "environment_variables": _environment_variables(mapping["environment_variables"]),
        "input_artifacts": _artifacts(mapping["input_artifacts"], name="input_artifacts"),
        "output_artifacts": outputs,
        "execution_evidence_mode": evidence_mode,
        "execution_evidence_sha256": evidence_sha,
        "prediction_provider_manifest_id": manifest_id,
        "started_at_utc": started_text,
        "completed_at_utc": completed_text,
        "terminal_status": status,
        "metadata": plain_json(metadata),
    }


def build_provider_execution_attestation(value: object) -> Mapping[str, Any]:
    """Build one canonical content-addressed execution attestation from a strict spec."""

    specification = _normalize_spec(value)
    runtime = cast(Mapping[str, object], specification["runtime"])
    runtime_identity_bound = bool(
        runtime["container_image_digest"] is not None
        or runtime["environment_lock_sha256"] is not None
    )
    evidence_complete = bool(
        specification["execution_evidence_mode"] == "wrapper-observed-v1"
        and specification["execution_evidence_sha256"] is not None
        and specification["terminal_status"] == "succeeded"
        and specification["prediction_provider_manifest_id"] is not None
        and specification["input_artifacts"]
        and specification["output_artifacts"]
        and runtime_identity_bound
    )
    payload: dict[str, Any] = {
        "schema": PROVIDER_EXECUTION_ATTESTATION_SCHEMA,
        "schema_version": PROVIDER_EXECUTION_ATTESTATION_VERSION,
        **specification,
        "execution_evidence_complete": evidence_complete,
        "claim_boundary": PROVIDER_EXECUTION_ATTESTATION_CLAIM_BOUNDARY,
    }
    payload["provider_execution_attestation_id"] = _attestation_id(payload)
    return frozen_finite_json_mapping(payload, name="provider execution attestation")


def validate_provider_execution_attestation(value: object) -> Mapping[str, Any]:
    """Validate an attestation, including canonical ordering and its content address."""

    mapping = require_mapping(value, name="provider execution attestation")
    require_exact_fields(mapping, _ATTESTATION_FIELDS, name="provider execution attestation")
    if mapping["schema"] != PROVIDER_EXECUTION_ATTESTATION_SCHEMA:
        raise ValueError("unsupported provider execution attestation schema")
    schema_version = require_exact_integer(
        mapping["schema_version"],
        name="provider execution attestation schema_version",
        minimum=1,
    )
    if schema_version != PROVIDER_EXECUTION_ATTESTATION_VERSION:
        raise ValueError("unsupported provider execution attestation version")
    if mapping["claim_boundary"] != PROVIDER_EXECUTION_ATTESTATION_CLAIM_BOUNDARY:
        raise ValueError("provider execution attestation claim boundary changed")
    _exact_bool(
        mapping["execution_evidence_complete"],
        name="execution_evidence_complete",
    )

    specification = {name: mapping[name] for name in _SPEC_FIELDS}
    rebuilt = build_provider_execution_attestation(specification)
    expected = plain_json(rebuilt)
    observed = plain_json(mapping)
    if observed != expected:
        declared = require_sha256(
            mapping["provider_execution_attestation_id"],
            name="provider_execution_attestation_id",
        )
        if declared != _attestation_id(mapping):
            raise ValueError("provider execution attestation ID does not match its content")
        raise ValueError("provider execution attestation is not in canonical form")
    return rebuilt


def load_provider_execution_attestation(path: str | Path) -> Mapping[str, Any]:
    """Load strict JSON and validate one execution attestation."""

    payload = load_json_object(path, name="provider execution attestation")
    return validate_provider_execution_attestation(payload)


def write_provider_execution_attestation(
    path: str | Path,
    attestation: object,
) -> Mapping[str, Any]:
    """Publish one canonical attestation atomically without replacing different bytes."""

    validated = validate_provider_execution_attestation(attestation)
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("provider execution attestation destination must not be a symbolic link")
    content = _canonical_json(validated, pretty=True).decode("utf-8")
    if destination.exists():
        existing = load_provider_execution_attestation(destination)
        if plain_json(existing) != plain_json(validated):
            raise FileExistsError(
                f"refusing to replace a different provider execution attestation: {destination}"
            )
        return existing
    atomic_write_text(destination, content, overwrite=False)
    reloaded = load_provider_execution_attestation(destination)
    if plain_json(reloaded) != plain_json(validated):
        raise RuntimeError("published provider execution attestation changed during verification")
    return reloaded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="build an attestation from a strict JSON spec")
    create.add_argument("specification", type=Path)
    create.add_argument("--output", type=Path, required=True)
    create.set_defaults(handler=_create_command)

    verify = subparsers.add_parser("verify", help="validate a persisted execution attestation")
    verify.add_argument("attestation", type=Path)
    verify.add_argument("--require-complete", action="store_true")
    verify.set_defaults(handler=_verify_command)
    return parser


def _create_command(arguments: argparse.Namespace) -> int:
    specification = load_json_object(
        arguments.specification,
        name="provider execution attestation specification",
    )
    attestation = build_provider_execution_attestation(specification)
    written = write_provider_execution_attestation(arguments.output, attestation)
    print(written["provider_execution_attestation_id"])
    return 0


def _verify_command(arguments: argparse.Namespace) -> int:
    attestation = load_provider_execution_attestation(arguments.attestation)
    if arguments.require_complete and attestation["execution_evidence_complete"] is not True:
        raise ValueError("provider execution attestation lacks complete wrapper-observed evidence")
    print(attestation["provider_execution_attestation_id"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify provider-neutral execution attestations."""

    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    return int(arguments.handler(arguments))


__all__ = [
    "PROVIDER_EXECUTION_ATTESTATION_CLAIM_BOUNDARY",
    "PROVIDER_EXECUTION_ATTESTATION_SCHEMA",
    "PROVIDER_EXECUTION_ATTESTATION_VERSION",
    "build_provider_execution_attestation",
    "load_provider_execution_attestation",
    "main",
    "validate_provider_execution_attestation",
    "write_provider_execution_attestation",
]


if __name__ == "__main__":
    raise SystemExit(main())
