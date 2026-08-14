"""Freeze, evaluate, replay, and authorize comparative provider readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ._atomic_file import atomic_write_text
from ._immutable_json import plain_json
from ._provider_readiness_matrix_core import (
    PROVIDER_READINESS_MATRIX_AUTHORIZATION_SCHEMA,
    PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY,
    PROVIDER_READINESS_MATRIX_DECISION_SCHEMA,
    PROVIDER_READINESS_MATRIX_LOCK_SCHEMA,
    PROVIDER_READINESS_MATRIX_REQUEST_SCHEMA,
    PROVIDER_READINESS_MATRIX_SELECTION_RULE,
    PROVIDER_READINESS_MATRIX_VERSION,
    ProviderReadinessMatrixAuthorizationV1,
    ProviderReadinessMatrixDecisionV1,
    ProviderReadinessMatrixEntryV1,
    ProviderReadinessMatrixLockV1,
    ProviderReadinessMatrixProviderV1,
    ProviderReadinessMatrixRequestV1,
    authorize_provider_readiness_matrix_target,
    evaluate_provider_readiness_matrix,
    readiness_matrix_provider_metadata,
)
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_finite_json_mapping,
    require_mapping,
)
from .fresh_provider_readiness import load_fresh_provider_readiness_decision

PROVIDER_READINESS_MATRIX_LOCK_SPEC_SCHEMA: Final = (
    "prob4d.provider-readiness-matrix-lock-spec"
)
PROVIDER_READINESS_MATRIX_DECISION_SPEC_SCHEMA: Final = (
    "prob4d.provider-readiness-matrix-decision-spec"
)

_LOCK_SPEC_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "matrix_id",
        "selection_rule",
        "maximum_target_evaluations",
        "source_repository",
        "source_revision",
        "cohort_binding_id",
        "query_definition_id",
        "fallback_identity_id",
        "development_group_ids",
        "calibration_group_ids",
        "target_group_ids",
        "confirmation_group_ids",
        "comparison_policy",
        "providers",
        "source_payloads_opened",
        "source_outcomes_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "confirmation_payloads_opened",
        "metadata",
    }
)
_PROVIDER_SPEC_FIELDS: Final = frozenset(
    {
        "provider_id",
        "priority",
        "provider_repository",
        "provider_revision",
        "model_set_id",
        "loader_id",
        "promotion_lock_id",
        "adapter_identity_id",
        "adapter_conformance_id",
        "metadata",
    }
)
_DECISION_SPEC_FIELDS: Final = frozenset(
    {"schema", "schema_version", "matrix_lock_path", "entries", "metadata"}
)
_DECISION_ENTRY_SPEC_FIELDS: Final = frozenset(
    {"provider_id", "decision_path", "metadata"}
)


def _read_bytes(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name} {path}") from error


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_bytes(path, name="matrix input")).hexdigest()


def _json_array(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    if any(type(item) is not str or not item for item in value):
        raise ValueError(f"{name} must contain nonempty strings")
    return tuple(value)


def _safe_member(root: Path, value: object, *, name: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty relative path")
    if "\\" in value:
        raise ValueError(f"{name} must use POSIX separators")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a confined relative path")
    candidate = root.joinpath(*pure.parts)
    current = root.resolve()
    for part in pure.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} escapes the specification directory") from error
    return resolved


def _write(path: str | Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def build_provider_readiness_matrix_lock(
    specification_path: str | Path,
) -> ProviderReadinessMatrixLockV1:
    """Freeze one complete provider comparison before source execution."""

    path = Path(specification_path)
    record = load_json_object(path, name="provider-readiness matrix lock specification")
    require_exact_fields(
        record,
        _LOCK_SPEC_FIELDS,
        name="provider-readiness matrix lock specification",
    )
    if record["schema"] != PROVIDER_READINESS_MATRIX_LOCK_SPEC_SCHEMA:
        raise ValueError("provider-readiness matrix lock specification schema changed")
    if record["schema_version"] != PROVIDER_READINESS_MATRIX_VERSION:
        raise ValueError("provider-readiness matrix lock specification version changed")
    raw_providers = record["providers"]
    if not isinstance(raw_providers, list) or not raw_providers:
        raise ValueError("matrix lock specification requires providers")
    providers: list[ProviderReadinessMatrixProviderV1] = []
    for index, value in enumerate(raw_providers):
        provider = require_mapping(value, name=f"matrix provider specification {index}")
        require_exact_fields(
            provider,
            _PROVIDER_SPEC_FIELDS,
            name=f"matrix provider specification {index}",
        )
        providers.append(
            ProviderReadinessMatrixProviderV1(
                provider_id=provider["provider_id"],
                priority=provider["priority"],
                provider_repository=provider["provider_repository"],
                provider_revision=provider["provider_revision"],
                model_set_id=provider["model_set_id"],
                loader_id=provider["loader_id"],
                promotion_lock_id=provider["promotion_lock_id"],
                adapter_identity_id=provider["adapter_identity_id"],
                adapter_conformance_id=provider["adapter_conformance_id"],
                metadata=require_finite_json_mapping(
                    provider["metadata"],
                    name=f"matrix provider specification {index} metadata",
                ),
            )
        )
    return ProviderReadinessMatrixLockV1(
        matrix_id=record["matrix_id"],
        source_spec_sha256=_sha256_file(path),
        selection_rule=record["selection_rule"],
        maximum_target_evaluations=record["maximum_target_evaluations"],
        source_repository=record["source_repository"],
        source_revision=record["source_revision"],
        cohort_binding_id=record["cohort_binding_id"],
        query_definition_id=record["query_definition_id"],
        fallback_identity_id=record["fallback_identity_id"],
        development_group_ids=_json_array(
            record["development_group_ids"],
            name="development_group_ids",
        ),
        calibration_group_ids=_json_array(
            record["calibration_group_ids"],
            name="calibration_group_ids",
        ),
        target_group_ids=_json_array(
            record["target_group_ids"],
            name="target_group_ids",
        ),
        confirmation_group_ids=_json_array(
            record["confirmation_group_ids"],
            name="confirmation_group_ids",
        ),
        comparison_policy=require_finite_json_mapping(
            record["comparison_policy"],
            name="matrix comparison policy",
        ),
        providers=tuple(providers),
        source_payloads_opened=record["source_payloads_opened"],
        source_outcomes_opened=record["source_outcomes_opened"],
        target_payloads_opened=record["target_payloads_opened"],
        target_outcomes_opened=record["target_outcomes_opened"],
        confirmation_payloads_opened=record["confirmation_payloads_opened"],
        metadata=require_finite_json_mapping(
            record["metadata"],
            name="matrix lock metadata",
        ),
    )


def write_provider_readiness_matrix_lock(
    path: str | Path,
    lock: ProviderReadinessMatrixLockV1,
    *,
    overwrite: bool = False,
) -> None:
    _write(path, lock.to_dict(), overwrite=overwrite)


def load_provider_readiness_matrix_lock(path: str | Path) -> ProviderReadinessMatrixLockV1:
    return ProviderReadinessMatrixLockV1.from_dict(
        load_json_object(path, name="provider-readiness matrix lock")
    )


def build_provider_readiness_matrix_request(
    specification_path: str | Path,
) -> ProviderReadinessMatrixRequestV1:
    """Bind the frozen matrix to one exact source-only decision per provider."""

    path = Path(specification_path)
    record = load_json_object(path, name="provider-readiness matrix decision specification")
    require_exact_fields(
        record,
        _DECISION_SPEC_FIELDS,
        name="provider-readiness matrix decision specification",
    )
    if record["schema"] != PROVIDER_READINESS_MATRIX_DECISION_SPEC_SCHEMA:
        raise ValueError("provider-readiness matrix decision specification schema changed")
    if record["schema_version"] != PROVIDER_READINESS_MATRIX_VERSION:
        raise ValueError("provider-readiness matrix decision specification version changed")
    root = path.parent.resolve()
    lock_path = _safe_member(root, record["matrix_lock_path"], name="matrix_lock_path")
    lock = load_provider_readiness_matrix_lock(lock_path)
    raw_entries = record["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("matrix decision specification requires entries")
    entries: list[ProviderReadinessMatrixEntryV1] = []
    for index, value in enumerate(raw_entries):
        entry = require_mapping(value, name=f"matrix decision entry {index}")
        require_exact_fields(
            entry,
            _DECISION_ENTRY_SPEC_FIELDS,
            name=f"matrix decision entry {index}",
        )
        provider_id = entry["provider_id"]
        provider = lock.provider(provider_id)
        decision_path = _safe_member(
            root,
            entry["decision_path"],
            name=f"matrix decision entry {index} path",
        )
        entries.append(
            ProviderReadinessMatrixEntryV1(
                provider_id=provider_id,
                priority=provider.priority,
                decision_file_sha256=_sha256_file(decision_path),
                decision=load_fresh_provider_readiness_decision(decision_path),
                metadata=require_finite_json_mapping(
                    entry["metadata"],
                    name=f"matrix decision entry {index} metadata",
                ),
            )
        )
    return ProviderReadinessMatrixRequestV1(
        matrix_lock_file_sha256=_sha256_file(lock_path),
        matrix_lock=lock,
        source_spec_sha256=_sha256_file(path),
        entries=tuple(entries),
        metadata=require_finite_json_mapping(
            record["metadata"],
            name="matrix request metadata",
        ),
    )


def write_provider_readiness_matrix_request(
    path: str | Path,
    request: ProviderReadinessMatrixRequestV1,
    *,
    overwrite: bool = False,
) -> None:
    _write(path, request.to_dict(), overwrite=overwrite)


def load_provider_readiness_matrix_request(
    path: str | Path,
) -> ProviderReadinessMatrixRequestV1:
    return ProviderReadinessMatrixRequestV1.from_dict(
        load_json_object(path, name="provider-readiness matrix request")
    )


def write_provider_readiness_matrix_decision(
    path: str | Path,
    decision: ProviderReadinessMatrixDecisionV1,
    *,
    overwrite: bool = False,
) -> None:
    _write(path, decision.to_dict(), overwrite=overwrite)


def load_provider_readiness_matrix_decision(
    path: str | Path,
) -> ProviderReadinessMatrixDecisionV1:
    return ProviderReadinessMatrixDecisionV1.from_dict(
        load_json_object(path, name="provider-readiness matrix decision")
    )


def write_provider_readiness_matrix_authorization(
    path: str | Path,
    authorization: ProviderReadinessMatrixAuthorizationV1,
    *,
    overwrite: bool = False,
) -> None:
    _write(path, authorization.to_dict(), overwrite=overwrite)


def load_provider_readiness_matrix_authorization(
    path: str | Path,
) -> ProviderReadinessMatrixAuthorizationV1:
    return ProviderReadinessMatrixAuthorizationV1.from_dict(
        load_json_object(path, name="provider-readiness matrix authorization")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--overwrite", action="store_true")
    verify_lock = subparsers.add_parser("verify-lock")
    verify_lock.add_argument("--artifact", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--spec", type=Path, required=True)
    evaluate.add_argument("--request-output", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    verify_request = subparsers.add_parser("verify-request")
    verify_request.add_argument("--artifact", type=Path, required=True)
    verify_decision = subparsers.add_parser("verify-decision")
    verify_decision.add_argument("--artifact", type=Path, required=True)
    authorize = subparsers.add_parser("authorize-target")
    authorize.add_argument("--decision", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--overwrite", action="store_true")
    verify_authorization = subparsers.add_parser("verify-authorization")
    verify_authorization.add_argument("--artifact", type=Path, required=True)
    return parser


def _summary(value: Mapping[str, Any]) -> None:
    print(json.dumps(plain_json(value), sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        lock = build_provider_readiness_matrix_lock(arguments.spec)
        write_provider_readiness_matrix_lock(
            arguments.output,
            lock,
            overwrite=arguments.overwrite,
        )
        _summary({"lock_id": lock.provider_readiness_matrix_lock_id})
        return 0
    if arguments.command == "verify-lock":
        lock = load_provider_readiness_matrix_lock(arguments.artifact)
        _summary({"lock_id": lock.provider_readiness_matrix_lock_id})
        return 0
    if arguments.command == "evaluate":
        request = build_provider_readiness_matrix_request(arguments.spec)
        decision = evaluate_provider_readiness_matrix(request)
        write_provider_readiness_matrix_request(
            arguments.request_output,
            request,
            overwrite=arguments.overwrite,
        )
        write_provider_readiness_matrix_decision(
            arguments.output,
            decision,
            overwrite=arguments.overwrite,
        )
        _summary(
            {
                "decision_id": decision.provider_readiness_matrix_decision_id,
                "matrix_status": decision.matrix_status,
                "selected_provider_id": decision.selected_provider_id,
            }
        )
        return 0 if decision.selected_provider_id is not None else 2
    if arguments.command == "verify-request":
        request = load_provider_readiness_matrix_request(arguments.artifact)
        _summary({"request_id": request.provider_readiness_matrix_request_id})
        return 0
    if arguments.command == "verify-decision":
        decision = load_provider_readiness_matrix_decision(arguments.artifact)
        _summary(
            {
                "decision_id": decision.provider_readiness_matrix_decision_id,
                "matrix_status": decision.matrix_status,
            }
        )
        return 0 if decision.selected_provider_id is not None else 2
    if arguments.command == "authorize-target":
        authorization = authorize_provider_readiness_matrix_target(
            load_provider_readiness_matrix_decision(arguments.decision)
        )
        write_provider_readiness_matrix_authorization(
            arguments.output,
            authorization,
            overwrite=arguments.overwrite,
        )
        _summary(
            {
                "authorization_id": authorization.provider_readiness_matrix_authorization_id,
                "selected_provider_id": authorization.selected_provider_id,
            }
        )
        return 0
    authorization = load_provider_readiness_matrix_authorization(arguments.artifact)
    _summary(
        {
            "authorization_id": authorization.provider_readiness_matrix_authorization_id,
            "selected_provider_id": authorization.selected_provider_id,
        }
    )
    return 0


__all__ = [
    "PROVIDER_READINESS_MATRIX_AUTHORIZATION_SCHEMA",
    "PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY",
    "PROVIDER_READINESS_MATRIX_DECISION_SCHEMA",
    "PROVIDER_READINESS_MATRIX_DECISION_SPEC_SCHEMA",
    "PROVIDER_READINESS_MATRIX_LOCK_SCHEMA",
    "PROVIDER_READINESS_MATRIX_LOCK_SPEC_SCHEMA",
    "PROVIDER_READINESS_MATRIX_REQUEST_SCHEMA",
    "PROVIDER_READINESS_MATRIX_SELECTION_RULE",
    "PROVIDER_READINESS_MATRIX_VERSION",
    "ProviderReadinessMatrixAuthorizationV1",
    "ProviderReadinessMatrixDecisionV1",
    "ProviderReadinessMatrixEntryV1",
    "ProviderReadinessMatrixLockV1",
    "ProviderReadinessMatrixProviderV1",
    "ProviderReadinessMatrixRequestV1",
    "authorize_provider_readiness_matrix_target",
    "build_provider_readiness_matrix_lock",
    "build_provider_readiness_matrix_request",
    "evaluate_provider_readiness_matrix",
    "load_provider_readiness_matrix_authorization",
    "load_provider_readiness_matrix_decision",
    "load_provider_readiness_matrix_lock",
    "load_provider_readiness_matrix_request",
    "main",
    "readiness_matrix_provider_metadata",
    "write_provider_readiness_matrix_authorization",
    "write_provider_readiness_matrix_decision",
    "write_provider_readiness_matrix_lock",
    "write_provider_readiness_matrix_request",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
