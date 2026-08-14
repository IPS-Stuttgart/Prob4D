"""Content-addressed provider terminal decisions and scientific stop rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

PROVIDER_TERMINAL_DECISION_SCHEMA: Final = "prob4d.provider-terminal-decision"
PROVIDER_TERMINAL_DECISION_VERSION: Final = 1

CLASSIFICATION_SUPPORT_NEGATIVE: Final = "support-negative"
CLASSIFICATION_BATCH_INCOMPATIBLE: Final = "batch-incompatible"
CLASSIFICATION_SCIENTIFIC_NEGATIVE: Final = "scientific-negative"
CLASSIFICATION_TECHNICAL_FAILURE: Final = "technical-failure"
CLASSIFICATION_COMPLETED_POSITIVE: Final = "completed-positive"

_VALID_CLASSIFICATIONS: Final = frozenset(
    {
        CLASSIFICATION_SUPPORT_NEGATIVE,
        CLASSIFICATION_BATCH_INCOMPATIBLE,
        CLASSIFICATION_SCIENTIFIC_NEGATIVE,
        CLASSIFICATION_TECHNICAL_FAILURE,
        CLASSIFICATION_COMPLETED_POSITIVE,
    }
)
_INFRASTRUCTURE_CLASSIFICATIONS: Final = frozenset(
    {CLASSIFICATION_BATCH_INCOMPATIBLE, CLASSIFICATION_TECHNICAL_FAILURE}
)
REQUIRED_INFRASTRUCTURE_NONCLAIMS: Final = frozenset(
    {
        "provider-competence",
        "provider-calibration",
        "bayesian-phystwin-benefit",
        "causal4d-intervention-benefit",
        "deployment-safety",
        "state-of-the-art",
    }
)


def _canonical_bytes(record: Mapping[str, object]) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_id(record: Mapping[str, object]) -> str:
    identity_record = dict(record)
    identity_record.pop("artifact_id", None)
    return hashlib.sha256(_canonical_bytes(identity_record)).hexdigest()


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    return None if value is None else _string(value, name=name)


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    result = tuple(_string(item, name=f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_fields(
    record: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    actual = frozenset(record)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"{name} contains non-finite number {token!r}")

    if path.is_symlink():
        raise ValueError(f"{name} is a symbolic link")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}") from error
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


@dataclass(frozen=True)
class ProviderTerminalDecisionV1:
    """Classification of one completed provider information boundary."""

    protocol_id: str
    provider_manifest_id: str
    classification: str
    failed_stage: str | None
    source_payloads_accessed: bool
    source_outcomes_accessed: bool
    target_payloads_accessed: bool
    target_outcomes_accessed: bool
    rerun_authorized: bool
    successor_protocol_required: bool
    evidence_ids: tuple[str, ...]
    authorized_inferences: tuple[str, ...]
    forbidden_inferences: tuple[str, ...]
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        _string(self.protocol_id, name="protocol_id")
        _string(self.provider_manifest_id, name="provider_manifest_id")
        _string(self.summary, name="summary")
        if self.classification not in _VALID_CLASSIFICATIONS:
            raise ValueError("unsupported provider terminal classification")
        if self.failed_stage is not None:
            _string(self.failed_stage, name="failed_stage")
        for name, roster in (
            ("evidence_ids", self.evidence_ids),
            ("authorized_inferences", self.authorized_inferences),
            ("forbidden_inferences", self.forbidden_inferences),
        ):
            if len(set(roster)) != len(roster):
                raise ValueError(f"{name} must not contain duplicates")
            for item in roster:
                _string(item, name=f"{name} item")

        # Report the scientifically important infrastructure rule before generic
        # overlap diagnostics, even when a malformed specification both
        # authorizes and forbids the same claim.
        if self.classification in _INFRASTRUCTURE_CLASSIFICATIONS:
            if self.authorized_inferences:
                raise ValueError(
                    "infrastructure terminal decisions authorize no scientific inference"
                )
            missing = REQUIRED_INFRASTRUCTURE_NONCLAIMS - set(
                self.forbidden_inferences
            )
            if missing:
                raise ValueError(
                    "infrastructure terminal decision omits forbidden claims: "
                    + ", ".join(sorted(missing))
                )
            if self.target_outcomes_accessed:
                raise ValueError(
                    "batch or technical failure must precede target outcomes"
                )

        if set(self.authorized_inferences) & set(self.forbidden_inferences):
            raise ValueError("an inference cannot be both authorized and forbidden")
        if self.source_outcomes_accessed and not self.source_payloads_accessed:
            raise ValueError("source outcomes require source payload access")
        if self.target_outcomes_accessed and not self.target_payloads_accessed:
            raise ValueError("target outcomes require target payload access")
        if self.target_payloads_accessed and not self.source_payloads_accessed:
            raise ValueError("target payload access requires source payload access")

        if self.classification == CLASSIFICATION_SUPPORT_NEGATIVE:
            if self.source_payloads_accessed or self.target_payloads_accessed:
                raise ValueError("support-negative decision must precede payload access")
            if self.authorized_inferences != ("provider-support-negative",):
                raise ValueError(
                    "support-negative authorizes only provider-support-negative"
                )
        if self.classification == CLASSIFICATION_SCIENTIFIC_NEGATIVE:
            if not (self.source_outcomes_accessed or self.target_outcomes_accessed):
                raise ValueError("scientific-negative requires opened outcome evidence")
            if not self.authorized_inferences:
                raise ValueError("scientific-negative requires a declared inference")
        if self.classification == CLASSIFICATION_COMPLETED_POSITIVE:
            if not self.target_outcomes_accessed:
                raise ValueError("completed-positive requires target outcome access")
            if not self.authorized_inferences:
                raise ValueError("completed-positive requires a declared inference")
        if self.target_outcomes_accessed and self.rerun_authorized:
            raise ValueError("opened target outcomes cannot authorize a rerun")
        if (
            self.classification
            in {
                CLASSIFICATION_SUPPORT_NEGATIVE,
                CLASSIFICATION_BATCH_INCOMPATIBLE,
                CLASSIFICATION_TECHNICAL_FAILURE,
            }
            and not self.successor_protocol_required
        ):
            raise ValueError("terminal infrastructure boundary requires a successor protocol")

        _canonical_bytes(dict(self.metadata))
        expected = _artifact_id(self.to_record(include_artifact_id=False))
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected)
        elif self.artifact_id != expected:
            raise ValueError("provider terminal decision artifact ID mismatch")

    @property
    def scientific_result(self) -> bool:
        return self.classification in {
            CLASSIFICATION_SCIENTIFIC_NEGATIVE,
            CLASSIFICATION_COMPLETED_POSITIVE,
        }

    def to_record(self, *, include_artifact_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": PROVIDER_TERMINAL_DECISION_SCHEMA,
            "schema_version": PROVIDER_TERMINAL_DECISION_VERSION,
            "protocol_id": self.protocol_id,
            "provider_manifest_id": self.provider_manifest_id,
            "classification": self.classification,
            "failed_stage": self.failed_stage,
            "source_payloads_accessed": self.source_payloads_accessed,
            "source_outcomes_accessed": self.source_outcomes_accessed,
            "target_payloads_accessed": self.target_payloads_accessed,
            "target_outcomes_accessed": self.target_outcomes_accessed,
            "rerun_authorized": self.rerun_authorized,
            "successor_protocol_required": self.successor_protocol_required,
            "evidence_ids": list(self.evidence_ids),
            "authorized_inferences": list(self.authorized_inferences),
            "forbidden_inferences": list(self.forbidden_inferences),
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }
        if include_artifact_id:
            record["artifact_id"] = self.artifact_id
        return record

    @classmethod
    def from_record(cls, value: object) -> ProviderTerminalDecisionV1:
        record = _mapping(value, name="provider terminal decision")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "protocol_id",
                "provider_manifest_id",
                "classification",
                "failed_stage",
                "source_payloads_accessed",
                "source_outcomes_accessed",
                "target_payloads_accessed",
                "target_outcomes_accessed",
                "rerun_authorized",
                "successor_protocol_required",
                "evidence_ids",
                "authorized_inferences",
                "forbidden_inferences",
                "summary",
                "metadata",
                "artifact_id",
            }
        )
        _exact_fields(record, expected, name="provider terminal decision")
        if record["schema"] != PROVIDER_TERMINAL_DECISION_SCHEMA:
            raise ValueError("unsupported provider terminal decision schema")
        if record["schema_version"] != PROVIDER_TERMINAL_DECISION_VERSION:
            raise ValueError("unsupported provider terminal decision version")
        return cls(
            protocol_id=_string(record["protocol_id"], name="protocol_id"),
            provider_manifest_id=_string(
                record["provider_manifest_id"], name="provider_manifest_id"
            ),
            classification=_string(record["classification"], name="classification"),
            failed_stage=_optional_string(record["failed_stage"], name="failed_stage"),
            source_payloads_accessed=_boolean(
                record["source_payloads_accessed"], name="source_payloads_accessed"
            ),
            source_outcomes_accessed=_boolean(
                record["source_outcomes_accessed"], name="source_outcomes_accessed"
            ),
            target_payloads_accessed=_boolean(
                record["target_payloads_accessed"], name="target_payloads_accessed"
            ),
            target_outcomes_accessed=_boolean(
                record["target_outcomes_accessed"], name="target_outcomes_accessed"
            ),
            rerun_authorized=_boolean(
                record["rerun_authorized"], name="rerun_authorized"
            ),
            successor_protocol_required=_boolean(
                record["successor_protocol_required"],
                name="successor_protocol_required",
            ),
            evidence_ids=_string_tuple(record["evidence_ids"], name="evidence_ids"),
            authorized_inferences=_string_tuple(
                record["authorized_inferences"], name="authorized_inferences"
            ),
            forbidden_inferences=_string_tuple(
                record["forbidden_inferences"], name="forbidden_inferences"
            ),
            summary=_string(record["summary"], name="summary"),
            metadata=dict(_mapping(record["metadata"], name="metadata")),
            artifact_id=_string(record["artifact_id"], name="artifact_id"),
        )


def write_provider_terminal_decision(
    path: str | Path, decision: ProviderTerminalDecisionV1
) -> Path:
    destination_input = Path(path)
    if destination_input.is_symlink():
        raise ValueError("terminal decision output is a symbolic link")
    destination = destination_input.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(decision.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") == content:
            return destination
        raise FileExistsError("refusing to replace a different terminal decision")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_provider_terminal_decision(path: str | Path) -> ProviderTerminalDecisionV1:
    return ProviderTerminalDecisionV1.from_record(
        _strict_json(Path(path), name="provider terminal decision artifact")
    )


def build_provider_terminal_decision(
    specification_path: str | Path,
) -> ProviderTerminalDecisionV1:
    specification = _strict_json(
        Path(specification_path), name="provider terminal decision specification"
    )
    record = dict(specification)
    record["artifact_id"] = _artifact_id(record)
    return ProviderTerminalDecisionV1.from_record(record)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or verify provider terminal decision")
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build")
    build.add_argument("specification")
    build.add_argument("output")
    verify = actions.add_parser("verify")
    verify.add_argument("artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    if arguments.action == "verify":
        decision = load_provider_terminal_decision(arguments.artifact)
    else:
        decision = build_provider_terminal_decision(arguments.specification)
        write_provider_terminal_decision(arguments.output, decision)
    print(
        json.dumps(
            {
                "artifact_id": decision.artifact_id,
                "classification": decision.classification,
                "scientific_result": decision.scientific_result,
                "target_outcomes_accessed": decision.target_outcomes_accessed,
                "authorized_inferences": list(decision.authorized_inferences),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CLASSIFICATION_BATCH_INCOMPATIBLE",
    "CLASSIFICATION_COMPLETED_POSITIVE",
    "CLASSIFICATION_SCIENTIFIC_NEGATIVE",
    "CLASSIFICATION_SUPPORT_NEGATIVE",
    "CLASSIFICATION_TECHNICAL_FAILURE",
    "PROVIDER_TERMINAL_DECISION_SCHEMA",
    "PROVIDER_TERMINAL_DECISION_VERSION",
    "REQUIRED_INFRASTRUCTURE_NONCLAIMS",
    "ProviderTerminalDecisionV1",
    "build_provider_terminal_decision",
    "load_provider_terminal_decision",
    "main",
    "write_provider_terminal_decision",
]


if __name__ == "__main__":
    raise SystemExit(main())
