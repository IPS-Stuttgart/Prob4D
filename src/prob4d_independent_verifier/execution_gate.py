"""Caller-bound execution gates for independently verified Proof4D artifacts.

An internally valid certificate is not sufficient for execution: without a
caller-owned expected context, a certificate could be replayed for a different
input, query, action, policy, support receipt, or fallback. This module verifies
the certificate first and then requires exact equality with a content-addressed
execution context supplied by the caller.

The module intentionally does not import :mod:`prob4d`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .orbit_advantage import (
    PROOF_CARRYING_ORBIT_KIND,
    ProofCarryingOrbitVerification,
    verify_proof_carrying_orbit_action,
)
from .proof_carrying import (
    PROOF_CARRYING_FACTOR_KIND,
    ProofCarryingFactorVerification,
    verify_proof_carrying_factor,
)

EXECUTION_CONTEXT_SCHEMA = "prob4d.proof4d-execution-context"
EXECUTION_CONTEXT_VERSION = 1
EXECUTION_GATE_SCHEMA = "prob4d.proof4d-execution-gate-verification"
EXECUTION_GATE_VERSION = 1
EXECUTION_GATE_IMPLEMENTATION = "prob4d-independent-execution-gate-v1"
EXECUTION_GATE_CLAIM_BOUNDARY = (
    "The gate verifies the certificate and exact equality with a content-addressed, "
    "caller-supplied execution context.",
    "This prevents an internally valid certificate from authorizing a different input, "
    "query, action, policy, support receipt, factor, or fallback.",
    "The gate does not authenticate the caller, establish that the context describes the "
    "physical world, or prevent replay when the caller intentionally reuses the same context.",
)

_MAX_JSON_BYTES = 16 * 1024**2
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOP_LEVEL_CONTEXT_KEYS = {
    "schema",
    "schema_version",
    "certificate_kind",
    "bindings",
    "context_id",
}
_LOCAL_BINDING_KEYS = {
    "certificate_id",
    "input_digest",
    "source_factor_digest",
    "query_id",
    "query_program_digest",
    "fallback_id",
}
_ORBIT_BINDING_KEYS = {
    "certificate_id",
    "input_digest",
    "shared_gauge_id",
    "support_receipt_digest",
    "candidate_action_id",
    "fallback_action_id",
    "candidate_loss_program_digest",
    "fallback_loss_program_digest",
    "fallback_id",
    "fallback_digest",
    "admission_policy_digest",
}
_DIGEST_BINDING_KEYS = {
    "certificate_id",
    "input_digest",
    "source_factor_digest",
    "query_program_digest",
    "support_receipt_digest",
    "candidate_loss_program_digest",
    "fallback_loss_program_digest",
    "fallback_digest",
    "admission_policy_digest",
}


class _InvalidExecutionGate(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Proof4DExecutionGateVerification:
    """Fail-closed result after certificate and caller-context verification."""

    decision: str
    valid: bool
    admitted: bool
    certificate_id: str | None
    context_id: str | None
    certificate_decision: str | None
    reason_codes: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EXECUTION_GATE_SCHEMA,
            "schema_version": EXECUTION_GATE_VERSION,
            "gate_implementation": EXECUTION_GATE_IMPLEMENTATION,
            "decision": self.decision,
            "valid": self.valid,
            "admitted": self.admitted,
            "certificate_id": self.certificate_id,
            "context_id": self.context_id,
            "certificate_decision": self.certificate_decision,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "claim_boundary": list(EXECUTION_GATE_CLAIM_BOUNDARY),
        }


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_execution_context_id(context: Mapping[str, object]) -> str:
    """Return the content ID after excluding ``context_id``."""

    payload = dict(context)
    payload.pop("context_id", None)
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def render_execution_context(context: Mapping[str, object]) -> str:
    """Render one execution context in canonical human-readable JSON form."""

    expected = compute_execution_context_id(context)
    if context.get("context_id") != expected:
        raise ValueError("context_id does not match the execution context contents")
    return (
        json.dumps(
            context,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _reject_constant(value: str) -> None:
    raise _InvalidExecutionGate(
        "non-finite-json-number",
        f"non-finite JSON number {value!r} is forbidden",
    )


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidExecutionGate("duplicate-json-key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_clone(value: Mapping[str, object]) -> dict[str, object]:
    try:
        cloned = json.loads(
            _canonical_bytes(dict(value)).decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_object,
        )
    except _InvalidExecutionGate:
        raise
    except (TypeError, ValueError) as error:
        raise _InvalidExecutionGate("invalid-json-value", str(error)) from error
    if type(cloned) is not dict:
        raise _InvalidExecutionGate("top-level-not-object", "value must be one JSON object")
    return cloned


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        size = path.stat().st_size
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _InvalidExecutionGate(f"{name}-unreadable", str(error)) from error
    if size > _MAX_JSON_BYTES:
        raise _InvalidExecutionGate(
            f"{name}-too-large",
            f"{name} exceeds {_MAX_JSON_BYTES} bytes",
        )
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_object,
        )
    except _InvalidExecutionGate:
        raise
    except json.JSONDecodeError as error:
        raise _InvalidExecutionGate(f"invalid-{name}-json", str(error)) from error
    if type(value) is not dict:
        raise _InvalidExecutionGate(
            "top-level-not-object",
            f"{name} must contain one JSON object",
        )
    return value


def _nonempty_text(value: object, *, name: str, maximum_length: int = 512) -> str:
    if type(value) is not str:
        raise _InvalidExecutionGate("invalid-string", f"{name} must be a string")
    if not value or value.strip() != value or len(value) > maximum_length:
        raise _InvalidExecutionGate(
            "invalid-string",
            f"{name} must be nonempty, trimmed, and at most {maximum_length} characters",
        )
    return value


def _digest(value: object, *, name: str) -> str:
    text = _nonempty_text(value, name=name, maximum_length=71)
    if _DIGEST_PATTERN.fullmatch(text) is None:
        raise _InvalidExecutionGate(
            "invalid-digest",
            f"{name} must be a lowercase sha256: digest",
        )
    return text


def _require_object(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _InvalidExecutionGate("invalid-object", f"{name} must be an object")
    return value


def _expected_binding_keys(certificate_kind: str) -> set[str]:
    if certificate_kind == PROOF_CARRYING_FACTOR_KIND:
        return _LOCAL_BINDING_KEYS
    if certificate_kind == PROOF_CARRYING_ORBIT_KIND:
        return _ORBIT_BINDING_KEYS
    raise _InvalidExecutionGate(
        "unsupported-certificate-kind",
        f"unsupported certificate kind {certificate_kind!r}",
    )


def _validate_bindings(
    certificate_kind: str,
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    bindings = _require_object(value, name=name)
    expected = _expected_binding_keys(certificate_kind)
    actual = set(bindings)
    if actual != expected:
        raise _InvalidExecutionGate(
            "unexpected-binding-keys",
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}",
        )
    normalized: dict[str, object] = {}
    for key in sorted(expected):
        item = bindings[key]
        if key == "support_receipt_digest" and item is None:
            normalized[key] = None
        elif key in _DIGEST_BINDING_KEYS:
            normalized[key] = _digest(item, name=f"{name}.{key}")
        else:
            normalized[key] = _nonempty_text(item, name=f"{name}.{key}")
    return normalized


def build_execution_context(
    *,
    certificate_kind: str,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    """Build a content-addressed caller-owned execution context."""

    kind = _nonempty_text(certificate_kind, name="certificate_kind")
    normalized = _validate_bindings(kind, dict(bindings), name="bindings")
    context: dict[str, object] = {
        "schema": EXECUTION_CONTEXT_SCHEMA,
        "schema_version": EXECUTION_CONTEXT_VERSION,
        "certificate_kind": kind,
        "bindings": normalized,
    }
    context["context_id"] = compute_execution_context_id(context)
    return context


def _verify_context(value: dict[str, object]) -> tuple[str, dict[str, object], str]:
    actual_keys = set(value)
    if actual_keys != _TOP_LEVEL_CONTEXT_KEYS:
        raise _InvalidExecutionGate(
            "unexpected-context-keys",
            f"execution context keys differ; missing="
            f"{sorted(_TOP_LEVEL_CONTEXT_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _TOP_LEVEL_CONTEXT_KEYS)}",
        )
    if value["schema"] != EXECUTION_CONTEXT_SCHEMA:
        raise _InvalidExecutionGate("unsupported-context-schema", "unsupported context schema")
    if value["schema_version"] != EXECUTION_CONTEXT_VERSION:
        raise _InvalidExecutionGate(
            "unsupported-context-version",
            "unsupported context schema version",
        )
    kind = _nonempty_text(value["certificate_kind"], name="certificate_kind")
    bindings = _validate_bindings(kind, value["bindings"], name="bindings")
    context_id = _digest(value["context_id"], name="context_id")
    expected_id = compute_execution_context_id(value)
    if context_id != expected_id:
        raise _InvalidExecutionGate(
            "context-id-mismatch",
            f"expected {expected_id}, got {context_id}",
        )
    return kind, bindings, context_id


def _local_bindings(certificate: dict[str, object]) -> dict[str, object]:
    query = _require_object(certificate.get("query"), name="certificate.query")
    decision = _require_object(certificate.get("decision"), name="certificate.decision")
    provenance = _require_object(certificate.get("provenance"), name="certificate.provenance")
    return {
        "certificate_id": certificate.get("certificate_id"),
        "input_digest": provenance.get("input_digest"),
        "source_factor_digest": provenance.get("source_factor_digest"),
        "query_id": query.get("query_id"),
        "query_program_digest": query.get("query_program_digest"),
        "fallback_id": decision.get("fallback_id"),
    }


def _orbit_bindings(certificate: dict[str, object]) -> dict[str, object]:
    orbit = _require_object(certificate.get("orbit"), name="certificate.orbit")
    actions = _require_object(certificate.get("actions"), name="certificate.actions")
    candidate = _require_object(actions.get("candidate"), name="certificate.actions.candidate")
    fallback = _require_object(actions.get("fallback"), name="certificate.actions.fallback")
    decision = _require_object(certificate.get("decision"), name="certificate.decision")
    provenance = _require_object(certificate.get("provenance"), name="certificate.provenance")
    return {
        "certificate_id": certificate.get("certificate_id"),
        "input_digest": provenance.get("input_digest"),
        "shared_gauge_id": orbit.get("shared_gauge_id"),
        "support_receipt_digest": orbit.get("support_receipt_digest"),
        "candidate_action_id": candidate.get("action_id"),
        "fallback_action_id": fallback.get("action_id"),
        "candidate_loss_program_digest": candidate.get("loss_program_digest"),
        "fallback_loss_program_digest": fallback.get("loss_program_digest"),
        "fallback_id": decision.get("fallback_id"),
        "fallback_digest": decision.get("fallback_digest"),
        "admission_policy_digest": provenance.get("admission_policy_digest"),
    }


def _certificate_bindings(
    certificate_kind: str,
    certificate: dict[str, object],
) -> dict[str, object]:
    if certificate_kind == PROOF_CARRYING_FACTOR_KIND:
        return _validate_bindings(
            certificate_kind,
            _local_bindings(certificate),
            name="certificate_bindings",
        )
    if certificate_kind == PROOF_CARRYING_ORBIT_KIND:
        return _validate_bindings(
            certificate_kind,
            _orbit_bindings(certificate),
            name="certificate_bindings",
        )
    raise _InvalidExecutionGate(
        "unsupported-certificate-kind",
        f"unsupported certificate kind {certificate_kind!r}",
    )


def _invalid_report(
    *,
    code: str,
    detail: str,
    certificate_id: str | None,
    context_id: str | None,
    certificate_decision: str | None,
) -> Proof4DExecutionGateVerification:
    return Proof4DExecutionGateVerification(
        decision="invalid-fail-closed",
        valid=False,
        admitted=False,
        certificate_id=certificate_id,
        context_id=context_id,
        certificate_decision=certificate_decision,
        reason_codes=(code,),
        detail=detail,
    )


def verify_for_execution(
    certificate_artifact: str | Path | Mapping[str, object],
    context_artifact: str | Path | Mapping[str, object],
) -> Proof4DExecutionGateVerification:
    """Verify a certificate and exact caller-owned execution bindings."""

    certificate_id: str | None = None
    context_id: str | None = None
    certificate_decision: str | None = None
    try:
        context = (
            _strict_clone(context_artifact)
            if isinstance(context_artifact, Mapping)
            else _load_json(Path(context_artifact), name="context")
        )
        context_kind, expected_bindings, context_id = _verify_context(context)
        certificate = (
            _strict_clone(certificate_artifact)
            if isinstance(certificate_artifact, Mapping)
            else _load_json(Path(certificate_artifact), name="certificate")
        )
        raw_certificate_id = certificate.get("certificate_id")
        if type(raw_certificate_id) is str:
            certificate_id = raw_certificate_id
        certificate_kind = certificate.get("certificate_kind")
        if certificate_kind != context_kind:
            raise _InvalidExecutionGate(
                "execution-context-kind-mismatch",
                "certificate and caller context use different certificate kinds",
            )
        verification: ProofCarryingFactorVerification | ProofCarryingOrbitVerification
        if certificate_kind == PROOF_CARRYING_FACTOR_KIND:
            verification = verify_proof_carrying_factor(certificate)
        elif certificate_kind == PROOF_CARRYING_ORBIT_KIND:
            verification = verify_proof_carrying_orbit_action(certificate)
        else:
            raise _InvalidExecutionGate(
                "unsupported-certificate-kind",
                f"unsupported certificate kind {certificate_kind!r}",
            )
        certificate_decision = verification.decision
        if not verification.valid:
            reason = (
                verification.reason_codes[0] if verification.reason_codes else "invalid-certificate"
            )
            return _invalid_report(
                code=f"certificate:{reason}",
                detail=verification.detail,
                certificate_id=verification.certificate_id,
                context_id=context_id,
                certificate_decision=certificate_decision,
            )
        actual_bindings = _certificate_bindings(context_kind, certificate)
        mismatches = [
            key
            for key in sorted(expected_bindings)
            if expected_bindings[key] != actual_bindings[key]
        ]
        if mismatches:
            raise _InvalidExecutionGate(
                "execution-context-mismatch",
                "certificate differs from caller context for: " + ", ".join(mismatches),
            )
        if verification.admitted:
            return Proof4DExecutionGateVerification(
                decision="verified-admit",
                valid=True,
                admitted=True,
                certificate_id=verification.certificate_id,
                context_id=context_id,
                certificate_decision=certificate_decision,
                reason_codes=(),
                detail=(
                    "certificate is valid, admitted, and exactly matches the caller-owned "
                    "execution context"
                ),
            )
        return Proof4DExecutionGateVerification(
            decision="verified-reject",
            valid=True,
            admitted=False,
            certificate_id=verification.certificate_id,
            context_id=context_id,
            certificate_decision=certificate_decision,
            reason_codes=verification.reason_codes,
            detail=(
                "certificate is valid and matches the caller context, but its declared "
                "consequence is unsupported"
            ),
        )
    except _InvalidExecutionGate as error:
        return _invalid_report(
            code=error.code,
            detail=error.message,
            certificate_id=certificate_id,
            context_id=context_id,
            certificate_decision=certificate_decision,
        )
    except (TypeError, ValueError) as error:
        return _invalid_report(
            code="unexpected-execution-gate-error",
            detail=str(error),
            certificate_id=certificate_id,
            context_id=context_id,
            certificate_decision=certificate_decision,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the caller-bound execution gate."""

    parser = argparse.ArgumentParser(
        prog="python -m prob4d_independent_verifier.execution_gate",
        description=(
            "Verify a Proof4D certificate against a caller-owned execution context. "
            "Exit 0 admits, 2 rejects a valid unsupported consequence, and 3 fails closed."
        ),
    )
    parser.add_argument("certificate", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    report = verify_for_execution(arguments.certificate, arguments.context)
    print(
        json.dumps(
            report.to_dict(),
            sort_keys=True,
            indent=None if arguments.compact else 2,
            separators=(",", ":") if arguments.compact else None,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if report.admitted:
        return 0
    if report.valid:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXECUTION_CONTEXT_SCHEMA",
    "EXECUTION_CONTEXT_VERSION",
    "EXECUTION_GATE_CLAIM_BOUNDARY",
    "EXECUTION_GATE_IMPLEMENTATION",
    "EXECUTION_GATE_SCHEMA",
    "EXECUTION_GATE_VERSION",
    "Proof4DExecutionGateVerification",
    "build_execution_context",
    "compute_execution_context_id",
    "main",
    "render_execution_context",
    "verify_for_execution",
]
