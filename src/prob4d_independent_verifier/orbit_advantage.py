"""Independent verification of Proof4D axial-orbit action certificates.

This module intentionally does not import :mod:`prob4d`. It recomputes the
analytic extrema of one fallback-minus-candidate harmonic loss over the entire
declared axial-rotation arc and fails closed on malformed, tampered, or
unsupported artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROOF_CARRYING_ORBIT_SCHEMA = "prob4d.proof-carrying-axial-orbit-action"
PROOF_CARRYING_ORBIT_VERSION = 1
PROOF_CARRYING_ORBIT_KIND = "shared-axial-orbit-robust-advantage-v1"
PROOF_CARRYING_ORBIT_CLAIM_SCOPE = "conditional-global-axial-orbit-action-advantage-only"
EXACT_FALLBACK_POLICY = "exact-caller-owned-fallback"

VERIFICATION_SCHEMA = "prob4d.proof-carrying-orbit-action-verification"
VERIFICATION_VERSION = 1
VERIFIER_IMPLEMENTATION = "prob4d-independent-orbit-advantage-v1"
VERIFICATION_CLAIM_BOUNDARY = (
    "The verifier proves a robust fallback-minus-candidate advantage over every angle in "
    "the declared shared axial-rotation arc.",
    "It checks the harmonic loss algebra, analytic extrema, bounded-error margin, "
    "content identity, policy provenance, and exact-fallback declaration.",
    "It does not establish that the declared orbit exhausts physical uncertainty, that "
    "the support receipt or loss programs are truthful, or that execution is safe outside "
    "the stated conditional model.",
)

_MAX_JSON_BYTES = 1024**2
_REPORTED_VALUE_ATOL = 1e-12
_REPORTED_VALUE_RTOL = 1e-10
_AXIS_ATOL = 1e-12
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "certificate_kind",
    "claim_scope",
    "orbit",
    "actions",
    "advantage",
    "decision",
    "provenance",
    "certificate_id",
}
_ORBIT_KEYS = {
    "shared_gauge_id",
    "origin",
    "axis",
    "scope_admitted",
    "support_receipt_digest",
    "arc",
}
_ARC_KEYS = {"center", "half_width"}
_ACTIONS_KEYS = {"fallback", "candidate"}
_ACTION_KEYS = {"action_id", "loss_program_digest", "loss_harmonic"}
_HARMONIC_KEYS = {"constant", "cosine", "sine"}
_ADVANTAGE_KEYS = {
    "difference_harmonic",
    "advantage_error_bound",
    "required_margin",
    "numerical_slack",
    "reported_nominal_lower",
    "reported_nominal_upper",
    "reported_robust_lower",
    "reported_robust_upper",
}
_DECISION_KEYS = {
    "admitted",
    "reason_codes",
    "fallback_policy",
    "fallback_id",
    "fallback_digest",
}
_PROVENANCE_KEYS = {
    "producer",
    "producer_version",
    "input_digest",
    "admission_policy_digest",
    "assumption_ids",
    "calibration_receipt_digest",
}
_ALLOWED_REASONS = (
    "orbit-model-scope-not-admitted",
    "infeasible-anchor-support",
    "nonpositive-robust-advantage",
)


class _InvalidCertificate(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ProofCarryingOrbitVerification:
    """Fail-closed result of independent axial-orbit verification."""

    decision: str
    valid: bool
    admitted: bool
    certificate_id: str | None
    reason_codes: tuple[str, ...]
    detail: str
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": VERIFICATION_SCHEMA,
            "schema_version": VERIFICATION_VERSION,
            "verifier_implementation": VERIFIER_IMPLEMENTATION,
            "decision": self.decision,
            "valid": self.valid,
            "admitted": self.admitted,
            "certificate_id": self.certificate_id,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "metrics": dict(self.metrics),
            "claim_boundary": list(VERIFICATION_CLAIM_BOUNDARY),
        }


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _certificate_id(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("certificate_id", None)
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _reject_constant(value: str) -> None:
    raise _InvalidCertificate(
        "non-finite-json-number",
        f"non-finite JSON number {value!r} is forbidden",
    )


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidCertificate("duplicate-json-key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, object]:
    try:
        size = path.stat().st_size
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _InvalidCertificate("artifact-unreadable", str(error)) from error
    if size > _MAX_JSON_BYTES:
        raise _InvalidCertificate(
            "artifact-too-large",
            f"certificate exceeds {_MAX_JSON_BYTES} bytes",
        )
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_object,
        )
    except _InvalidCertificate:
        raise
    except json.JSONDecodeError as error:
        raise _InvalidCertificate("invalid-json", str(error)) from error
    if type(value) is not dict:
        raise _InvalidCertificate("top-level-not-object", "certificate must be one JSON object")
    return value


def _require_keys(value: object, expected: set[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _InvalidCertificate("invalid-object", f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise _InvalidCertificate(
            "unexpected-object-keys",
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}",
        )
    return value


def _nonempty_text(value: object, *, name: str, maximum_length: int = 512) -> str:
    if type(value) is not str:
        raise _InvalidCertificate("invalid-string", f"{name} must be a string")
    if not value or value.strip() != value or len(value) > maximum_length:
        raise _InvalidCertificate(
            "invalid-string",
            f"{name} must be nonempty, trimmed, and at most {maximum_length} characters",
        )
    return value


def _digest(value: object, *, name: str) -> str:
    text = _nonempty_text(value, name=name, maximum_length=71)
    if _DIGEST_PATTERN.fullmatch(text) is None:
        raise _InvalidCertificate(
            "invalid-digest",
            f"{name} must be a lowercase sha256: digest",
        )
    return text


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidCertificate("invalid-number", f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _InvalidCertificate("non-finite-number", f"{name} must be finite")
    return numeric


def _nonnegative_float(value: object, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric < 0.0:
        raise _InvalidCertificate("number-out-of-range", f"{name} must be nonnegative")
    return numeric


def _nullable_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name=name)


def _vector3(value: object, *, name: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise _InvalidCertificate("invalid-vector", f"{name} must contain three numbers")
    parsed = [_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    return parsed[0], parsed[1], parsed[2]


def _harmonic(value: object, *, name: str) -> tuple[float, float, float]:
    record = _require_keys(value, _HARMONIC_KEYS, name=name)
    return (
        _finite_float(record["constant"], name=f"{name}.constant"),
        _finite_float(record["cosine"], name=f"{name}.cosine"),
        _finite_float(record["sine"], name=f"{name}.sine"),
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        abs_tol=_REPORTED_VALUE_ATOL,
        rel_tol=_REPORTED_VALUE_RTOL,
    )


def _require_reported(
    reported: float | None,
    expected: float | None,
    *,
    code: str,
    name: str,
) -> None:
    if expected is None:
        if reported is not None:
            raise _InvalidCertificate(code, f"{name} must be null")
        return
    if reported is None or not _close(reported, expected):
        raise _InvalidCertificate(code, f"{name} does not match analytic recomputation")


def _contains(center: float, half_width: float, angle: float) -> bool:
    displacement = math.remainder(angle - center, 2.0 * math.pi)
    return abs(displacement) <= half_width


def _evaluate(harmonic: tuple[float, float, float], angle: float) -> float:
    constant, cosine, sine = harmonic
    value = math.fsum((constant, cosine * math.cos(angle), sine * math.sin(angle)))
    if not math.isfinite(value):
        raise _InvalidCertificate("numeric-overflow", "harmonic evaluation overflowed")
    return value


def _bounds(
    harmonic: tuple[float, float, float],
    arc: tuple[float, float],
) -> tuple[float, float]:
    constant, cosine, sine = harmonic
    radius = math.hypot(cosine, sine)
    if not math.isfinite(radius):
        raise _InvalidCertificate("numeric-overflow", "harmonic amplitude overflowed")
    center, half_width = arc
    if radius == 0.0:
        return constant, constant
    if half_width == math.pi:
        lower = constant - radius
        upper = constant + radius
        if not all(math.isfinite(item) for item in (lower, upper)):
            raise _InvalidCertificate("numeric-overflow", "harmonic extrema overflowed")
        return lower, upper
    candidates = [center - half_width, center + half_width]
    maximum_angle = math.atan2(sine, cosine)
    minimum_angle = math.remainder(maximum_angle + math.pi, 2.0 * math.pi)
    for angle in (minimum_angle, maximum_angle):
        if _contains(center, half_width, angle):
            candidates.append(angle)
    values = [_evaluate(harmonic, angle) for angle in candidates]
    return min(values), max(values)


def _verify_orbit(value: object) -> tuple[bool, tuple[float, float] | None, dict[str, float]]:
    orbit = _require_keys(value, _ORBIT_KEYS, name="orbit")
    _nonempty_text(orbit["shared_gauge_id"], name="orbit.shared_gauge_id")
    origin = _vector3(orbit["origin"], name="orbit.origin")
    axis = _vector3(orbit["axis"], name="orbit.axis")
    axis_norm = math.hypot(*axis)
    if not math.isfinite(axis_norm):
        raise _InvalidCertificate("numeric-overflow", "orbit axis norm overflowed")
    if abs(axis_norm - 1.0) > _AXIS_ATOL:
        raise _InvalidCertificate("orbit-axis-not-unit", "orbit.axis must be unit length")
    origin_norm = math.hypot(*origin)
    if not math.isfinite(origin_norm):
        raise _InvalidCertificate("numeric-overflow", "orbit origin norm overflowed")
    if type(orbit["scope_admitted"]) is not bool:
        raise _InvalidCertificate(
            "invalid-scope-decision",
            "orbit.scope_admitted must be a bool",
        )
    scope_admitted = orbit["scope_admitted"]
    support_receipt = orbit["support_receipt_digest"]
    if support_receipt is not None:
        _digest(support_receipt, name="orbit.support_receipt_digest")
    if scope_admitted and support_receipt is None:
        raise _InvalidCertificate(
            "missing-support-receipt",
            "an admitted orbit scope requires a support receipt digest",
        )
    arc_value = orbit["arc"]
    arc: tuple[float, float] | None
    if arc_value is None:
        arc = None
    else:
        arc_record = _require_keys(arc_value, _ARC_KEYS, name="orbit.arc")
        center = _finite_float(arc_record["center"], name="orbit.arc.center")
        half_width = _nonnegative_float(
            arc_record["half_width"],
            name="orbit.arc.half_width",
        )
        if half_width > math.pi:
            raise _InvalidCertificate(
                "invalid-orbit-arc",
                "orbit.arc.half_width must not exceed pi",
            )
        canonical_center = math.remainder(center, 2.0 * math.pi)
        if not _close(center, canonical_center):
            raise _InvalidCertificate(
                "noncanonical-orbit-angle",
                "orbit.arc.center must be reduced modulo 2*pi",
            )
        arc = (center, half_width)
    metrics = {
        "orbit_axis_norm_residual": abs(axis_norm - 1.0),
        "orbit_origin_norm": origin_norm,
        "orbit_arc_feasible": 0.0 if arc is None else 1.0,
    }
    if arc is not None:
        metrics["orbit_arc_half_width"] = arc[1]
    return scope_admitted, arc, metrics


def _verify_actions(
    value: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    actions = _require_keys(value, _ACTIONS_KEYS, name="actions")
    parsed: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for role in ("fallback", "candidate"):
        action = _require_keys(actions[role], _ACTION_KEYS, name=f"actions.{role}")
        action_id = _nonempty_text(action["action_id"], name=f"actions.{role}.action_id")
        _digest(
            action["loss_program_digest"],
            name=f"actions.{role}.loss_program_digest",
        )
        parsed[role] = (
            action_id,
            _harmonic(action["loss_harmonic"], name=f"actions.{role}.loss_harmonic"),
        )
    if parsed["fallback"][0] == parsed["candidate"][0]:
        raise _InvalidCertificate(
            "identical-action-identities",
            "candidate and fallback action IDs must differ",
        )
    return parsed["fallback"][1], parsed["candidate"][1]


def _verify_advantage(
    value: object,
    *,
    fallback: tuple[float, float, float],
    candidate: tuple[float, float, float],
    scope_admitted: bool,
    arc: tuple[float, float] | None,
) -> tuple[bool, tuple[str, ...], dict[str, float]]:
    advantage = _require_keys(value, _ADVANTAGE_KEYS, name="advantage")
    difference = _harmonic(
        advantage["difference_harmonic"],
        name="advantage.difference_harmonic",
    )
    expected_difference = (
        fallback[0] - candidate[0],
        fallback[1] - candidate[1],
        fallback[2] - candidate[2],
    )
    if not all(math.isfinite(item) for item in expected_difference):
        raise _InvalidCertificate("numeric-overflow", "fallback-minus-candidate loss overflowed")
    if any(
        not _close(got, expected)
        for got, expected in zip(difference, expected_difference, strict=True)
    ):
        raise _InvalidCertificate(
            "advantage-witness-mismatch",
            "advantage.difference_harmonic does not equal fallback minus candidate",
        )
    error = _nonnegative_float(
        advantage["advantage_error_bound"],
        name="advantage.advantage_error_bound",
    )
    margin = _nonnegative_float(
        advantage["required_margin"],
        name="advantage.required_margin",
    )
    slack = _nonnegative_float(
        advantage["numerical_slack"],
        name="advantage.numerical_slack",
    )
    reported_nominal_lower = _nullable_float(
        advantage["reported_nominal_lower"],
        name="advantage.reported_nominal_lower",
    )
    reported_nominal_upper = _nullable_float(
        advantage["reported_nominal_upper"],
        name="advantage.reported_nominal_upper",
    )
    reported_robust_lower = _nullable_float(
        advantage["reported_robust_lower"],
        name="advantage.reported_robust_lower",
    )
    reported_robust_upper = _nullable_float(
        advantage["reported_robust_upper"],
        name="advantage.reported_robust_upper",
    )

    nominal_lower: float | None
    nominal_upper: float | None
    robust_lower: float | None
    robust_upper: float | None
    reasons: list[str] = []
    if not scope_admitted:
        reasons.append("orbit-model-scope-not-admitted")
    if arc is None:
        nominal_lower = nominal_upper = robust_lower = robust_upper = None
        reasons.append("infeasible-anchor-support")
    else:
        nominal_lower, nominal_upper = _bounds(expected_difference, arc)
        robust_lower = nominal_lower - error
        robust_upper = nominal_upper + error
        if not all(math.isfinite(item) for item in (robust_lower, robust_upper)):
            raise _InvalidCertificate("numeric-overflow", "robust advantage bounds overflowed")
        if not robust_lower > margin + slack:
            reasons.append("nonpositive-robust-advantage")

    _require_reported(
        reported_nominal_lower,
        nominal_lower,
        code="reported-nominal-lower-mismatch",
        name="advantage.reported_nominal_lower",
    )
    _require_reported(
        reported_nominal_upper,
        nominal_upper,
        code="reported-nominal-upper-mismatch",
        name="advantage.reported_nominal_upper",
    )
    _require_reported(
        reported_robust_lower,
        robust_lower,
        code="reported-robust-lower-mismatch",
        name="advantage.reported_robust_lower",
    )
    _require_reported(
        reported_robust_upper,
        robust_upper,
        code="reported-robust-upper-mismatch",
        name="advantage.reported_robust_upper",
    )
    admitted = not reasons
    metrics = {
        "advantage_bounds_feasible": 0.0 if robust_lower is None else 1.0,
        "advantage_error_bound": error,
        "required_margin": margin,
        "numerical_slack": slack,
    }
    if nominal_lower is not None and nominal_upper is not None:
        if robust_lower is None or robust_upper is None:
            raise _InvalidCertificate(
                "inconsistent-advantage-bounds",
                "nominal and robust advantage bounds must be present together",
            )
        metrics.update(
            {
                "nominal_lower_advantage": nominal_lower,
                "nominal_upper_advantage": nominal_upper,
                "robust_lower_advantage": robust_lower,
                "robust_upper_advantage": robust_upper,
                "robust_admission_margin": robust_lower - margin - slack,
            }
        )
    return admitted, tuple(reasons), metrics


def _verify_decision(
    value: object,
    *,
    admitted: bool,
    reasons: tuple[str, ...],
) -> None:
    decision = _require_keys(value, _DECISION_KEYS, name="decision")
    if type(decision["admitted"]) is not bool:
        raise _InvalidCertificate(
            "invalid-producer-decision",
            "decision.admitted must be a bool",
        )
    if decision["admitted"] is not admitted:
        raise _InvalidCertificate(
            "producer-decision-mismatch",
            "producer admission does not match analytic verification",
        )
    reported_reasons = decision["reason_codes"]
    if type(reported_reasons) is not list:
        raise _InvalidCertificate(
            "invalid-reason-codes",
            "decision.reason_codes must be a list",
        )
    normalized = tuple(
        _nonempty_text(item, name="decision.reason_code") for item in reported_reasons
    )
    if any(item not in _ALLOWED_REASONS for item in normalized):
        raise _InvalidCertificate("invalid-reason-codes", "unknown decision reason code")
    if normalized != reasons:
        raise _InvalidCertificate(
            "producer-reason-mismatch",
            "producer reason codes do not match analytic verification",
        )
    if decision["fallback_policy"] != EXACT_FALLBACK_POLICY:
        raise _InvalidCertificate(
            "unsupported-fallback-policy",
            "decision.fallback_policy must require exact caller-owned fallback",
        )
    _nonempty_text(decision["fallback_id"], name="decision.fallback_id")
    _digest(decision["fallback_digest"], name="decision.fallback_digest")


def _verify_provenance(value: object) -> None:
    provenance = _require_keys(value, _PROVENANCE_KEYS, name="provenance")
    _nonempty_text(provenance["producer"], name="provenance.producer")
    _nonempty_text(
        provenance["producer_version"],
        name="provenance.producer_version",
    )
    _digest(provenance["input_digest"], name="provenance.input_digest")
    _digest(
        provenance["admission_policy_digest"],
        name="provenance.admission_policy_digest",
    )
    assumptions = provenance["assumption_ids"]
    if type(assumptions) is not list or not assumptions:
        raise _InvalidCertificate(
            "invalid-assumption-list",
            "provenance.assumption_ids must be a nonempty list",
        )
    normalized = [_nonempty_text(item, name="provenance.assumption_id") for item in assumptions]
    if normalized != sorted(set(normalized)):
        raise _InvalidCertificate(
            "invalid-assumption-list",
            "assumption IDs must be sorted and unique",
        )
    calibration = provenance["calibration_receipt_digest"]
    if calibration is not None:
        _digest(calibration, name="provenance.calibration_receipt_digest")


def _semantic_verify(certificate: dict[str, object]) -> ProofCarryingOrbitVerification:
    root = _require_keys(certificate, _TOP_LEVEL_KEYS, name="certificate")
    certificate_id = _digest(root["certificate_id"], name="certificate_id")
    expected_id = _certificate_id(root)
    if certificate_id != expected_id:
        raise _InvalidCertificate(
            "certificate-id-mismatch",
            f"expected {expected_id}, got {certificate_id}",
        )
    if root["schema"] != PROOF_CARRYING_ORBIT_SCHEMA:
        raise _InvalidCertificate("unsupported-schema", "unsupported certificate schema")
    if root["schema_version"] != PROOF_CARRYING_ORBIT_VERSION:
        raise _InvalidCertificate(
            "unsupported-schema-version",
            "unsupported certificate schema version",
        )
    if root["certificate_kind"] != PROOF_CARRYING_ORBIT_KIND:
        raise _InvalidCertificate("unsupported-kind", "unsupported certificate kind")
    if root["claim_scope"] != PROOF_CARRYING_ORBIT_CLAIM_SCOPE:
        raise _InvalidCertificate("invalid-claim-scope", "claim scope was widened")

    scope_admitted, arc, orbit_metrics = _verify_orbit(root["orbit"])
    fallback, candidate = _verify_actions(root["actions"])
    admitted, reasons, advantage_metrics = _verify_advantage(
        root["advantage"],
        fallback=fallback,
        candidate=candidate,
        scope_admitted=scope_admitted,
        arc=arc,
    )
    _verify_decision(root["decision"], admitted=admitted, reasons=reasons)
    _verify_provenance(root["provenance"])
    metrics = {**orbit_metrics, **advantage_metrics}
    if admitted:
        return ProofCarryingOrbitVerification(
            decision="verified-admit",
            valid=True,
            admitted=True,
            certificate_id=certificate_id,
            reason_codes=(),
            detail=(
                "certificate is internally valid and the candidate action has robust "
                "advantage over fallback across the complete declared axial orbit arc"
            ),
            metrics=metrics,
        )
    return ProofCarryingOrbitVerification(
        decision="verified-reject",
        valid=True,
        admitted=False,
        certificate_id=certificate_id,
        reason_codes=reasons,
        detail=(
            "certificate is internally valid but the candidate action is not authorized "
            "over the complete declared axial orbit arc"
        ),
        metrics=metrics,
    )


def verify_proof_carrying_orbit_action(
    artifact: str | Path | Mapping[str, object],
) -> ProofCarryingOrbitVerification:
    """Verify one axial-orbit action certificate and never admit malformed input."""

    certificate_id: str | None = None
    try:
        if isinstance(artifact, Mapping):
            value = json.loads(
                _canonical_bytes(dict(artifact)).decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_no_duplicate_object,
            )
            if type(value) is not dict:
                raise _InvalidCertificate(
                    "top-level-not-object",
                    "certificate must contain one JSON object",
                )
            certificate = value
        else:
            certificate = _load_json(Path(artifact))
        raw_id = certificate.get("certificate_id")
        if type(raw_id) is str:
            certificate_id = raw_id
        return _semantic_verify(certificate)
    except _InvalidCertificate as error:
        return ProofCarryingOrbitVerification(
            decision="invalid-fail-closed",
            valid=False,
            admitted=False,
            certificate_id=certificate_id,
            reason_codes=(error.code,),
            detail=error.message,
            metrics={},
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return ProofCarryingOrbitVerification(
            decision="invalid-fail-closed",
            valid=False,
            admitted=False,
            certificate_id=certificate_id,
            reason_codes=("unexpected-verification-error",),
            detail=str(error),
            metrics={},
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Verify an axial-orbit certificate as a fail-closed execution gate."""

    parser = argparse.ArgumentParser(
        prog="python -m prob4d_independent_verifier.orbit_advantage",
        description=(
            "Independently verify a Proof4D axial-orbit action certificate. "
            "Exit 0 admits, 2 rejects a valid action, and 3 marks the certificate invalid."
        ),
    )
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    report = verify_proof_carrying_orbit_action(arguments.certificate)
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
    "EXACT_FALLBACK_POLICY",
    "PROOF_CARRYING_ORBIT_CLAIM_SCOPE",
    "PROOF_CARRYING_ORBIT_KIND",
    "PROOF_CARRYING_ORBIT_SCHEMA",
    "PROOF_CARRYING_ORBIT_VERSION",
    "ProofCarryingOrbitVerification",
    "VERIFICATION_CLAIM_BOUNDARY",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_VERSION",
    "VERIFIER_IMPLEMENTATION",
    "main",
    "verify_proof_carrying_orbit_action",
]
