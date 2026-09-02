"""Build portable Proof4D certificates for robust actions on axial orbits.

The certificate proves one deliberately conditional global statement: over one
declared shared axial-rotation ambiguity arc, the candidate action has uniformly
positive fallback-minus-candidate advantage after a declared absolute error
bound. The artifact is content-addressed and independently verifiable without
importing the producing :mod:`prob4d` package.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path

from ._atomic_file import atomic_write_text
from ._version import __version__
from .axial_query_certificate import (
    AngleArc,
    HarmonicQuery,
    certify_shared_orbit_advantage,
)
from .proof_carrying_factor import EXACT_FALLBACK_POLICY

PROOF_CARRYING_ORBIT_SCHEMA = "prob4d.proof-carrying-axial-orbit-action"
PROOF_CARRYING_ORBIT_VERSION = 1
PROOF_CARRYING_ORBIT_KIND = "shared-axial-orbit-robust-advantage-v1"
PROOF_CARRYING_ORBIT_CLAIM_SCOPE = "conditional-global-axial-orbit-action-advantage-only"
DEFAULT_ORBIT_ASSUMPTION_IDS = (
    "declared-shared-axial-orbit-v1",
    "declared-affine-action-loss-on-orbit-v1",
    "declared-uniform-advantage-error-bound-v1",
    "external-orbit-support-scope-v1",
)
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def compute_proof_carrying_orbit_id(certificate: Mapping[str, object]) -> str:
    """Return the content ID after excluding ``certificate_id``."""

    payload = dict(certificate)
    payload.pop("certificate_id", None)
    return _sha256(payload)


def render_proof_carrying_orbit(certificate: Mapping[str, object]) -> str:
    """Render one certificate in canonical human-readable JSON form."""

    expected = compute_proof_carrying_orbit_id(certificate)
    if certificate.get("certificate_id") != expected:
        raise ValueError("certificate_id does not match the certificate contents")
    return (
        json.dumps(
            certificate,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def write_proof_carrying_orbit(
    path: str | Path,
    certificate: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one complete axial-orbit certificate."""

    atomic_write_text(path, render_proof_carrying_orbit(certificate), overwrite=overwrite)


def _nonempty_text(value: object, *, name: str, maximum_length: int = 512) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty and have no surrounding whitespace")
    if len(value) > maximum_length:
        raise ValueError(f"{name} is too long")
    return value


def _digest(value: object, *, name: str) -> str:
    text = _nonempty_text(value, name=name, maximum_length=71)
    if _DIGEST_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase sha256: digest")
    return text


def _nonnegative_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _assumption_ids(values: Iterable[str]) -> list[str]:
    result = sorted({_nonempty_text(value, name="assumption_id") for value in values})
    if not result:
        raise ValueError("assumption_ids must not be empty")
    return result


def _harmonic_record(query: HarmonicQuery) -> dict[str, float]:
    return {
        "constant": query.constant,
        "cosine": query.cosine,
        "sine": query.sine,
    }


def build_axial_orbit_action_certificate(
    *,
    fallback_loss: HarmonicQuery,
    candidate_loss: HarmonicQuery,
    scope_admitted: bool,
    support_receipt_digest: str | None,
    candidate_action_id: str,
    fallback_action_id: str,
    candidate_loss_program_digest: str,
    fallback_loss_program_digest: str,
    fallback_id: str,
    fallback_digest: str,
    input_digest: str,
    admission_policy_digest: str,
    arc: AngleArc | None = AngleArc(),
    advantage_error_bound: float = 0.0,
    required_margin: float = 0.0,
    numerical_slack: float = 1e-12,
    producer: str = "prob4d",
    producer_version: str = __version__,
    assumption_ids: Iterable[str] = DEFAULT_ORBIT_ASSUMPTION_IDS,
    calibration_receipt_digest: str | None = None,
) -> dict[str, object]:
    """Build a portable robust-action certificate on one shared axial orbit.

    Admission proves only the following conditional statement: on every angle
    in the declared arc, the candidate action's loss is smaller than the
    fallback action's loss by more than the required margin after subtracting
    the declared uniform error bound and numerical slack. The external support
    receipt is bound but not interpreted by this producer or its verifier.
    """

    if type(scope_admitted) is not bool:
        raise TypeError("scope_admitted must be a bool")
    if not isinstance(fallback_loss, HarmonicQuery):
        raise TypeError("fallback_loss must be a HarmonicQuery")
    if not isinstance(candidate_loss, HarmonicQuery):
        raise TypeError("candidate_loss must be a HarmonicQuery")
    if fallback_loss.orbit_key != candidate_loss.orbit_key:
        raise ValueError("both losses must use the same shared orbit")
    candidate_action = _nonempty_text(candidate_action_id, name="candidate_action_id")
    fallback_action = _nonempty_text(fallback_action_id, name="fallback_action_id")
    if candidate_action == fallback_action:
        raise ValueError("candidate and fallback action IDs must differ")
    candidate_program = _digest(
        candidate_loss_program_digest,
        name="candidate_loss_program_digest",
    )
    fallback_program = _digest(
        fallback_loss_program_digest,
        name="fallback_loss_program_digest",
    )
    declared_fallback_id = _nonempty_text(fallback_id, name="fallback_id")
    declared_fallback_digest = _digest(fallback_digest, name="fallback_digest")
    source_digest = _digest(input_digest, name="input_digest")
    policy_digest = _digest(admission_policy_digest, name="admission_policy_digest")
    support_digest = (
        None
        if support_receipt_digest is None
        else _digest(support_receipt_digest, name="support_receipt_digest")
    )
    if scope_admitted and support_digest is None:
        raise ValueError("scope_admitted requires a support_receipt_digest")
    calibration_digest = (
        None
        if calibration_receipt_digest is None
        else _digest(
            calibration_receipt_digest,
            name="calibration_receipt_digest",
        )
    )
    error = _nonnegative_real(advantage_error_bound, name="advantage_error_bound")
    margin = _nonnegative_real(required_margin, name="required_margin")
    slack = _nonnegative_real(numerical_slack, name="numerical_slack")
    provider = _nonempty_text(producer, name="producer")
    provider_version = _nonempty_text(producer_version, name="producer_version")
    assumptions = _assumption_ids(assumption_ids)

    orbit_identity, geometry = fallback_loss.orbit_key
    origin = list(geometry[:3])
    axis = list(geometry[3:])
    difference = fallback_loss.minus(candidate_loss)
    nominal_bounds = None if arc is None else difference.bounds(arc)
    result = certify_shared_orbit_advantage(
        fallback_loss=fallback_loss,
        candidate_loss=candidate_loss,
        scope_admitted=scope_admitted,
        arc=arc,
        advantage_error_bound=error,
        required_margin=margin,
        numerical_slack=slack,
    )

    certificate: dict[str, object] = {
        "schema": PROOF_CARRYING_ORBIT_SCHEMA,
        "schema_version": PROOF_CARRYING_ORBIT_VERSION,
        "certificate_kind": PROOF_CARRYING_ORBIT_KIND,
        "claim_scope": PROOF_CARRYING_ORBIT_CLAIM_SCOPE,
        "orbit": {
            "shared_gauge_id": orbit_identity,
            "origin": origin,
            "axis": axis,
            "scope_admitted": scope_admitted,
            "support_receipt_digest": support_digest,
            "arc": (
                None
                if arc is None
                else {
                    "center": arc.center,
                    "half_width": arc.half_width,
                }
            ),
        },
        "actions": {
            "fallback": {
                "action_id": fallback_action,
                "loss_program_digest": fallback_program,
                "loss_harmonic": _harmonic_record(fallback_loss),
            },
            "candidate": {
                "action_id": candidate_action,
                "loss_program_digest": candidate_program,
                "loss_harmonic": _harmonic_record(candidate_loss),
            },
        },
        "advantage": {
            "difference_harmonic": _harmonic_record(difference),
            "advantage_error_bound": error,
            "required_margin": margin,
            "numerical_slack": slack,
            "reported_nominal_lower": (
                None if nominal_bounds is None else nominal_bounds.lower
            ),
            "reported_nominal_upper": (
                None if nominal_bounds is None else nominal_bounds.upper
            ),
            "reported_robust_lower": result.lower_advantage,
            "reported_robust_upper": result.upper_advantage,
        },
        "decision": {
            "admitted": result.admitted,
            "reason_codes": list(result.reason_codes),
            "fallback_policy": EXACT_FALLBACK_POLICY,
            "fallback_id": declared_fallback_id,
            "fallback_digest": declared_fallback_digest,
        },
        "provenance": {
            "producer": provider,
            "producer_version": provider_version,
            "input_digest": source_digest,
            "admission_policy_digest": policy_digest,
            "assumption_ids": assumptions,
            "calibration_receipt_digest": calibration_digest,
        },
    }
    certificate["certificate_id"] = compute_proof_carrying_orbit_id(certificate)
    return certificate


__all__ = [
    "DEFAULT_ORBIT_ASSUMPTION_IDS",
    "PROOF_CARRYING_ORBIT_CLAIM_SCOPE",
    "PROOF_CARRYING_ORBIT_KIND",
    "PROOF_CARRYING_ORBIT_SCHEMA",
    "PROOF_CARRYING_ORBIT_VERSION",
    "build_axial_orbit_action_certificate",
    "compute_proof_carrying_orbit_id",
    "render_proof_carrying_orbit",
    "write_proof_carrying_orbit",
]
