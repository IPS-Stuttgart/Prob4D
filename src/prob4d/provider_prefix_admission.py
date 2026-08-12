"""Fail-closed admission binding support feasibility to calibration transport.

Provider support and calibration-transport evidence intentionally answer separate
questions.  This module creates one small content-addressed certificate that
requires both to pass for the same provider, cohort, and causal target prefix.
It remains upstream of BayesianPhysTwin's independent physical-update guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)
from .calibration_transport import (
    CalibrationTransportEvidenceV1,
    load_calibration_transport_evidence,
    load_calibration_transport_model,
)
from .provider_support_feasibility import (
    ProviderSupportFeasibilityV1,
    load_provider_support_feasibility,
)

PROVIDER_PREFIX_ADMISSION_SCHEMA = "prob4d.provider-prefix-admission"
PROVIDER_PREFIX_ADMISSION_VERSION = 1
PROVIDER_PREFIX_BINDING_METADATA_KEY = "provider_prefix_binding"
PROVIDER_PREFIX_ADMISSION_CLAIM_BOUNDARY = (
    "This artifact verifies that an exact causal provider prefix passes both a "
    "pre-residual support-feasibility gate and a source-only calibration-transport "
    "gate with matching provider/cohort/prefix bindings. It does not establish "
    "point accuracy, covariance coverage, authorize a BayesianPhysTwin state update, "
    "establish physical-query or Causal4D benefit, deployment safety, or state of the art."
)

_BINDING_FIELDS = frozenset(
    {
        "provider_manifest_id",
        "cohort_binding_id",
        "target_prefix_id",
        "causal_prefix_only",
        "target_residuals_used",
        "target_outcomes_used",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "provider_manifest_id",
        "cohort_binding_id",
        "target_prefix_id",
        "provider_support_feasibility_id",
        "calibration_transport_model_id",
        "calibration_transport_evidence_id",
        "calibration_transport_feature_contract_id",
        "support_feasible",
        "calibration_transport_accepted",
        "admitted",
        "exact_fallback_required",
        "decision_reasons",
        "metadata",
        "claim_boundary",
        "provider_prefix_admission_id",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _sorted_unique_strings(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _strings_from_json(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return _sorted_unique_strings(tuple(value), name=name)


def _binding(evidence: CalibrationTransportEvidenceV1) -> Mapping[str, Any]:
    raw = evidence.metadata.get(PROVIDER_PREFIX_BINDING_METADATA_KEY)
    mapping = require_mapping(raw, name=PROVIDER_PREFIX_BINDING_METADATA_KEY)
    require_exact_fields(
        mapping,
        _BINDING_FIELDS,
        name=PROVIDER_PREFIX_BINDING_METADATA_KEY,
    )
    provider_manifest_id = require_sha256(
        mapping["provider_manifest_id"],
        name="provider_prefix_binding.provider_manifest_id",
    )
    cohort_binding_id = require_sha256(
        mapping["cohort_binding_id"],
        name="provider_prefix_binding.cohort_binding_id",
    )
    target_prefix_id = require_sha256(
        mapping["target_prefix_id"],
        name="provider_prefix_binding.target_prefix_id",
    )
    causal_prefix_only = _strict_bool(
        mapping["causal_prefix_only"],
        name="provider_prefix_binding.causal_prefix_only",
    )
    residuals_used = _strict_bool(
        mapping["target_residuals_used"],
        name="provider_prefix_binding.target_residuals_used",
    )
    outcomes_used = _strict_bool(
        mapping["target_outcomes_used"],
        name="provider_prefix_binding.target_outcomes_used",
    )
    if not causal_prefix_only:
        raise ValueError("calibration transport is not bound to a causal prefix")
    if residuals_used or outcomes_used:
        raise ValueError("calibration transport admission must not use target residuals/outcomes")
    return {
        "provider_manifest_id": provider_manifest_id,
        "cohort_binding_id": cohort_binding_id,
        "target_prefix_id": target_prefix_id,
        "causal_prefix_only": True,
        "target_residuals_used": False,
        "target_outcomes_used": False,
    }


@dataclass(frozen=True, slots=True)
class ProviderPrefixAdmissionV1:
    """Conjunctive provider-prefix admission and exact-fallback certificate."""

    provider_manifest_id: str
    cohort_binding_id: str
    target_prefix_id: str
    provider_support_feasibility_id: str
    calibration_transport_model_id: str
    calibration_transport_evidence_id: str
    calibration_transport_feature_contract_id: str
    support_feasible: bool
    calibration_transport_accepted: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    admitted: bool = field(init=False)
    exact_fallback_required: bool = field(init=False)
    decision_reasons: tuple[str, ...] = field(init=False)
    provider_prefix_admission_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "provider_manifest_id",
            "cohort_binding_id",
            "target_prefix_id",
            "provider_support_feasibility_id",
            "calibration_transport_model_id",
            "calibration_transport_evidence_id",
            "calibration_transport_feature_contract_id",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        support = _strict_bool(self.support_feasible, name="support_feasible")
        transport = _strict_bool(
            self.calibration_transport_accepted,
            name="calibration_transport_accepted",
        )
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )
        reasons: list[str] = []
        if not support:
            reasons.append("support-feasibility-negative")
        if not transport:
            reasons.append("calibration-transport-negative")
        admitted = support and transport
        object.__setattr__(self, "support_feasible", support)
        object.__setattr__(self, "calibration_transport_accepted", transport)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "exact_fallback_required", not admitted)
        object.__setattr__(self, "decision_reasons", tuple(reasons))
        object.__setattr__(
            self,
            "provider_prefix_admission_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_PREFIX_ADMISSION_SCHEMA,
            "schema_version": PROVIDER_PREFIX_ADMISSION_VERSION,
            "provider_manifest_id": self.provider_manifest_id,
            "cohort_binding_id": self.cohort_binding_id,
            "target_prefix_id": self.target_prefix_id,
            "provider_support_feasibility_id": self.provider_support_feasibility_id,
            "calibration_transport_model_id": self.calibration_transport_model_id,
            "calibration_transport_evidence_id": self.calibration_transport_evidence_id,
            "calibration_transport_feature_contract_id": (
                self.calibration_transport_feature_contract_id
            ),
            "support_feasible": self.support_feasible,
            "calibration_transport_accepted": self.calibration_transport_accepted,
            "admitted": self.admitted,
            "exact_fallback_required": self.exact_fallback_required,
            "decision_reasons": list(self.decision_reasons),
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_PREFIX_ADMISSION_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "provider_prefix_admission_id": self.provider_prefix_admission_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderPrefixAdmissionV1:
        mapping = require_mapping(value, name="provider prefix admission")
        require_exact_fields(mapping, _ARTIFACT_FIELDS, name="provider prefix admission")
        if mapping["schema"] != PROVIDER_PREFIX_ADMISSION_SCHEMA:
            raise ValueError("provider prefix admission schema changed")
        if mapping["schema_version"] != PROVIDER_PREFIX_ADMISSION_VERSION:
            raise ValueError("provider prefix admission version changed")
        if mapping["claim_boundary"] != PROVIDER_PREFIX_ADMISSION_CLAIM_BOUNDARY:
            raise ValueError("provider prefix admission claim boundary changed")
        result = cls(
            provider_manifest_id=mapping["provider_manifest_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            target_prefix_id=mapping["target_prefix_id"],
            provider_support_feasibility_id=mapping[
                "provider_support_feasibility_id"
            ],
            calibration_transport_model_id=mapping["calibration_transport_model_id"],
            calibration_transport_evidence_id=mapping[
                "calibration_transport_evidence_id"
            ],
            calibration_transport_feature_contract_id=mapping[
                "calibration_transport_feature_contract_id"
            ],
            support_feasible=mapping["support_feasible"],
            calibration_transport_accepted=mapping[
                "calibration_transport_accepted"
            ],
            metadata=require_finite_json_mapping(mapping["metadata"], name="metadata"),
        )
        supplied_id = require_sha256(
            mapping["provider_prefix_admission_id"],
            name="provider_prefix_admission_id",
        )
        if supplied_id != result.provider_prefix_admission_id:
            raise ValueError("provider prefix admission identity mismatch")
        if plain_json(mapping) != result.to_dict():
            raise ValueError("provider prefix admission derived fields changed")
        return result


def build_provider_prefix_admission(
    support: ProviderSupportFeasibilityV1,
    transport: CalibrationTransportEvidenceV1,
    *,
    provider_manifest_id: str,
    target_prefix_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> ProviderPrefixAdmissionV1:
    """Bind two validated upstream gates to the same exact causal provider prefix."""

    if not isinstance(support, ProviderSupportFeasibilityV1):
        raise TypeError("support must be ProviderSupportFeasibilityV1")
    if not isinstance(transport, CalibrationTransportEvidenceV1):
        raise TypeError("transport must be CalibrationTransportEvidenceV1")
    manifest_id = require_sha256(provider_manifest_id, name="provider_manifest_id")
    prefix_id = require_sha256(target_prefix_id, name="target_prefix_id")
    binding = _binding(transport)
    if binding["provider_manifest_id"] != manifest_id:
        raise ValueError("calibration transport provider_manifest_id binding changed")
    if binding["target_prefix_id"] != prefix_id:
        raise ValueError("calibration transport target_prefix_id binding changed")
    cohort_id = support.request.cohort_binding_id
    if binding["cohort_binding_id"] != cohort_id:
        raise ValueError("support and calibration transport cohort bindings differ")
    return ProviderPrefixAdmissionV1(
        provider_manifest_id=manifest_id,
        cohort_binding_id=cohort_id,
        target_prefix_id=prefix_id,
        provider_support_feasibility_id=support.provider_support_feasibility_id,
        calibration_transport_model_id=transport.model.model_id,
        calibration_transport_evidence_id=transport.evidence_id,
        calibration_transport_feature_contract_id=transport.model.feature_contract_id,
        support_feasible=support.support_feasible,
        calibration_transport_accepted=transport.accepted,
        metadata={} if metadata is None else metadata,
    )


def write_provider_prefix_admission(
    path: str | Path,
    admission: ProviderPrefixAdmissionV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(admission, ProviderPrefixAdmissionV1):
        raise TypeError("admission must be ProviderPrefixAdmissionV1")
    payload = json.dumps(admission.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_provider_prefix_admission(path: str | Path) -> ProviderPrefixAdmissionV1:
    return ProviderPrefixAdmissionV1.from_dict(
        load_json_object(path, name="provider prefix admission")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--support", type=Path, required=True)
    build.add_argument("--transport-model", type=Path, required=True)
    build.add_argument("--transport-evidence", type=Path, required=True)
    build.add_argument("--provider-manifest-id", required=True)
    build.add_argument("--target-prefix-id", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        support = load_provider_support_feasibility(arguments.support)
        model = load_calibration_transport_model(arguments.transport_model)
        evidence = load_calibration_transport_evidence(
            arguments.transport_evidence,
            model=model,
        )
        admission = build_provider_prefix_admission(
            support,
            evidence,
            provider_manifest_id=arguments.provider_manifest_id,
            target_prefix_id=arguments.target_prefix_id,
        )
        write_provider_prefix_admission(
            arguments.output,
            admission,
            overwrite=arguments.overwrite,
        )
    else:
        admission = load_provider_prefix_admission(arguments.artifact)
    print(
        json.dumps(
            {
                "provider_prefix_admission_id": admission.provider_prefix_admission_id,
                "admitted": admission.admitted,
                "exact_fallback_required": admission.exact_fallback_required,
                "decision_reasons": list(admission.decision_reasons),
            },
            sort_keys=True,
        )
    )
    return 0 if admission.admitted else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PROVIDER_PREFIX_ADMISSION_CLAIM_BOUNDARY",
    "PROVIDER_PREFIX_ADMISSION_SCHEMA",
    "PROVIDER_PREFIX_ADMISSION_VERSION",
    "PROVIDER_PREFIX_BINDING_METADATA_KEY",
    "ProviderPrefixAdmissionV1",
    "build_provider_prefix_admission",
    "load_provider_prefix_admission",
    "write_provider_prefix_admission",
]
