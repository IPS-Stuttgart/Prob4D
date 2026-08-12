"""Bind Sim(3) propagation adequacy into prospective provider readiness.

Source covariance localization can establish that shared gauge/dependence energy is
well behaved before conditional point covariance is inspected.  A provider that then
marginalizes the gauge through a first-order approximation also needs evidence that
that approximation is adequate for the exact source cohort and frozen physical query.
This module makes that prerequisite content-addressed and fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)
from .diagnostics.sim3_linearization import GaussianLinearizationAdequacyV1
from .diagnostics.sim3_linearization_certificate import (
    gaussian_linearization_adequacy_from_dict,
    load_gaussian_linearization_adequacy,
)
from .fresh_provider_readiness import (
    FreshProviderCohortLockV1,
    ReadinessGateV1,
    unevaluated_gate,
)
from .source_covariance_localization import (
    SourceCovarianceLocalizationV1,
    load_source_covariance_localization,
)

GAUGE_PROPAGATION_READINESS_SCHEMA = "prob4d.gauge-propagation-readiness"
GAUGE_PROPAGATION_READINESS_VERSION = 1
GAUGE_PROPAGATION_BINDING_METADATA_KEY = "fresh_provider_readiness_binding"
GAUGE_PROPAGATION_READINESS_CLAIM_BOUNDARY = (
    "This artifact binds one declared gauge-propagation mode to an exact source "
    "covariance localization, provider/cohort identity, and frozen physical query. "
    "It authorizes first-order gauge marginalization only when a strict source-only "
    "Sim(3) certificate is adequate under the frozen policy. It does not establish "
    "provider competence, target calibration, BayesianPhysTwin or Causal4D benefit, "
    "deployment safety, or state of the art. A rejected approximation requires the "
    "explicit gauge latent or an already-declared exact fallback; it does not authorize "
    "target-side covariance inflation."
)

PropagationMode = Literal["explicit-gauge-latent", "first-order-marginalized"]
PropagationClassification = Literal[
    "explicit-gauge-latent-retained",
    "first-order-adequate",
    "first-order-inadequate",
    "technical-failure",
]
GateStatus = Literal["pass", "fail", "technical-failure"]
PerturbationSide = Literal["left", "right"]

_PARAMETER_BLOCKS = ("scale", "rotation", "translation")
_POLICY_FIELDS = frozenset(
    {
        "propagation_mode",
        "expected_perturbation_side",
        "expected_parameter_order",
        "minimum_sample_count",
        "require_query_projection",
        "require_supplied_jacobian_validation",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "provider_manifest_id",
        "cohort_binding_id",
        "query_definition_id",
        "source_covariance_localization_id",
        "source_group_ids",
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
        "query_definition_id",
        "source_covariance_localization_id",
        "source_group_ids",
        "policy",
        "certificate",
        "gaussian_linearization_adequacy_id",
        "classification",
        "gate_status",
        "reason_codes",
        "linearized_marginalization_authorized",
        "explicit_latent_or_exact_fallback_required",
        "next_action",
        "metadata",
        "claim_boundary",
        "gauge_propagation_readiness_id",
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


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_exact_string(value, name=name)


def _mode(value: object) -> PropagationMode:
    result = require_exact_string(value, name="propagation_mode")
    if result not in {"explicit-gauge-latent", "first-order-marginalized"}:
        raise ValueError("propagation_mode is not supported")
    return cast(PropagationMode, result)


def _side(value: object) -> PerturbationSide | None:
    result = _optional_string(value, name="expected_perturbation_side")
    if result is not None and result not in {"left", "right"}:
        raise ValueError("expected_perturbation_side must be left, right, or null")
    return cast(PerturbationSide | None, result)


def _parameter_order(value: object, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("expected_parameter_order must be a sequence")
    result = tuple(
        require_exact_string(item, name=f"expected_parameter_order[{index}]")
        for index, item in enumerate(value)
    )
    if not result and allow_empty:
        return ()
    if len(result) != 3 or set(result) != set(_PARAMETER_BLOCKS):
        raise ValueError("expected_parameter_order must permute scale, rotation, translation")
    return result


def _sorted_unique_strings(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _mean_transform_vector(certificate: GaussianLinearizationAdequacyV1) -> tuple[float, ...]:
    value = certificate.metadata.get("mean_transform_vector")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 7:
        raise ValueError("certificate metadata mean_transform_vector must contain 7 values")
    result = tuple(float(item) for item in value)
    if not np.all(np.isfinite(result)):
        raise ValueError("certificate metadata mean_transform_vector must be finite")
    return result


def _certificate_binding(
    certificate: GaussianLinearizationAdequacyV1,
) -> Mapping[str, Any]:
    raw = certificate.metadata.get(GAUGE_PROPAGATION_BINDING_METADATA_KEY)
    mapping = require_mapping(raw, name=GAUGE_PROPAGATION_BINDING_METADATA_KEY)
    require_exact_fields(
        mapping,
        _BINDING_FIELDS,
        name=GAUGE_PROPAGATION_BINDING_METADATA_KEY,
    )
    source_groups = _sorted_unique_strings(
        mapping["source_group_ids"],
        name="source_group_ids",
    )
    causal_prefix_only = _strict_bool(
        mapping["causal_prefix_only"],
        name="causal_prefix_only",
    )
    residuals_used = _strict_bool(
        mapping["target_residuals_used"],
        name="target_residuals_used",
    )
    outcomes_used = _strict_bool(
        mapping["target_outcomes_used"],
        name="target_outcomes_used",
    )
    if not causal_prefix_only:
        raise ValueError("linearization certificate is not bound to a causal prefix")
    if residuals_used or outcomes_used:
        raise ValueError("linearization readiness must not use target residuals or outcomes")
    return {
        "provider_manifest_id": require_sha256(
            mapping["provider_manifest_id"],
            name="provider_manifest_id",
        ),
        "cohort_binding_id": require_sha256(
            mapping["cohort_binding_id"],
            name="cohort_binding_id",
        ),
        "query_definition_id": require_sha256(
            mapping["query_definition_id"],
            name="query_definition_id",
        ),
        "source_covariance_localization_id": require_sha256(
            mapping["source_covariance_localization_id"],
            name="source_covariance_localization_id",
        ),
        "source_group_ids": source_groups,
        "causal_prefix_only": True,
        "target_residuals_used": False,
        "target_outcomes_used": False,
    }


@dataclass(frozen=True, slots=True)
class GaugePropagationReadinessPolicyV1:
    """Frozen decision policy for one declared gauge-propagation implementation."""

    propagation_mode: PropagationMode
    expected_perturbation_side: PerturbationSide | None
    expected_parameter_order: tuple[str, ...]
    minimum_sample_count: int
    require_query_projection: bool
    require_supplied_jacobian_validation: bool

    def __post_init__(self) -> None:
        mode = _mode(self.propagation_mode)
        side = _side(self.expected_perturbation_side)
        order = _parameter_order(
            self.expected_parameter_order,
            allow_empty=mode == "explicit-gauge-latent",
        )
        minimum = require_exact_integer(
            self.minimum_sample_count,
            name="minimum_sample_count",
            minimum=0,
        )
        require_query = _strict_bool(
            self.require_query_projection,
            name="require_query_projection",
        )
        require_jacobian = _strict_bool(
            self.require_supplied_jacobian_validation,
            name="require_supplied_jacobian_validation",
        )
        if mode == "explicit-gauge-latent":
            if side is not None or order or minimum != 0 or require_query or require_jacobian:
                raise ValueError(
                    "explicit-gauge-latent policy must not declare linearization requirements"
                )
        else:
            if side is None or not order or minimum < 2:
                raise ValueError(
                    "first-order-marginalized policy requires side, order, and samples"
                )
        object.__setattr__(self, "propagation_mode", mode)
        object.__setattr__(self, "expected_perturbation_side", side)
        object.__setattr__(self, "expected_parameter_order", order)
        object.__setattr__(self, "minimum_sample_count", minimum)
        object.__setattr__(self, "require_query_projection", require_query)
        object.__setattr__(
            self,
            "require_supplied_jacobian_validation",
            require_jacobian,
        )

    @classmethod
    def explicit_latent(cls) -> GaugePropagationReadinessPolicyV1:
        return cls(
            propagation_mode="explicit-gauge-latent",
            expected_perturbation_side=None,
            expected_parameter_order=(),
            minimum_sample_count=0,
            require_query_projection=False,
            require_supplied_jacobian_validation=False,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "propagation_mode": self.propagation_mode,
            "expected_perturbation_side": self.expected_perturbation_side,
            "expected_parameter_order": list(self.expected_parameter_order),
            "minimum_sample_count": self.minimum_sample_count,
            "require_query_projection": self.require_query_projection,
            "require_supplied_jacobian_validation": (
                self.require_supplied_jacobian_validation
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> GaugePropagationReadinessPolicyV1:
        mapping = require_mapping(value, name="gauge propagation readiness policy")
        require_exact_fields(
            mapping,
            _POLICY_FIELDS,
            name="gauge propagation readiness policy",
        )
        return cls(
            propagation_mode=mapping["propagation_mode"],
            expected_perturbation_side=mapping["expected_perturbation_side"],
            expected_parameter_order=_parameter_order(
                mapping["expected_parameter_order"],
                allow_empty=mapping["propagation_mode"] == "explicit-gauge-latent",
            ),
            minimum_sample_count=mapping["minimum_sample_count"],
            require_query_projection=mapping["require_query_projection"],
            require_supplied_jacobian_validation=mapping[
                "require_supplied_jacobian_validation"
            ],
        )


@dataclass(frozen=True, slots=True)
class GaugePropagationReadinessV1:
    """Content-addressed propagation decision bound to exact source evidence."""

    provider_manifest_id: str
    cohort_binding_id: str
    query_definition_id: str
    source_covariance_localization_id: str
    source_group_ids: tuple[str, ...]
    policy: GaugePropagationReadinessPolicyV1
    certificate: GaussianLinearizationAdequacyV1 | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    gaussian_linearization_adequacy_id: str | None = field(init=False)
    classification: PropagationClassification = field(init=False)
    gate_status: GateStatus = field(init=False)
    reason_codes: tuple[str, ...] = field(init=False)
    linearized_marginalization_authorized: bool = field(init=False)
    explicit_latent_or_exact_fallback_required: bool = field(init=False)
    next_action: str = field(init=False)
    gauge_propagation_readiness_id: str = field(init=False)

    def __post_init__(self) -> None:
        provider_id = require_sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        cohort_id = require_sha256(self.cohort_binding_id, name="cohort_binding_id")
        query_id = require_sha256(self.query_definition_id, name="query_definition_id")
        localization_id = require_sha256(
            self.source_covariance_localization_id,
            name="source_covariance_localization_id",
        )
        groups = _sorted_unique_strings(self.source_group_ids, name="source_group_ids")
        if not isinstance(self.policy, GaugePropagationReadinessPolicyV1):
            raise TypeError("policy must be GaugePropagationReadinessPolicyV1")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )

        reasons: list[str] = []
        certificate_id: str | None = None
        if self.policy.propagation_mode == "explicit-gauge-latent":
            if self.certificate is not None:
                raise ValueError("explicit-gauge-latent mode must not carry a certificate")
            classification: PropagationClassification = "explicit-gauge-latent-retained"
            gate_status: GateStatus = "pass"
            authorized = False
            fallback_required = False
            next_action = "retain-explicit-gauge-latent"
        else:
            if not isinstance(self.certificate, GaussianLinearizationAdequacyV1):
                raise TypeError(
                    "first-order-marginalized mode requires a strict linearization certificate"
                )
            certificate = self.certificate
            certificate_id = certificate.gaussian_linearization_adequacy_id
            binding = _certificate_binding(certificate)
            expected_binding = {
                "provider_manifest_id": provider_id,
                "cohort_binding_id": cohort_id,
                "query_definition_id": query_id,
                "source_covariance_localization_id": localization_id,
                "source_group_ids": groups,
                "causal_prefix_only": True,
                "target_residuals_used": False,
                "target_outcomes_used": False,
            }
            if plain_json(binding) != plain_json(expected_binding):
                raise ValueError("linearization certificate readiness binding changed")
            expected_side = self.policy.expected_perturbation_side
            assert expected_side is not None
            if certificate.parameterization != f"sim3-{expected_side}-perturbation":
                raise ValueError("linearization certificate parameterization changed")
            if certificate.parameter_dimension != 7:
                raise ValueError("Sim(3) linearization certificate dimension must be seven")
            if certificate.output_shape[1] != 3:
                raise ValueError("Sim(3) linearization output must contain 3-D points")
            if tuple(certificate.parameter_order) != self.policy.expected_parameter_order:
                raise ValueError("linearization certificate parameter order changed")
            if certificate.metadata.get("perturbation_side") != expected_side:
                raise ValueError("linearization certificate perturbation-side metadata changed")
            point_count = certificate.metadata.get("point_count")
            if type(point_count) is not int or point_count != certificate.output_shape[0]:
                raise ValueError("linearization certificate point_count metadata changed")
            _mean_transform_vector(certificate)

            if certificate.sample_count < self.policy.minimum_sample_count:
                reasons.append("insufficient-linearization-sample-count")
            if self.policy.require_query_projection and certificate.query_diagnostics is None:
                reasons.append("missing-required-query-projection")
            if (
                self.policy.require_supplied_jacobian_validation
                and not certificate.jacobian_validated
            ):
                reasons.append("supplied-jacobian-not-validated")

            if reasons:
                classification = "technical-failure"
                gate_status = "technical-failure"
                authorized = False
                fallback_required = True
                next_action = "retain-explicit-gauge-latent-or-exact-fallback"
            elif not certificate.adequate:
                classification = "first-order-inadequate"
                gate_status = "fail"
                reasons.extend(
                    f"linearization:{reason}" for reason in certificate.failure_reasons
                )
                authorized = False
                fallback_required = True
                next_action = "retain-explicit-gauge-latent-or-exact-fallback"
            else:
                classification = "first-order-adequate"
                gate_status = "pass"
                authorized = True
                fallback_required = False
                next_action = "permit-declared-first-order-marginalization"

        object.__setattr__(self, "provider_manifest_id", provider_id)
        object.__setattr__(self, "cohort_binding_id", cohort_id)
        object.__setattr__(self, "query_definition_id", query_id)
        object.__setattr__(self, "source_covariance_localization_id", localization_id)
        object.__setattr__(self, "source_group_ids", groups)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "gaussian_linearization_adequacy_id", certificate_id)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "gate_status", gate_status)
        object.__setattr__(self, "reason_codes", tuple(sorted(set(reasons))))
        object.__setattr__(self, "linearized_marginalization_authorized", authorized)
        object.__setattr__(
            self,
            "explicit_latent_or_exact_fallback_required",
            fallback_required,
        )
        object.__setattr__(self, "next_action", next_action)
        object.__setattr__(
            self,
            "gauge_propagation_readiness_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": GAUGE_PROPAGATION_READINESS_SCHEMA,
            "schema_version": GAUGE_PROPAGATION_READINESS_VERSION,
            "provider_manifest_id": self.provider_manifest_id,
            "cohort_binding_id": self.cohort_binding_id,
            "query_definition_id": self.query_definition_id,
            "source_covariance_localization_id": (
                self.source_covariance_localization_id
            ),
            "source_group_ids": list(self.source_group_ids),
            "policy": self.policy.to_dict(),
            "certificate": None if self.certificate is None else self.certificate.to_dict(),
            "gaussian_linearization_adequacy_id": (
                self.gaussian_linearization_adequacy_id
            ),
            "classification": self.classification,
            "gate_status": self.gate_status,
            "reason_codes": list(self.reason_codes),
            "linearized_marginalization_authorized": (
                self.linearized_marginalization_authorized
            ),
            "explicit_latent_or_exact_fallback_required": (
                self.explicit_latent_or_exact_fallback_required
            ),
            "next_action": self.next_action,
            "metadata": plain_json(self.metadata),
            "claim_boundary": GAUGE_PROPAGATION_READINESS_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "gauge_propagation_readiness_id": self.gauge_propagation_readiness_id,
        }

    def readiness_gate(self) -> ReadinessGateV1:
        metadata = {
            "propagation_mode": self.policy.propagation_mode,
            "classification": self.classification,
            "query_definition_id": self.query_definition_id,
            "source_covariance_localization_id": (
                self.source_covariance_localization_id
            ),
            "linearized_marginalization_authorized": (
                self.linearized_marginalization_authorized
            ),
        }
        return ReadinessGateV1(
            gate_name="gauge-dependence",
            status=self.gate_status,
            evidence_id=self.gauge_propagation_readiness_id,
            reason_codes=() if self.gate_status == "pass" else self.reason_codes,
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, value: object) -> GaugePropagationReadinessV1:
        mapping = require_mapping(value, name="gauge propagation readiness")
        require_exact_fields(mapping, _ARTIFACT_FIELDS, name="gauge propagation readiness")
        if mapping["schema"] != GAUGE_PROPAGATION_READINESS_SCHEMA:
            raise ValueError("gauge propagation readiness schema changed")
        if mapping["schema_version"] != GAUGE_PROPAGATION_READINESS_VERSION:
            raise ValueError("gauge propagation readiness version changed")
        if mapping["claim_boundary"] != GAUGE_PROPAGATION_READINESS_CLAIM_BOUNDARY:
            raise ValueError("gauge propagation readiness claim boundary changed")
        raw_certificate = mapping["certificate"]
        certificate = (
            None
            if raw_certificate is None
            else gaussian_linearization_adequacy_from_dict(raw_certificate)
        )
        result = cls(
            provider_manifest_id=mapping["provider_manifest_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            query_definition_id=mapping["query_definition_id"],
            source_covariance_localization_id=mapping[
                "source_covariance_localization_id"
            ],
            source_group_ids=_sorted_unique_strings(
                mapping["source_group_ids"],
                name="source_group_ids",
            ),
            policy=GaugePropagationReadinessPolicyV1.from_dict(mapping["policy"]),
            certificate=certificate,
            metadata=require_finite_json_mapping(mapping["metadata"], name="metadata"),
        )
        supplied_id = require_sha256(
            mapping["gauge_propagation_readiness_id"],
            name="gauge_propagation_readiness_id",
        )
        if supplied_id != result.gauge_propagation_readiness_id:
            raise ValueError("gauge propagation readiness identity mismatch")
        if plain_json(mapping) != plain_json(result.to_dict()):
            raise ValueError("gauge propagation readiness derived fields changed")
        return result


def build_gauge_propagation_readiness(
    localization: SourceCovarianceLocalizationV1,
    policy: GaugePropagationReadinessPolicyV1,
    *,
    query_definition_id: str,
    certificate: GaussianLinearizationAdequacyV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GaugePropagationReadinessV1:
    """Build one propagation decision from exact source-localization evidence."""

    if not isinstance(localization, SourceCovarianceLocalizationV1):
        raise TypeError("localization must be SourceCovarianceLocalizationV1")
    if not isinstance(policy, GaugePropagationReadinessPolicyV1):
        raise TypeError("policy must be GaugePropagationReadinessPolicyV1")
    return GaugePropagationReadinessV1(
        provider_manifest_id=localization.provider_manifest_id,
        cohort_binding_id=localization.cohort_binding_id,
        query_definition_id=query_definition_id,
        source_covariance_localization_id=(
            localization.source_covariance_localization_id
        ),
        source_group_ids=tuple(group.group_id for group in localization.groups),
        policy=policy,
        certificate=certificate,
        metadata={} if metadata is None else metadata,
    )


def compose_source_covariance_readiness_gates(
    cohort_lock: FreshProviderCohortLockV1,
    localization: SourceCovarianceLocalizationV1,
    propagation: GaugePropagationReadinessV1 | None,
) -> tuple[ReadinessGateV1, ReadinessGateV1]:
    """Compose source covariance and propagation evidence in strict stage order."""

    if not isinstance(cohort_lock, FreshProviderCohortLockV1):
        raise TypeError("cohort_lock must be FreshProviderCohortLockV1")
    if not isinstance(localization, SourceCovarianceLocalizationV1):
        raise TypeError("localization must be SourceCovarianceLocalizationV1")
    if cohort_lock.cohort_binding_id != localization.cohort_binding_id:
        raise ValueError("cohort lock and source covariance localization differ")
    source_gauge, source_point = localization.readiness_gates()
    if source_gauge.status != "pass":
        if propagation is not None:
            _validate_composition_bindings(cohort_lock, localization, propagation)
        return source_gauge, source_point
    if propagation is None:
        raise ValueError(
            "a passing gauge/dependence localization requires propagation readiness"
        )
    _validate_composition_bindings(cohort_lock, localization, propagation)
    propagation_gate = propagation.readiness_gate()
    if propagation_gate.status != "pass":
        return propagation_gate, unevaluated_gate("point-covariance")
    return propagation_gate, source_point


def _validate_composition_bindings(
    cohort_lock: FreshProviderCohortLockV1,
    localization: SourceCovarianceLocalizationV1,
    propagation: GaugePropagationReadinessV1,
) -> None:
    if not isinstance(propagation, GaugePropagationReadinessV1):
        raise TypeError("propagation must be GaugePropagationReadinessV1")
    if propagation.provider_manifest_id != localization.provider_manifest_id:
        raise ValueError("propagation and covariance localization provider IDs differ")
    if propagation.cohort_binding_id != localization.cohort_binding_id:
        raise ValueError("propagation and covariance localization cohort IDs differ")
    if propagation.cohort_binding_id != cohort_lock.cohort_binding_id:
        raise ValueError("propagation and cohort lock bindings differ")
    if propagation.query_definition_id != cohort_lock.query_definition_id:
        raise ValueError("propagation and cohort lock query definitions differ")
    if propagation.source_covariance_localization_id != (
        localization.source_covariance_localization_id
    ):
        raise ValueError("propagation references a different covariance localization")
    localization_groups = tuple(group.group_id for group in localization.groups)
    if propagation.source_group_ids != localization_groups:
        raise ValueError("propagation and covariance localization source groups differ")


def write_gauge_propagation_readiness(
    path: str | Path,
    readiness: GaugePropagationReadinessV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(readiness, GaugePropagationReadinessV1):
        raise TypeError("readiness must be GaugePropagationReadinessV1")
    payload = json.dumps(readiness.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_gauge_propagation_readiness(
    path: str | Path,
) -> GaugePropagationReadinessV1:
    return GaugePropagationReadinessV1.from_dict(
        load_json_object(path, name="gauge propagation readiness")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--localization", type=Path, required=True)
    build.add_argument("--policy", type=Path, required=True)
    build.add_argument("--query-definition-id", required=True)
    build.add_argument("--certificate", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        localization = load_source_covariance_localization(arguments.localization)
        policy = GaugePropagationReadinessPolicyV1.from_dict(
            load_json_object(arguments.policy, name="gauge propagation readiness policy")
        )
        certificate = (
            None
            if arguments.certificate is None
            else load_gaussian_linearization_adequacy(arguments.certificate)
        )
        readiness = build_gauge_propagation_readiness(
            localization,
            policy,
            query_definition_id=arguments.query_definition_id,
            certificate=certificate,
        )
        write_gauge_propagation_readiness(
            arguments.output,
            readiness,
            overwrite=arguments.overwrite,
        )
    else:
        readiness = load_gauge_propagation_readiness(arguments.artifact)
    print(
        json.dumps(
            {
                "gauge_propagation_readiness_id": (
                    readiness.gauge_propagation_readiness_id
                ),
                "classification": readiness.classification,
                "gate_status": readiness.gate_status,
                "linearized_marginalization_authorized": (
                    readiness.linearized_marginalization_authorized
                ),
                "next_action": readiness.next_action,
            },
            sort_keys=True,
        )
    )
    return 0 if readiness.gate_status == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "GAUGE_PROPAGATION_BINDING_METADATA_KEY",
    "GAUGE_PROPAGATION_READINESS_CLAIM_BOUNDARY",
    "GAUGE_PROPAGATION_READINESS_SCHEMA",
    "GAUGE_PROPAGATION_READINESS_VERSION",
    "GaugePropagationReadinessPolicyV1",
    "GaugePropagationReadinessV1",
    "build_gauge_propagation_readiness",
    "compose_source_covariance_readiness_gates",
    "load_gauge_propagation_readiness",
    "write_gauge_propagation_readiness",
]
