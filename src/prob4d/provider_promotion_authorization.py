"""Bind target-free provider support to held-out promotion authorization.

The support-feasibility artifact is computed before prediction payloads, residuals,
calibration outcomes, or target outcomes are opened. This module turns a positive
result into a content-addressed authorization that must match the exact promotion
lock and complete target-group roster before later evidence can be bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._heldout_promotion_common import (
    _atomic_write_json,
    _exact_keys,
    _load_json,
    _strict_bool,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    load_promotion_lock,
    promotion_lock_from_dict,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import _sha256_json
from .heldout_provider_evidence import (
    HeldoutProviderEvidenceV2,
    heldout_provider_evidence_from_dict,
    load_heldout_provider_evidence,
)
from .provider_support_feasibility import (
    ProviderSupportFeasibilityV1,
    load_provider_support_feasibility,
)

PROVIDER_PROMOTION_AUTHORIZATION_SCHEMA = "prob4d.provider-promotion-authorization"
PROVIDER_PROMOTION_AUTHORIZATION_VERSION = 2
PROVIDER_PROMOTION_AUTHORIZATION_CLAIM_BOUNDARY = (
    "This artifact establishes only that one exact target-free promotion lock "
    "has a positive, complete, pre-residual support-feasibility result for every "
    "frozen target group before calibration or target payloads are opened. It "
    "does not establish provider competence, calibrated uncertainty, "
    "BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of "
    "the art."
)
AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_SCHEMA = "prob4d.authorized-heldout-provider-evidence"
AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_VERSION = 1
AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY = (
    "This envelope binds replay-complete held-out evidence to the exact earlier "
    "target-free provider-promotion authorization. A passing envelope preserves "
    "the scientific claim boundary of its enclosed evidence."
)

_AUTHORIZATION_FIELDS = {
    "schema_name",
    "schema_version",
    "promotion_lock",
    "support_feasibility",
    "promotion_lock_id",
    "support_request_id",
    "support_feasibility_id",
    "target_group_ids",
    "supported_target_group_ids",
    "stream_roster_id",
    "technical_exclusion_policy_id",
    "intrinsics_ids",
    "extrinsics_ids",
    "metric_anchor_ids",
    "calibration_payloads_opened",
    "target_payloads_opened",
    "provider_residuals_computed",
    "authorized",
    "metadata",
    "claim_boundary",
    "authorization_id",
}
_AUTHORIZED_EVIDENCE_FIELDS = {
    "schema_name",
    "schema_version",
    "authorization",
    "evidence",
    "authorization_id",
    "evidence_id",
    "promotion_lock_id",
    "overall_passed",
    "claim_boundary",
    "authorized_evidence_id",
}


def _canonical_string_tuple(
    values: Sequence[str],
    *,
    name: str,
    nonempty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _strict_string(value, name=f"{name}[{index}]") for index, value in enumerate(values)
    )
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _string_tuple_from_json(
    value: object,
    *,
    name: str,
    nonempty: bool = False,
) -> tuple[str, ...]:
    return _canonical_string_tuple(
        tuple(_strict_list(value, name=name)),
        name=name,
        nonempty=nonempty,
    )


def _stream_roster_descriptor(
    feasibility: ProviderSupportFeasibilityV1,
) -> dict[str, object]:
    return {
        "request_id": feasibility.request.request_id,
        "streams": [
            {
                "group_id": stream.group_id,
                "stream_id": stream.stream_id,
                "causal_frame_start": stream.causal_frame_start,
                "causal_frame_stop_exclusive": (stream.causal_frame_stop_exclusive),
                "required_frame_ids": list(stream.required_frame_ids),
                "intrinsics_id": stream.intrinsics_id,
                "extrinsics_id": stream.extrinsics_id,
                "metric_anchor_id": stream.metric_anchor_id,
                "minimum_geometry_support_fraction": (stream.minimum_geometry_support_fraction),
                "technical_failure_code": stream.technical_failure_code,
            }
            for stream in feasibility.request.streams
        ],
    }


def _technical_exclusion_policy_descriptor(
    feasibility: ProviderSupportFeasibilityV1,
) -> dict[str, object]:
    request = feasibility.request
    return {
        "permitted_technical_exclusion_codes": list(request.permitted_technical_exclusion_codes),
        "maximum_technical_exclusions": request.maximum_technical_exclusions,
        "technical_exclusion_count": feasibility.technical_exclusion_count,
    }


def _request_group_ids(
    feasibility: ProviderSupportFeasibilityV1,
) -> tuple[str, ...]:
    return tuple(sorted({stream.group_id for stream in feasibility.request.streams}))


def _supported_group_ids(
    feasibility: ProviderSupportFeasibilityV1,
) -> tuple[str, ...]:
    return tuple(
        sorted({result.group_id for result in feasibility.stream_results if result.supported})
    )


def _digest_ids(
    feasibility: ProviderSupportFeasibilityV1,
    attribute: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for stream in feasibility.request.streams
                if (value := getattr(stream, attribute)) is not None
            }
        )
    )


@dataclass(frozen=True, slots=True)
class ProviderPromotionAuthorizationV2:
    """Positive target-free authorization for one exact promotion lock."""

    promotion_lock: HeldoutProviderPromotionLockV1
    support_feasibility: ProviderSupportFeasibilityV1
    calibration_payloads_opened: bool = False
    target_payloads_opened: bool = False
    provider_residuals_computed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.promotion_lock,
            HeldoutProviderPromotionLockV1,
        ):
            raise TypeError("promotion_lock must be HeldoutProviderPromotionLockV1")
        if not isinstance(
            self.support_feasibility,
            ProviderSupportFeasibilityV1,
        ):
            raise TypeError("support_feasibility must be ProviderSupportFeasibilityV1")
        for name in (
            "calibration_payloads_opened",
            "target_payloads_opened",
            "provider_residuals_computed",
        ):
            value = _strict_bool(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
            if value:
                raise ValueError(f"{name} must be false when authorization is created")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="provider promotion authorization metadata",
            ),
        )

        lock = self.promotion_lock
        feasibility = self.support_feasibility
        request = feasibility.request
        if not feasibility.support_feasible:
            raise PermissionError(
                "provider support feasibility is negative; promotion is not authorized"
            )
        if request.prediction_payloads_opened:
            raise ValueError("support request opened prediction payloads")
        if request.residuals_used:
            raise ValueError("support request used provider residuals")
        if request.target_outcomes_used:
            raise ValueError("support request used target outcomes")
        if request.promotion_lock_id != lock.promotion_lock_id:
            raise ValueError("support request references a different promotion lock")
        if request.source_repository != lock.source_repository:
            raise ValueError("support request and promotion lock source repositories differ")
        if request.source_revision != lock.source_revision:
            raise ValueError("support request and promotion lock source revisions differ")
        if request.model_set_id != lock.model_set_id:
            raise ValueError("support request and promotion lock model-set identities differ")
        if self.target_group_ids != lock.target_group_ids:
            raise ValueError("support request does not cover the complete frozen target roster")
        if self.supported_target_group_ids != lock.target_group_ids:
            missing = sorted(set(lock.target_group_ids) - set(self.supported_target_group_ids))
            raise PermissionError(
                f"support feasibility lacks a supported stream for target groups: {missing}"
            )

    @property
    def promotion_lock_id(self) -> str:
        return self.promotion_lock.promotion_lock_id

    @property
    def support_request_id(self) -> str:
        return self.support_feasibility.request.request_id

    @property
    def support_feasibility_id(self) -> str:
        return self.support_feasibility.provider_support_feasibility_id

    @property
    def target_group_ids(self) -> tuple[str, ...]:
        return _request_group_ids(self.support_feasibility)

    @property
    def supported_target_group_ids(self) -> tuple[str, ...]:
        return _supported_group_ids(self.support_feasibility)

    @property
    def stream_roster_id(self) -> str:
        return _sha256_json(_stream_roster_descriptor(self.support_feasibility))

    @property
    def technical_exclusion_policy_id(self) -> str:
        return _sha256_json(_technical_exclusion_policy_descriptor(self.support_feasibility))

    @property
    def intrinsics_ids(self) -> tuple[str, ...]:
        return _digest_ids(self.support_feasibility, "intrinsics_id")

    @property
    def extrinsics_ids(self) -> tuple[str, ...]:
        return _digest_ids(self.support_feasibility, "extrinsics_id")

    @property
    def metric_anchor_ids(self) -> tuple[str, ...]:
        return _digest_ids(self.support_feasibility, "metric_anchor_id")

    @property
    def authorized(self) -> bool:
        return True

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": PROVIDER_PROMOTION_AUTHORIZATION_SCHEMA,
            "schema_version": PROVIDER_PROMOTION_AUTHORIZATION_VERSION,
            "promotion_lock": self.promotion_lock.to_dict(),
            "support_feasibility": self.support_feasibility.to_dict(),
            "promotion_lock_id": self.promotion_lock_id,
            "support_request_id": self.support_request_id,
            "support_feasibility_id": self.support_feasibility_id,
            "target_group_ids": list(self.target_group_ids),
            "supported_target_group_ids": list(self.supported_target_group_ids),
            "stream_roster_id": self.stream_roster_id,
            "technical_exclusion_policy_id": (self.technical_exclusion_policy_id),
            "intrinsics_ids": list(self.intrinsics_ids),
            "extrinsics_ids": list(self.extrinsics_ids),
            "metric_anchor_ids": list(self.metric_anchor_ids),
            "calibration_payloads_opened": (self.calibration_payloads_opened),
            "target_payloads_opened": self.target_payloads_opened,
            "provider_residuals_computed": (self.provider_residuals_computed),
            "authorized": True,
            "metadata": plain_json(self.metadata),
            "claim_boundary": (PROVIDER_PROMOTION_AUTHORIZATION_CLAIM_BOUNDARY),
        }

    @property
    def authorization_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "authorization_id": self.authorization_id,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> ProviderPromotionAuthorizationV2:
        mapping = _strict_mapping(
            value,
            name="provider promotion authorization",
        )
        _exact_keys(
            mapping,
            _AUTHORIZATION_FIELDS,
            name="provider promotion authorization",
        )
        if mapping["schema_name"] != PROVIDER_PROMOTION_AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported provider promotion authorization schema")
        if mapping["schema_version"] != PROVIDER_PROMOTION_AUTHORIZATION_VERSION:
            raise ValueError("unsupported provider promotion authorization version")
        if mapping["claim_boundary"] != PROVIDER_PROMOTION_AUTHORIZATION_CLAIM_BOUNDARY:
            raise ValueError("provider promotion authorization claim boundary changed")
        if mapping["authorized"] is not True:
            raise ValueError("provider promotion authorization is not positive")
        authorization = cls(
            promotion_lock=promotion_lock_from_dict(mapping["promotion_lock"]),
            support_feasibility=ProviderSupportFeasibilityV1.from_dict(
                mapping["support_feasibility"]
            ),
            calibration_payloads_opened=mapping["calibration_payloads_opened"],
            target_payloads_opened=mapping["target_payloads_opened"],
            provider_residuals_computed=mapping["provider_residuals_computed"],
            metadata=_strict_mapping(
                mapping["metadata"],
                name="provider promotion authorization metadata",
            ),
        )
        derived = {
            "promotion_lock_id": authorization.promotion_lock_id,
            "support_request_id": authorization.support_request_id,
            "support_feasibility_id": authorization.support_feasibility_id,
            "target_group_ids": list(authorization.target_group_ids),
            "supported_target_group_ids": list(authorization.supported_target_group_ids),
            "stream_roster_id": authorization.stream_roster_id,
            "technical_exclusion_policy_id": (authorization.technical_exclusion_policy_id),
            "intrinsics_ids": list(authorization.intrinsics_ids),
            "extrinsics_ids": list(authorization.extrinsics_ids),
            "metric_anchor_ids": list(authorization.metric_anchor_ids),
        }
        for name, expected in derived.items():
            if mapping[name] != expected:
                raise ValueError(f"provider promotion authorization {name} changed")
        if mapping["authorization_id"] != authorization.authorization_id:
            raise ValueError("provider promotion authorization identity changed")
        return authorization


@dataclass(frozen=True, slots=True)
class AuthorizedHeldoutProviderEvidenceV1:
    """Replay-complete evidence bound to its earlier support authorization."""

    authorization: ProviderPromotionAuthorizationV2
    evidence: HeldoutProviderEvidenceV2

    def __post_init__(self) -> None:
        if not isinstance(
            self.authorization,
            ProviderPromotionAuthorizationV2,
        ):
            raise TypeError("authorization must be ProviderPromotionAuthorizationV2")
        if not isinstance(self.evidence, HeldoutProviderEvidenceV2):
            raise TypeError("evidence must be HeldoutProviderEvidenceV2")
        lock = self.authorization.promotion_lock
        evidence = self.evidence
        if evidence.promotion_lock.to_dict() != lock.to_dict():
            raise ValueError("held-out evidence uses a different promotion lock")
        selection = evidence.selection_evidence
        if selection.experiment_id != lock.experiment_id:
            raise ValueError("held-out evidence identifies a different experiment")
        if selection.source_repository != lock.source_repository:
            raise ValueError("held-out evidence identifies a different source repository")
        if selection.source_revision != lock.source_revision:
            raise ValueError("held-out evidence identifies a different source revision")
        calibration_groups = tuple(sorted({row.group_id for row in selection.calibration_rows}))
        if calibration_groups != lock.calibration_group_ids:
            raise ValueError("held-out evidence changed the calibration group roster")
        target_groups = tuple(decision.group_id for decision in selection.deployment_decisions)
        if target_groups != lock.target_group_ids:
            raise ValueError("held-out evidence changed the target group roster")

    @property
    def authorization_id(self) -> str:
        return self.authorization.authorization_id

    @property
    def evidence_id(self) -> str:
        return self.evidence.evidence_id

    @property
    def promotion_lock_id(self) -> str:
        return self.authorization.promotion_lock_id

    @property
    def overall_passed(self) -> bool:
        return self.evidence.promotion_report.overall_passed

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": (AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_SCHEMA),
            "schema_version": (AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_VERSION),
            "authorization": self.authorization.to_dict(),
            "evidence": self.evidence.to_dict(),
            "authorization_id": self.authorization_id,
            "evidence_id": self.evidence_id,
            "promotion_lock_id": self.promotion_lock_id,
            "overall_passed": self.overall_passed,
            "claim_boundary": (AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY),
        }

    @property
    def authorized_evidence_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "authorized_evidence_id": self.authorized_evidence_id,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> AuthorizedHeldoutProviderEvidenceV1:
        mapping = _strict_mapping(
            value,
            name="authorized held-out provider evidence",
        )
        _exact_keys(
            mapping,
            _AUTHORIZED_EVIDENCE_FIELDS,
            name="authorized held-out provider evidence",
        )
        if mapping["schema_name"] != AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_SCHEMA:
            raise ValueError("unsupported authorized held-out evidence schema")
        if mapping["schema_version"] != AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_VERSION:
            raise ValueError("unsupported authorized held-out evidence version")
        if mapping["claim_boundary"] != AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY:
            raise ValueError("authorized held-out evidence claim boundary changed")
        result = cls(
            authorization=ProviderPromotionAuthorizationV2.from_dict(mapping["authorization"]),
            evidence=heldout_provider_evidence_from_dict(mapping["evidence"]),
        )
        derived = {
            "authorization_id": result.authorization_id,
            "evidence_id": result.evidence_id,
            "promotion_lock_id": result.promotion_lock_id,
            "overall_passed": result.overall_passed,
            "authorized_evidence_id": result.authorized_evidence_id,
        }
        for name, expected in derived.items():
            if mapping[name] != expected:
                raise ValueError(f"authorized held-out provider evidence {name} changed")
        return result


def authorize_provider_promotion(
    promotion_lock: HeldoutProviderPromotionLockV1,
    support_feasibility: ProviderSupportFeasibilityV1,
    *,
    calibration_payloads_opened: bool = False,
    target_payloads_opened: bool = False,
    provider_residuals_computed: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ProviderPromotionAuthorizationV2:
    """Create a positive authorization before calibration or target access."""

    return ProviderPromotionAuthorizationV2(
        promotion_lock=promotion_lock,
        support_feasibility=support_feasibility,
        calibration_payloads_opened=calibration_payloads_opened,
        target_payloads_opened=target_payloads_opened,
        provider_residuals_computed=provider_residuals_computed,
        metadata={} if metadata is None else metadata,
    )


def require_provider_promotion_authorization(
    authorization: ProviderPromotionAuthorizationV2,
    promotion_lock: HeldoutProviderPromotionLockV1,
) -> ProviderPromotionAuthorizationV2:
    """Fail closed unless one authorization belongs to the exact lock."""

    if not isinstance(
        authorization,
        ProviderPromotionAuthorizationV2,
    ):
        raise TypeError("authorization must be ProviderPromotionAuthorizationV2")
    if not isinstance(
        promotion_lock,
        HeldoutProviderPromotionLockV1,
    ):
        raise TypeError("promotion_lock must be HeldoutProviderPromotionLockV1")
    if authorization.promotion_lock.to_dict() != promotion_lock.to_dict():
        raise ValueError("provider promotion authorization belongs to a different lock")
    return authorization


def bind_authorized_heldout_provider_evidence(
    authorization: ProviderPromotionAuthorizationV2,
    evidence: HeldoutProviderEvidenceV2,
) -> AuthorizedHeldoutProviderEvidenceV1:
    """Bind final replay evidence to its earlier target-free authorization."""

    require_provider_promotion_authorization(
        authorization,
        evidence.promotion_lock,
    )
    return AuthorizedHeldoutProviderEvidenceV1(
        authorization=authorization,
        evidence=evidence,
    )


def write_provider_promotion_authorization(
    path: str | Path,
    authorization: ProviderPromotionAuthorizationV2,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(
        authorization,
        ProviderPromotionAuthorizationV2,
    ):
        raise TypeError("authorization must be ProviderPromotionAuthorizationV2")
    _atomic_write_json(
        Path(path),
        authorization.to_dict(),
        overwrite=overwrite,
    )


def load_provider_promotion_authorization(
    path: str | Path,
) -> ProviderPromotionAuthorizationV2:
    mapping, _ = _load_json(
        Path(path),
        name="provider promotion authorization",
    )
    return ProviderPromotionAuthorizationV2.from_dict(mapping)


def write_authorized_heldout_provider_evidence(
    path: str | Path,
    evidence: AuthorizedHeldoutProviderEvidenceV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(
        evidence,
        AuthorizedHeldoutProviderEvidenceV1,
    ):
        raise TypeError("evidence must be AuthorizedHeldoutProviderEvidenceV1")
    _atomic_write_json(
        Path(path),
        evidence.to_dict(),
        overwrite=overwrite,
    )


def load_authorized_heldout_provider_evidence(
    path: str | Path,
) -> AuthorizedHeldoutProviderEvidenceV1:
    mapping, _ = _load_json(
        Path(path),
        name="authorized held-out provider evidence",
    )
    return AuthorizedHeldoutProviderEvidenceV1.from_dict(mapping)


def _summary(
    authorization: ProviderPromotionAuthorizationV2,
) -> dict[str, object]:
    return {
        "authorization_id": authorization.authorization_id,
        "promotion_lock_id": authorization.promotion_lock_id,
        "support_request_id": authorization.support_request_id,
        "support_feasibility_id": authorization.support_feasibility_id,
        "target_group_count": len(authorization.target_group_ids),
        "stream_count": authorization.support_feasibility.stream_count,
        "authorized": True,
    }


def _print_json(value: Mapping[str, object], *, compact: bool) -> None:
    print(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
            allow_nan=False,
        )
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prob4d.provider_promotion_authorization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--promotion-lock", required=True)
    authorize.add_argument("--support-feasibility", required=True)
    authorize.add_argument("--output", required=True)
    authorize.add_argument("--compact", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--compact", action="store_true")

    bind = subparsers.add_parser("bind-evidence")
    bind.add_argument("--authorization", required=True)
    bind.add_argument("--evidence", required=True)
    bind.add_argument("--output", required=True)
    bind.add_argument("--compact", action="store_true")

    parsed = parser.parse_args(arguments)
    if parsed.command == "authorize":
        try:
            authorization = authorize_provider_promotion(
                load_promotion_lock(parsed.promotion_lock),
                load_provider_support_feasibility(parsed.support_feasibility),
            )
        except PermissionError as error:
            print(str(error), file=sys.stderr)
            return 2
        write_provider_promotion_authorization(
            parsed.output,
            authorization,
        )
        _print_json(
            _summary(authorization),
            compact=parsed.compact,
        )
        return 0
    if parsed.command == "verify":
        authorization = load_provider_promotion_authorization(parsed.artifact)
        _print_json(
            _summary(authorization),
            compact=parsed.compact,
        )
        return 0
    authorization = load_provider_promotion_authorization(parsed.authorization)
    evidence = load_heldout_provider_evidence(parsed.evidence)
    bound = bind_authorized_heldout_provider_evidence(
        authorization,
        evidence,
    )
    write_authorized_heldout_provider_evidence(
        parsed.output,
        bound,
    )
    _print_json(
        {
            "authorized_evidence_id": bound.authorized_evidence_id,
            "authorization_id": bound.authorization_id,
            "evidence_id": bound.evidence_id,
            "promotion_lock_id": bound.promotion_lock_id,
            "overall_passed": bound.overall_passed,
        },
        compact=parsed.compact,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY",
    "AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_SCHEMA",
    "AUTHORIZED_HELDOUT_PROVIDER_EVIDENCE_VERSION",
    "PROVIDER_PROMOTION_AUTHORIZATION_CLAIM_BOUNDARY",
    "PROVIDER_PROMOTION_AUTHORIZATION_SCHEMA",
    "PROVIDER_PROMOTION_AUTHORIZATION_VERSION",
    "AuthorizedHeldoutProviderEvidenceV1",
    "ProviderPromotionAuthorizationV2",
    "authorize_provider_promotion",
    "bind_authorized_heldout_provider_evidence",
    "load_authorized_heldout_provider_evidence",
    "load_provider_promotion_authorization",
    "main",
    "require_provider_promotion_authorization",
    "write_authorized_heldout_provider_evidence",
    "write_provider_promotion_authorization",
]
