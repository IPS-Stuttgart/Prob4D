"""Source-only provider mean-quality and identity-competence evidence.

The report in this module treats complete physical objects or acquisition sessions
as the statistical units. It deliberately separates observation-mean competence
from identity/reliability competence so downstream readiness logic cannot respond
to an inaccurate mean by merely fitting a richer covariance model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_sha256,
)

SOURCE_PROVIDER_COMPETENCE_SCHEMA = "prob4d.source-provider-competence"
SOURCE_PROVIDER_COMPETENCE_VERSION = 1
SOURCE_PROVIDER_COMPETENCE_CLAIM_BOUNDARY = (
    "This source-only report evaluates observation-mean and identity/reliability "
    "competence on complete development or calibration objects/sessions. It does "
    "not use target payloads or outcomes, prove target transfer, authorize a "
    "BayesianPhysTwin update, establish Causal4D benefit, or establish state of the art."
)

GateStatus = Literal["pass", "fail", "technical-failure"]

_POLICY_FIELDS = frozenset(
    {
        "minimum_evaluable_groups",
        "maximum_technical_failures",
        "permitted_technical_failure_codes",
        "maximum_mean_proper_score_delta",
        "maximum_mean_point_rmse_ratio",
        "maximum_mean_endpoint_rmse_ratio",
        "maximum_worst_group_point_rmse_ratio",
        "maximum_mean_absolute_drift_slope_m_per_frame",
        "maximum_mean_seam_rmse_m",
        "minimum_mean_quality_group_pass_fraction",
        "minimum_mean_association_precision",
        "minimum_mean_identity_retention",
        "minimum_mean_support_retention",
        "minimum_identity_group_pass_fraction",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "group_id",
        "candidate_proper_score",
        "baseline_proper_score",
        "candidate_point_rmse_m",
        "baseline_point_rmse_m",
        "candidate_endpoint_rmse_m",
        "baseline_endpoint_rmse_m",
        "absolute_drift_slope_m_per_frame",
        "seam_rmse_m",
        "association_precision",
        "identity_retention",
        "support_retention",
        "technical_failure_code",
        "metadata",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "provider_manifest_id",
        "cohort_binding_id",
        "group_definition",
        "policy",
        "groups",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "group_count",
        "technical_failure_count",
        "evaluable_group_count",
        "technical_integrity_pass",
        "mean_proper_score_delta",
        "mean_point_rmse_ratio",
        "mean_endpoint_rmse_ratio",
        "worst_group_point_rmse_ratio",
        "mean_absolute_drift_slope_m_per_frame",
        "mean_seam_rmse_m",
        "mean_association_precision",
        "mean_identity_retention",
        "mean_support_retention",
        "mean_quality_group_pass_fraction",
        "identity_group_pass_fraction",
        "mean_quality_status",
        "identity_reliability_status",
        "mean_quality_reasons",
        "identity_reliability_reasons",
        "source_competence_pass",
        "metadata",
        "claim_boundary",
        "source_provider_competence_id",
    }
)


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _probability(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_exact_string(value, name=name)


def _sorted_unique_strings(
    value: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _strings_from_json(value: object, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return _sorted_unique_strings(tuple(value), name=name, allow_empty=allow_empty)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


@dataclass(frozen=True, slots=True)
class SourceProviderCompetencePolicyV1:
    """Frozen equal-group source-competence decision policy."""

    minimum_evaluable_groups: int
    maximum_technical_failures: int
    permitted_technical_failure_codes: tuple[str, ...]
    maximum_mean_proper_score_delta: float
    maximum_mean_point_rmse_ratio: float
    maximum_mean_endpoint_rmse_ratio: float
    maximum_worst_group_point_rmse_ratio: float
    maximum_mean_absolute_drift_slope_m_per_frame: float
    maximum_mean_seam_rmse_m: float
    minimum_mean_quality_group_pass_fraction: float
    minimum_mean_association_precision: float
    minimum_mean_identity_retention: float
    minimum_mean_support_retention: float
    minimum_identity_group_pass_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_evaluable_groups",
            require_exact_integer(
                self.minimum_evaluable_groups,
                name="minimum_evaluable_groups",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "maximum_technical_failures",
            require_exact_integer(
                self.maximum_technical_failures,
                name="maximum_technical_failures",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "permitted_technical_failure_codes",
            _sorted_unique_strings(
                self.permitted_technical_failure_codes,
                name="permitted_technical_failure_codes",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_mean_proper_score_delta",
            require_json_number(
                self.maximum_mean_proper_score_delta,
                name="maximum_mean_proper_score_delta",
            ),
        )
        for name in (
            "maximum_mean_point_rmse_ratio",
            "maximum_mean_endpoint_rmse_ratio",
            "maximum_worst_group_point_rmse_ratio",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        for name in (
            "maximum_mean_absolute_drift_slope_m_per_frame",
            "maximum_mean_seam_rmse_m",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))
        for name in (
            "minimum_mean_quality_group_pass_fraction",
            "minimum_mean_association_precision",
            "minimum_mean_identity_retention",
            "minimum_mean_support_retention",
            "minimum_identity_group_pass_fraction",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_evaluable_groups": self.minimum_evaluable_groups,
            "maximum_technical_failures": self.maximum_technical_failures,
            "permitted_technical_failure_codes": list(
                self.permitted_technical_failure_codes
            ),
            "maximum_mean_proper_score_delta": self.maximum_mean_proper_score_delta,
            "maximum_mean_point_rmse_ratio": self.maximum_mean_point_rmse_ratio,
            "maximum_mean_endpoint_rmse_ratio": self.maximum_mean_endpoint_rmse_ratio,
            "maximum_worst_group_point_rmse_ratio": (
                self.maximum_worst_group_point_rmse_ratio
            ),
            "maximum_mean_absolute_drift_slope_m_per_frame": (
                self.maximum_mean_absolute_drift_slope_m_per_frame
            ),
            "maximum_mean_seam_rmse_m": self.maximum_mean_seam_rmse_m,
            "minimum_mean_quality_group_pass_fraction": (
                self.minimum_mean_quality_group_pass_fraction
            ),
            "minimum_mean_association_precision": (
                self.minimum_mean_association_precision
            ),
            "minimum_mean_identity_retention": self.minimum_mean_identity_retention,
            "minimum_mean_support_retention": self.minimum_mean_support_retention,
            "minimum_identity_group_pass_fraction": (
                self.minimum_identity_group_pass_fraction
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceProviderCompetencePolicyV1:
        mapping = require_mapping(value, name="source competence policy")
        require_exact_fields(mapping, _POLICY_FIELDS, name="source competence policy")
        return cls(
            minimum_evaluable_groups=mapping["minimum_evaluable_groups"],
            maximum_technical_failures=mapping["maximum_technical_failures"],
            permitted_technical_failure_codes=_strings_from_json(
                mapping["permitted_technical_failure_codes"],
                name="permitted_technical_failure_codes",
                allow_empty=True,
            ),
            maximum_mean_proper_score_delta=mapping[
                "maximum_mean_proper_score_delta"
            ],
            maximum_mean_point_rmse_ratio=mapping[
                "maximum_mean_point_rmse_ratio"
            ],
            maximum_mean_endpoint_rmse_ratio=mapping[
                "maximum_mean_endpoint_rmse_ratio"
            ],
            maximum_worst_group_point_rmse_ratio=mapping[
                "maximum_worst_group_point_rmse_ratio"
            ],
            maximum_mean_absolute_drift_slope_m_per_frame=mapping[
                "maximum_mean_absolute_drift_slope_m_per_frame"
            ],
            maximum_mean_seam_rmse_m=mapping["maximum_mean_seam_rmse_m"],
            minimum_mean_quality_group_pass_fraction=mapping[
                "minimum_mean_quality_group_pass_fraction"
            ],
            minimum_mean_association_precision=mapping[
                "minimum_mean_association_precision"
            ],
            minimum_mean_identity_retention=mapping[
                "minimum_mean_identity_retention"
            ],
            minimum_mean_support_retention=mapping[
                "minimum_mean_support_retention"
            ],
            minimum_identity_group_pass_fraction=mapping[
                "minimum_identity_group_pass_fraction"
            ],
        )


@dataclass(frozen=True, slots=True)
class SourceProviderGroupResultV1:
    """One complete object/session source result."""

    group_id: str
    candidate_proper_score: float | None
    baseline_proper_score: float | None
    candidate_point_rmse_m: float | None
    baseline_point_rmse_m: float | None
    candidate_endpoint_rmse_m: float | None
    baseline_endpoint_rmse_m: float | None
    absolute_drift_slope_m_per_frame: float | None
    seam_rmse_m: float | None
    association_precision: float | None
    identity_retention: float | None
    support_retention: float | None
    technical_failure_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_id",
            require_exact_string(self.group_id, name="group_id"),
        )
        failure = _optional_string(
            self.technical_failure_code,
            name="technical_failure_code",
        )
        metric_names = (
            "candidate_proper_score",
            "baseline_proper_score",
            "candidate_point_rmse_m",
            "baseline_point_rmse_m",
            "candidate_endpoint_rmse_m",
            "baseline_endpoint_rmse_m",
            "absolute_drift_slope_m_per_frame",
            "seam_rmse_m",
            "association_precision",
            "identity_retention",
            "support_retention",
        )
        if failure is not None:
            if any(getattr(self, name) is not None for name in metric_names):
                raise ValueError("technical-failure groups must not contain scored metrics")
        else:
            if any(getattr(self, name) is None for name in metric_names):
                raise ValueError("evaluable groups require every scored metric")
            for name in ("candidate_proper_score", "baseline_proper_score"):
                object.__setattr__(
                    self,
                    name,
                    require_json_number(getattr(self, name), name=name),
                )
            for name in (
                "candidate_point_rmse_m",
                "candidate_endpoint_rmse_m",
                "absolute_drift_slope_m_per_frame",
                "seam_rmse_m",
            ):
                object.__setattr__(
                    self,
                    name,
                    _nonnegative(getattr(self, name), name=name),
                )
            for name in ("baseline_point_rmse_m", "baseline_endpoint_rmse_m"):
                object.__setattr__(
                    self,
                    name,
                    _positive(getattr(self, name), name=name),
                )
            for name in (
                "association_precision",
                "identity_retention",
                "support_retention",
            ):
                object.__setattr__(
                    self,
                    name,
                    _probability(getattr(self, name), name=name),
                )
        object.__setattr__(self, "technical_failure_code", failure)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="group metadata"),
        )

    @property
    def evaluable(self) -> bool:
        return self.technical_failure_code is None

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "candidate_proper_score": self.candidate_proper_score,
            "baseline_proper_score": self.baseline_proper_score,
            "candidate_point_rmse_m": self.candidate_point_rmse_m,
            "baseline_point_rmse_m": self.baseline_point_rmse_m,
            "candidate_endpoint_rmse_m": self.candidate_endpoint_rmse_m,
            "baseline_endpoint_rmse_m": self.baseline_endpoint_rmse_m,
            "absolute_drift_slope_m_per_frame": (
                self.absolute_drift_slope_m_per_frame
            ),
            "seam_rmse_m": self.seam_rmse_m,
            "association_precision": self.association_precision,
            "identity_retention": self.identity_retention,
            "support_retention": self.support_retention,
            "technical_failure_code": self.technical_failure_code,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceProviderGroupResultV1:
        mapping = require_mapping(value, name="source provider group")
        require_exact_fields(mapping, _GROUP_FIELDS, name="source provider group")
        return cls(
            group_id=mapping["group_id"],
            candidate_proper_score=mapping["candidate_proper_score"],
            baseline_proper_score=mapping["baseline_proper_score"],
            candidate_point_rmse_m=mapping["candidate_point_rmse_m"],
            baseline_point_rmse_m=mapping["baseline_point_rmse_m"],
            candidate_endpoint_rmse_m=mapping["candidate_endpoint_rmse_m"],
            baseline_endpoint_rmse_m=mapping["baseline_endpoint_rmse_m"],
            absolute_drift_slope_m_per_frame=mapping[
                "absolute_drift_slope_m_per_frame"
            ],
            seam_rmse_m=mapping["seam_rmse_m"],
            association_precision=mapping["association_precision"],
            identity_retention=mapping["identity_retention"],
            support_retention=mapping["support_retention"],
            technical_failure_code=mapping["technical_failure_code"],
            metadata=require_finite_json_mapping(
                mapping["metadata"], name="group metadata"
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceProviderCompetenceReportV1:
    """Replay-complete equal-group source competence decision."""

    provider_manifest_id: str
    cohort_binding_id: str
    group_definition: str
    policy: SourceProviderCompetencePolicyV1
    groups: tuple[SourceProviderGroupResultV1, ...]
    source_truth_used: bool = True
    target_payloads_opened: bool = False
    target_outcomes_opened: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    group_count: int = field(init=False)
    technical_failure_count: int = field(init=False)
    evaluable_group_count: int = field(init=False)
    technical_integrity_pass: bool = field(init=False)
    mean_proper_score_delta: float | None = field(init=False)
    mean_point_rmse_ratio: float | None = field(init=False)
    mean_endpoint_rmse_ratio: float | None = field(init=False)
    worst_group_point_rmse_ratio: float | None = field(init=False)
    mean_absolute_drift_slope_m_per_frame: float | None = field(init=False)
    mean_seam_rmse_m: float | None = field(init=False)
    mean_association_precision: float | None = field(init=False)
    mean_identity_retention: float | None = field(init=False)
    mean_support_retention: float | None = field(init=False)
    mean_quality_group_pass_fraction: float | None = field(init=False)
    identity_group_pass_fraction: float | None = field(init=False)
    mean_quality_status: GateStatus = field(init=False)
    identity_reliability_status: GateStatus = field(init=False)
    mean_quality_reasons: tuple[str, ...] = field(init=False)
    identity_reliability_reasons: tuple[str, ...] = field(init=False)
    source_competence_pass: bool = field(init=False)
    source_provider_competence_id: str = field(init=False)

    def __post_init__(self) -> None:
        provider_id = require_sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        cohort_id = require_sha256(self.cohort_binding_id, name="cohort_binding_id")
        group_definition = require_exact_string(
            self.group_definition,
            name="group_definition",
        )
        if not isinstance(self.policy, SourceProviderCompetencePolicyV1):
            raise TypeError("policy must be SourceProviderCompetencePolicyV1")
        if type(self.groups) is not tuple or not self.groups:
            raise ValueError("groups must be a nonempty canonical tuple")
        if any(not isinstance(item, SourceProviderGroupResultV1) for item in self.groups):
            raise TypeError("groups must contain SourceProviderGroupResultV1 values")
        groups = tuple(sorted(self.groups, key=lambda item: item.group_id))
        if tuple(item.group_id for item in groups) != tuple(
            sorted({item.group_id for item in groups})
        ):
            raise ValueError("group IDs must be unique")
        source_truth_used = _strict_bool(self.source_truth_used, name="source_truth_used")
        target_payloads_opened = _strict_bool(
            self.target_payloads_opened,
            name="target_payloads_opened",
        )
        target_outcomes_opened = _strict_bool(
            self.target_outcomes_opened,
            name="target_outcomes_opened",
        )
        if not source_truth_used:
            raise ValueError("source competence scoring requires declared source truth")
        if target_payloads_opened or target_outcomes_opened:
            raise ValueError("source competence evidence must remain target closed")
        metadata = frozen_finite_json_mapping(self.metadata, name="report metadata")

        technical = tuple(item for item in groups if not item.evaluable)
        evaluable = tuple(item for item in groups if item.evaluable)
        permitted = set(self.policy.permitted_technical_failure_codes)
        unknown_failures = tuple(
            item.technical_failure_code
            for item in technical
            if item.technical_failure_code not in permitted
        )
        technical_integrity = (
            len(technical) <= self.policy.maximum_technical_failures
            and not unknown_failures
        )

        metrics = self._aggregate(evaluable)
        mean_reasons: list[str] = []
        identity_reasons: list[str] = []
        if not technical_integrity:
            if len(technical) > self.policy.maximum_technical_failures:
                mean_reasons.append("technical-failure-budget-exceeded")
                identity_reasons.append("technical-failure-budget-exceeded")
            if unknown_failures:
                mean_reasons.append("unpermitted-technical-failure-code")
                identity_reasons.append("unpermitted-technical-failure-code")
        if len(evaluable) < self.policy.minimum_evaluable_groups:
            mean_reasons.append("insufficient-evaluable-groups")
            identity_reasons.append("insufficient-evaluable-groups")

        if evaluable:
            assert metrics["mean_proper_score_delta"] is not None
            assert metrics["mean_point_rmse_ratio"] is not None
            assert metrics["mean_endpoint_rmse_ratio"] is not None
            assert metrics["worst_group_point_rmse_ratio"] is not None
            assert metrics["mean_absolute_drift_slope_m_per_frame"] is not None
            assert metrics["mean_seam_rmse_m"] is not None
            assert metrics["mean_quality_group_pass_fraction"] is not None
            assert metrics["mean_association_precision"] is not None
            assert metrics["mean_identity_retention"] is not None
            assert metrics["mean_support_retention"] is not None
            assert metrics["identity_group_pass_fraction"] is not None
            if (
                metrics["mean_proper_score_delta"]
                > self.policy.maximum_mean_proper_score_delta
            ):
                mean_reasons.append("mean-proper-score-regression")
            if (
                metrics["mean_point_rmse_ratio"]
                > self.policy.maximum_mean_point_rmse_ratio
            ):
                mean_reasons.append("mean-point-rmse-regression")
            if (
                metrics["mean_endpoint_rmse_ratio"]
                > self.policy.maximum_mean_endpoint_rmse_ratio
            ):
                mean_reasons.append("mean-endpoint-rmse-regression")
            if (
                metrics["worst_group_point_rmse_ratio"]
                > self.policy.maximum_worst_group_point_rmse_ratio
            ):
                mean_reasons.append("worst-group-point-rmse-regression")
            if (
                metrics["mean_absolute_drift_slope_m_per_frame"]
                > self.policy.maximum_mean_absolute_drift_slope_m_per_frame
            ):
                mean_reasons.append("drift-slope-exceeded")
            if metrics["mean_seam_rmse_m"] > self.policy.maximum_mean_seam_rmse_m:
                mean_reasons.append("seam-error-exceeded")
            if (
                metrics["mean_quality_group_pass_fraction"]
                < self.policy.minimum_mean_quality_group_pass_fraction
            ):
                mean_reasons.append("mean-quality-group-pass-fraction-below-minimum")
            if (
                metrics["mean_association_precision"]
                < self.policy.minimum_mean_association_precision
            ):
                identity_reasons.append("association-precision-below-minimum")
            if (
                metrics["mean_identity_retention"]
                < self.policy.minimum_mean_identity_retention
            ):
                identity_reasons.append("identity-retention-below-minimum")
            if (
                metrics["mean_support_retention"]
                < self.policy.minimum_mean_support_retention
            ):
                identity_reasons.append("support-retention-below-minimum")
            if (
                metrics["identity_group_pass_fraction"]
                < self.policy.minimum_identity_group_pass_fraction
            ):
                identity_reasons.append("identity-group-pass-fraction-below-minimum")

        mean_status: GateStatus
        identity_status: GateStatus
        if not technical_integrity:
            mean_status = "technical-failure"
            identity_status = "technical-failure"
        else:
            mean_status = "pass" if not mean_reasons else "fail"
            identity_status = "pass" if not identity_reasons else "fail"

        object.__setattr__(self, "provider_manifest_id", provider_id)
        object.__setattr__(self, "cohort_binding_id", cohort_id)
        object.__setattr__(self, "group_definition", group_definition)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "source_truth_used", source_truth_used)
        object.__setattr__(self, "target_payloads_opened", target_payloads_opened)
        object.__setattr__(self, "target_outcomes_opened", target_outcomes_opened)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "group_count", len(groups))
        object.__setattr__(self, "technical_failure_count", len(technical))
        object.__setattr__(self, "evaluable_group_count", len(evaluable))
        object.__setattr__(self, "technical_integrity_pass", technical_integrity)
        for name, value in metrics.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "mean_quality_status", mean_status)
        object.__setattr__(self, "identity_reliability_status", identity_status)
        object.__setattr__(self, "mean_quality_reasons", tuple(sorted(mean_reasons)))
        object.__setattr__(
            self,
            "identity_reliability_reasons",
            tuple(sorted(identity_reasons)),
        )
        object.__setattr__(
            self,
            "source_competence_pass",
            mean_status == "pass" and identity_status == "pass",
        )
        object.__setattr__(
            self,
            "source_provider_competence_id",
            _sha256_json(self._content_dict()),
        )

    def _aggregate(
        self,
        groups: tuple[SourceProviderGroupResultV1, ...],
    ) -> dict[str, float | None]:
        names = {
            "mean_proper_score_delta",
            "mean_point_rmse_ratio",
            "mean_endpoint_rmse_ratio",
            "worst_group_point_rmse_ratio",
            "mean_absolute_drift_slope_m_per_frame",
            "mean_seam_rmse_m",
            "mean_association_precision",
            "mean_identity_retention",
            "mean_support_retention",
            "mean_quality_group_pass_fraction",
            "identity_group_pass_fraction",
        }
        if not groups:
            return {name: None for name in names}

        proper_deltas = []
        point_ratios = []
        endpoint_ratios = []
        mean_quality_pass = []
        identity_pass = []
        drifts = []
        seams = []
        associations = []
        identities = []
        supports = []
        for group in groups:
            assert group.candidate_proper_score is not None
            assert group.baseline_proper_score is not None
            assert group.candidate_point_rmse_m is not None
            assert group.baseline_point_rmse_m is not None
            assert group.candidate_endpoint_rmse_m is not None
            assert group.baseline_endpoint_rmse_m is not None
            assert group.absolute_drift_slope_m_per_frame is not None
            assert group.seam_rmse_m is not None
            assert group.association_precision is not None
            assert group.identity_retention is not None
            assert group.support_retention is not None
            proper_delta = group.candidate_proper_score - group.baseline_proper_score
            point_ratio = group.candidate_point_rmse_m / group.baseline_point_rmse_m
            endpoint_ratio = (
                group.candidate_endpoint_rmse_m / group.baseline_endpoint_rmse_m
            )
            proper_deltas.append(proper_delta)
            point_ratios.append(point_ratio)
            endpoint_ratios.append(endpoint_ratio)
            drifts.append(group.absolute_drift_slope_m_per_frame)
            seams.append(group.seam_rmse_m)
            associations.append(group.association_precision)
            identities.append(group.identity_retention)
            supports.append(group.support_retention)
            mean_quality_pass.append(
                proper_delta <= self.policy.maximum_mean_proper_score_delta
                and point_ratio <= self.policy.maximum_worst_group_point_rmse_ratio
                and endpoint_ratio <= self.policy.maximum_mean_endpoint_rmse_ratio
                and group.absolute_drift_slope_m_per_frame
                <= self.policy.maximum_mean_absolute_drift_slope_m_per_frame
                and group.seam_rmse_m <= self.policy.maximum_mean_seam_rmse_m
            )
            identity_pass.append(
                group.association_precision
                >= self.policy.minimum_mean_association_precision
                and group.identity_retention
                >= self.policy.minimum_mean_identity_retention
                and group.support_retention
                >= self.policy.minimum_mean_support_retention
            )
        return {
            "mean_proper_score_delta": float(np.mean(proper_deltas)),
            "mean_point_rmse_ratio": float(np.mean(point_ratios)),
            "mean_endpoint_rmse_ratio": float(np.mean(endpoint_ratios)),
            "worst_group_point_rmse_ratio": float(np.max(point_ratios)),
            "mean_absolute_drift_slope_m_per_frame": float(np.mean(drifts)),
            "mean_seam_rmse_m": float(np.mean(seams)),
            "mean_association_precision": float(np.mean(associations)),
            "mean_identity_retention": float(np.mean(identities)),
            "mean_support_retention": float(np.mean(supports)),
            "mean_quality_group_pass_fraction": float(np.mean(mean_quality_pass)),
            "identity_group_pass_fraction": float(np.mean(identity_pass)),
        }

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": SOURCE_PROVIDER_COMPETENCE_SCHEMA,
            "schema_version": SOURCE_PROVIDER_COMPETENCE_VERSION,
            "provider_manifest_id": self.provider_manifest_id,
            "cohort_binding_id": self.cohort_binding_id,
            "group_definition": self.group_definition,
            "policy": self.policy.to_dict(),
            "groups": [item.to_dict() for item in self.groups],
            "source_truth_used": self.source_truth_used,
            "target_payloads_opened": self.target_payloads_opened,
            "target_outcomes_opened": self.target_outcomes_opened,
            "group_count": self.group_count,
            "technical_failure_count": self.technical_failure_count,
            "evaluable_group_count": self.evaluable_group_count,
            "technical_integrity_pass": self.technical_integrity_pass,
            "mean_proper_score_delta": self.mean_proper_score_delta,
            "mean_point_rmse_ratio": self.mean_point_rmse_ratio,
            "mean_endpoint_rmse_ratio": self.mean_endpoint_rmse_ratio,
            "worst_group_point_rmse_ratio": self.worst_group_point_rmse_ratio,
            "mean_absolute_drift_slope_m_per_frame": (
                self.mean_absolute_drift_slope_m_per_frame
            ),
            "mean_seam_rmse_m": self.mean_seam_rmse_m,
            "mean_association_precision": self.mean_association_precision,
            "mean_identity_retention": self.mean_identity_retention,
            "mean_support_retention": self.mean_support_retention,
            "mean_quality_group_pass_fraction": (
                self.mean_quality_group_pass_fraction
            ),
            "identity_group_pass_fraction": self.identity_group_pass_fraction,
            "mean_quality_status": self.mean_quality_status,
            "identity_reliability_status": self.identity_reliability_status,
            "mean_quality_reasons": list(self.mean_quality_reasons),
            "identity_reliability_reasons": list(
                self.identity_reliability_reasons
            ),
            "source_competence_pass": self.source_competence_pass,
            "metadata": plain_json(self.metadata),
            "claim_boundary": SOURCE_PROVIDER_COMPETENCE_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["source_provider_competence_id"] = self.source_provider_competence_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> SourceProviderCompetenceReportV1:
        mapping = require_mapping(value, name="source provider competence report")
        require_exact_fields(
            mapping,
            _REPORT_FIELDS,
            name="source provider competence report",
        )
        if mapping["schema"] != SOURCE_PROVIDER_COMPETENCE_SCHEMA:
            raise ValueError("source provider competence schema changed")
        if mapping["schema_version"] != SOURCE_PROVIDER_COMPETENCE_VERSION:
            raise ValueError("source provider competence version changed")
        if mapping["claim_boundary"] != SOURCE_PROVIDER_COMPETENCE_CLAIM_BOUNDARY:
            raise ValueError("source provider competence claim boundary changed")
        raw_groups = mapping["groups"]
        if not isinstance(raw_groups, list):
            raise ValueError("groups must be a JSON array")
        result = cls(
            provider_manifest_id=mapping["provider_manifest_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            group_definition=mapping["group_definition"],
            policy=SourceProviderCompetencePolicyV1.from_dict(mapping["policy"]),
            groups=tuple(SourceProviderGroupResultV1.from_dict(item) for item in raw_groups),
            source_truth_used=mapping["source_truth_used"],
            target_payloads_opened=mapping["target_payloads_opened"],
            target_outcomes_opened=mapping["target_outcomes_opened"],
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="report metadata",
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("source provider competence derived fields changed")
        return result


def write_source_provider_competence(
    path: str | Path,
    report: SourceProviderCompetenceReportV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(report, SourceProviderCompetenceReportV1):
        raise TypeError("report must be SourceProviderCompetenceReportV1")
    _atomic_write_json(path, report.to_dict(), overwrite=overwrite)


def load_source_provider_competence(
    path: str | Path,
) -> SourceProviderCompetenceReportV1:
    return SourceProviderCompetenceReportV1.from_dict(
        load_json_object(path, name="source provider competence report")
    )


__all__ = [
    "SOURCE_PROVIDER_COMPETENCE_CLAIM_BOUNDARY",
    "SOURCE_PROVIDER_COMPETENCE_SCHEMA",
    "SOURCE_PROVIDER_COMPETENCE_VERSION",
    "SourceProviderCompetencePolicyV1",
    "SourceProviderCompetenceReportV1",
    "SourceProviderGroupResultV1",
    "load_source_provider_competence",
    "write_source_provider_competence",
]
