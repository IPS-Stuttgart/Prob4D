"""Content-addressed replayable report for point uncertainty v2 promotion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ._atomic_file import atomic_write_text
from ._immutable_json import plain_json
from ._point_uncertainty_v2_promotion_metrics import (
    _CRITERIA_FIELDS,
    _SUMMARY_FIELDS,
    _criteria,
    _summary,
)
from ._point_uncertainty_v2_promotion_types import (
    POINT_UNCERTAINTY_PROMOTION_CLAIM_BOUNDARY,
    POINT_UNCERTAINTY_PROMOTION_SCHEMA,
    POINT_UNCERTAINTY_PROMOTION_VERSION,
    PointUncertaintyGroupMetricsV1,
    PointUncertaintyPromotionPolicyV1,
)
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_string,
    require_mapping,
    require_sha256,
)

_REPORT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "point_uncertainty_calibration_id",
        "baseline_point_calibration_id",
        "provider_manifest_id",
        "cohort_binding_id",
        "validation_sha256",
        "training_group_ids",
        "baseline_training_group_ids",
        "validation_group_ids",
        "candidate_fit_converged",
        "policy",
        "groups",
        "summary",
        "criteria",
        "promote_candidate",
        "claim_boundary",
        "point_uncertainty_promotion_id",
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


@dataclass(frozen=True, slots=True)
class PointUncertaintyPromotionReportV1:
    point_uncertainty_calibration_id: str
    baseline_point_calibration_id: str
    provider_manifest_id: str
    cohort_binding_id: str
    validation_sha256: str
    training_group_ids: tuple[str, ...]
    baseline_training_group_ids: tuple[str, ...]
    groups: tuple[PointUncertaintyGroupMetricsV1, ...]
    policy: PointUncertaintyPromotionPolicyV1
    fit_converged: bool

    def __post_init__(self) -> None:
        for name in (
            "point_uncertainty_calibration_id",
            "baseline_point_calibration_id",
            "provider_manifest_id",
            "cohort_binding_id",
            "validation_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        training = tuple(
            require_exact_string(item, name=f"training_group_ids[{index}]")
            for index, item in enumerate(self.training_group_ids)
        )
        if not training or training != tuple(sorted(set(training))):
            raise ValueError("training_group_ids must be non-empty, sorted, and unique")
        object.__setattr__(self, "training_group_ids", training)
        baseline_training = tuple(
            require_exact_string(item, name=f"baseline_training_group_ids[{index}]")
            for index, item in enumerate(self.baseline_training_group_ids)
        )
        if baseline_training != training:
            raise ValueError(
                "baseline v1 and candidate v2 must use the same independent training groups"
            )
        object.__setattr__(self, "baseline_training_group_ids", baseline_training)
        groups = tuple(self.groups)
        if not groups:
            raise ValueError("groups must not be empty")
        validation_ids = tuple(group.group_id for group in groups)
        if validation_ids != tuple(sorted(set(validation_ids))):
            raise ValueError("validation group metrics must be sorted and unique")
        overlap = set(training).intersection(validation_ids)
        if overlap:
            raise ValueError(
                "point uncertainty promotion requires disjoint training and validation "
                f"groups; overlap={sorted(overlap)}"
            )
        object.__setattr__(self, "groups", groups)
        if not isinstance(self.policy, PointUncertaintyPromotionPolicyV1):
            raise TypeError("policy must be PointUncertaintyPromotionPolicyV1")
        if type(self.fit_converged) is not bool:
            raise TypeError("fit_converged must be a Boolean")

    @property
    def validation_group_ids(self) -> tuple[str, ...]:
        return tuple(group.group_id for group in self.groups)

    @property
    def summary(self) -> dict[str, float | int]:
        return _summary(self.groups)

    @property
    def criteria(self) -> dict[str, bool]:
        return _criteria(self.fit_converged, self.summary, self.policy)

    @property
    def promote_candidate(self) -> bool:
        return all(self.criteria.values())

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": POINT_UNCERTAINTY_PROMOTION_SCHEMA,
            "schema_version": POINT_UNCERTAINTY_PROMOTION_VERSION,
            "point_uncertainty_calibration_id": self.point_uncertainty_calibration_id,
            "baseline_point_calibration_id": self.baseline_point_calibration_id,
            "provider_manifest_id": self.provider_manifest_id,
            "cohort_binding_id": self.cohort_binding_id,
            "validation_sha256": self.validation_sha256,
            "training_group_ids": list(self.training_group_ids),
            "baseline_training_group_ids": list(self.baseline_training_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "candidate_fit_converged": self.fit_converged,
            "policy": self.policy.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "summary": self.summary,
            "criteria": self.criteria,
            "promote_candidate": self.promote_candidate,
            "claim_boundary": POINT_UNCERTAINTY_PROMOTION_CLAIM_BOUNDARY,
        }

    @property
    def point_uncertainty_promotion_id(self) -> str:
        return _sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "point_uncertainty_promotion_id": self.point_uncertainty_promotion_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> PointUncertaintyPromotionReportV1:
        mapping = require_mapping(value, name="point uncertainty promotion report")
        require_exact_fields(mapping, _REPORT_FIELDS, name="point uncertainty promotion report")
        if mapping["schema"] != POINT_UNCERTAINTY_PROMOTION_SCHEMA:
            raise ValueError("point uncertainty promotion schema changed")
        if mapping["schema_version"] != POINT_UNCERTAINTY_PROMOTION_VERSION:
            raise ValueError("point uncertainty promotion version changed")
        if mapping["claim_boundary"] != POINT_UNCERTAINTY_PROMOTION_CLAIM_BOUNDARY:
            raise ValueError("point uncertainty promotion claim boundary changed")
        criteria = require_mapping(mapping["criteria"], name="promotion criteria")
        require_exact_fields(criteria, _CRITERIA_FIELDS, name="promotion criteria")
        summary = require_mapping(mapping["summary"], name="promotion summary")
        require_exact_fields(summary, _SUMMARY_FIELDS, name="promotion summary")
        groups_raw = mapping["groups"]
        if type(groups_raw) is not list:
            raise ValueError("groups must be a JSON array")
        training_raw = mapping["training_group_ids"]
        if type(training_raw) is not list:
            raise ValueError("training_group_ids must be a JSON array")
        baseline_training_raw = mapping["baseline_training_group_ids"]
        if type(baseline_training_raw) is not list:
            raise ValueError("baseline_training_group_ids must be a JSON array")
        result = cls(
            point_uncertainty_calibration_id=mapping[
                "point_uncertainty_calibration_id"
            ],
            baseline_point_calibration_id=mapping["baseline_point_calibration_id"],
            provider_manifest_id=mapping["provider_manifest_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            validation_sha256=mapping["validation_sha256"],
            training_group_ids=tuple(cast(list[str], training_raw)),
            baseline_training_group_ids=tuple(cast(list[str], baseline_training_raw)),
            groups=tuple(
                PointUncertaintyGroupMetricsV1.from_dict(item)
                for item in cast(list[object], groups_raw)
            ),
            policy=PointUncertaintyPromotionPolicyV1.from_dict(mapping["policy"]),
            fit_converged=mapping["candidate_fit_converged"],
        )
        supplied_id = require_sha256(
            mapping["point_uncertainty_promotion_id"],
            name="point_uncertainty_promotion_id",
        )
        if supplied_id != result.point_uncertainty_promotion_id:
            raise ValueError("point uncertainty promotion identity mismatch")
        if plain_json(mapping) != result.to_dict():
            raise ValueError("point uncertainty promotion derived fields changed")
        return result


def write_point_uncertainty_promotion_report_v1(
    path: str | Path,
    report: PointUncertaintyPromotionReportV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(report, PointUncertaintyPromotionReportV1):
        raise TypeError("report must be PointUncertaintyPromotionReportV1")
    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_point_uncertainty_promotion_report_v1(
    path: str | Path,
) -> PointUncertaintyPromotionReportV1:
    return PointUncertaintyPromotionReportV1.from_dict(
        load_json_object(path, name="point uncertainty promotion report")
    )


__all__ = [
    "PointUncertaintyPromotionReportV1",
    "load_point_uncertainty_promotion_report_v1",
    "write_point_uncertainty_promotion_report_v1",
]
