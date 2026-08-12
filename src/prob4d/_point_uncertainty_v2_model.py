"""Artifact model and prediction path for point uncertainty calibration v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._point_uncertainty_v2_common import (
    PointUncertaintyCalibrationPolicyV2,
    float_matrix,
    float_tuple,
    local_point_basis,
    string_tuple,
)
from ._scientific_scalars import require_genuine_integer
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_mapping,
    require_sha256,
)

POINT_UNCERTAINTY_CALIBRATION_SCHEMA = "prob4d.point-uncertainty-calibration"
POINT_UNCERTAINTY_CALIBRATION_VERSION = 2
POINT_UNCERTAINTY_CALIBRATION_CLAIM_BOUNDARY = (
    "This source/calibration-only artifact is an experimental conditional point-covariance "
    "model authorized by an explicit point-covariance-localized source diagnostic and a "
    "passing gauge-propagation readiness decision for the same provider, cohort, source "
    "groups, and physical query. It does not absorb shared Sim(3) gauge uncertainty, use "
    "protected target outcomes, establish provider competence or transfer, authorize a "
    "BayesianPhysTwin update, establish Causal4D intervention benefit, deployment safety, "
    "or state of the art."
)

_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "provider_manifest_id",
        "cohort_binding_id",
        "source_covariance_localization_id",
        "gauge_propagation_readiness_id",
        "source_training_sha256",
        "feature_names",
        "feature_mean",
        "feature_scale",
        "parallel_coefficients",
        "lateral_reference_coefficients",
        "lateral_orthogonal_coefficients",
        "group_ids",
        "group_counts",
        "policy",
        "training_normalized_energy",
        "fit_iterations",
        "fit_converged",
        "metadata",
        "claim_boundary",
        "point_uncertainty_calibration_id",
    }
)


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PointUncertaintyCalibrationV2:
    """Content-addressed three-axis conditional point-covariance calibration."""

    provider_manifest_id: str
    cohort_binding_id: str
    source_covariance_localization_id: str
    gauge_propagation_readiness_id: str
    source_training_sha256: str
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    parallel_coefficients: tuple[float, ...]
    lateral_reference_coefficients: tuple[float, ...]
    lateral_orthogonal_coefficients: tuple[float, ...]
    group_ids: tuple[str, ...]
    group_counts: tuple[int, ...]
    policy: PointUncertaintyCalibrationPolicyV2
    training_normalized_energy: tuple[float, float, float]
    fit_iterations: int
    fit_converged: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    point_uncertainty_calibration_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "provider_manifest_id",
            "cohort_binding_id",
            "source_covariance_localization_id",
            "gauge_propagation_readiness_id",
            "source_training_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))

        names = string_tuple(self.feature_names, name="feature_names")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(
            self,
            "feature_mean",
            float_tuple(self.feature_mean, name="feature_mean", length=len(names)),
        )
        object.__setattr__(
            self,
            "feature_scale",
            float_tuple(
                self.feature_scale,
                name="feature_scale",
                length=len(names),
                positive_only=True,
            ),
        )
        for name in (
            "parallel_coefficients",
            "lateral_reference_coefficients",
            "lateral_orthogonal_coefficients",
        ):
            object.__setattr__(
                self,
                name,
                float_tuple(
                    getattr(self, name),
                    name=name,
                    length=len(names) + 1,
                ),
            )

        group_ids = string_tuple(
            self.group_ids,
            name="group_ids",
            sorted_unique=True,
        )
        if type(self.group_counts) not in {tuple, list}:
            raise TypeError("group_counts must be a sequence")
        group_counts = tuple(
            require_genuine_integer(item, name=f"group_counts[{index}]", minimum=1)
            for index, item in enumerate(self.group_counts)
        )
        if len(group_ids) != len(group_counts):
            raise ValueError("group_ids and group_counts must have matching lengths")
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "group_counts", group_counts)

        if not isinstance(self.policy, PointUncertaintyCalibrationPolicyV2):
            raise TypeError("policy must be PointUncertaintyCalibrationPolicyV2")
        object.__setattr__(
            self,
            "training_normalized_energy",
            float_tuple(
                self.training_normalized_energy,
                name="training_normalized_energy",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "fit_iterations",
            require_genuine_integer(self.fit_iterations, name="fit_iterations", minimum=1),
        )
        if type(self.fit_converged) is not bool:
            raise TypeError("fit_converged must be a Boolean")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(dict(self.metadata), name="metadata"),
        )
        object.__setattr__(
            self,
            "point_uncertainty_calibration_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
            "schema_version": POINT_UNCERTAINTY_CALIBRATION_VERSION,
            "provider_manifest_id": self.provider_manifest_id,
            "cohort_binding_id": self.cohort_binding_id,
            "source_covariance_localization_id": self.source_covariance_localization_id,
            "gauge_propagation_readiness_id": self.gauge_propagation_readiness_id,
            "source_training_sha256": self.source_training_sha256,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "parallel_coefficients": list(self.parallel_coefficients),
            "lateral_reference_coefficients": list(self.lateral_reference_coefficients),
            "lateral_orthogonal_coefficients": list(self.lateral_orthogonal_coefficients),
            "group_ids": list(self.group_ids),
            "group_counts": list(self.group_counts),
            "policy": self.policy.to_dict(),
            "training_normalized_energy": list(self.training_normalized_energy),
            "fit_iterations": self.fit_iterations,
            "fit_converged": self.fit_converged,
            "metadata": plain_json(self.metadata),
            "claim_boundary": POINT_UNCERTAINTY_CALIBRATION_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "point_uncertainty_calibration_id": self.point_uncertainty_calibration_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> PointUncertaintyCalibrationV2:
        mapping = require_mapping(value, name="point uncertainty v2 calibration")
        require_exact_fields(
            mapping,
            _ARTIFACT_FIELDS,
            name="point uncertainty v2 calibration",
        )
        if mapping["schema"] != POINT_UNCERTAINTY_CALIBRATION_SCHEMA:
            raise ValueError("point uncertainty calibration schema changed")
        if mapping["schema_version"] != POINT_UNCERTAINTY_CALIBRATION_VERSION:
            raise ValueError("point uncertainty calibration version changed")
        if mapping["claim_boundary"] != POINT_UNCERTAINTY_CALIBRATION_CLAIM_BOUNDARY:
            raise ValueError("point uncertainty calibration claim boundary changed")

        result = cls(
            provider_manifest_id=mapping["provider_manifest_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            source_covariance_localization_id=mapping[
                "source_covariance_localization_id"
            ],
            gauge_propagation_readiness_id=mapping[
                "gauge_propagation_readiness_id"
            ],
            source_training_sha256=mapping["source_training_sha256"],
            feature_names=tuple(cast(list[Any], mapping["feature_names"])),
            feature_mean=tuple(cast(list[Any], mapping["feature_mean"])),
            feature_scale=tuple(cast(list[Any], mapping["feature_scale"])),
            parallel_coefficients=tuple(
                cast(list[Any], mapping["parallel_coefficients"])
            ),
            lateral_reference_coefficients=tuple(
                cast(list[Any], mapping["lateral_reference_coefficients"])
            ),
            lateral_orthogonal_coefficients=tuple(
                cast(list[Any], mapping["lateral_orthogonal_coefficients"])
            ),
            group_ids=tuple(cast(list[Any], mapping["group_ids"])),
            group_counts=tuple(cast(list[Any], mapping["group_counts"])),
            policy=PointUncertaintyCalibrationPolicyV2.from_dict(mapping["policy"]),
            training_normalized_energy=tuple(
                cast(list[Any], mapping["training_normalized_energy"])
            ),
            fit_iterations=mapping["fit_iterations"],
            fit_converged=mapping["fit_converged"],
            metadata=cast(Mapping[str, Any], mapping["metadata"]),
        )
        supplied_id = require_sha256(
            mapping["point_uncertainty_calibration_id"],
            name="point_uncertainty_calibration_id",
        )
        if supplied_id != result.point_uncertainty_calibration_id:
            raise ValueError("point uncertainty calibration identity mismatch")
        if plain_json(mapping) != result.to_dict():
            raise ValueError("point uncertainty calibration derived fields changed")
        return result

    def _design(self, features: object) -> np.ndarray:
        matrix = float_matrix(features, name="features")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("features have the wrong number of columns")
        standardized = (
            matrix - np.asarray(self.feature_mean, dtype=np.float64)[None, :]
        ) / np.asarray(self.feature_scale, dtype=np.float64)[None, :]
        return np.column_stack((np.ones(matrix.shape[0]), standardized))

    def predict_variances(self, features: object) -> np.ndarray:
        """Predict ray, reference-tangent, and orthogonal-tangent variances."""

        design = self._design(features)
        coefficients = np.asarray(
            [
                self.parallel_coefficients,
                self.lateral_reference_coefficients,
                self.lateral_orthogonal_coefficients,
            ],
            dtype=np.float64,
        )
        log_variance = np.clip(
            design @ coefficients.T,
            self.policy.log_variance_lower,
            self.policy.log_variance_upper,
        )
        return np.maximum(np.exp(log_variance), self.policy.variance_floor)

    def covariance_matrices(
        self,
        ray_directions: object,
        tangent_reference: object,
        features: object,
    ) -> np.ndarray:
        """Construct conditional 3x3 covariances without shared gauge terms."""

        rays, tangent_one, tangent_two = local_point_basis(
            ray_directions,
            tangent_reference,
        )
        variances = self.predict_variances(features)
        if variances.shape[0] != rays.shape[0]:
            raise ValueError("features and basis inputs must have matching rows")
        return (
            variances[:, 0, None, None] * np.einsum("ni,nj->nij", rays, rays)
            + variances[:, 1, None, None]
            * np.einsum("ni,nj->nij", tangent_one, tangent_one)
            + variances[:, 2, None, None]
            * np.einsum("ni,nj->nij", tangent_two, tangent_two)
        )


def write_point_uncertainty_calibration_v2(
    path: str | Path,
    calibration: PointUncertaintyCalibrationV2,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(calibration, PointUncertaintyCalibrationV2):
        raise TypeError("calibration must be PointUncertaintyCalibrationV2")
    payload = json.dumps(
        calibration.to_dict(),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_point_uncertainty_calibration_v2(
    path: str | Path,
) -> PointUncertaintyCalibrationV2:
    return PointUncertaintyCalibrationV2.from_dict(
        load_json_object(path, name="point uncertainty v2 calibration")
    )


__all__ = [
    "POINT_UNCERTAINTY_CALIBRATION_CLAIM_BOUNDARY",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "PointUncertaintyCalibrationV2",
    "load_point_uncertainty_calibration_v2",
    "write_point_uncertainty_calibration_v2",
]
