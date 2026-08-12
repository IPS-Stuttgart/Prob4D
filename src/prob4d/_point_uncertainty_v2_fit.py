"""Equal-group Gaussian-NLL fitting for point uncertainty calibration v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._point_uncertainty_v2_common import (
    PointUncertaintyCalibrationPolicyV2,
    float_matrix,
    local_point_basis,
)
from ._point_uncertainty_v2_model import PointUncertaintyCalibrationV2
from ._strict_json import require_exact_string, require_sha256
from .gauge_propagation_readiness import GaugePropagationReadinessV1
from .source_covariance_localization import SourceCovarianceLocalizationV1


def _group_weights(
    group_ids: Sequence[str],
    *,
    required_groups: tuple[str, ...],
    policy: PointUncertaintyCalibrationPolicyV2,
) -> tuple[np.ndarray, tuple[int, ...]]:
    values = tuple(require_exact_string(item, name="group_id") for item in group_ids)
    actual_groups = tuple(sorted(set(values)))
    if actual_groups != required_groups:
        raise ValueError("training group IDs must exactly match localization groups")
    if len(actual_groups) < policy.minimum_group_count:
        raise ValueError("too few independent groups for point uncertainty calibration")
    counts = tuple(values.count(group_id) for group_id in actual_groups)
    if any(count < policy.minimum_rows_per_group for count in counts):
        raise ValueError("one or more groups have too few calibration rows")
    count_by_group = dict(zip(actual_groups, counts, strict=True))
    weights = np.asarray(
        [
            1.0 / (len(actual_groups) * count_by_group[group_id])
            for group_id in values
        ],
        dtype=np.float64,
    )
    return weights, counts


def _weighted_design(
    features: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.sum(weights[:, None] * features, axis=0)
    centered = features - mean[None, :]
    scale = np.sqrt(
        np.maximum(np.sum(weights[:, None] * centered * centered, axis=0), 1e-12)
    )
    return np.column_stack((np.ones(features.shape[0]), centered / scale)), mean, scale


def _axis_terms(
    design: np.ndarray,
    residual_squared: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    *,
    policy: PointUncertaintyCalibrationPolicyV2,
) -> tuple[float, np.ndarray, np.ndarray]:
    eta = np.clip(
        design @ coefficients,
        policy.log_variance_lower,
        policy.log_variance_upper,
    )
    variance = np.maximum(np.exp(eta), policy.variance_floor)
    ratio = residual_squared / variance
    penalty = coefficients.copy()
    penalty[0] = 0.0
    objective = 0.5 * float(
        np.sum(weights * (np.log(variance) + ratio))
        + policy.ridge_strength * penalty @ penalty
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    gradient = (
        0.5 * design.T @ (weights * (1.0 - ratio))
        + policy.ridge_strength * regularizer @ coefficients
    )
    hessian = (
        0.5 * design.T @ ((weights * ratio)[:, None] * design)
        + policy.ridge_strength * regularizer
    )
    return objective, gradient, hessian


def _fit_axis(
    design: np.ndarray,
    residual_squared: np.ndarray,
    weights: np.ndarray,
    *,
    policy: PointUncertaintyCalibrationPolicyV2,
) -> tuple[np.ndarray, int, bool]:
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = np.log(
        max(float(np.sum(weights * residual_squared)), policy.variance_floor)
    )
    for iteration in range(1, policy.maximum_iterations + 1):
        objective, gradient, hessian = _axis_terms(
            design,
            residual_squared,
            weights,
            coefficients,
            policy=policy,
        )
        step = np.linalg.solve(
            hessian + 1e-12 * np.eye(design.shape[1]),
            gradient,
        )
        if np.linalg.norm(step, ord=np.inf) <= policy.newton_tolerance:
            return coefficients, iteration, True
        damping = 1.0
        while damping >= 2.0**-20:
            candidate = coefficients - damping * step
            candidate_objective, _, _ = _axis_terms(
                design,
                residual_squared,
                weights,
                candidate,
                policy=policy,
            )
            if candidate_objective <= objective:
                coefficients = candidate
                break
            damping *= 0.5
        else:
            return coefficients, iteration, False
    return coefficients, policy.maximum_iterations, False


def _fit_lateral(
    design: np.ndarray,
    residual_one: np.ndarray,
    residual_two: np.ndarray,
    weights: np.ndarray,
    *,
    policy: PointUncertaintyCalibrationPolicyV2,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    first, _, _ = _fit_axis(design, residual_one, weights, policy=policy)
    second, _, _ = _fit_axis(design, residual_two, weights, policy=policy)
    size = design.shape[1]
    coupling = policy.lateral_coupling_strength * np.eye(size, dtype=np.float64)

    for iteration in range(1, policy.maximum_iterations + 1):
        obj_one, grad_one, hess_one = _axis_terms(
            design, residual_one, weights, first, policy=policy
        )
        obj_two, grad_two, hess_two = _axis_terms(
            design, residual_two, weights, second, policy=policy
        )
        difference = first - second
        objective = (
            obj_one
            + obj_two
            + 0.5 * policy.lateral_coupling_strength * difference @ difference
        )
        gradient = np.concatenate(
            (grad_one + coupling @ difference, grad_two - coupling @ difference)
        )
        hessian = np.block(
            [
                [hess_one + coupling, -coupling],
                [-coupling, hess_two + coupling],
            ]
        )
        step = np.linalg.solve(
            hessian + 1e-12 * np.eye(2 * size),
            gradient,
        )
        if np.linalg.norm(step, ord=np.inf) <= policy.newton_tolerance:
            return first, second, iteration, True

        damping = 1.0
        while damping >= 2.0**-20:
            candidate_one = first - damping * step[:size]
            candidate_two = second - damping * step[size:]
            cand_obj_one, _, _ = _axis_terms(
                design, residual_one, weights, candidate_one, policy=policy
            )
            cand_obj_two, _, _ = _axis_terms(
                design, residual_two, weights, candidate_two, policy=policy
            )
            cand_difference = candidate_one - candidate_two
            candidate_objective = (
                cand_obj_one
                + cand_obj_two
                + 0.5
                * policy.lateral_coupling_strength
                * cand_difference
                @ cand_difference
            )
            if candidate_objective <= objective:
                first, second = candidate_one, candidate_two
                break
            damping *= 0.5
        else:
            return first, second, iteration, False
    return first, second, policy.maximum_iterations, False


def validate_point_uncertainty_v2_eligibility(
    localization: SourceCovarianceLocalizationV1,
    propagation: GaugePropagationReadinessV1,
) -> tuple[str, ...]:
    """Validate source localization and gauge propagation before residual access."""

    if not isinstance(localization, SourceCovarianceLocalizationV1):
        raise TypeError("localization must be SourceCovarianceLocalizationV1")
    if (
        localization.classification != "point-covariance-localized"
        or not localization.authorize_point_uncertainty_development
    ):
        raise ValueError(
            "PointUncertaintyCalibrationV2 requires point-covariance-localized authorization"
        )
    if not isinstance(propagation, GaugePropagationReadinessV1):
        raise TypeError("propagation must be GaugePropagationReadinessV1")
    if propagation.provider_manifest_id != localization.provider_manifest_id:
        raise ValueError("propagation and localization provider IDs differ")
    if propagation.cohort_binding_id != localization.cohort_binding_id:
        raise ValueError("propagation and localization cohort IDs differ")
    if propagation.source_covariance_localization_id != (
        localization.source_covariance_localization_id
    ):
        raise ValueError("propagation references a different covariance localization")
    required_groups = tuple(group.group_id for group in localization.groups)
    if propagation.source_group_ids != required_groups:
        raise ValueError("propagation and localization source groups differ")
    if propagation.gate_status != "pass":
        raise ValueError(
            "PointUncertaintyCalibrationV2 requires passing gauge propagation readiness"
        )
    if propagation.explicit_latent_or_exact_fallback_required:
        raise ValueError("gauge propagation requires the explicit latent or exact fallback")
    if propagation.classification not in {
        "explicit-gauge-latent-retained",
        "first-order-adequate",
    }:
        raise ValueError("gauge propagation classification does not authorize fitting")
    return required_groups


def fit_point_uncertainty_calibration_v2(
    localization: SourceCovarianceLocalizationV1,
    propagation: GaugePropagationReadinessV1,
    *,
    residual_xyz: object,
    ray_directions: object,
    tangent_reference: object,
    features: object,
    feature_names: Sequence[str],
    group_ids: Sequence[str],
    source_training_sha256: str,
    policy: PointUncertaintyCalibrationPolicyV2 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PointUncertaintyCalibrationV2:
    """Fit the gated equal-group Gaussian variance model."""

    required_groups = validate_point_uncertainty_v2_eligibility(
        localization,
        propagation,
    )
    fit_policy = PointUncertaintyCalibrationPolicyV2() if policy is None else policy
    if not isinstance(fit_policy, PointUncertaintyCalibrationPolicyV2):
        raise TypeError("policy must be PointUncertaintyCalibrationPolicyV2")

    residuals = float_matrix(residual_xyz, name="residual_xyz", columns=3)
    feature_matrix = float_matrix(features, name="features")
    if feature_matrix.shape[0] != residuals.shape[0]:
        raise ValueError("features and residual_xyz must have matching rows")
    names = tuple(
        require_exact_string(item, name=f"feature_names[{index}]")
        for index, item in enumerate(feature_names)
    )
    if len(names) != feature_matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match feature columns")
    if len(group_ids) != residuals.shape[0]:
        raise ValueError("group_ids and residual_xyz must have matching rows")

    weights, group_counts = _group_weights(
        group_ids,
        required_groups=required_groups,
        policy=fit_policy,
    )
    rays, tangent_one, tangent_two = local_point_basis(
        ray_directions,
        tangent_reference,
    )
    if rays.shape[0] != residuals.shape[0]:
        raise ValueError("basis inputs and residual_xyz must have matching rows")
    projected = np.column_stack(
        (
            np.sum(residuals * rays, axis=1),
            np.sum(residuals * tangent_one, axis=1),
            np.sum(residuals * tangent_two, axis=1),
        )
    )
    residual_squared = projected * projected
    design, feature_mean, feature_scale = _weighted_design(feature_matrix, weights)

    parallel, parallel_iterations, parallel_converged = _fit_axis(
        design,
        residual_squared[:, 0],
        weights,
        policy=fit_policy,
    )
    lateral_one, lateral_two, lateral_iterations, lateral_converged = _fit_lateral(
        design,
        residual_squared[:, 1],
        residual_squared[:, 2],
        weights,
        policy=fit_policy,
    )
    coefficients = np.vstack((parallel, lateral_one, lateral_two))
    predicted = np.maximum(
        np.exp(
            np.clip(
                design @ coefficients.T,
                fit_policy.log_variance_lower,
                fit_policy.log_variance_upper,
            )
        ),
        fit_policy.variance_floor,
    )
    normalized_energy = np.sum(
        weights[:, None] * residual_squared / predicted,
        axis=0,
    )

    return PointUncertaintyCalibrationV2(
        provider_manifest_id=localization.provider_manifest_id,
        cohort_binding_id=localization.cohort_binding_id,
        source_covariance_localization_id=localization.source_covariance_localization_id,
        gauge_propagation_readiness_id=propagation.gauge_propagation_readiness_id,
        source_training_sha256=require_sha256(
            source_training_sha256,
            name="source_training_sha256",
        ),
        feature_names=names,
        feature_mean=tuple(float(item) for item in feature_mean),
        feature_scale=tuple(float(item) for item in feature_scale),
        parallel_coefficients=tuple(float(item) for item in parallel),
        lateral_reference_coefficients=tuple(float(item) for item in lateral_one),
        lateral_orthogonal_coefficients=tuple(float(item) for item in lateral_two),
        group_ids=required_groups,
        group_counts=group_counts,
        policy=fit_policy,
        training_normalized_energy=tuple(float(item) for item in normalized_energy),
        fit_iterations=max(parallel_iterations, lateral_iterations),
        fit_converged=parallel_converged and lateral_converged,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "fit_point_uncertainty_calibration_v2",
    "validate_point_uncertainty_v2_eligibility",
]
