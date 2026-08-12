from __future__ import annotations

import copy

import numpy as np
import pytest

from prob4d.gauge_propagation_readiness import (
    GaugePropagationReadinessPolicyV1,
    build_gauge_propagation_readiness,
)
from prob4d.point_uncertainty_v2 import (
    PointUncertaintyCalibrationPolicyV2,
    PointUncertaintyCalibrationV2,
    fit_point_uncertainty_calibration_v2,
    local_point_basis,
)
from prob4d.source_covariance_localization import (
    SourceCovarianceLocalizationGroupV1,
    SourceCovarianceLocalizationPolicyV1,
    SourceCovarianceLocalizationV1,
)


def _localization(*, conditional_energy: float) -> SourceCovarianceLocalizationV1:
    policy = SourceCovarianceLocalizationPolicyV1(
        minimum_group_count=4,
        normalized_nees_lower=0.5,
        normalized_nees_upper=1.5,
        minimum_joint_pass_fraction=0.75,
        shared_energy_lower=0.5,
        shared_energy_upper=1.5,
        minimum_shared_pass_fraction=0.75,
        conditional_energy_lower=0.8,
        conditional_energy_upper=1.2,
        minimum_conditional_pass_fraction=0.75,
        require_shared_subspace=True,
    )
    groups = tuple(
        SourceCovarianceLocalizationGroupV1(
            group_id=f"group-{index}",
            normalized_nees=1.0 if conditional_energy == 1.0 else 2.0,
            shared_subspace_normalized_energy=1.0,
            conditional_subspace_normalized_energy=conditional_energy,
            joint_in_band=conditional_energy == 1.0,
            shared_in_band=True,
            conditional_in_band=conditional_energy == 1.0,
        )
        for index in range(4)
    )
    return SourceCovarianceLocalizationV1(
        provider_manifest_id="1" * 64,
        cohort_binding_id="2" * 64,
        source_provider_competence_id="3" * 64,
        joint_diagnostic_sha256="4" * 64,
        joint_residual_source_sha256="5" * 64,
        policy=policy,
        groups=groups,
        source_mean_status="pass",
        identity_reliability_status="pass",
    )


def _propagation(localization: SourceCovarianceLocalizationV1):
    return build_gauge_propagation_readiness(
        localization,
        GaugePropagationReadinessPolicyV1.explicit_latent(),
        query_definition_id="7" * 64,
    )


def _synthetic_training(
    localization: SourceCovarianceLocalizationV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(19)
    rows_per_group = 96
    group_ids = [
        group.group_id
        for group in localization.groups
        for _ in range(rows_per_group)
    ]
    row_count = len(group_ids)
    features = rng.normal(size=(row_count, 3))
    rays = rng.normal(size=(row_count, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    tangent_reference = rng.normal(size=(row_count, 3))
    rays, tangent_one, tangent_two = local_point_basis(
        rays,
        tangent_reference,
    )

    design = np.column_stack((np.ones(row_count), features))
    coefficients = np.asarray(
        [
            [-5.0, 0.25, -0.10, 0.05],
            [-5.4, 0.10, 0.20, -0.08],
            [-6.2, -0.05, 0.12, 0.10],
        ],
        dtype=np.float64,
    )
    variances = np.exp(design @ coefficients.T)
    local_residuals = rng.normal(size=(row_count, 3)) * np.sqrt(variances)
    residual_xyz = (
        local_residuals[:, 0, None] * rays
        + local_residuals[:, 1, None] * tangent_one
        + local_residuals[:, 2, None] * tangent_two
    )
    return residual_xyz, rays, tangent_reference, features, group_ids


def test_local_point_basis_is_orthonormal_with_zero_reference() -> None:
    rays = np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    references = np.zeros_like(rays)
    ray, tangent_one, tangent_two = local_point_basis(rays, references)

    for basis in (ray, tangent_one, tangent_two):
        np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0)
    np.testing.assert_allclose(np.sum(ray * tangent_one, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(ray * tangent_two, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(tangent_one * tangent_two, axis=1),
        0.0,
        atol=1e-12,
    )


def test_fit_requires_explicit_point_covariance_localization() -> None:
    localization = _localization(conditional_energy=1.0)
    residuals, rays, references, features, group_ids = _synthetic_training(
        localization
    )
    assert localization.classification == "covariance-adequate"
    assert not localization.authorize_point_uncertainty_development

    with pytest.raises(ValueError, match="point-covariance-localized"):
        fit_point_uncertainty_calibration_v2(
            localization,
            _propagation(localization),
            residual_xyz=residuals,
            ray_directions=rays,
            tangent_reference=references,
            features=features,
            feature_names=("depth", "flow", "disagreement"),
            group_ids=group_ids,
            source_training_sha256="6" * 64,
            policy=PointUncertaintyCalibrationPolicyV2(
                minimum_group_count=4,
                minimum_rows_per_group=64,
            ),
        )


def test_fit_is_group_balanced_anisotropic_and_roundtrips() -> None:
    localization = _localization(conditional_energy=2.0)
    propagation = _propagation(localization)
    residuals, rays, references, features, group_ids = _synthetic_training(
        localization
    )
    calibration = fit_point_uncertainty_calibration_v2(
        localization,
        propagation,
        residual_xyz=residuals,
        ray_directions=rays,
        tangent_reference=references,
        features=features,
        feature_names=("depth", "flow", "disagreement"),
        group_ids=group_ids,
        source_training_sha256="6" * 64,
        policy=PointUncertaintyCalibrationPolicyV2(
            minimum_group_count=4,
            minimum_rows_per_group=64,
        ),
    )

    assert calibration.fit_converged
    assert calibration.gauge_propagation_readiness_id == (
        propagation.gauge_propagation_readiness_id
    )
    assert calibration.group_counts == (96, 96, 96, 96)
    np.testing.assert_allclose(
        calibration.training_normalized_energy,
        np.ones(3),
        atol=0.04,
    )
    predicted = calibration.predict_variances(features)
    assert predicted.shape == (features.shape[0], 3)
    assert np.all(predicted > 0.0)
    assert np.mean(predicted[:, 1]) > 1.4 * np.mean(predicted[:, 2])

    covariance = calibration.covariance_matrices(
        rays[:32],
        references[:32],
        features[:32],
    )
    np.testing.assert_allclose(
        covariance,
        np.transpose(covariance, (0, 2, 1)),
        atol=1e-12,
    )
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)

    restored = PointUncertaintyCalibrationV2.from_dict(calibration.to_dict())
    assert (
        restored.point_uncertainty_calibration_id
        == calibration.point_uncertainty_calibration_id
    )
    assert restored.gauge_propagation_readiness_id == (
        propagation.gauge_propagation_readiness_id
    )


def test_fit_rejects_group_roster_mismatch() -> None:
    localization = _localization(conditional_energy=2.0)
    residuals, rays, references, features, group_ids = _synthetic_training(
        localization
    )
    group_ids[-1] = "unexpected-group"

    with pytest.raises(ValueError, match="exactly match"):
        fit_point_uncertainty_calibration_v2(
            localization,
            _propagation(localization),
            residual_xyz=residuals,
            ray_directions=rays,
            tangent_reference=references,
            features=features,
            feature_names=("depth", "flow", "disagreement"),
            group_ids=group_ids,
            source_training_sha256="6" * 64,
            policy=PointUncertaintyCalibrationPolicyV2(
                minimum_group_count=4,
                minimum_rows_per_group=64,
            ),
        )


def test_calibration_content_address_detects_tampering() -> None:
    localization = _localization(conditional_energy=2.0)
    residuals, rays, references, features, group_ids = _synthetic_training(
        localization
    )
    calibration = fit_point_uncertainty_calibration_v2(
        localization,
        _propagation(localization),
        residual_xyz=residuals,
        ray_directions=rays,
        tangent_reference=references,
        features=features,
        feature_names=("depth", "flow", "disagreement"),
        group_ids=group_ids,
        source_training_sha256="6" * 64,
        policy=PointUncertaintyCalibrationPolicyV2(
            minimum_group_count=4,
            minimum_rows_per_group=64,
        ),
    )
    tampered = copy.deepcopy(calibration.to_dict())
    tampered["parallel_coefficients"][0] += 0.1

    with pytest.raises(ValueError, match="identity mismatch"):
        PointUncertaintyCalibrationV2.from_dict(tampered)


def test_fit_rejects_mismatched_propagation_before_training_access() -> None:
    localization = _localization(conditional_energy=2.0)
    propagation = _propagation(_localization(conditional_energy=1.0))

    with pytest.raises(ValueError, match="different covariance localization"):
        fit_point_uncertainty_calibration_v2(
            localization,
            propagation,
            residual_xyz=object(),
            ray_directions=object(),
            tangent_reference=object(),
            features=object(),
            feature_names=(),
            group_ids=(),
            source_training_sha256="6" * 64,
        )
