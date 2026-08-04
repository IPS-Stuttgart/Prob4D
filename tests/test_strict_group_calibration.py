from __future__ import annotations

import numpy as np
import pytest

from prob4d.calibration import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    fit_group_balanced_point_uncertainty_calibration,
)
from prob4d.uncertainty import DepthDisagreementModel, StructuredCovariance

_PROVENANCE = {
    "calibration_case_ids": ("case-a", "case-b"),
    "source_repository": "FlorianPfaff/Prob4D",
    "source_revision": "a" * 40,
    "motioncrafter_revision": "b" * 40,
    "model_identifier": "motioncrafter@sha256:cafebabe",
    "covariance_method": "depth_disagreement_anisotropic_v1",
    "image_resolution": (2, 2),
    "window_size": 3,
    "window_overlap": 1,
    "covariance_cluster_size": 2,
    "input_artifact_sha256": ("c" * 64,),
}


def _group_inputs() -> tuple[np.ndarray, StructuredCovariance, np.ndarray]:
    rays = np.zeros((6, 3), dtype=np.float64)
    rays[:, 2] = 1.0
    covariance = StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.ones(6),
        lateral_variance=np.ones(6),
    )
    errors = np.zeros((6, 3), dtype=np.float64)
    errors[:, 2] = np.asarray([1.0, 2.0, 10.0, 1.5, 2.5, 20.0])
    groups = np.asarray(["case-a"] * 3 + ["case-b"] * 3)
    return errors, covariance, groups


def test_group_balanced_fit_returns_strict_public_artifact() -> None:
    errors, covariance, groups = _group_inputs()

    artifact, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition=" object ",
        **_PROVENANCE,
    )

    assert isinstance(artifact, PointUncertaintyCalibrationV1)
    assert artifact.metadata[
        "group_balanced_point_uncertainty_calibration"
    ]["group_definition"] == "object"


def test_group_balanced_fit_rejects_provenance_coercion() -> None:
    errors, covariance, groups = _group_inputs()

    with pytest.raises(ValueError, match="group_definition must be a non-empty string"):
        fit_group_balanced_point_uncertainty_calibration(
            DepthDisagreementModel(),
            errors,
            covariance,
            groups,
            group_definition=7,
            **_PROVENANCE,
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        fit_group_balanced_point_uncertainty_calibration(
            DepthDisagreementModel(),
            errors,
            covariance,
            groups,
            group_definition="object",
            metadata=0,
            **_PROVENANCE,
        )


def test_oversized_calibration_number_fails_as_value_error() -> None:
    with pytest.raises(ValueError, match="scale must be finite"):
        GaugeCovarianceCalibrationV1(
            scale=10**10000,
            rotation=1.0,
            translation=1.0,
            count=2,
            trim_quantile=0.99,
            **_PROVENANCE,
        )
