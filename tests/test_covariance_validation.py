from __future__ import annotations

import numpy as np
import pytest

from prob4d.covariance import (
    covariance_statistics,
    regularized_inverse_psd,
)
from prob4d.fusion import (
    FusedSequence,
    fuse_gaussians_covariance_intersection,
    fuse_gaussians_independent,
)
from prob4d.metrics import TruthSequence, evaluate_sequence, uncertainty_diagnostics


def test_tiny_negative_eigenvalue_is_tolerated_and_regularized() -> None:
    covariance = np.diag([1.0, 0.5, -1e-15])

    inverse = regularized_inverse_psd(covariance)
    _, _, log_determinant = covariance_statistics(covariance)

    assert np.all(np.isfinite(inverse))
    assert np.isfinite(log_determinant)
    np.testing.assert_allclose(inverse, inverse.T)


def test_materially_indefinite_covariance_is_rejected() -> None:
    covariance = np.diag([1.0, 0.5, -1e-3])

    with pytest.raises(ValueError, match="positive semidefinite"):
        regularized_inverse_psd(covariance)


def test_non_symmetric_covariance_is_rejected() -> None:
    covariance = np.eye(3)
    covariance[0, 1] = 0.1

    with pytest.raises(ValueError, match="symmetric"):
        regularized_inverse_psd(covariance)


def test_independent_fusion_rejects_indefinite_covariance() -> None:
    mean = np.zeros((1, 3))
    valid = np.eye(3)[None]
    invalid = np.diag([1.0, 1.0, -0.1])[None]

    with pytest.raises(ValueError, match="positive semidefinite"):
        fuse_gaussians_independent(mean, valid, mean, invalid)


def test_covariance_intersection_rejects_indefinite_covariance() -> None:
    mean = np.zeros((1, 3))
    valid = np.eye(3)[None]
    invalid = np.diag([1.0, 1.0, -0.1])[None]

    with pytest.raises(ValueError, match="positive semidefinite"):
        fuse_gaussians_covariance_intersection(mean, valid, mean, invalid)


def _sequence(covariance: np.ndarray) -> tuple[FusedSequence, TruthSequence]:
    points = np.zeros((1, 1, 1, 3))
    mask = np.ones((1, 1, 1), dtype=bool)
    prediction = FusedSequence(
        frame_indices=np.asarray([0]),
        point_map=points,
        valid_mask=mask,
        point_covariance=covariance.reshape(1, 1, 1, 3, 3),
        contributors=np.ones_like(mask, dtype=np.uint16),
    )
    truth = TruthSequence(
        frame_indices=np.asarray([0]),
        point_map=points,
        valid_mask=mask,
    )
    return prediction, truth


def test_fused_sequence_rejects_indefinite_active_covariance() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        _sequence(np.diag([1.0, 1.0, -0.1]))


def test_fused_sequence_projects_tolerated_negative_roundoff() -> None:
    prediction, truth = _sequence(np.diag([1.0, 1.0, -1e-15]))

    assert np.min(np.linalg.eigvalsh(prediction.point_covariance[0, 0, 0])) >= 0.0
    result = evaluate_sequence(prediction, truth, align_scale_translation=False)
    assert result.point_rmse == 0.0


def test_fused_sequence_rejects_asymmetric_active_covariance() -> None:
    covariance = np.eye(3)
    covariance[0, 1] = 0.1

    with pytest.raises(ValueError, match="symmetric"):
        _sequence(covariance)


def test_uncertainty_diagnostics_reject_indefinite_covariance() -> None:
    errors = np.asarray([[0.1, 0.0, 0.0]])
    covariance = np.diag([1.0, 1.0, -0.1])[None]

    with pytest.raises(ValueError, match="positive semidefinite"):
        uncertainty_diagnostics(errors, covariance, np.ones(1))


def test_legacy_flow_metric_excludes_invalid_geometry() -> None:
    points = np.asarray([[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]])
    valid = np.asarray([[[True, False]]])
    covariance = np.broadcast_to(
        np.eye(3),
        points.shape + (3,),
    ).copy()
    prediction_flow = np.asarray(
        [[[[1.0, 0.0, 0.0], [999.0, 0.0, 0.0]]]]
    )
    truth_flow = np.asarray(
        [[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    )
    deform = np.ones(valid.shape, dtype=bool)
    prediction = FusedSequence(
        frame_indices=np.asarray([0]),
        point_map=points,
        valid_mask=valid,
        point_covariance=covariance,
        contributors=np.ones_like(valid, dtype=np.uint16),
        scene_flow=prediction_flow,
        deform_mask=deform,
        flow_covariance=covariance,
    )
    truth = TruthSequence(
        frame_indices=np.asarray([0]),
        point_map=points,
        valid_mask=valid,
        scene_flow=truth_flow,
        deform_mask=deform,
    )

    result = evaluate_sequence(
        prediction,
        truth,
        align_scale_translation=False,
    )

    assert result.flow_epe == 0.0
