import numpy as np

from prob4d.fusion import FusedSequence
from prob4d.metrics import TruthSequence, evaluate_sequence, uncertainty_diagnostics


def test_metrics_detect_boundary_jump_and_calibrated_coverage() -> None:
    truth_points = np.zeros((4, 1, 2, 3))
    truth_points[..., 0] = np.arange(4)[:, None, None]
    prediction_points = truth_points.copy()
    prediction_points[2:, ..., 0] += 0.2
    covariance = np.broadcast_to(np.eye(3) * 0.04, truth_points.shape + (3,)).copy()
    mask = np.ones((4, 1, 2), dtype=bool)
    prediction = FusedSequence(
        np.arange(4),
        prediction_points,
        mask,
        covariance,
        np.ones_like(mask, dtype=np.uint16),
    )
    truth = TruthSequence(np.arange(4), truth_points, mask)

    metrics = evaluate_sequence(
        prediction,
        truth,
        boundary_frames=[2],
        align_scale_translation=False,
    )

    assert metrics.point_rmse > 0
    np.testing.assert_allclose(metrics.seam_rmse, 0.2)
    assert metrics.coverage_95 == 1.0


def test_uncertainty_diagnostics_detect_calibration_and_rank_failures() -> None:
    errors = np.zeros((100, 3))
    errors[:, 0] = np.linspace(0.01, 1.0, 100)
    variances = np.linspace(0.01, 1.0, 100) ** 2
    covariances = variances[:, None, None] * np.eye(3)

    diagnostics = uncertainty_diagnostics(errors, covariances, np.ones(100))

    assert diagnostics.uncertainty_error_spearman > 0.99
    assert diagnostics.relative_error_retained_80 < diagnostics.mean_relative_error
    assert diagnostics.selective_gain_80 > 0.0
    assert diagnostics.count == 100


def test_uncertainty_diagnostics_ranks_relative_predictive_variance() -> None:
    errors = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    covariances = np.stack([np.eye(3), 0.2 * np.eye(3)])
    target_norms = np.array([10.0, 1.0])

    diagnostics = uncertainty_diagnostics(
        errors,
        covariances,
        target_norms,
        uncertainty_normalizers=target_norms,
    )

    np.testing.assert_allclose(diagnostics.uncertainty_error_spearman, 1.0)
    assert diagnostics.selective_gain_80 > 0.0
