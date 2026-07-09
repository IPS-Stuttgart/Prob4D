import numpy as np

from prob4d.fusion import FusedSequence
from prob4d.metrics import TruthSequence, evaluate_sequence


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

