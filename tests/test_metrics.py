import numpy as np
import pytest

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


def _assert_metric_dicts_close(
    first: dict[str, float | int | None],
    second: dict[str, float | int | None],
) -> None:
    assert first.keys() == second.keys()
    for key in first:
        left = first[key]
        right = second[key]
        if left is None or right is None or isinstance(left, int):
            assert left == right
        else:
            np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)


def test_chunked_evaluation_matches_large_chunk_with_flow_and_support() -> None:
    generator = np.random.default_rng(17)
    shape = (4, 3, 5)
    truth_points = generator.normal(size=shape + (3,))
    truth_points[..., 2] += 3.0
    prediction_points = (truth_points - np.asarray([0.2, -0.1, 0.05])) / 1.03
    prediction_points += generator.normal(scale=0.01, size=truth_points.shape)
    valid = generator.random(shape) > 0.15
    support = generator.random(shape) > 0.2
    flow_support = generator.random(shape) > 0.25
    covariance = np.zeros(shape + (3, 3))
    diagonal = np.arange(3)
    covariance[..., diagonal, diagonal] = (
        0.02 + 0.01 * generator.random(shape + (1,))
    )
    truth_flow = generator.normal(scale=0.05, size=truth_points.shape)
    prediction_flow = truth_flow / 1.03
    deform = valid & (generator.random(shape) > 0.2)
    flow_covariance = np.zeros_like(covariance)
    flow_covariance[..., diagonal, diagonal] = 0.005

    prediction = FusedSequence(
        np.arange(shape[0]),
        prediction_points,
        valid,
        covariance,
        np.ones(shape, dtype=np.uint16),
        prediction_flow,
        deform,
        flow_covariance,
    )
    truth = TruthSequence(
        np.arange(shape[0]),
        truth_points,
        valid,
        truth_flow,
        deform,
    )

    small = evaluate_sequence(
        prediction,
        truth,
        boundary_frames=[2],
        truth_support_mask=support,
        truth_flow_support_mask=flow_support,
        evaluation_chunk_size=3,
    )
    large = evaluate_sequence(
        prediction,
        truth,
        boundary_frames=[2],
        truth_support_mask=support,
        truth_flow_support_mask=flow_support,
        evaluation_chunk_size=10_000,
    )

    _assert_metric_dicts_close(small.to_dict(), large.to_dict())


def test_pretransformed_evaluation_matches_materialized_sequence() -> None:
    truth_points = np.asarray(
        [
            [[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]],
            [[[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]],
        ]
    )
    prediction_points = truth_points + np.asarray([0.1, -0.2, 0.05])
    valid = np.ones((2, 1, 2), dtype=bool)
    covariance = np.broadcast_to(
        np.diag([0.02, 0.03, 0.04]),
        prediction_points.shape + (3,),
    ).copy()
    prediction = FusedSequence(
        np.arange(2),
        prediction_points,
        valid,
        covariance,
        np.ones_like(valid, dtype=np.uint16),
    )
    truth = TruthSequence(np.arange(2), truth_points, valid)
    scale = 1.07
    translation = np.asarray([0.3, -0.1, 0.2])
    materialized = FusedSequence(
        prediction.frame_indices,
        scale * prediction.point_map + translation,
        prediction.valid_mask,
        scale**2 * prediction.point_covariance,
        prediction.contributors,
    )

    expected = evaluate_sequence(
        materialized,
        truth,
        align_scale_translation=False,
        evaluation_chunk_size=2,
    )
    actual = evaluate_sequence(
        prediction,
        truth,
        align_scale_translation=False,
        prediction_scale=scale,
        prediction_translation=translation,
        evaluation_chunk_size=2,
    )

    _assert_metric_dicts_close(actual.to_dict(), expected.to_dict())


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("evaluation_chunk_size", 0, "evaluation_chunk_size must be positive"),
        ("prediction_scale", 0.0, "prediction_scale must be finite and positive"),
        ("prediction_translation", [0.0, 0.0], "finite three-vector"),
    ],
)
def test_evaluate_sequence_rejects_invalid_execution_transform(
    keyword: str,
    value: object,
    message: str,
) -> None:
    points = np.zeros((1, 1, 1, 3))
    mask = np.ones((1, 1, 1), dtype=bool)
    covariance = np.broadcast_to(np.eye(3), points.shape + (3,)).copy()
    prediction = FusedSequence(
        np.array([0]),
        points,
        mask,
        covariance,
        np.ones_like(mask, dtype=np.uint16),
    )
    truth = TruthSequence(np.array([0]), points, mask)
    with pytest.raises(ValueError, match=message):
        evaluate_sequence(prediction, truth, **{keyword: value})
