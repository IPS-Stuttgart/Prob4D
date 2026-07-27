import itertools

import numpy as np
import pytest

from prob4d.gauge import (
    RelativeGaugeConstraint,
    SequentialGaugeEstimator,
    fuse_sim3_covariance_intersection,
)
from prob4d.sim3 import Sim3, so3_log


def _translation_candidate(
    translation_x: float,
    covariance_scale: float,
) -> tuple[Sim3, np.ndarray]:
    transform = Sim3.from_vector(
        np.array([0.0, 0.0, 0.0, 0.0, translation_x, 0.0, 0.0])
    )
    return transform, np.eye(7) * covariance_scale


def _constraint(
    reference_id: str,
    moving_id: str,
    reference: Sim3,
    moving: Sim3,
    *,
    translation_noise: float = 0.0,
    covariance_scale: float = 1e-3,
) -> RelativeGaugeConstraint:
    noise = Sim3.from_vector(
        np.array([0.0, 0.0, 0.0, 0.0, translation_noise, 0.0, 0.0])
    )
    relative = reference.inverse().compose(moving).compose(noise)
    return RelativeGaugeConstraint(
        reference_id=reference_id,
        moving_id=moving_id,
        reference_from_moving=relative,
        covariance=np.eye(7) * covariance_scale,
    )


def test_multi_estimate_ci_is_invariant_to_candidate_order() -> None:
    candidates = [
        _translation_candidate(0.00, 1e-3),
        _translation_candidate(0.05, 2e-3),
        _translation_candidate(-0.02, 5e-4),
    ]
    baseline_transform, baseline_covariance, _ = (
        fuse_sim3_covariance_intersection(
            candidates,
            minimum_weight=0.05,
        )
    )

    for permutation in itertools.permutations(candidates):
        transform, covariance, weights = fuse_sim3_covariance_intersection(
            permutation,
            minimum_weight=0.05,
        )
        np.testing.assert_allclose(
            transform.as_vector(),
            baseline_transform.as_vector(),
            atol=1e-12,
        )
        np.testing.assert_allclose(covariance, baseline_covariance, atol=1e-15)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-15)
        assert np.all(weights >= 0.05 - 1e-15)


def test_sequential_estimator_is_invariant_to_constraint_order() -> None:
    truth = {
        "w0": Sim3.identity(),
        "w1": _translation_candidate(1.0, 1e-6)[0],
        "w2": _translation_candidate(2.0, 1e-6)[0],
        "w3": _translation_candidate(3.0, 1e-6)[0],
    }
    fixed = [
        _constraint("w0", "w1", truth["w0"], truth["w1"], covariance_scale=1e-6),
        _constraint("w1", "w2", truth["w1"], truth["w2"], covariance_scale=1e-6),
    ]
    alternatives = [
        _constraint("w0", "w3", truth["w0"], truth["w3"], covariance_scale=1e-3),
        _constraint(
            "w1",
            "w3",
            truth["w1"],
            truth["w3"],
            translation_noise=0.05,
            covariance_scale=2e-3,
        ),
        _constraint(
            "w2",
            "w3",
            truth["w2"],
            truth["w3"],
            translation_noise=-0.02,
            covariance_scale=5e-4,
        ),
    ]

    baseline = None
    for permutation in itertools.permutations(alternatives):
        result = SequentialGaugeEstimator().estimate(
            list(truth),
            fixed + list(permutation),
        )["w3"]
        if baseline is None:
            baseline = result
            continue
        np.testing.assert_allclose(
            result.global_from_local.as_vector(),
            baseline.global_from_local.as_vector(),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            result.covariance,
            baseline.covariance,
            atol=1e-14,
        )


def test_ci_uses_a_local_chart_across_the_rotation_log_branch() -> None:
    angle = np.deg2rad(179.0)
    first = Sim3.from_vector(np.array([0.0, 0.0, 0.0, angle, 0.0, 0.0, 0.0]))
    second = Sim3.from_vector(
        np.array([0.0, 0.0, 0.0, -angle, 0.0, 0.0, 0.0])
    )

    fused, covariance, _ = fuse_sim3_covariance_intersection(
        [
            (first, np.eye(7) * 1e-4),
            (second, np.eye(7) * 1e-4),
        ],
        minimum_weight=0.05,
    )

    assert np.linalg.norm(so3_log(fused.rotation)) > np.deg2rad(170.0)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0


def test_ci_rejects_materially_indefinite_covariance() -> None:
    covariance = np.eye(7)
    covariance[0, 0] = -1.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        fuse_sim3_covariance_intersection([(Sim3.identity(), covariance)])
