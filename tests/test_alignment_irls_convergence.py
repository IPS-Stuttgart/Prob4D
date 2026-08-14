from __future__ import annotations

import numpy as np
import pytest

import prob4d.alignment as alignment
from prob4d.sim3 import Sim3


def _outlier_problem(seed: int = 91) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    source = generator.normal(size=(600, 3))
    truth = Sim3.from_vector(
        np.array([0.11, 0.08, -0.04, 0.06, 0.7, -0.35, 0.18])
    )
    target = truth.transform_points(source)
    target += generator.normal(scale=0.004, size=source.shape)
    target[:75] += generator.normal(scale=1.5, size=(75, 3))
    weights = generator.uniform(0.5, 1.5, size=source.shape[0])
    return source, target, weights


def test_covariance_uses_weights_that_fit_returned_transform(monkeypatch) -> None:
    source, target, weights = _outlier_problem()
    fit_weights: list[np.ndarray] = []
    covariance_weights: list[np.ndarray] = []
    original_fit = alignment._weighted_umeyama
    original_covariance = alignment._alignment_covariance_estimate

    def recorded_fit(
        source_array: np.ndarray,
        target_array: np.ndarray,
        weight_array: np.ndarray,
    ) -> Sim3:
        fit_weights.append(np.asarray(weight_array).copy())
        return original_fit(source_array, target_array, weight_array)

    def recorded_covariance(
        source_array: np.ndarray,
        target_array: np.ndarray,
        weight_array: np.ndarray,
        transform: Sim3,
        *,
        cluster_ids: np.ndarray | None = None,
    ) -> alignment._CovarianceEstimate:
        covariance_weights.append(np.asarray(weight_array).copy())
        return original_covariance(
            source_array,
            target_array,
            weight_array,
            transform,
            cluster_ids=cluster_ids,
        )

    monkeypatch.setattr(alignment, "_weighted_umeyama", recorded_fit)
    monkeypatch.setattr(
        alignment,
        "_alignment_covariance_estimate",
        recorded_covariance,
    )

    result = alignment.estimate_sim3_robust(source, target, weights=weights)

    assert result.inlier_fraction < 1.0
    assert len(fit_weights) >= 2
    assert len(covariance_weights) == 1
    np.testing.assert_array_equal(covariance_weights[0], fit_weights[-1])


def test_nonconverged_irls_fails_closed() -> None:
    source, target, weights = _outlier_problem()

    with pytest.raises(ValueError, match=r"did not converge.*max_iterations=1"):
        alignment.estimate_sim3_robust(
            source,
            target,
            weights=weights,
            max_iterations=1,
        )


def test_uniform_weight_scaling_is_metamorphically_invariant() -> None:
    source, target, weights = _outlier_problem()

    baseline = alignment.estimate_sim3_robust(source, target, weights=weights)
    scaled = alignment.estimate_sim3_robust(source, target, weights=7.25 * weights)

    np.testing.assert_allclose(
        scaled.transform.as_vector(),
        baseline.transform.as_vector(),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        scaled.covariance,
        baseline.covariance,
        rtol=1e-9,
        atol=1e-11,
    )
    assert scaled.inlier_fraction == baseline.inlier_fraction
    assert scaled.residual_rms == pytest.approx(
        baseline.residual_rms,
        rel=1e-12,
        abs=1e-12,
    )
