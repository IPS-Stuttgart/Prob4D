from __future__ import annotations

import numpy as np

import prob4d.alignment as alignment
from prob4d.alignment import (
    DENSE_ALIGNMENT_COVARIANCE_METHOD,
    estimate_sim3_robust,
)
from prob4d.sim3 import Sim3


def _problem(seed: int = 31) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    source = generator.normal(size=(480, 3))
    transform = Sim3.from_vector(
        np.array([0.08, 0.12, -0.04, 0.06, 0.5, -0.3, 0.2])
    )
    target = transform.transform_points(source)
    target += generator.normal(scale=0.01, size=target.shape)
    return source, target


def test_alignment_covariance_uses_factorized_solves(monkeypatch) -> None:
    source, target = _problem()

    def forbidden_pseudoinverse(*args, **kwargs):
        raise AssertionError("alignment covariance must not use np.linalg.pinv")

    monkeypatch.setattr(alignment.np.linalg, "pinv", forbidden_pseudoinverse)
    result = estimate_sim3_robust(source, target)

    assert result.covariance_method == "iid_gauss_newton"
    assert np.all(np.isfinite(result.covariance))
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-14)
    assert float(np.min(np.linalg.eigvalsh(result.covariance))) >= -1e-12


def test_cluster_robust_sandwich_uses_factorized_solves(monkeypatch) -> None:
    source, target = _problem(41)
    cluster_ids = np.arange(source.shape[0], dtype=np.int64) // 12

    def forbidden_pseudoinverse(*args, **kwargs):
        raise AssertionError("alignment covariance must not use np.linalg.pinv")

    monkeypatch.setattr(alignment.np.linalg, "pinv", forbidden_pseudoinverse)
    result = estimate_sim3_robust(
        source,
        target,
        covariance_cluster_ids=cluster_ids,
    )

    assert result.covariance_method == DENSE_ALIGNMENT_COVARIANCE_METHOD
    assert result.num_covariance_clusters == 40
    assert result.information_rank == 7
    assert np.all(np.isfinite(result.covariance))
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-14)
    assert float(np.min(np.linalg.eigvalsh(result.covariance))) >= -1e-12


def test_factorized_solver_has_valid_eigendecomposition_fallback(monkeypatch) -> None:
    information = np.diag(np.linspace(1.0, 7.0, 7))
    right_hand_side = np.arange(49, dtype=np.float64).reshape(7, 7)

    def fail_cholesky(*args, **kwargs):
        raise np.linalg.LinAlgError("forced fallback")

    monkeypatch.setattr(alignment.np.linalg, "cholesky", fail_cholesky)
    solve = alignment._factorized_information_solver(information)

    np.testing.assert_allclose(
        solve(right_hand_side),
        np.linalg.solve(information, right_hand_side),
    )
