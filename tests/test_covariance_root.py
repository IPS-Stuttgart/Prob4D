from __future__ import annotations

import numpy as np
import pytest

import prob4d.covariance_root as roots
from prob4d import observation_export


def test_canonical_root_reconstructs_covariance_and_reports_trace() -> None:
    covariance = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    root, retained = roots.canonical_covariance_root(covariance)

    np.testing.assert_allclose(root @ root.T, covariance, atol=1e-12)
    assert retained == pytest.approx(1.0)


def test_canonical_root_is_invariant_to_rotated_repeated_eigenvectors(
    monkeypatch,
) -> None:
    covariance = np.diag([4.0, 4.0, 1.0])

    def eigendecomposition(angle: float):
        cosine = np.cos(angle)
        sine = np.sin(angle)

        def fake_eigh(matrix: np.ndarray):
            np.testing.assert_allclose(matrix, covariance)
            return (
                np.asarray([1.0, 4.0, 4.0]),
                np.asarray(
                    [
                        [0.0, cosine, -sine],
                        [0.0, sine, cosine],
                        [1.0, 0.0, 0.0],
                    ]
                ),
            )

        return fake_eigh

    monkeypatch.setattr(roots.np.linalg, "eigh", eigendecomposition(0.0))
    first, _ = roots.canonical_covariance_root(covariance)
    monkeypatch.setattr(roots.np.linalg, "eigh", eigendecomposition(np.pi / 5.0))
    second, _ = roots.canonical_covariance_root(covariance)

    np.testing.assert_allclose(first, second, atol=1e-12)


def test_canonical_root_rejects_rank_cut_through_repeated_eigenspace() -> None:
    with pytest.raises(ValueError, match="max_rank cuts through"):
        roots.canonical_covariance_root(
            np.diag([4.0, 4.0, 1.0]),
            max_rank=1,
        )


def test_context_dispatch_preserves_legacy_default_and_resets() -> None:
    roots.install_covariance_root_dispatch()
    covariance = np.diag([4.0, 4.0, 1.0])

    legacy_root, _ = observation_export.deterministic_covariance_root(
        covariance,
        max_rank=1,
    )
    assert legacy_root.shape == (3, 1)
    assert roots.current_covariance_root_mode() == "legacy_eigenvectors"

    with roots.covariance_root_mode("canonical_eigenspaces"):
        assert roots.current_covariance_root_mode() == "canonical_eigenspaces"
        with pytest.raises(ValueError, match="max_rank cuts through"):
            observation_export.deterministic_covariance_root(
                covariance,
                max_rank=1,
            )

    assert roots.current_covariance_root_mode() == "legacy_eigenvectors"
