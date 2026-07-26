import numpy as np

from prob4d.observation_export import gauge_covariance_factor


class _Transform:
    scale = 1.0
    rotation = np.eye(3)


def test_gauge_factor_recovers_linearized_marginal(monkeypatch) -> None:
    monkeypatch.setattr(
        "prob4d.observation_export.so3_log", lambda rotation: np.zeros(3)
    )
    monkeypatch.setattr(
        "prob4d.observation_export.so3_right_jacobian",
        lambda vector: np.eye(3),
    )
    covariance = np.diag([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    points = np.asarray([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    factor = gauge_covariance_factor(
        points,
        _Transform(),
        covariance,
        include_translation=True,
    )
    marginal = np.einsum("nir,njr->nij", factor, factor)

    expected = []
    for point in points:
        skew = np.asarray(
            [
                [0.0, -point[2], point[1]],
                [point[2], 0.0, -point[0]],
                [-point[1], point[0], 0.0],
            ]
        )
        jacobian = np.zeros((3, 7))
        jacobian[:, 0] = point
        jacobian[:, 1:4] = -skew
        jacobian[:, 4:7] = np.eye(3)
        expected.append(jacobian @ covariance @ jacobian.T)
    np.testing.assert_allclose(marginal, np.asarray(expected), atol=1e-12)


def test_vector_factor_excludes_translation(monkeypatch) -> None:
    monkeypatch.setattr(
        "prob4d.observation_export.so3_log", lambda rotation: np.zeros(3)
    )
    monkeypatch.setattr(
        "prob4d.observation_export.so3_right_jacobian",
        lambda vector: np.eye(3),
    )
    covariance = np.diag([0.01, 0.02, 0.03, 0.04, 5.0, 6.0, 7.0])
    vector = np.asarray([[1.0, 0.0, 0.0]])
    factor = gauge_covariance_factor(
        vector,
        _Transform(),
        covariance,
        include_translation=False,
    )
    assert np.allclose(factor[..., 4:7], 0.0)
