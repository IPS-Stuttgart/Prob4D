import numpy as np

from prob4d.observation_export import gauge_covariance_factor
from prob4d.sim3 import Sim3, skew


def test_gauge_factor_recovers_identity_linearization() -> None:
    covariance = np.diag(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    )
    points = np.asarray([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    factor = gauge_covariance_factor(
        points,
        Sim3(),
        covariance,
        include_translation=True,
    )
    actual = np.einsum("nir,njr->nij", factor, factor)

    expected = []
    for point in points:
        jacobian = np.zeros((3, 7))
        jacobian[:, 0] = point
        jacobian[:, 1:4] = -skew(point)
        jacobian[:, 4:7] = np.eye(3)
        expected.append(jacobian @ covariance @ jacobian.T)
    np.testing.assert_allclose(actual, np.asarray(expected), atol=1e-12)


def test_vector_factor_excludes_translation_covariance() -> None:
    covariance = np.diag(
        [0.01, 0.02, 0.03, 0.04, 5.0, 6.0, 7.0]
    )
    without_translation = covariance.copy()
    without_translation[4:7, 4:7] = 0.0
    vector = np.asarray([[1.0, 0.0, 0.0]])
    first = gauge_covariance_factor(
        vector,
        Sim3(),
        covariance,
        include_translation=False,
    )
    second = gauge_covariance_factor(
        vector,
        Sim3(),
        without_translation,
        include_translation=False,
    )
    np.testing.assert_allclose(
        np.einsum("nir,njr->nij", first, first),
        np.einsum("nir,njr->nij", second, second),
        atol=1e-12,
    )
