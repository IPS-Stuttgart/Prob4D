from __future__ import annotations

import numpy as np
import pytest

from prob4d.sim3 import Sim3
from prob4d.uncertainty import DepthDisagreementModel, StructuredCovariance


def test_sim3_defensively_copies_and_freezes_arrays() -> None:
    rotation = np.eye(3)
    translation = np.array([1.0, 2.0, 3.0])
    transform = Sim3(rotation=rotation, translation=translation)
    rotation[0, 0] = 7.0
    translation[0] = 9.0

    np.testing.assert_array_equal(transform.rotation, np.eye(3))
    np.testing.assert_array_equal(transform.translation, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        transform.rotation[0, 0] = 2.0
    with pytest.raises(ValueError):
        transform.translation[0] = 2.0


def test_sim3_rejects_nonfinite_translation() -> None:
    with pytest.raises(ValueError, match="finite"):
        Sim3(translation=np.array([0.0, np.nan, 0.0]))


def test_structured_covariance_defensively_copies_and_freezes_arrays() -> None:
    rays = np.ones((2, 3))
    parallel = np.ones(2)
    lateral = np.full(2, 0.5)
    covariance = StructuredCovariance(rays, parallel, lateral)
    rays[:] = 0.0
    parallel[:] = 4.0
    lateral[:] = 3.0

    assert np.all(np.linalg.norm(covariance.ray_directions, axis=-1) == pytest.approx(1.0))
    np.testing.assert_array_equal(covariance.parallel_variance, [1.0, 1.0])
    np.testing.assert_array_equal(covariance.lateral_variance, [0.5, 0.5])
    with pytest.raises(ValueError):
        covariance.parallel_variance[0] = 2.0


def test_depth_disagreement_model_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DepthDisagreementModel(disagreement_gain=-1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        DepthDisagreementModel(parallel_scale=0.0)


def test_structured_covariance_rejects_zero_rays() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        StructuredCovariance(np.zeros((1, 3)), np.ones(1), np.ones(1))
