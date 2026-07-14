import numpy as np
import pytest

from prob4d.sim3 import Sim3
from scripts.export_vggt_prob4d_blends import (
    alpha_name,
    blend_point_maps,
    parse_alphas,
    sampled_correspondences,
)


def test_parse_alphas_sorts_and_deduplicates() -> None:
    assert parse_alphas("0.75,0.25,0.75") == [0.25, 0.75]
    assert alpha_name(0.25) == "prob4d_0p25"


def test_parse_alphas_rejects_endpoints() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        parse_alphas("0,0.5")


def test_sampled_correspondences_is_deterministic() -> None:
    source = np.arange(90, dtype=float).reshape(2, 3, 5, 3)
    target = source + 1.0
    mask = np.ones((2, 3, 5), dtype=bool)

    first = sampled_correspondences(source, target, mask, maximum=8, seed=4)
    second = sampled_correspondences(source, target, mask, maximum=8, seed=4)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_blend_point_maps_applies_external_transform() -> None:
    prob4d = np.full((1, 1, 1, 3), 4.0)
    external = np.full((1, 1, 1, 3), 1.0)
    transform = Sim3(scale=2.0, rotation=np.eye(3), translation=np.ones(3))

    blended = blend_point_maps(prob4d, external, transform, 0.5)

    np.testing.assert_allclose(blended, 3.5)
