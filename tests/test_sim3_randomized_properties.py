from __future__ import annotations

import numpy as np

from prob4d.sim3 import Sim3


def _random_sim3(rng: np.random.Generator) -> Sim3:
    return Sim3.from_vector(
        np.concatenate(
            (
                rng.normal(scale=0.25, size=1),
                rng.normal(scale=0.35, size=3),
                rng.normal(scale=2.0, size=3),
            )
        )
    )


def test_randomized_sim3_group_properties() -> None:
    rng = np.random.default_rng(20260814)
    for _ in range(64):
        first = _random_sim3(rng)
        second = _random_sim3(rng)
        third = _random_sim3(rng)
        points = rng.normal(size=(17, 3))

        np.testing.assert_allclose(
            first.compose(second).transform_points(points),
            first.transform_points(second.transform_points(points)),
            atol=2e-12,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            first.compose(second).compose(third).transform_points(points),
            first.compose(second.compose(third)).transform_points(points),
            atol=4e-12,
            rtol=4e-12,
        )
        np.testing.assert_allclose(
            first.inverse().transform_points(first.transform_points(points)),
            points,
            atol=4e-12,
            rtol=4e-12,
        )
        np.testing.assert_allclose(
            first.compose(first.inverse()).transform_points(points),
            points,
            atol=4e-12,
            rtol=4e-12,
        )
