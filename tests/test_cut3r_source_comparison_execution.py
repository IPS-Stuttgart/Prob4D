from __future__ import annotations

import numpy as np

from prob4d.cut3r_source_comparison_execution import (
    build_native_product,
    build_restarted_comparison_products,
    causal_window_schedule,
    newest_eligible_windows,
)
from prob4d.data import PredictionWindow
from prob4d.sim3 import Sim3


def _window(
    window_id: str,
    frames: list[int],
    *,
    local_from_global: Sim3 | None = None,
    invalid: dict[int, tuple[int, int]] | None = None,
) -> PredictionWindow:
    height, width = 3, 4
    rows, columns = np.indices((height, width), dtype=np.float64)
    points = []
    rays = []
    for frame in frames:
        global_points = np.stack(
            (
                0.02 * columns + 0.001 * frame,
                0.02 * rows,
                np.ones_like(rows) + 0.002 * frame,
            ),
            axis=-1,
        )
        local_points = (
            global_points
            if local_from_global is None
            else local_from_global.transform_points(global_points)
        )
        points.append(local_points)
        rays.append(local_points / np.linalg.norm(local_points, axis=-1, keepdims=True))
    point_map = np.asarray(points, dtype=np.float32)
    valid = np.ones(point_map.shape[:-1], dtype=bool)
    for frame, pixel in (invalid or {}).items():
        valid[frames.index(frame)][pixel] = False
    return PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray(frames),
        point_map=point_map,
        valid_mask=valid,
        ray_directions=np.asarray(rays, dtype=np.float32),
        dense_storage_dtype="float32",
    )


def test_end_anchored_window_schedule_is_exact() -> None:
    schedule = causal_window_schedule(0, 58, window_size=25, overlap=8)

    assert [(span.start, span.stop) for span in schedule] == [
        (0, 25),
        (17, 42),
        (33, 58),
    ]
    assert [span.window_id for span in schedule] == [
        "window-000000-000025",
        "window-000017-000042",
        "window-000033-000058",
    ]


def test_newest_eligible_falls_back_per_pixel() -> None:
    older = _window("older", [0, 1, 2])
    newer = _window("newer", [2, 3, 4], invalid={2: (1, 2)})

    owned = newest_eligible_windows([older, newer])
    older_owned, newer_owned = owned

    assert np.all(older_owned.valid_mask[:2])
    assert np.count_nonzero(older_owned.valid_mask[2]) == 1
    assert older_owned.valid_mask[2, 1, 2]
    assert np.count_nonzero(newer_owned.valid_mask[0]) == 11
    assert not newer_owned.valid_mask[0, 1, 2]
    assert np.all(newer_owned.valid_mask[1:])


def test_restarted_arms_share_gauges_and_newest_has_one_contributor() -> None:
    first = _window("w0", [0, 1, 2])
    second = _window(
        "w1",
        [2, 3, 4],
        local_from_global=Sim3(scale=1.2, translation=np.asarray([0.1, -0.05, 0.2])),
    )

    products = build_restarted_comparison_products([first, second], random_seed=7)

    assert products.newest.frame_indices.tolist() == [0, 1, 2, 3, 4]
    assert products.fused.frame_indices.tolist() == [0, 1, 2, 3, 4]
    assert np.max(products.newest.contributors) == 1
    assert np.max(products.fused.contributors) == 2
    assert products.alignments[0].reference_id == "w0"
    assert products.alignments[0].moving_id == "w1"
    assert [estimate.window_id for estimate in products.gauges] == ["w0", "w1"]
    overlap_index = products.fused.frame_indices.tolist().index(2)
    np.testing.assert_allclose(
        products.newest.point_map[overlap_index],
        products.fused.point_map[overlap_index],
        atol=1e-5,
    )


def test_native_product_preserves_point_mean_and_support() -> None:
    window = _window("continuous", [0, 1, 2])

    native = build_native_product(window)

    np.testing.assert_allclose(native.point_map, window.point_map)
    np.testing.assert_array_equal(native.valid_mask, window.valid_mask)
    assert np.all(native.contributors[native.valid_mask] == 1)
