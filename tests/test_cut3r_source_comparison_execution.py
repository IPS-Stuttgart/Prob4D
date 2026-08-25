from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from prob4d.cut3r_source_comparison_execution import (
    build_native_product,
    build_restarted_comparison_products,
    causal_window_schedule,
    newest_eligible_windows,
)
from prob4d.data import PredictionWindow
from prob4d.sim3 import Sim3


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/science/run_cut3r_source_comparison.py"
    spec = importlib.util.spec_from_file_location("cut3r_source_comparison_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_cut3r_runtime_exposes_repository_and_internal_package(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_runner_module()
    checkout = tmp_path / "CUT3R"
    (checkout / "src").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))

    module._prepend_cut3r_import_paths(checkout)

    assert sys.path[:2] == [str(checkout / "src"), str(checkout)]


def test_runtime_bootstrap_failure_is_retained_without_progress(tmp_path: Path) -> None:
    module = _load_runner_module()
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    checkout = tmp_path / "CUT3R"
    checkpoint = tmp_path / "model.pth"
    processed.mkdir()
    checkout.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    case = {
        "case_id": "development-case",
        "group_id": "development-group",
        "role": "development",
    }

    manifest = module._retain_runtime_failure(
        error=ModuleNotFoundError(f"missing package below {checkout}"),
        traceback_text=f"trace below {processed}",
        case=case,
        plan={"plan_id": "frozen-plan"},
        processed_root=processed,
        output_root=output,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
        shard_index=0,
    )

    assert manifest["status"] == "retained-technical-failure"
    assert manifest["source_rgb_frames_decoded"] is False
    assert manifest["cut3r_inference_executed"] is False
    assert manifest["source_predictions_written"] is False
    assert str(checkout) not in manifest["failure"]
    assert (output / "cases/development-case/case_manifest.json").is_file()


def test_amended_plan_selects_only_the_registered_replacement_smoke() -> None:
    module = _load_runner_module()
    plan = {
        "schema_version": 2,
        "execution": {
            "shard_count": 2,
            "smoke_policy": {"registered_case_id": "development-new"},
        },
        "cases": [
            {"case_id": "development-new", "role": "development"},
            {"case_id": "development-old", "role": "development"},
        ],
    }

    selected = module._selected_cases(
        plan,
        shard_index=0,
        shard_count=2,
        smoke_case_id="development-new",
    )
    assert [case["case_id"] for case in selected] == ["development-new"]

    with pytest.raises(ValueError, match="amended registered case"):
        module._selected_cases(
            plan,
            shard_index=0,
            shard_count=2,
            smoke_case_id="development-old",
        )
