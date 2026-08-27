from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.pointworld_sparse_adapter import (
    POINTWORLD_SPARSE_SOURCE_SCHEMA,
    POINTWORLD_SPARSE_SOURCE_VERSION,
    POINTWORLD_UNCERTAINTY_SEMANTICS,
    export_pointworld_source_snapshot,
    main,
    pointworld_output_to_persistent_window,
)
from prob4d.persistent_point_prediction import (
    PersistentPointPredictionWindow,
)


def _pointworld_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory = np.zeros((1, 3, 4, 3), dtype=np.float32)
    trajectory[0, :, :, 0] = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.1, 1.1, 2.1, 3.1],
            [0.2, 1.2, 2.2, 3.2],
        ]
    )
    exists = np.ones((1, 3, 4), dtype=bool)
    exists[0, 0, 3] = False
    exists[0, 1:, 2] = False
    log_var = np.zeros((1, 3, 4, 1), dtype=np.float32)
    return trajectory, exists, log_var


def test_conversion_filters_padding_and_preserves_point_order() -> None:
    trajectory, exists, log_var = _pointworld_arrays()
    window = pointworld_output_to_persistent_window(
        window_id="w0",
        frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
        scene_flows=trajectory,
        scene_exists=exists,
        log_var=log_var,
        point_ids=np.asarray([30, 10, 20, 40], dtype=np.int64),
        storage_dtype="float32",
    )

    assert window.point_ids.tolist() == [10, 20, 30]
    assert window.shape == (3, 3)
    assert window.valid_mask[:, 1].tolist() == [True, False, False]
    assert window.point_trajectory[0, :, 0].tolist() == [1.0, 2.0, 0.0]
    assert window.uncertainty_semantics == POINTWORLD_UNCERTAINTY_SEMANTICS


def test_strict_source_snapshot_exports_and_refuses_clobber(
    tmp_path: Path,
    capsys,
) -> None:
    trajectory, exists, log_var = _pointworld_arrays()
    source = tmp_path / "source.npz"
    output = tmp_path / "persistent.npz"
    np.savez_compressed(
        source,
        schema_name=np.asarray(POINTWORLD_SPARSE_SOURCE_SCHEMA),
        schema_version=np.asarray(
            POINTWORLD_SPARSE_SOURCE_VERSION,
            dtype=np.int64,
        ),
        frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
        scene_flows=trajectory,
        scene_exists=exists,
        log_var=log_var,
        context_frame_count=np.asarray(1, dtype=np.int64),
    )

    assert (
        main(
            [
                str(source),
                str(output),
                "--window-id",
                "pointworld-window-0000",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["point_count"] == 3
    loaded = PersistentPointPredictionWindow.from_npz(output)
    assert loaded.window_id == "pointworld-window-0000"

    with pytest.raises(FileExistsError, match="refusing to replace"):
        export_pointworld_source_snapshot(
            source,
            output,
            window_id="pointworld-window-0000",
        )


def test_source_snapshot_rejects_unversioned_or_ambiguous_payload(
    tmp_path: Path,
) -> None:
    trajectory, exists, log_var = _pointworld_arrays()
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source,
        frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
        scene_flows=trajectory,
        scene_exists=exists,
        log_var=log_var,
        context_frame_count=np.asarray(1, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="fields changed"):
        export_pointworld_source_snapshot(
            source,
            tmp_path / "output.npz",
            window_id="w0",
        )

    with pytest.raises(ValueError, match="one singleton batch dimension"):
        pointworld_output_to_persistent_window(
            window_id="w0",
            frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
            scene_flows=np.zeros((2, 3, 4, 3), dtype=np.float32),
            scene_exists=np.ones((2, 3, 4), dtype=bool),
            log_var=np.zeros((2, 3, 4, 1), dtype=np.float32),
        )
