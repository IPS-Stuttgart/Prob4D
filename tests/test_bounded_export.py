from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import prob4d.bounded_export as bounded_export
from prob4d.bounded_export import save_fused_prediction_bounded
from prob4d.fusion import FusedSequence
from prob4d.io import (
    load_fused_prediction_artifact,
    load_fused_prediction_metadata,
    save_fused_prediction,
)


def _sequence() -> FusedSequence:
    generator = np.random.default_rng(20260806)
    shape = (4, 3, 5)
    point_map = generator.normal(size=shape + (3,))
    valid_mask = generator.random(shape) > 0.2
    point_covariance = np.empty(shape + (3, 3), dtype=np.float64)
    for index in np.ndindex(shape):
        factor = generator.normal(size=(3, 3))
        point_covariance[index] = factor @ factor.T + 0.01 * np.eye(3)
    scene_flow = generator.normal(scale=0.05, size=shape + (3,))
    deform_mask = valid_mask & (generator.random(shape) > 0.25)
    flow_covariance = np.empty_like(point_covariance)
    for index in np.ndindex(shape):
        factor = generator.normal(scale=0.1, size=(3, 3))
        flow_covariance[index] = factor @ factor.T + 0.001 * np.eye(3)
    return FusedSequence(
        frame_indices=np.arange(10, 14, dtype=np.int64),
        point_map=point_map,
        valid_mask=valid_mask,
        point_covariance=point_covariance,
        contributors=generator.integers(1, 4, size=shape, dtype=np.uint16),
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        flow_covariance=flow_covariance,
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_bounded_export_matches_ordinary_field_contract(
    tmp_path: Path,
    compressed: bool,
) -> None:
    sequence = _sequence()
    ordinary = tmp_path / "ordinary.npz"
    bounded = tmp_path / "bounded.npz"
    kwargs = {
        "method_id": "bounded-ci",
        "fusion_method": "covariance_intersection",
        "metadata": {"calibration": "source-only", "rank": 2},
        "compressed": compressed,
    }

    ordinary_metadata = save_fused_prediction(ordinary, sequence, **kwargs)
    bounded_metadata = save_fused_prediction_bounded(
        bounded,
        sequence,
        chunk_rows=7,
        **kwargs,
    )

    assert bounded_metadata.to_dict() == ordinary_metadata.to_dict()
    with (
        np.load(ordinary, allow_pickle=False) as ordinary_data,
        np.load(
            bounded,
            allow_pickle=False,
        ) as bounded_data,
    ):
        assert bounded_data.files == ordinary_data.files
        for field in ordinary_data.files:
            ordinary_value = ordinary_data[field]
            bounded_value = bounded_data[field]
            assert bounded_value.dtype == ordinary_value.dtype
            assert bounded_value.shape == ordinary_value.shape
            np.testing.assert_array_equal(bounded_value, ordinary_value)

    ordinary_artifact = load_fused_prediction_artifact(ordinary)
    bounded_artifact = load_fused_prediction_artifact(bounded)
    assert bounded_artifact.metadata.to_dict() == ordinary_artifact.metadata.to_dict()
    np.testing.assert_array_equal(
        bounded_artifact.sequence.point_map,
        ordinary_artifact.sequence.point_map,
    )
    np.testing.assert_array_equal(
        bounded_artifact.sequence.point_covariance,
        ordinary_artifact.sequence.point_covariance,
    )
    np.testing.assert_array_equal(
        bounded_artifact.sequence.scene_flow,
        ordinary_artifact.sequence.scene_flow,
    )
    np.testing.assert_array_equal(
        bounded_artifact.sequence.flow_covariance,
        ordinary_artifact.sequence.flow_covariance,
    )


def test_bounded_export_without_covariance_preserves_optional_fields(
    tmp_path: Path,
) -> None:
    sequence = _sequence()
    destination = tmp_path / "without-covariance"

    save_fused_prediction_bounded(
        destination,
        sequence,
        method_id="uniform",
        fusion_method="uniform",
        include_covariance=False,
        chunk_rows=5,
    )

    archive = destination.with_suffix(".npz")
    assert archive.is_file()
    with np.load(archive, allow_pickle=False) as data:
        assert "point_covariance_packed" not in data.files
        assert "flow_covariance_packed" not in data.files
        assert "contributors" not in data.files
        assert "scene_flow" in data.files
        assert "deform_mask" in data.files
        np.testing.assert_array_equal(
            data["point_map"],
            sequence.point_map.astype(np.float32),
        )
        np.testing.assert_array_equal(
            data["scene_flow"],
            sequence.scene_flow.astype(np.float32),
        )
    metadata = load_fused_prediction_metadata(archive)
    assert metadata.fusion_method == "uniform"


@pytest.mark.parametrize("chunk_rows", [0, -1, True, 1.5])
def test_bounded_export_rejects_invalid_chunk_rows(
    tmp_path: Path,
    chunk_rows: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        save_fused_prediction_bounded(
            tmp_path / "invalid.npz",
            _sequence(),
            method_id="uniform",
            fusion_method="uniform",
            chunk_rows=chunk_rows,  # type: ignore[arg-type]
        )


def test_bounded_export_keeps_existing_destination_on_archive_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "existing.npz"
    original = b"pre-existing-artifact"
    destination.write_bytes(original)

    def fail_archive(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected archive failure")

    monkeypatch.setattr(bounded_export, "_archive_members", fail_archive)
    with pytest.raises(RuntimeError, match="injected archive failure"):
        save_fused_prediction_bounded(
            destination,
            _sequence(),
            method_id="uniform",
            fusion_method="uniform",
            chunk_rows=3,
        )

    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(".existing.npz.members.*"))
    assert not list(tmp_path.glob(".existing.npz.*.tmp"))
