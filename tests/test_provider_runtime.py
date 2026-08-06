from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
)
from prob4d.provider_runtime import (
    fuse_provider_exploratory,
    load_prediction_provider_runtime,
    resolve_provider_gauges,
)
from prob4d.sim3 import Sim3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window(
    window_id: str,
    *,
    frame: int,
    value: float,
) -> PredictionWindow:
    points = np.full((1, 2, 2, 3), value, dtype=np.float32)
    points[..., 2] += 1.0
    return PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray([frame], dtype=np.int64),
        point_map=points,
        valid_mask=np.ones((1, 2, 2), dtype=bool),
        dense_storage_dtype="float32",
    )


def _descriptor(
    root: Path,
    window: PredictionWindow,
    *,
    product_role: str = "independent-window",
    source_start: int | None = None,
    source_stop: int | None = None,
    stochastic_member_id: str = "member-0",
    dependence_groups: tuple[str, ...] = ("model:test", "input:test"),
) -> PredictionPayloadDescriptorV1:
    path = root / f"{window.window_id}.npz"
    window.to_npz(path)
    start = window.start_frame if source_start is None else source_start
    stop = window.stop_frame if source_stop is None else source_stop
    return PredictionPayloadDescriptorV1(
        product_role=product_role,
        window_id=window.window_id,
        path=path.name,
        sha256=_sha256(path),
        byte_count=path.stat().st_size,
        view_id="camera-0",
        stochastic_member_id=stochastic_member_id,
        dependence_group_ids=dependence_groups,
        dense_storage_dtype=window.dense_storage_dtype,
        has_scene_flow=False,
        has_ray_directions=False,
        frame_lineage=(
            PredictionFrameLineageV1(
                output_frame_id=window.start_frame,
                source_frame_start=start,
                source_frame_stop_exclusive=stop,
                contributor_ids=(window.window_id,),
            ),
        ),
    )


def _manifest(
    root: Path,
    descriptors: tuple[PredictionPayloadDescriptorV1, ...],
    *,
    coordinate_semantics: str = "metric-world",
) -> Path:
    manifest = PredictionProviderManifestV1(
        sequence_id="sequence-a",
        provider_family="test-provider",
        provider_repository="example/test-provider",
        provider_revision="1" * 40,
        provider_run_id="2" * 64,
        model_set_id="3" * 64,
        loader_id="4" * 64,
        coordinate_semantics=coordinate_semantics,
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        payloads=descriptors,
        metadata={"uses_truth": False},
    )
    path = root / "provider.json"
    save_prediction_provider_manifest(path, manifest)
    return path


def test_causal_runtime_does_not_open_future_payloads(tmp_path: Path) -> None:
    prefix = _window("prefix", frame=0, value=1.0)
    prefix_descriptor = _descriptor(
        tmp_path,
        prefix,
        source_start=0,
        source_stop=1,
    )

    future_path = tmp_path / "future.npz"
    future_path.write_bytes(b"not a prediction archive")
    future_descriptor = PredictionPayloadDescriptorV1(
        product_role="independent-window",
        window_id="future",
        path=future_path.name,
        sha256=_sha256(future_path),
        byte_count=future_path.stat().st_size,
        view_id="camera-0",
        stochastic_member_id="member-1",
        dependence_group_ids=("model:test", "input:test"),
        dense_storage_dtype="float32",
        has_scene_flow=False,
        has_ray_directions=False,
        frame_lineage=(
            PredictionFrameLineageV1(
                output_frame_id=2,
                source_frame_start=2,
                source_frame_stop_exclusive=3,
                contributor_ids=("future",),
            ),
        ),
    )
    manifest = _manifest(tmp_path, (prefix_descriptor, future_descriptor))

    runtime = load_prediction_provider_runtime(manifest, causal_frame_stop=2)
    assert runtime.window_ids == ("prefix",)
    assert runtime.summary()["future_prediction_payloads_opened"] == 0

    with pytest.raises(ValueError):
        load_prediction_provider_runtime(manifest)


def test_runtime_rejects_dependent_alternative_constructions(tmp_path: Path) -> None:
    first = _window("world-points", frame=0, value=1.0)
    second = _window("depth-unprojected", frame=0, value=2.0)
    descriptors = (
        _descriptor(
            tmp_path,
            first,
            product_role="external-sequence",
            stochastic_member_id="same-run",
        ),
        _descriptor(
            tmp_path,
            second,
            product_role="external-sequence",
            stochastic_member_id="same-run",
        ),
    )
    manifest = _manifest(
        tmp_path,
        descriptors,
        coordinate_semantics="sequence-local-sim3",
    )

    with pytest.raises(ValueError, match="dependent alternative constructions"):
        load_prediction_provider_runtime(manifest)

    runtime = load_prediction_provider_runtime(
        manifest,
        allow_dependent_alternatives=True,
    )
    assert runtime.dependent_alternatives_admitted


def test_coordinate_semantics_are_resolved_fail_closed(tmp_path: Path) -> None:
    window = _window("sample", frame=0, value=1.0)
    descriptor = _descriptor(tmp_path, window)

    metric_runtime = load_prediction_provider_runtime(
        _manifest(tmp_path, (descriptor,), coordinate_semantics="metric-world")
    )
    metric_gauges = resolve_provider_gauges(metric_runtime)
    np.testing.assert_allclose(metric_gauges["sample"].as_vector(), np.zeros(7))
    with pytest.raises(ValueError, match="must not be transformed"):
        resolve_provider_gauges(metric_runtime, sequence_gauge=Sim3.identity())

    sequence_manifest = tmp_path / "sequence-provider.json"
    sequence = PredictionProviderManifestV1(
        sequence_id="sequence-a",
        provider_family="test-provider",
        provider_repository="example/test-provider",
        provider_revision="1" * 40,
        provider_run_id="5" * 64,
        model_set_id="3" * 64,
        loader_id="4" * 64,
        coordinate_semantics="sequence-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        payloads=(descriptor,),
        metadata={"uses_truth": False},
    )
    save_prediction_provider_manifest(sequence_manifest, sequence)
    sequence_runtime = load_prediction_provider_runtime(sequence_manifest)
    with pytest.raises(ValueError, match="require a metric sequence gauge"):
        resolve_provider_gauges(sequence_runtime)
    shared = Sim3.from_vector(np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))
    gauges = resolve_provider_gauges(sequence_runtime, sequence_gauge=shared)
    assert gauges["sample"] is shared


def test_exploratory_metric_provider_runs_existing_fusion_core(
    tmp_path: Path,
) -> None:
    first = _window("first", frame=0, value=0.0)
    second = _window("second", frame=0, value=2.0)
    manifest = _manifest(
        tmp_path,
        (_descriptor(tmp_path, first), _descriptor(tmp_path, second)),
    )
    runtime = load_prediction_provider_runtime(manifest)
    fused = fuse_provider_exploratory(
        runtime,
        point_standard_deviation_m=0.01,
        method="uniform",
    )
    np.testing.assert_allclose(fused.point_map[..., :2], 1.0)
    np.testing.assert_allclose(fused.point_map[..., 2], 2.0)
    assert np.all(fused.contributors == 2)
