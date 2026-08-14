from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.adapter.v1 import (
    ProviderAdapterIdentityV1,
    ProviderAdapterRequestV1,
    ProviderAdapterWindowV1,
    StaticPredictionProviderAdapterV1,
    load_provider_adapter_request,
    materialize_provider_adapter,
    write_provider_adapter_request,
)
from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    verify_prediction_provider_manifest,
)


def _identity() -> ProviderAdapterIdentityV1:
    return ProviderAdapterIdentityV1(
        adapter_name="example-adapter",
        adapter_version=1,
        adapter_implementation_id="a" * 64,
        provider_family="Example4D",
        provider_repository="example/Example4D",
        provider_revision="b" * 40,
        provider_run_id="c" * 64,
        model_set_id="d" * 64,
        loader_id="e" * 64,
        coordinate_semantics="window-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        metadata={"native_format": "example-cache-v1"},
    )


def _request(*, cutoff: int = 6) -> ProviderAdapterRequestV1:
    return ProviderAdapterRequestV1(
        sequence_id="case-a",
        causal_frame_stop=cutoff,
        input_family_id="f" * 64,
        input_snapshot_id=("1" if cutoff == 6 else "2") * 64,
        metadata={"fixture": "unit"},
    )


def _window(
    *,
    window_id: str,
    frames: tuple[int, ...],
    relative_path: str,
    source_stop: int,
) -> ProviderAdapterWindowV1:
    points = np.zeros((len(frames), 2, 3, 3), dtype=np.float32)
    points[..., 0] = np.asarray(frames, dtype=np.float32)[:, None, None]
    points[..., 2] = 1.0
    valid = np.ones(points.shape[:-1], dtype=bool)
    prediction = PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray(frames, dtype=np.int64),
        point_map=points,
        valid_mask=valid,
        dense_storage_dtype="float32",
    )
    lineage = tuple(
        PredictionFrameLineageV1(
            output_frame_id=frame,
            source_frame_start=frames[0],
            source_frame_stop_exclusive=source_stop,
            contributor_ids=(window_id,),
        )
        for frame in frames
    )
    return ProviderAdapterWindowV1(
        window=prediction,
        relative_path=relative_path,
        product_role="independent-window",
        view_id="camera-0",
        stochastic_member_id=f"member:{window_id}",
        dependence_group_ids=("model:shared", f"window:{window_id}"),
        frame_lineage=lineage,
    )


def test_adapter_materialization_is_canonical_and_idempotent(tmp_path: Path) -> None:
    later = _window(
        window_id="window-0001",
        frames=(3, 4, 5),
        relative_path="payloads/window-0001.npz",
        source_stop=6,
    )
    earlier = _window(
        window_id="window-0000",
        frames=(0, 1, 2),
        relative_path="payloads/window-0000.npz",
        source_stop=3,
    )
    adapter = StaticPredictionProviderAdapterV1(
        identity=_identity(),
        windows=(later, earlier),
    )
    output = tmp_path / "provider-neutral.json"

    first = materialize_provider_adapter(adapter, _request(), output)
    second = materialize_provider_adapter(adapter, _request(), output)

    assert first.artifact_id == second.artifact_id
    assert [payload.window_id for payload in first.payloads] == [
        "window-0000",
        "window-0001",
    ]
    assert first.metadata["source_adapter"] == "prob4d-provider-adapter-v1"
    assert first.metadata["provider_adapter_identity_id"] == (
        adapter.identity.provider_adapter_identity_id
    )
    assert first.metadata["provider_adapter_request_id"] == (
        _request().provider_adapter_request_id
    )
    _, report = verify_prediction_provider_manifest(output, causal_frame_stop=6)
    assert report["verified_payload_count"] == 2
    assert report["admitted_payload_count"] == 2


def test_adapter_request_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    request = _request()
    write_provider_adapter_request(path, request)
    assert load_provider_adapter_request(path).provider_adapter_request_id == (
        request.provider_adapter_request_id
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    record["causal_frame_stop"] = 7
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="derived fields changed"):
        load_provider_adapter_request(path)


def test_adapter_rejects_future_dependent_output(tmp_path: Path) -> None:
    future = _window(
        window_id="window-0000",
        frames=(0, 1, 2),
        relative_path="payloads/window-0000.npz",
        source_stop=7,
    )
    adapter = StaticPredictionProviderAdapterV1(identity=_identity(), windows=(future,))
    with pytest.raises(ValueError, match="crosses the requested causal"):
        materialize_provider_adapter(
            adapter,
            _request(cutoff=6),
            tmp_path / "provider-neutral.json",
        )


def test_adapter_refuses_conflicting_existing_payload(tmp_path: Path) -> None:
    original = _window(
        window_id="window-0000",
        frames=(0, 1, 2),
        relative_path="payloads/window-0000.npz",
        source_stop=3,
    )
    materialize_provider_adapter(
        StaticPredictionProviderAdapterV1(identity=_identity(), windows=(original,)),
        _request(),
        tmp_path / "provider-neutral.json",
    )

    changed_points = np.array(original.window.point_map, copy=True)
    changed_points[..., 0] += 1.0
    changed_window = PredictionWindow(
        window_id=original.window.window_id,
        frame_indices=original.window.frame_indices,
        point_map=changed_points,
        valid_mask=original.window.valid_mask,
        dense_storage_dtype="float32",
    )
    changed = ProviderAdapterWindowV1(
        window=changed_window,
        relative_path=original.relative_path,
        product_role=original.product_role,
        view_id=original.view_id,
        stochastic_member_id=original.stochastic_member_id,
        dependence_group_ids=original.dependence_group_ids,
        frame_lineage=original.frame_lineage,
    )
    with pytest.raises(ValueError, match="refusing to replace different adapter payload"):
        materialize_provider_adapter(
            StaticPredictionProviderAdapterV1(identity=_identity(), windows=(changed,)),
            _request(),
            tmp_path / "other-manifest.json",
        )
