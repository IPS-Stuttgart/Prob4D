from __future__ import annotations

from pathlib import Path

import numpy as np

from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import PredictionFrameLineageV1
from prob4d.provider_adapter import (
    ProviderAdapterIdentityV1,
    ProviderAdapterRequestV1,
    ProviderAdapterWindowV1,
)
from prob4d.provider_adapter_conformance import (
    load_provider_adapter_conformance,
    run_provider_adapter_conformance,
    write_provider_adapter_conformance,
)


def _identity() -> ProviderAdapterIdentityV1:
    return ProviderAdapterIdentityV1(
        adapter_name="fixture-adapter",
        adapter_version=1,
        adapter_implementation_id="a" * 64,
        provider_family="Fixture4D",
        provider_repository="example/Fixture4D",
        provider_revision="b" * 40,
        provider_run_id="c" * 64,
        model_set_id="d" * 64,
        loader_id="e" * 64,
        coordinate_semantics="window-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        metadata={"native_format": "fixture-cache-v1"},
    )


def _request(cutoff: int) -> ProviderAdapterRequestV1:
    return ProviderAdapterRequestV1(
        sequence_id="fixture",
        causal_frame_stop=cutoff,
        input_family_id="f" * 64,
        input_snapshot_id=("1" if cutoff == 3 else "2") * 64,
    )


def _window(
    window_id: str,
    frames: tuple[int, ...],
    *,
    value_offset: float = 0.0,
) -> ProviderAdapterWindowV1:
    points = np.zeros((len(frames), 1, 2, 3), dtype=np.float32)
    points[..., 0] = np.asarray(frames, dtype=np.float32)[:, None, None] + value_offset
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
            source_frame_stop_exclusive=frames[-1] + 1,
            contributor_ids=(window_id,),
        )
        for frame in frames
    )
    return ProviderAdapterWindowV1(
        window=prediction,
        relative_path=f"payloads/{window_id}.npz",
        product_role="independent-window",
        view_id="camera-0",
        stochastic_member_id=f"member:{window_id}",
        dependence_group_ids=("input:fixture", f"window:{window_id}"),
        frame_lineage=lineage,
    )


class _FixtureAdapter:
    identity = _identity()

    def produce(
        self,
        request: ProviderAdapterRequestV1,
    ) -> tuple[ProviderAdapterWindowV1, ...]:
        windows = [_window("window-0000", (0, 1, 2))]
        if request.causal_frame_stop >= 5:
            windows.append(_window("window-0001", (3, 4)))
        return tuple(reversed(windows))


class _FutureLeakingAdapter(_FixtureAdapter):
    def produce(
        self,
        request: ProviderAdapterRequestV1,
    ) -> tuple[ProviderAdapterWindowV1, ...]:
        offset = 1.0 if request.causal_frame_stop >= 5 else 0.0
        windows = [_window("window-0000", (0, 1, 2), value_offset=offset)]
        if request.causal_frame_stop >= 5:
            windows.append(_window("window-0001", (3, 4)))
        return tuple(windows)


def test_full_adapter_conformance_passes_and_replays(tmp_path: Path) -> None:
    result = run_provider_adapter_conformance(
        _FixtureAdapter(),
        _request(3),
        _request(5),
        tmp_path / "runs",
    )
    assert result.conformance_pass
    assert all(result.checks.values())

    path = tmp_path / "conformance.json"
    write_provider_adapter_conformance(path, result)
    loaded = load_provider_adapter_conformance(path)
    assert loaded.provider_adapter_conformance_id == (
        result.provider_adapter_conformance_id
    )


def test_future_prefix_change_is_retained_as_valid_conformance_failure(
    tmp_path: Path,
) -> None:
    result = run_provider_adapter_conformance(
        _FutureLeakingAdapter(),
        _request(3),
        _request(5),
        tmp_path / "runs",
    )
    assert not result.conformance_pass
    assert not result.checks["causal_prefix_invariant"]
    assert "causal-prefix-invariant-failed" in result.reason_codes


class _ChangingIdentityAdapter(_FixtureAdapter):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def identity(self) -> ProviderAdapterIdentityV1:
        implementation = "a" * 64 if self.calls == 0 else "9" * 64
        return ProviderAdapterIdentityV1(
            adapter_name="fixture-adapter",
            adapter_version=1,
            adapter_implementation_id=implementation,
            provider_family="Fixture4D",
            provider_repository="example/Fixture4D",
            provider_revision="b" * 40,
            provider_run_id="c" * 64,
            model_set_id="d" * 64,
            loader_id="e" * 64,
            coordinate_semantics="window-local-sim3",
            point_semantics="dense-point-map",
            flow_semantics="absent",
            ray_semantics="absent",
            metadata={"native_format": "fixture-cache-v1"},
        )

    def produce(
        self,
        request: ProviderAdapterRequestV1,
    ) -> tuple[ProviderAdapterWindowV1, ...]:
        self.calls += 1
        return super().produce(request)


def test_adapter_identity_change_is_retained_as_conformance_failure(
    tmp_path: Path,
) -> None:
    result = run_provider_adapter_conformance(
        _ChangingIdentityAdapter(),
        _request(3),
        _request(5),
        tmp_path / "runs",
    )
    assert not result.conformance_pass
    assert not result.checks["adapter_identity_stable"]
    assert "adapter-identity-stable-failed" in result.reason_codes
