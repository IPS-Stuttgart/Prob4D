from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.prediction_batch_preflight import (
    PredictionBatchIntegrityError,
    PredictionBatchPolicyV1,
    load_prediction_batch_preflight,
    main,
    preflight_prediction_batch,
    write_prediction_batch_preflight,
)
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window(
    path: Path,
    *,
    window_id: str,
    frames: tuple[int, ...] = (0, 1),
    height: int = 2,
    width: int = 3,
) -> PredictionWindow:
    point_map = np.zeros((len(frames), height, width, 3), dtype=np.float32)
    point_map[..., 2] = 1.0
    valid = np.ones(point_map.shape[:-1], dtype=bool)
    window = PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray(frames, dtype=np.int64),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=np.zeros_like(point_map),
        deform_mask=valid,
        dense_storage_dtype="float32",
    )
    window.to_npz(path)
    return window


def _descriptor(
    path: Path,
    window: PredictionWindow,
    *,
    relative_path: str,
    source_stop: int,
    member: str,
) -> PredictionPayloadDescriptorV1:
    frame_ids = tuple(int(value) for value in window.frame_indices)
    return PredictionPayloadDescriptorV1(
        product_role="independent-window",
        window_id=window.window_id,
        path=relative_path,
        sha256=_sha256(path),
        byte_count=path.stat().st_size,
        view_id="camera-0",
        stochastic_member_id=member,
        dependence_group_ids=("model:shared", f"member:{member}"),
        dense_storage_dtype=window.dense_storage_dtype,
        has_scene_flow=True,
        has_ray_directions=False,
        frame_lineage=tuple(
            PredictionFrameLineageV1(
                output_frame_id=frame_id,
                source_frame_start=0,
                source_frame_stop_exclusive=source_stop,
                contributor_ids=(window.window_id,),
            )
            for frame_id in frame_ids
        ),
    )


def _future_descriptor() -> PredictionPayloadDescriptorV1:
    return PredictionPayloadDescriptorV1(
        product_role="independent-window",
        window_id="future",
        path="future-missing.npz",
        sha256="f" * 64,
        byte_count=1,
        view_id="camera-0",
        stochastic_member_id="future-member",
        dependence_group_ids=("model:shared", "member:future"),
        dense_storage_dtype="float32",
        has_scene_flow=True,
        has_ray_directions=False,
        frame_lineage=tuple(
            PredictionFrameLineageV1(
                output_frame_id=frame_id,
                source_frame_start=0,
                source_frame_stop_exclusive=20,
                contributor_ids=("future",),
            )
            for frame_id in (0, 1)
        ),
    )


def _manifest(
    path: Path, descriptors: tuple[PredictionPayloadDescriptorV1, ...]
) -> Path:
    manifest = PredictionProviderManifestV1(
        sequence_id="case-a",
        provider_family="Example4D",
        provider_repository="example/provider",
        provider_revision="a" * 40,
        provider_run_id="b" * 64,
        model_set_id="c" * 64,
        loader_id="d" * 64,
        coordinate_semantics="window-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="forward-point-displacement",
        ray_semantics="absent",
        payloads=descriptors,
        metadata={
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
        },
    )
    save_prediction_provider_manifest(path, manifest)
    return path


def _two_payload_manifest(
    tmp_path: Path, *, first_height: int = 2, second_height: int = 2
) -> Path:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    first = _window(first_path, window_id="first", height=first_height)
    second = _window(second_path, window_id="second", height=second_height)
    return _manifest(
        tmp_path / "provider.json",
        (
            _descriptor(
                first_path,
                first,
                relative_path="first.npz",
                source_stop=2,
                member="one",
            ),
            _descriptor(
                second_path,
                second,
                relative_path="second.npz",
                source_stop=2,
                member="two",
            ),
        ),
    )


def test_compatible_batch_passes_and_roundtrips(tmp_path: Path) -> None:
    manifest_path = _two_payload_manifest(tmp_path)

    result = preflight_prediction_batch(manifest_path, causal_frame_stop=2)

    assert result.compatible is True
    assert result.status == "pass"
    assert len(result.entries) == 2
    assert result.violations == ()
    assert result.future_prediction_payloads_opened == 0

    output = tmp_path / "preflight.json"
    write_prediction_batch_preflight(output, result)
    write_prediction_batch_preflight(output, result)
    assert load_prediction_batch_preflight(output) == result


def test_spatial_mismatch_is_a_structured_negative(tmp_path: Path) -> None:
    manifest_path = _two_payload_manifest(tmp_path, second_height=4)

    result = preflight_prediction_batch(manifest_path, causal_frame_stop=2)

    assert result.compatible is False
    assert result.status == "batch-incompatible"
    assert [item.code for item in result.violations] == ["spatial-shape-mismatch"]
    assert result.violations[0].expected == [2, 3]
    assert result.violations[0].observed == [4, 3]


def test_policy_can_admit_a_prospectively_declared_ragged_grid(
    tmp_path: Path,
) -> None:
    manifest_path = _two_payload_manifest(tmp_path, second_height=4)

    result = preflight_prediction_batch(
        manifest_path,
        causal_frame_stop=2,
        policy=PredictionBatchPolicyV1(require_common_spatial_shape=False),
    )

    assert result.compatible is True
    assert result.violations == ()


def test_missing_future_payload_is_not_opened(tmp_path: Path) -> None:
    current_path = tmp_path / "current.npz"
    current = _window(current_path, window_id="current")
    manifest_path = _manifest(
        tmp_path / "provider.json",
        (
            _descriptor(
                current_path,
                current,
                relative_path="current.npz",
                source_stop=2,
                member="current",
            ),
            _future_descriptor(),
        ),
    )

    result = preflight_prediction_batch(manifest_path, causal_frame_stop=2)

    assert result.compatible is True
    assert len(result.entries) == 1
    assert result.excluded_future_payload_ids
    assert result.future_prediction_payloads_opened == 0
    assert not (tmp_path / "future-missing.npz").exists()


def test_selected_payload_tamper_is_an_integrity_failure(tmp_path: Path) -> None:
    payload_path = tmp_path / "current.npz"
    window = _window(payload_path, window_id="current")
    manifest_path = _manifest(
        tmp_path / "provider.json",
        (
            _descriptor(
                payload_path,
                window,
                relative_path="current.npz",
                source_stop=2,
                member="current",
            ),
        ),
    )
    payload_path.write_bytes(payload_path.read_bytes() + b"tamper")

    with pytest.raises(PredictionBatchIntegrityError, match="byte count mismatch"):
        preflight_prediction_batch(manifest_path, causal_frame_stop=2)


def test_cli_writes_negative_artifact_and_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = _two_payload_manifest(tmp_path, second_height=4)
    output = tmp_path / "preflight.json"

    assert (
        main(
            [
                "build",
                str(manifest_path),
                str(output),
                "--causal-frame-stop",
                "2",
            ]
        )
        == 2
    )
    assert output.is_file()
    assert '"status": "batch-incompatible"' in capsys.readouterr().out
    assert load_prediction_batch_preflight(output).compatible is False
