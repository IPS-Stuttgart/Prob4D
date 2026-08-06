from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.visual_bias import VisualBiasNuisanceV1
from prob4d.visual_bias_stream import (
    VisualBiasNuisanceStreamV1,
    append_visual_bias_nuisance,
    build_visual_bias_nuisance_stream,
    load_visual_bias_nuisance_stream,
    write_visual_bias_nuisance_stream,
)


def _sha(character: str) -> str:
    return character * 64


def _nuisance(
    *,
    observation_character: str,
    identity_character: str,
    row_count: int = 2,
    covariance: np.ndarray | None = None,
    jacobian_scale: float = 1.0,
    bias_ids: tuple[str, ...] = ("camera-0",),
) -> VisualBiasNuisanceV1:
    row_bias_indices = np.zeros(row_count, dtype=np.int64)
    bias_jacobian = np.zeros((row_count, 3, 1), dtype=np.float64)
    bias_jacobian[:, 0, 0] = jacobian_scale
    prior = (
        np.asarray([[4.0]], dtype=np.float64)
        if covariance is None
        else np.asarray(covariance, dtype=np.float64)
    )
    return VisualBiasNuisanceV1(
        observation_artifact_id=_sha(observation_character),
        observation_identity_sha256=_sha(identity_character),
        bias_ids=bias_ids,
        basis_names=("depth-offset",),
        row_bias_indices=row_bias_indices,
        bias_jacobian=bias_jacobian,
        joint_bias_covariance=prior,
        orthogonalization_semantics=("conditional-whitened-global-gauge-projection-v1"),
        maximum_gauge_projection=1e-14,
        gauge_projection_tolerance=1e-10,
        metadata={"source": "calibration-only"},
    )


def _stream() -> VisualBiasNuisanceStreamV1:
    return build_visual_bias_nuisance_stream(
        stream_key="camera-0-recursive",
        nuisances=(
            _nuisance(observation_character="a", identity_character="b"),
            _nuisance(
                observation_character="c",
                identity_character="d",
                jacobian_scale=2.0,
            ),
        ),
        observation_stream_update_ids=(_sha("1"), _sha("2")),
        frame_intervals=((0, 5), (5, 10)),
        model_metadata={"calibration_artifact_id": _sha("e")},
        metadata={"case_id": "case-001"},
    )


def test_shared_prior_produces_cross_update_covariance() -> None:
    stream = _stream()

    factor = stream.low_rank_factor()
    dense = factor.reshape(3 * stream.observation_count, -1)
    covariance = dense @ dense.T

    first_x = 0
    first_row_of_second_update_x = 3 * stream.updates[1].row_start
    assert covariance[first_x, first_row_of_second_update_x] == pytest.approx(8.0)
    assert stream.latent_dimension == 1
    assert stream.row_update_indices.tolist() == [0, 0, 1, 1]


def test_builder_rejects_incompatible_shared_prior() -> None:
    first = _nuisance(observation_character="a", identity_character="b")
    second = _nuisance(
        observation_character="c",
        identity_character="d",
        covariance=np.asarray([[5.0]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="exact joint prior"):
        build_visual_bias_nuisance_stream(
            stream_key="incompatible",
            nuisances=(first, second),
            observation_stream_update_ids=(_sha("1"), _sha("2")),
            frame_intervals=((0, 5), (5, 10)),
        )


def test_builder_rejects_overlapping_frames_and_duplicate_update_ids() -> None:
    nuisances = (
        _nuisance(observation_character="a", identity_character="b"),
        _nuisance(observation_character="c", identity_character="d"),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        build_visual_bias_nuisance_stream(
            stream_key="overlap",
            nuisances=nuisances,
            observation_stream_update_ids=(_sha("1"), _sha("2")),
            frame_intervals=((0, 6), (5, 10)),
        )
    with pytest.raises(ValueError, match="must be unique"):
        build_visual_bias_nuisance_stream(
            stream_key="duplicates",
            nuisances=nuisances,
            observation_stream_update_ids=(_sha("1"), _sha("1")),
            frame_intervals=((0, 5), (5, 10)),
        )


def test_builder_rejects_replayed_observation_evidence() -> None:
    first = _nuisance(observation_character="a", identity_character="b")
    repeated_artifact = _nuisance(
        observation_character="a",
        identity_character="d",
    )
    with pytest.raises(ValueError, match="observation_artifact_id values must be unique"):
        build_visual_bias_nuisance_stream(
            stream_key="replayed-artifact",
            nuisances=(first, repeated_artifact),
            observation_stream_update_ids=(_sha("1"), _sha("2")),
            frame_intervals=((0, 5), (5, 10)),
        )

    repeated_identity = _nuisance(
        observation_character="c",
        identity_character="b",
    )
    with pytest.raises(
        ValueError,
        match="observation_identity_sha256 values must be unique",
    ):
        build_visual_bias_nuisance_stream(
            stream_key="replayed-identity",
            nuisances=(first, repeated_identity),
            observation_stream_update_ids=(_sha("1"), _sha("2")),
            frame_intervals=((0, 5), (5, 10)),
        )


def test_append_preserves_retained_update_chain() -> None:
    original = _stream()
    retained_update_ids = tuple(update.update_id for update in original.updates)
    appended = append_visual_bias_nuisance(
        original,
        _nuisance(observation_character="f", identity_character="0"),
        observation_stream_update_id=_sha("3"),
        frame_interval=(12, 16),
    )

    assert tuple(update.update_id for update in appended.updates[:2]) == retained_update_ids
    assert appended.updates[-1].previous_update_id == retained_update_ids[-1]
    assert appended.artifact_id != original.artifact_id
    assert appended.row_update_indices.tolist()[-2:] == [2, 2]


def test_append_rejects_replayed_observation_update() -> None:
    stream = _stream()
    with pytest.raises(ValueError, match="already present"):
        append_visual_bias_nuisance(
            stream,
            _nuisance(observation_character="f", identity_character="0"),
            observation_stream_update_id=_sha("1"),
            frame_interval=(10, 15),
        )


def test_round_trip_and_idempotent_write(tmp_path: Path) -> None:
    stream = _stream()
    manifest = tmp_path / "stream.json"
    written_manifest, payload = write_visual_bias_nuisance_stream(stream, manifest)
    assert written_manifest == manifest
    assert payload.is_file()

    loaded = load_visual_bias_nuisance_stream(manifest)
    assert loaded.artifact_id == stream.artifact_id
    assert np.array_equal(loaded.bias_jacobian, stream.bias_jacobian)
    assert not loaded.bias_jacobian.flags.writeable

    write_visual_bias_nuisance_stream(stream, manifest)


def test_payload_and_manifest_tampering_fail_closed(tmp_path: Path) -> None:
    stream = _stream()
    manifest = tmp_path / "stream.json"
    _, payload = write_visual_bias_nuisance_stream(stream, manifest)

    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="byte count mismatch"):
        load_visual_bias_nuisance_stream(manifest)

    manifest.unlink()
    payload.unlink()
    write_visual_bias_nuisance_stream(stream, manifest)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["stream_key"] = "changed"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        load_visual_bias_nuisance_stream(manifest)


def test_strict_manifest_and_partial_destination_rejection(tmp_path: Path) -> None:
    stream = _stream()
    manifest = tmp_path / "stream.json"
    _, payload = write_visual_bias_nuisance_stream(stream, manifest)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace('"schema":', '"schema": "duplicate", "schema":', 1))
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_visual_bias_nuisance_stream(manifest)

    manifest.unlink()
    with pytest.raises(ValueError, match="partial artifact"):
        write_visual_bias_nuisance_stream(stream, manifest, payload_path=payload)


def test_different_stream_cannot_replace_retained_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "stream.json"
    write_visual_bias_nuisance_stream(_stream(), manifest)
    different = build_visual_bias_nuisance_stream(
        stream_key="different",
        nuisances=(_nuisance(observation_character="a", identity_character="b"),),
        observation_stream_update_ids=(_sha("9"),),
        frame_intervals=((0, 5),),
    )
    with pytest.raises(ValueError, match="refusing to replace"):
        write_visual_bias_nuisance_stream(different, manifest)
