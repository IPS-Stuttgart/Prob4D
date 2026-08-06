from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prob4d.observation_bias_binding import (
    build_observation_bias_binding,
    build_observation_bias_binding_from_paths,
    load_observation_bias_binding,
    verify_observation_bias_binding,
    write_observation_bias_binding,
)
from prob4d.observation_factor_stream import (
    ObservationFactorStreamUpdateV1,
    ObservationFactorStreamV1,
    write_observation_factor_stream,
)
from prob4d.visual_bias import VisualBiasNuisanceV1
from prob4d.visual_bias_stream import (
    build_visual_bias_nuisance_stream,
    write_visual_bias_nuisance_stream,
)


def _sha(character: str) -> str:
    return character * 64


def _observation_stream(
    *,
    observation_counts: tuple[int, ...] = (2, 3),
    identity_characters: tuple[str, ...] = ("a", "b"),
    frame_stops: tuple[int, ...] = (5, 10),
) -> ObservationFactorStreamV1:
    updates: list[ObservationFactorStreamUpdateV1] = []
    frame_start = 0
    previous_update_id: str | None = None
    for index, (count, identity, frame_stop) in enumerate(
        zip(observation_counts, identity_characters, frame_stops, strict=True)
    ):
        update = ObservationFactorStreamUpdateV1(
            update_index=index,
            admitted_frame_start=frame_start,
            causal_frame_stop=frame_stop,
            bundle_manifest_path=f"bundles/update-{index}.json",
            bundle_manifest_sha256=_sha(chr(ord("c") + index)),
            bundle_payload_sha256=_sha(chr(ord("e") + index)),
            bundle_sequence_id="sequence-1",
            case_id="case-1",
            stream_id="stream-1",
            source_repository="IPS-Stuttgart/Prob4D",
            source_revision="1" * 40,
            factor_count=1,
            observation_count=count,
            persistent_identity_count=count,
            observation_identity_sha256=_sha(identity),
            gauge_ids=(f"gauge-{index}",),
            previous_update_id=previous_update_id,
        )
        updates.append(update)
        previous_update_id = update.update_id
        frame_start = frame_stop
    return ObservationFactorStreamV1(
        sequence_id="sequence-1",
        case_id="case-1",
        stream_id="stream-1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="1" * 40,
        updates=tuple(updates),
        metadata={"protocol": "frozen"},
    )


def _visual_bias_stream(
    observation_stream: ObservationFactorStreamV1,
    *,
    identity_characters: tuple[str, ...] | None = None,
    observation_counts: tuple[int, ...] | None = None,
    frame_intervals: tuple[tuple[int, int], ...] | None = None,
    factor_update_ids: tuple[str, ...] | None = None,
):
    counts = (
        tuple(update.observation_count for update in observation_stream.updates)
        if observation_counts is None
        else observation_counts
    )
    intervals = (
        tuple(
            (update.admitted_frame_start, update.causal_frame_stop)
            for update in observation_stream.updates
        )
        if frame_intervals is None
        else frame_intervals
    )
    update_ids = (
        tuple(str(update.update_id) for update in observation_stream.updates)
        if factor_update_ids is None
        else factor_update_ids
    )
    identities = (
        tuple(update.observation_identity_sha256 for update in observation_stream.updates)
        if identity_characters is None
        else tuple(_sha(character) for character in identity_characters)
    )
    nuisances = []
    for index, (count, identity_sha256) in enumerate(
        zip(counts, identities, strict=True)
    ):
        jacobian = np.zeros((count, 3, 1), dtype=np.float64)
        jacobian[:, 0, 0] = index + 1.0
        nuisances.append(
            VisualBiasNuisanceV1(
                observation_artifact_id=_sha(chr(ord("7") + index)),
                observation_identity_sha256=identity_sha256,
                bias_ids=("camera-0",),
                basis_names=("depth-offset",),
                row_bias_indices=np.zeros(count, dtype=np.int64),
                bias_jacobian=jacobian,
                joint_bias_covariance=np.asarray([[4.0]], dtype=np.float64),
                orthogonalization_semantics=(
                    "conditional-whitened-global-gauge-projection-v1"
                ),
                maximum_gauge_projection=1e-14,
                gauge_projection_tolerance=1e-10,
                metadata={"uses_truth": False},
            )
        )
    return build_visual_bias_nuisance_stream(
        stream_key="sequence-1/camera-0",
        nuisances=tuple(nuisances),
        observation_stream_update_ids=update_ids,
        frame_intervals=intervals,
        model_metadata={"calibration_artifact_id": _sha("9")},
        metadata={"scope": "camera-0"},
    )


def test_binding_captures_exact_cross_stream_correspondence() -> None:
    observation_stream = _observation_stream()
    visual_bias_stream = _visual_bias_stream(observation_stream)

    binding = build_observation_bias_binding(
        observation_stream,
        visual_bias_stream,
        metadata={"consumer": "BayesianPhysTwin"},
    )

    assert binding.observation_factor_stream_artifact_id == observation_stream.artifact_id
    assert binding.visual_bias_stream_artifact_id == visual_bias_stream.artifact_id
    assert binding.observation_count == 5
    assert binding.causal_frame_stop == 10
    assert [update.observation_count for update in binding.updates] == [2, 3]
    assert binding.updates[1].previous_update_id == binding.updates[0].update_id
    assert verify_observation_bias_binding(
        binding,
        observation_stream,
        visual_bias_stream,
    ) is binding


def test_binding_rejects_wrong_observation_update_reference() -> None:
    observation_stream = _observation_stream()
    visual_bias_stream = _visual_bias_stream(
        observation_stream,
        factor_update_ids=(_sha("0"), _sha("1")),
    )
    with pytest.raises(ValueError, match="references another observation update"):
        build_observation_bias_binding(observation_stream, visual_bias_stream)


def test_binding_rejects_frame_row_and_identity_drift() -> None:
    observation_stream = _observation_stream()

    wrong_frames = _visual_bias_stream(
        observation_stream,
        frame_intervals=((0, 4), (4, 10)),
    )
    with pytest.raises(ValueError, match="frame interval differs"):
        build_observation_bias_binding(observation_stream, wrong_frames)

    wrong_rows = _visual_bias_stream(
        observation_stream,
        observation_counts=(1, 4),
    )
    with pytest.raises(ValueError, match="row count differs"):
        build_observation_bias_binding(observation_stream, wrong_rows)

    wrong_identity = _visual_bias_stream(
        observation_stream,
        identity_characters=("a", "c"),
    )
    with pytest.raises(ValueError, match="observation identity differs"):
        build_observation_bias_binding(observation_stream, wrong_identity)


def test_binding_rejects_update_count_mismatch() -> None:
    observation_stream = _observation_stream()
    shorter_observation_stream = _observation_stream(
        observation_counts=(2,),
        identity_characters=("a",),
        frame_stops=(5,),
    )
    visual_bias_stream = _visual_bias_stream(shorter_observation_stream)
    with pytest.raises(ValueError, match="different update counts"):
        build_observation_bias_binding(observation_stream, visual_bias_stream)


def test_roundtrip_and_append_only_rewrite(tmp_path: Path) -> None:
    first_observation_stream = _observation_stream(
        observation_counts=(2,),
        identity_characters=("a",),
        frame_stops=(5,),
    )
    first_visual_bias_stream = _visual_bias_stream(first_observation_stream)
    first = build_observation_bias_binding(
        first_observation_stream,
        first_visual_bias_stream,
        metadata={"consumer": "BayesianPhysTwin"},
    )

    full_observation_stream = _observation_stream()
    full_visual_bias_stream = _visual_bias_stream(full_observation_stream)
    full = build_observation_bias_binding(
        full_observation_stream,
        full_visual_bias_stream,
        metadata={"consumer": "BayesianPhysTwin"},
    )

    path = tmp_path / "binding.json"
    write_observation_bias_binding(first, path)
    write_observation_bias_binding(full, path)
    loaded = load_observation_bias_binding(path)
    assert loaded.artifact_id == full.artifact_id
    assert len(loaded.updates) == 2

    write_observation_bias_binding(full, path)

    forked_update = replace(
        full.updates[0],
        observation_artifact_id=_sha("f"),
        update_id=None,
    )
    forked_second = replace(
        full.updates[1],
        previous_update_id=forked_update.update_id,
        update_id=None,
    )
    forked = replace(
        full,
        updates=(forked_update, forked_second),
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="cannot fork"):
        write_observation_bias_binding(forked, path)


def test_strict_manifest_rejects_duplicate_key_and_identity_tamper(tmp_path: Path) -> None:
    observation_stream = _observation_stream()
    visual_bias_stream = _visual_bias_stream(observation_stream)
    binding = build_observation_bias_binding(observation_stream, visual_bias_stream)
    path = tmp_path / "binding.json"
    write_observation_bias_binding(binding, path)

    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace('"schema":', '"schema": "duplicate", "schema":', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_observation_bias_binding(path)

    write_observation_bias_binding(binding, tmp_path / "fresh.json")
    fresh = tmp_path / "fresh.json"
    record = fresh.read_text(encoding="utf-8").replace("sequence-1", "sequence-2", 1)
    fresh.write_text(record, encoding="utf-8")
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        load_observation_bias_binding(fresh)


def test_path_builder_can_replay_retained_stream_manifests(tmp_path: Path) -> None:
    observation_stream = _observation_stream()
    visual_bias_stream = _visual_bias_stream(observation_stream)
    observation_path = tmp_path / "observation-stream.json"
    visual_path = tmp_path / "visual-bias-stream.json"
    write_observation_factor_stream(observation_stream, observation_path)
    write_visual_bias_nuisance_stream(visual_bias_stream, visual_path)

    binding = build_observation_bias_binding_from_paths(
        observation_path,
        visual_path,
        validate_bundles=False,
    )
    assert binding.observation_count == observation_stream.observation_count
