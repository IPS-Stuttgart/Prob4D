from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from prob4d.lineage import motioncrafter_temporal_lineage_manifest
from prob4d.motioncrafter import (
    MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
    motioncrafter_seed_for_call,
)
from prob4d.motioncrafter_integrity import MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA
from prob4d.source import (
    MOTIONCRAFTER_SOURCE_ADAPTER_SCHEMA,
    WINDOWED_4D_SOURCE_MANIFEST_SCHEMA,
    WINDOWED_4D_SOURCE_MANIFEST_VERSION,
    Windowed4DSourceManifestV1,
    Windowed4DSourceWindowV1,
    adapt_motioncrafter_prediction_manifest,
    load_motioncrafter_source_manifest,
    load_windowed_4d_source_manifest,
    save_windowed_4d_source_manifest,
)


def _call(
    root_seed: int,
    *,
    policy: str,
    call_id: str,
    product: str,
    **metadata: object,
) -> dict[str, object]:
    return {
        "call_id": call_id,
        "product": product,
        "effective_seed": motioncrafter_seed_for_call(
            root_seed,
            policy=policy,  # type: ignore[arg-type]
            call_id=call_id,
        ),
        **metadata,
    }


def _motioncrafter_manifest(*, integrity: bool = True) -> dict[str, object]:
    seed = 17
    policy = MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL
    windows = [
        {
            "window_id": "window_0000",
            "path": "windows/window_0000.npz",
            "start_frame": 100,
            "stop_frame": 125,
        },
        {
            "window_id": "window_0001",
            "path": "windows/window_0001.npz",
            "start_frame": 117,
            "stop_frame": 142,
        },
    ]
    calls = [
        _call(
            seed,
            policy=policy,
            call_id="baseline-disjoint",
            product="disjoint_baseline",
        ),
        _call(
            seed,
            policy=policy,
            call_id="baseline-latent-linear",
            product="latent_linear_baseline",
        ),
    ]
    for window in windows:
        window_id = str(window["window_id"])
        start = int(window["start_frame"])
        stop = int(window["stop_frame"])
        calls.append(
            _call(
                seed,
                policy=policy,
                call_id=f"overlap-window:{window_id}:{start}:{stop}",
                product="independently_decoded_overlap_window",
                window_id=window_id,
                source_frame_start=start,
                source_frame_stop_exclusive=stop,
            )
        )
    manifest: dict[str, object] = {
        "format_version": 1,
        "video_path": "/nonportable/source.mp4",
        "motioncrafter_commit": "a" * 40,
        "config": {
            "window_size": 25,
            "overlap": 8,
            "frame_stride": 1,
            "seed": seed,
            "seed_policy": policy,
            "model_source_schema": "prob4d.motioncrafter-model-set.v2",
            "model_source_set_sha256": "b" * 64,
        },
        "stochastic_seed_schedule": {
            "schema": MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
            "policy": policy,
            "root_seed": seed,
            "calls": calls,
            "interpretation": "test fixture",
        },
        "temporal_lineage": motioncrafter_temporal_lineage_manifest(
            window_size=25,
            overlap=8,
        ),
        "overlap_windows": windows,
        "disjoint_baseline": "baseline_disjoint.npz",
        "latent_linear_baseline": "baseline_latent_linear.npz",
    }
    if integrity:
        members = [
            {
                "path": "baseline_disjoint.npz",
                "sha256": "c" * 64,
                "bytes": 101,
                "kind": "disjoint_baseline",
            },
            {
                "path": "baseline_latent_linear.npz",
                "sha256": "d" * 64,
                "bytes": 102,
                "kind": "latent_linear_baseline",
            },
            {
                "path": "windows/window_0000.npz",
                "sha256": "e" * 64,
                "bytes": 103,
                "kind": "independently_decoded_overlap_window",
            },
            {
                "path": "windows/window_0001.npz",
                "sha256": "f" * 64,
                "bytes": 104,
                "kind": "independently_decoded_overlap_window",
            },
        ]
        manifest["artifact_integrity"] = {
            "schema": MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA,
            "run_spec_sha256": "1" * 64,
            "members": members,
        }
    return manifest


def test_motioncrafter_adapter_builds_provider_neutral_contract() -> None:
    source = adapt_motioncrafter_prediction_manifest(_motioncrafter_manifest())

    assert source.schema_name == WINDOWED_4D_SOURCE_MANIFEST_SCHEMA
    assert source.schema_version == WINDOWED_4D_SOURCE_MANIFEST_VERSION
    assert source.source_provider_id == "motioncrafter"
    assert source.source_provider_revision == "a" * 40
    assert source.model_set_id == "prob4d.motioncrafter-model-set.v2:" + "b" * 64
    assert source.coordinate_frame == "window-local-sim3-gauge"
    assert source.length_unit == "provider-native-unscaled"
    assert source.geometry.nominal_window_size == 25
    assert source.geometry.nominal_overlap == 8
    assert source.geometry.frame_stride == 1
    assert tuple(window.window_id for window in source.windows) == (
        "window_0000",
        "window_0001",
    )
    assert source.windows[0].payload_sha256 == "e" * 64
    assert source.windows[0].payload_bytes == 103
    assert source.stochastic_policy["policy"] == "derived-per-call"
    assert source.provider_metadata["adapter_schema"] == MOTIONCRAFTER_SOURCE_ADAPTER_SCHEMA
    assert source.provider_metadata["artifact_integrity_bound"] is True
    assert source.source_manifest_file_sha256 is None
    assert source.claim_ready_source_identity is False


def test_file_adapter_binds_exact_manifest_bytes_and_is_claim_ready(tmp_path: Path) -> None:
    manifest = _motioncrafter_manifest()
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    source = load_motioncrafter_source_manifest(path)

    assert source.source_manifest_file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source.claim_ready_source_identity is True
    assert source.source_manifest_sha256 == adapt_motioncrafter_prediction_manifest(
        manifest
    ).source_manifest_sha256


def test_legacy_motioncrafter_manifest_is_explicitly_not_claim_ready() -> None:
    manifest = _motioncrafter_manifest(integrity=False)
    config = manifest["config"]
    assert isinstance(config, dict)
    config.pop("seed_policy")
    config.pop("model_source_schema")
    config.pop("model_source_set_sha256")
    manifest.pop("stochastic_seed_schedule")
    manifest.pop("temporal_lineage")

    source = adapt_motioncrafter_prediction_manifest(manifest)

    assert source.model_set_id is None
    assert source.stochastic_policy["policy"] == MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON
    assert source.stochastic_policy["schedule_source"] == "implicit-legacy-common"
    assert source.provider_metadata["temporal_lineage_source"] == (
        "reconstructed-from-legacy-config"
    )
    assert source.temporal_lineage["schema_version"] == 1
    assert all(not window.payload_identity_bound for window in source.windows)
    assert source.claim_ready_source_identity is False


def test_source_manifest_round_trip_is_content_addressed_and_append_only(
    tmp_path: Path,
) -> None:
    artifact = adapt_motioncrafter_prediction_manifest(_motioncrafter_manifest())
    path = tmp_path / "source.json"

    save_windowed_4d_source_manifest(artifact, path)
    save_windowed_4d_source_manifest(artifact, path)
    loaded = load_windowed_4d_source_manifest(path)

    assert loaded == artifact
    assert loaded.artifact_id == artifact.artifact_id
    conflicting = replace(
        artifact,
        provider_metadata={"adapter_schema": "different"},
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        save_windowed_4d_source_manifest(conflicting, path)


def test_source_manifest_rejects_tampered_content_address(tmp_path: Path) -> None:
    artifact = adapt_motioncrafter_prediction_manifest(_motioncrafter_manifest())
    payload = artifact.to_dict()
    payload["artifact_id"] = "0" * 64
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_id does not match"):
        load_windowed_4d_source_manifest(path)


def test_source_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text('{"artifact_id":"0","artifact_id":"1"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_windowed_4d_source_manifest(path)


def test_motioncrafter_adapter_rejects_unsafe_or_inconsistent_windows() -> None:
    unsafe = _motioncrafter_manifest()
    windows = unsafe["overlap_windows"]
    assert isinstance(windows, list)
    assert isinstance(windows[0], dict)
    windows[0]["path"] = "../escape.npz"
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        adapt_motioncrafter_prediction_manifest(unsafe)

    duplicate = _motioncrafter_manifest()
    duplicate_windows = duplicate["overlap_windows"]
    assert isinstance(duplicate_windows, list)
    assert isinstance(duplicate_windows[1], dict)
    duplicate_windows[1]["window_id"] = "window_0000"
    with pytest.raises(ValueError):
        adapt_motioncrafter_prediction_manifest(duplicate)

    missing_identity = _motioncrafter_manifest()
    integrity = missing_identity["artifact_integrity"]
    assert isinstance(integrity, dict)
    members = integrity["members"]
    assert isinstance(members, list)
    integrity["members"] = members[:-1]
    with pytest.raises(ValueError, match="lacks descriptor"):
        adapt_motioncrafter_prediction_manifest(missing_identity)


def test_window_contract_rejects_partial_identity_and_unsafe_path() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        Windowed4DSourceWindowV1(
            window_id="window",
            payload_path="window.npz",
            source_frame_start=0,
            source_frame_stop_exclusive=25,
            payload_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        Windowed4DSourceWindowV1(
            window_id="window",
            payload_path="/absolute/window.npz",
            source_frame_start=0,
            source_frame_stop_exclusive=25,
        )


def test_source_contract_defensively_freezes_nested_metadata() -> None:
    manifest = _motioncrafter_manifest()
    source = adapt_motioncrafter_prediction_manifest(manifest)

    with pytest.raises(TypeError, match="immutable"):
        source.provider_metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        source.stochastic_policy["policy"] = "changed"  # type: ignore[index]

    mutated = copy.deepcopy(manifest)
    assert isinstance(mutated["config"], dict)
    mutated["config"]["window_size"] = 99
    assert source.geometry.nominal_window_size == 25


def test_motioncrafter_adapter_rejects_noncanonical_scalar_aliases() -> None:
    manifest = _motioncrafter_manifest()
    manifest["format_version"] = True
    with pytest.raises(ValueError, match="format_version must be an integer"):
        adapt_motioncrafter_prediction_manifest(manifest)

    manifest = _motioncrafter_manifest()
    config = manifest["config"]
    assert isinstance(config, dict)
    config["window_size"] = 25.0
    with pytest.raises(ValueError, match="window_size must be an integer"):
        adapt_motioncrafter_prediction_manifest(manifest)


def test_motioncrafter_file_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        '{"format_version":1,"format_version":1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_motioncrafter_source_manifest(path)


def test_motioncrafter_adapter_rejects_coercive_seed_fields() -> None:
    manifest = _motioncrafter_manifest()
    config = manifest["config"]
    assert isinstance(config, dict)
    config["seed"] = True
    with pytest.raises(ValueError, match="config.seed must be an integer"):
        adapt_motioncrafter_prediction_manifest(manifest)

    manifest = _motioncrafter_manifest()
    schedule = manifest["stochastic_seed_schedule"]
    assert isinstance(schedule, dict)
    calls = schedule["calls"]
    assert isinstance(calls, list)
    assert isinstance(calls[0], dict)
    calls[0]["effective_seed"] = float(calls[0]["effective_seed"])
    with pytest.raises(ValueError, match="effective_seed must be an integer"):
        adapt_motioncrafter_prediction_manifest(manifest)


def test_source_manifest_requires_typed_nested_contracts() -> None:
    artifact = adapt_motioncrafter_prediction_manifest(_motioncrafter_manifest())

    with pytest.raises(ValueError, match="geometry must be"):
        replace(artifact, geometry={"nominal_window_size": 25})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="data_semantics must be"):
        replace(artifact, data_semantics={})  # type: ignore[arg-type]
