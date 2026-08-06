from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import prob4d.prediction_provider_import as provider_import
from prob4d.data import PredictionWindow
from prob4d.prediction_provider_import import (
    PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA,
    PREDICTION_PROVIDER_IMPORT_SPEC_VERSION,
    import_prediction_provider_specification,
)
from prob4d.prediction_provider_manifest import (
    PREDICTION_PROVIDER_MANIFEST_VERSION,
    SOURCE_DEPENDENCY_SEMANTICS,
    load_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_window(
    path: Path,
    *,
    window_id: str,
    frames: tuple[int, ...],
) -> None:
    point_map = np.zeros((len(frames), 2, 3, 3), dtype=np.float32)
    point_map[..., 0] = np.arange(len(frames), dtype=np.float32)[:, None, None]
    point_map[..., 2] = 1.0
    valid = np.ones(point_map.shape[:-1], dtype=bool)
    flow = np.full_like(point_map, 0.025)
    path.parent.mkdir(parents=True, exist_ok=True)
    PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray(frames, dtype=np.int64),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=valid,
        dense_storage_dtype="float32",
    ).to_npz(path)


def _lineage(
    frames: tuple[int, ...],
    *,
    source_start: int,
    source_stop: int,
    contributor: str,
) -> list[dict[str, object]]:
    return [
        {
            "output_frame_id": frame,
            "source_frame_start": source_start,
            "source_frame_stop_exclusive": source_stop,
            "contributor_ids": [contributor],
        }
        for frame in frames
    ]


def _payload(
    *,
    window_id: str,
    path: str,
    frames: tuple[int, ...],
    source_start: int,
    source_stop: int,
    seed: int,
) -> dict[str, object]:
    return {
        "product_role": "independent-window",
        "window_id": window_id,
        "path": path,
        "view_id": "camera-0",
        "stochastic_member_id": f"seed-{seed}",
        "dependence_group_ids": [
            "model-set:shared",
            "input-video:shared",
            f"stochastic-member:{seed}",
        ],
        "frame_lineage": _lineage(
            frames,
            source_start=source_start,
            source_stop=source_stop,
            contributor=window_id,
        ),
    }


def _specification(payloads: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA,
        "schema_version": PREDICTION_PROVIDER_IMPORT_SPEC_VERSION,
        "sequence_id": "case-a",
        "provider_family": "Example4D",
        "provider_repository": "example/Example4D",
        "provider_revision": "a" * 40,
        "provider_run_id": "b" * 64,
        "model_set_id": "c" * 64,
        "loader_id": "d" * 64,
        "coordinate_semantics": "window-local-sim3",
        "point_semantics": "dense-point-map",
        "flow_semantics": "forward-point-displacement",
        "ray_semantics": "absent",
        "source_dependency_semantics": SOURCE_DEPENDENCY_SEMANTICS,
        "payloads": payloads,
        "metadata": {
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
        },
    }


def _write_specification(root: Path, value: dict[str, object]) -> Path:
    path = root / "provider-import.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _two_window_bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    first_frames = (0, 1, 2)
    second_frames = (2, 3, 4)
    _write_window(
        root / "windows/window_0000.npz",
        window_id="window_0000",
        frames=first_frames,
    )
    _write_window(
        root / "windows/window_0001.npz",
        window_id="window_0001",
        frames=second_frames,
    )
    specification = _specification(
        [
            _payload(
                window_id="window_0001",
                path="windows/window_0001.npz",
                frames=second_frames,
                source_start=2,
                source_stop=5,
                seed=18,
            ),
            _payload(
                window_id="window_0000",
                path="windows/window_0000.npz",
                frames=first_frames,
                source_start=0,
                source_stop=3,
                seed=17,
            ),
        ]
    )
    return _write_specification(root, specification), root / "provider-neutral.json"


def test_generic_import_roundtrip_and_canonical_order(tmp_path: Path) -> None:
    specification, output = _two_window_bundle(tmp_path)
    manifest = import_prediction_provider_specification(specification, output)

    assert [item.window_id for item in manifest.payloads] == [
        "window_0000",
        "window_0001",
    ]
    assert manifest.metadata["source_adapter"] == (
        "prob4d-external-provider-import-spec-v1"
    )
    assert manifest.metadata["source_import_spec_sha256"] == _sha256(specification)
    assert manifest.metadata["source_import_spec_schema_version"] == (
        PREDICTION_PROVIDER_IMPORT_SPEC_VERSION
    )
    assert manifest.metadata["target_manifest_schema_version"] == (
        PREDICTION_PROVIDER_MANIFEST_VERSION
    )

    loaded = load_prediction_provider_manifest(output)
    assert loaded.artifact_id == manifest.artifact_id
    _, report = verify_prediction_provider_manifest(output, causal_frame_stop=5)
    assert report["verified_payload_count"] == 2
    assert report["admitted_payload_count"] == 2


def test_frame_lineage_must_match_payload_frames(tmp_path: Path) -> None:
    specification, output = _two_window_bundle(tmp_path)
    value = json.loads(specification.read_text(encoding="utf-8"))
    value["payloads"][0]["frame_lineage"][0]["output_frame_id"] = 1
    specification.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage differs"):
        import_prediction_provider_specification(specification, output)


def test_parent_traversal_and_reserved_metadata_are_rejected(tmp_path: Path) -> None:
    specification, output = _two_window_bundle(tmp_path)
    value = json.loads(specification.read_text(encoding="utf-8"))
    value["payloads"][0]["path"] = "../outside.npz"
    specification.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        import_prediction_provider_specification(specification, output)

    specification, output = _two_window_bundle(tmp_path / "reserved")
    value = json.loads(specification.read_text(encoding="utf-8"))
    value["metadata"]["source_adapter"] = "caller-controlled"
    specification.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="reserved fields"):
        import_prediction_provider_specification(specification, output)


def test_output_manifest_must_share_a_confined_payload_root(tmp_path: Path) -> None:
    specification, _ = _two_window_bundle(tmp_path)
    output = tmp_path / "elsewhere/provider-neutral.json"
    with pytest.raises(ValueError, match="must lie inside the manifest directory"):
        import_prediction_provider_specification(specification, output)


def test_payload_mutation_during_import_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification, output = _two_window_bundle(tmp_path)
    source_member = specification.parent / "windows/window_0001.npz"
    original = provider_import.PredictionWindow.from_npz
    mutated = False

    def mutating_loader(
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> PredictionWindow:
        nonlocal mutated
        window = original(path, *args, **kwargs)
        if not mutated:
            source_member.write_bytes(source_member.read_bytes() + b"tamper")
            mutated = True
        return window

    monkeypatch.setattr(
        provider_import.PredictionWindow,
        "from_npz",
        staticmethod(mutating_loader),
    )
    with pytest.raises(ValueError, match="changed during generic import"):
        import_prediction_provider_specification(specification, output)


def test_payload_is_parsed_from_a_private_exact_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification, output = _two_window_bundle(tmp_path)
    source_members = {
        (specification.parent / "windows/window_0000.npz").resolve(),
        (specification.parent / "windows/window_0001.npz").resolve(),
    }
    original = provider_import.PredictionWindow.from_npz
    parsed_paths: list[Path] = []

    def observing_loader(
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> PredictionWindow:
        parsed_paths.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(
        provider_import.PredictionWindow,
        "from_npz",
        staticmethod(observing_loader),
    )
    import_prediction_provider_specification(specification, output)

    assert len(parsed_paths) == 4  # import snapshots plus final manifest verification
    assert all(path not in source_members for path in parsed_paths[:2])
    assert all(path.name == "prediction-window.npz" for path in parsed_paths[:2])


def test_specification_mutation_during_import_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification, output = _two_window_bundle(tmp_path)
    original = provider_import._load_json_bytes

    def mutating_loader(payload: bytes, *, name: str) -> dict[str, object]:
        record = original(payload, name=name)
        value = json.loads(specification.read_text(encoding="utf-8"))
        value["sequence_id"] = "changed"
        specification.write_text(json.dumps(value), encoding="utf-8")
        return record

    monkeypatch.setattr(provider_import, "_load_json_bytes", mutating_loader)
    with pytest.raises(ValueError, match="changed during import"):
        import_prediction_provider_specification(specification, output)


def test_symlink_payload_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    outside = tmp_path / "outside.npz"
    _write_window(outside, window_id="window_0000", frames=(0, 1, 2))
    (root / "windows").mkdir(parents=True)
    try:
        (root / "windows/window_0000.npz").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    specification = _write_specification(
        root,
        _specification(
            [
                _payload(
                    window_id="window_0000",
                    path="windows/window_0000.npz",
                    frames=(0, 1, 2),
                    source_start=0,
                    source_stop=3,
                    seed=17,
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="symbolic link"):
        import_prediction_provider_specification(
            specification,
            root / "provider-neutral.json",
        )


def test_symlink_specification_and_output_are_rejected(tmp_path: Path) -> None:
    specification, output = _two_window_bundle(tmp_path)
    specification_link = tmp_path / "provider-import-link.json"
    output_target = tmp_path / "existing-output.json"
    output_target.write_text("do not replace\n", encoding="utf-8")
    output_link = tmp_path / "provider-output-link.json"
    try:
        specification_link.symlink_to(specification)
        output_link.symlink_to(output_target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="specification is a symbolic link"):
        import_prediction_provider_specification(specification_link, output)
    with pytest.raises(ValueError, match="output manifest is a symbolic link"):
        import_prediction_provider_specification(specification, output_link)
    assert output_target.read_text(encoding="utf-8") == "do not replace\n"
