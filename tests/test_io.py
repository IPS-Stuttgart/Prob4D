import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.fusion import FusedSequence, fuse_windows
from prob4d.io import (
    load_fused_prediction_artifact,
    load_fused_prediction_metadata,
    load_prediction_bundle,
    load_truth,
    pack_symmetric_covariance,
    save_fused_prediction,
    save_truth,
    unpack_symmetric_covariance,
)
from prob4d.sim3 import Sim3
from prob4d.synthetic import SyntheticProblem, make_synthetic_problem
from prob4d.uncertainty import DepthDisagreementModel


def _write_sequence(path: Path, sequence: FusedSequence) -> None:
    payload = {
        "frame_indices": sequence.frame_indices,
        "point_map": sequence.point_map,
        "valid_mask": sequence.valid_mask,
    }
    if sequence.scene_flow is not None:
        payload["scene_flow"] = sequence.scene_flow
        payload["deform_mask"] = sequence.deform_mask
    np.savez_compressed(path, **payload)


def write_problem_bundle(root: Path, problem: SyntheticProblem) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    windows_directory = root / "windows"
    windows_directory.mkdir()
    manifest_windows = []
    for window in problem.overlap_windows:
        relative = Path("windows") / f"{window.window_id}.npz"
        window.to_npz(root / relative)
        manifest_windows.append(
            {
                "window_id": window.window_id,
                "path": relative.as_posix(),
                "start_frame": window.start_frame,
                "stop_frame": window.stop_frame,
            }
        )
    model = DepthDisagreementModel()
    disjoint = fuse_windows(
        problem.disjoint_windows,
        {window.window_id: Sim3.identity() for window in problem.disjoint_windows},
        {window.window_id: model.predict(window) for window in problem.disjoint_windows},
        method="uniform",
    )
    latent = fuse_windows(
        problem.overlap_windows,
        {window.window_id: Sim3.identity() for window in problem.overlap_windows},
        {window.window_id: model.predict(window) for window in problem.overlap_windows},
        method="uniform",
    )
    _write_sequence(root / "baseline_disjoint.npz", disjoint)
    _write_sequence(root / "baseline_latent_linear.npz", latent)
    manifest = {
        "format_version": 1,
        "motioncrafter_commit": "synthetic-test",
        "overlap_windows": manifest_windows,
        "disjoint_baseline": "baseline_disjoint.npz",
        "latent_linear_baseline": "baseline_latent_linear.npz",
    }
    manifest_path = root / "predictions.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    truth_path = root / "truth.npz"
    save_truth(truth_path, problem.truth)
    return manifest_path, truth_path


def test_prediction_bundle_and_truth_round_trip(tmp_path: Path) -> None:
    problem = make_synthetic_problem(seed=21, num_frames=40, height=4, width=6, overlap=15)
    manifest_path, truth_path = write_problem_bundle(tmp_path / "bundle", problem)

    bundle = load_prediction_bundle(manifest_path)
    truth = load_truth(truth_path)

    assert len(bundle.overlap_windows) == len(problem.overlap_windows)
    assert bundle.metadata["motioncrafter_commit"] == "synthetic-test"
    np.testing.assert_allclose(truth.point_map, problem.truth.point_map, rtol=1e-6)


def test_prediction_bundle_rejects_explicit_derived_policy_without_schedule(
    tmp_path: Path,
) -> None:
    problem = make_synthetic_problem(seed=22, num_frames=40, height=4, width=6, overlap=15)
    manifest_path, _ = write_problem_bundle(tmp_path / "bundle", problem)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = {
        "seed": 42,
        "seed_policy": "derived-per-call",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks a seed schedule"):
        load_prediction_bundle(manifest_path)


def test_symmetric_covariance_pack_round_trip() -> None:
    covariance = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    covariance = covariance + np.swapaxes(covariance, -1, -2)

    restored = unpack_symmetric_covariance(pack_symmetric_covariance(covariance))
    np.testing.assert_array_equal(restored, covariance)


def test_fused_prediction_metadata_round_trip(tmp_path: Path) -> None:
    sequence = FusedSequence(
        frame_indices=np.array([0]),
        point_map=np.array([[[[0.0, 0.0, 1.0]]]]),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
        point_covariance=np.eye(3)[None, None, None],
        contributors=np.ones((1, 1, 1), dtype=np.uint16),
    )
    path = tmp_path / "fused.npz"

    save_fused_prediction(
        path,
        sequence,
        method_id="ci",
        fusion_method="covariance_intersection",
        metadata={"calibration": "held-out"},
    )

    metadata = load_fused_prediction_metadata(path)
    assert metadata.method_id == "ci"
    assert metadata.metadata == {"calibration": "held-out"}
    with pytest.raises(TypeError, match="immutable"):
        metadata.metadata["calibration"] = "changed"
    artifact = load_fused_prediction_artifact(path)
    np.testing.assert_allclose(artifact.sequence.point_map, sequence.point_map, rtol=1e-3)
    assert not artifact.sequence.frame_indices.flags.writeable
    assert not artifact.sequence.point_map.flags.writeable
    assert not artifact.sequence.valid_mask.flags.writeable
    assert not artifact.sequence.point_covariance.flags.writeable
    assert not artifact.sequence.contributors.flags.writeable


def test_legacy_fused_prediction_is_explicitly_unspecified(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        frame_indices=np.array([0]),
        point_map=np.array([[[[0.0, 0.0, 1.0]]]]),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
        point_covariance_packed=pack_symmetric_covariance(
            np.eye(3)[None, None, None]
        ),
        contributors=np.ones((1, 1, 1), dtype=np.uint16),
    )

    metadata = load_fused_prediction_metadata(path)

    assert metadata.legacy_unspecified
    assert metadata.fusion_method == "unspecified"


def test_prediction_bundle_explicit_float32_storage_is_auditable(
    tmp_path: Path,
) -> None:
    problem = make_synthetic_problem(
        seed=23,
        num_frames=40,
        height=4,
        width=6,
        overlap=15,
    )
    manifest_path, _ = write_problem_bundle(tmp_path / "bundle", problem)

    legacy = load_prediction_bundle(manifest_path)
    compact = load_prediction_bundle(
        manifest_path,
        dense_storage_dtype="float32",
    )
    legacy_summary = legacy.dense_storage_summary()
    compact_summary = compact.dense_storage_summary()

    assert legacy_summary["storage_dtypes"] == ["float64"]
    assert compact_summary["storage_dtypes"] == ["float32"]
    assert compact_summary["retained_fraction_of_float64"] == 0.5
    assert compact_summary["retained_bytes"] * 2 == legacy_summary["retained_bytes"]
    for expected, actual in zip(
        legacy.overlap_windows,
        compact.overlap_windows,
        strict=True,
    ):
        np.testing.assert_array_equal(actual.point_map, expected.point_map)
        if expected.scene_flow is not None:
            np.testing.assert_array_equal(actual.scene_flow, expected.scene_flow)
