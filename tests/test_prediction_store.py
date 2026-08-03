from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.benchmark import fuse_prediction_bundle_methods
from prob4d.data import PredictionWindow
from prob4d.fusion import FusedSequence, fuse_windows
from prob4d.io import load_prediction_bundle
from prob4d.prediction_store import (
    MMapPredictionWindow,
    load_prediction_bundle_store,
    load_prediction_window_store,
    materialize_prediction_bundle_store,
    prediction_bundle_store_summary,
    write_prediction_window_store,
)
from prob4d.sim3 import Sim3
from prob4d.synthetic import SyntheticProblem, make_synthetic_problem
from prob4d.uncertainty import DepthDisagreementModel


def _write_sequence(path: Path, sequence: FusedSequence) -> None:
    payload: dict[str, np.ndarray] = {
        "frame_indices": sequence.frame_indices,
        "point_map": sequence.point_map,
        "valid_mask": sequence.valid_mask,
    }
    if sequence.scene_flow is not None:
        payload["scene_flow"] = sequence.scene_flow
        payload["deform_mask"] = sequence.deform_mask
    np.savez_compressed(path, **payload)


def _write_problem_bundle(root: Path, problem: SyntheticProblem) -> Path:
    root.mkdir(parents=True)
    windows_directory = root / "windows"
    windows_directory.mkdir()
    manifest_windows: list[dict[str, object]] = []
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
    return manifest_path


def test_prediction_window_store_opens_read_only_memory_maps(tmp_path: Path) -> None:
    point_map = np.arange(72, dtype=np.float64).reshape(2, 3, 4, 3)
    valid = np.ones((2, 3, 4), dtype=bool)
    window = PredictionWindow(
        window_id="window-a",
        frame_indices=np.asarray([3, 4], dtype=np.int64),
        point_map=point_map,
        valid_mask=valid,
    )

    manifest = write_prediction_window_store(
        window,
        tmp_path / "window-store",
        dense_storage_dtype="float32",
    )
    loaded = load_prediction_window_store(tmp_path / "window-store")

    assert manifest.is_file()
    assert isinstance(loaded, MMapPredictionWindow)
    assert loaded.dense_storage_dtype == "float32"
    assert isinstance(loaded.point_map.base, np.memmap)
    assert not loaded.point_map.flags.writeable
    np.testing.assert_array_equal(loaded.frame_indices, window.frame_indices)
    np.testing.assert_allclose(loaded.point_map, window.point_map)


def test_prediction_window_store_detects_member_tampering(tmp_path: Path) -> None:
    window = PredictionWindow(
        window_id="window-a",
        frame_indices=np.asarray([0], dtype=np.int64),
        point_map=np.ones((1, 1, 1, 3), dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    write_prediction_window_store(window, tmp_path / "window-store")
    member = tmp_path / "window-store" / "point_map.npy"
    payload = bytearray(member.read_bytes())
    payload[-1] ^= 1
    member.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_prediction_window_store(tmp_path / "window-store")


def test_prediction_bundle_store_matches_eager_fusion(tmp_path: Path) -> None:
    problem = make_synthetic_problem(
        seed=31,
        num_frames=30,
        height=2,
        width=3,
        overlap=10,
    )
    source_manifest = _write_problem_bundle(tmp_path / "bundle", problem)
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    eager = load_prediction_bundle(source_manifest, dense_storage_dtype="float32")

    store_manifest = materialize_prediction_bundle_store(
        source_manifest,
        tmp_path / "bundle-store",
        dense_storage_dtype="float32",
    )
    stored = load_prediction_bundle_store(tmp_path / "bundle-store")

    assert store_manifest.is_file()
    assert stored.metadata["prediction_execution_store"][
        "source_manifest_sha256"
    ] == source_digest
    assert all(
        isinstance(window.point_map.base, np.memmap)
        for window in stored.overlap_windows
    )
    assert stored.dense_storage_summary() == eager.dense_storage_summary()

    methods = ("prob4d_uniform", "prob4d_ci")
    eager_fused = fuse_prediction_bundle_methods(eager, method_names=methods)
    stored_fused = fuse_prediction_bundle_methods(stored, method_names=methods)
    for method in methods:
        np.testing.assert_allclose(
            stored_fused[method].point_map,
            eager_fused[method].point_map,
        )
        np.testing.assert_allclose(
            stored_fused[method].point_covariance,
            eager_fused[method].point_covariance,
        )
        np.testing.assert_array_equal(
            stored_fused[method].contributors,
            eager_fused[method].contributors,
        )

    summary = prediction_bundle_store_summary(tmp_path / "bundle-store")
    assert summary["source_manifest_sha256"] == source_digest
    assert summary["storage_dtypes"] == ["float32"]


def test_prediction_bundle_store_is_create_once(tmp_path: Path) -> None:
    problem = make_synthetic_problem(
        seed=32,
        num_frames=20,
        height=1,
        width=2,
        overlap=5,
    )
    source_manifest = _write_problem_bundle(tmp_path / "bundle", problem)
    destination = tmp_path / "bundle-store"
    materialize_prediction_bundle_store(source_manifest, destination)

    with pytest.raises(ValueError, match="already exists"):
        materialize_prediction_bundle_store(source_manifest, destination)


def test_prediction_store_grouped_cli_materializes_and_validates(
    tmp_path: Path,
    capsys,
) -> None:
    from prob4d.cli import main as cli_main

    problem = make_synthetic_problem(
        seed=33,
        num_frames=20,
        height=1,
        width=2,
        overlap=5,
    )
    source_manifest = _write_problem_bundle(tmp_path / "bundle", problem)
    destination = tmp_path / "bundle-store"

    assert (
        cli_main(
            [
                "storage",
                "materialize",
                str(source_manifest),
                str(destination),
            ]
        )
        == 0
    )
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["storage_dtypes"] == ["float32"]

    assert cli_main(["storage", "validate", str(destination)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["store_id"] == materialized["store_id"]


def test_prediction_store_benchmark_reports_explicit_backends(tmp_path: Path) -> None:
    from argparse import Namespace

    from prob4d.prediction_store_benchmark import run_benchmark

    problem = make_synthetic_problem(
        seed=34,
        num_frames=20,
        height=1,
        width=2,
        overlap=5,
    )
    source_manifest = _write_problem_bundle(tmp_path / "bundle", problem)
    destination = tmp_path / "bundle-store"
    materialize_prediction_bundle_store(source_manifest, destination)

    eager = run_benchmark(
        Namespace(
            backend="eager_npz",
            input=source_manifest,
            dense_storage_dtype="float32",
        )
    )
    mapped = run_benchmark(
        Namespace(
            backend="mmap_npy",
            input=destination,
            dense_storage_dtype="float32",
        )
    )

    assert eager["configuration"]["backend"] == "eager_npz"
    assert mapped["configuration"]["backend"] == "mmap_npy"
    assert eager["identity"]["source_manifest_sha256"] == mapped["identity"][
        "source_manifest_sha256"
    ]
    assert mapped["identity"]["store_id"] is not None
    assert eager["memory_bytes"]["retained_dense_vectors"] == mapped[
        "memory_bytes"
    ]["retained_dense_vectors"]
