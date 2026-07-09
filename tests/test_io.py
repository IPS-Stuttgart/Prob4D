import json
from pathlib import Path

import numpy as np

from prob4d.fusion import FusedSequence, fuse_windows
from prob4d.io import load_prediction_bundle, load_truth, save_truth
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
