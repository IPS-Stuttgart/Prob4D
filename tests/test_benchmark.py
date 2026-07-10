from pathlib import Path

import numpy as np
from test_io import write_problem_bundle

from prob4d.benchmark import _write_fused_prediction, fuse_prediction_bundle
from prob4d.io import load_fused_prediction, load_prediction_bundle
from prob4d.synthetic import make_synthetic_problem


def test_fuse_prediction_bundle_exports_uniform_and_ci(tmp_path: Path) -> None:
    problem = make_synthetic_problem(
        seed=71,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    manifest, _ = write_problem_bundle(tmp_path / "bundle", problem)
    bundle = load_prediction_bundle(manifest)

    uniform, covariance_intersection = fuse_prediction_bundle(bundle)

    np.testing.assert_array_equal(uniform.frame_indices, problem.truth.frame_indices)
    np.testing.assert_array_equal(
        covariance_intersection.frame_indices, problem.truth.frame_indices
    )
    assert np.max(covariance_intersection.contributors) > 1


def test_fused_prediction_can_persist_compact_covariance(tmp_path: Path) -> None:
    problem = make_synthetic_problem(
        seed=72,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    manifest, _ = write_problem_bundle(tmp_path / "bundle", problem)
    uniform, _ = fuse_prediction_bundle(load_prediction_bundle(manifest))
    path = tmp_path / "prediction.npz"

    _write_fused_prediction(path, uniform, include_covariance=True)

    with np.load(path) as payload:
        assert payload["point_covariance_packed"].shape == uniform.valid_mask.shape + (6,)
        assert payload["contributors"].shape == uniform.valid_mask.shape
    restored = load_fused_prediction(path)
    np.testing.assert_allclose(restored.point_covariance, uniform.point_covariance, rtol=1e-6)
