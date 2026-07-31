from pathlib import Path

import numpy as np
from test_io import write_problem_bundle

from prob4d.benchmark import (
    _write_fused_prediction,
    benchmark_method_semantics,
    fuse_prediction_bundle,
    fuse_prediction_bundle_methods,
)
from prob4d.io import load_fused_prediction, load_fused_prediction_artifact, load_prediction_bundle
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


def test_fuse_prediction_bundle_exports_smoothed_uniform(tmp_path: Path) -> None:
    problem = make_synthetic_problem(
        seed=73,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    manifest, _ = write_problem_bundle(tmp_path / "bundle", problem)

    methods = fuse_prediction_bundle_methods(
        load_prediction_bundle(manifest),
        method_names={"prob4d_uniform_smoothed"},
    )

    assert set(methods) == {"prob4d_uniform_smoothed"}
    smoothed_uniform = methods["prob4d_uniform_smoothed"]
    np.testing.assert_array_equal(smoothed_uniform.frame_indices, problem.truth.frame_indices)
    assert np.max(smoothed_uniform.contributors) > 1


def test_fused_prediction_can_persist_compact_covariance_and_semantics(
    tmp_path: Path,
) -> None:
    problem = make_synthetic_problem(
        seed=72,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    manifest, _ = write_problem_bundle(tmp_path / "bundle", problem)
    _, covariance_intersection = fuse_prediction_bundle(load_prediction_bundle(manifest))
    path = tmp_path / "prediction.npz"

    _write_fused_prediction(
        path,
        covariance_intersection,
        method_id="prob4d_ci_smoothed_uncalibrated",
        include_covariance=True,
        metadata={"prob4d_revision": "a" * 40},
    )

    with np.load(path) as payload:
        assert payload["point_covariance_packed"].shape == (
            covariance_intersection.valid_mask.shape + (6,)
        )
        assert payload["contributors"].shape == covariance_intersection.valid_mask.shape
        assert payload["artifact_schema"].item() == "prob4d.fused-prediction"
        assert payload["method_id"].item() == "prob4d_ci_smoothed_uncalibrated"
    artifact = load_fused_prediction_artifact(path)
    assert artifact.metadata.fusion_method == "covariance_intersection"
    assert artifact.metadata.covariance_semantics == (
        "unknown_correlation_consistency_bound"
    )
    assert artifact.metadata.metadata["gauge_estimator"] == "fixed_lag"
    assert artifact.metadata.metadata["prob4d_revision"] == "a" * 40
    restored = load_fused_prediction(path)
    np.testing.assert_allclose(
        restored.point_covariance,
        covariance_intersection.point_covariance,
        rtol=1e-6,
    )


def test_benchmark_method_semantics_distinguish_covariance_meanings() -> None:
    uniform = benchmark_method_semantics("prob4d_uniform")
    precision = benchmark_method_semantics("prob4d_precision")
    covariance_intersection = benchmark_method_semantics("prob4d_ci")

    assert uniform["covariance_semantics"] == "gaussian_mixture_second_moment"
    assert precision["correlation_assumption"] == "contributors_treated_as_independent"
    assert covariance_intersection["covariance_semantics"] == (
        "unknown_correlation_consistency_bound"
    )
