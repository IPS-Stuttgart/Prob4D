from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prob4d.cut3r_bayesian_prefix_dev import (
    ARMS,
    NOISE_STD,
    PRIOR_STD,
    Rows,
    bayesian_residual,
    kernel,
    last_residual_shift,
    predict_arms,
    score_arms,
)
from prob4d.dot_rope_cut3r_study import parse_coordinate_text


def _observations() -> tuple[Rows, np.ndarray, np.ndarray]:
    points = np.asarray([[0.0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]])
    rows = Rows(np.asarray([3, 4, 5]), np.asarray([0, 1, 2]), points)
    residual = np.asarray([[0.05, 0, 0], [0.04, 0.01, 0], [0.06, 0, 0]])
    return rows, residual, points + 0.01


@pytest.mark.parametrize("correlation", [0.0, 0.8])
def test_existing_operator_matches_dense_conditioning(correlation: float) -> None:
    rows, residual, query = _observations()
    mean, covariance = bayesian_residual(rows, residual, query, shared_correlation=correlation)
    noise = NOISE_STD**2 * ((1 - correlation) * np.eye(3) + correlation * np.ones((3, 3)))
    innovation = kernel(rows.points, rows.points) + noise
    cross = kernel(query, rows.points)
    expected_mean = cross @ np.linalg.solve(innovation, residual)
    expected_variance = PRIOR_STD**2 - np.sum(
        cross * np.linalg.solve(innovation, cross.T).T, axis=1
    )
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(
        covariance, (expected_variance + NOISE_STD**2)[:, None, None] * np.eye(3), atol=1e-12
    )
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_shared_noise_is_consumed_and_zero_innovation_stays_zero() -> None:
    rows, residual, query = _observations()
    independent = bayesian_residual(rows, residual, query, shared_correlation=0)
    shared = bayesian_residual(rows, residual, query, shared_correlation=0.8)
    assert not np.allclose(independent[0], shared[0])
    assert not np.allclose(independent[1], shared[1])
    mean, _ = bayesian_residual(rows, np.zeros_like(residual), query, shared_correlation=0.8)
    np.testing.assert_array_equal(mean, 0)


def test_duplicate_evidence_does_not_change_posterior() -> None:
    rows, residual, query = _observations()
    repeated = Rows(
        np.tile(rows.frame, 10), np.tile(rows.identity, 10), np.tile(rows.points, (10, 1))
    )
    first = bayesian_residual(rows, residual, query, shared_correlation=0.8)
    second = bayesian_residual(repeated, np.tile(residual, (10, 1)), query, shared_correlation=0.8)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_conflicting_duplicate_and_future_observation_rejected() -> None:
    rows, residual, query = _observations()
    conflict = Rows(np.asarray([3, 3, 5]), np.asarray([0, 0, 2]), rows.points)
    with pytest.raises(ValueError, match="conflicting"):
        bayesian_residual(conflict, residual, query, shared_correlation=0.8)
    future = Rows(np.asarray([3, 4, 6]), rows.identity, rows.points)
    with pytest.raises(ValueError, match="future"):
        bayesian_residual(future, residual, query, shared_correlation=0.8)


def test_last_valid_residual_not_only_final_frame() -> None:
    rows = Rows(np.asarray([5, 3, 4]), np.asarray([0, 1, 0]), np.zeros((3, 3)))
    residual = np.asarray([[5.0, 0, 0], [3, 0, 0], [4, 0, 0]])
    result = last_residual_shift(rows, residual, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(result, [[5, 0, 0], [3, 0, 0], [0, 0, 0]])


def _prefix() -> tuple[Rows, np.ndarray, Rows]:
    points = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    prefix = Rows(np.repeat(np.arange(1, 6), 3), np.tile(np.arange(3), 5), np.tile(points, (5, 1)))
    truth = prefix.points.copy()
    truth[prefix.frame >= 3, 2] += 0.05
    query = Rows(np.repeat([6, 7], 3), np.tile(np.arange(3), 2), np.tile(points, (2, 1)))
    return prefix, truth, query


def test_matched_controls_and_prediction_shape() -> None:
    prefix, truth, query = _prefix()
    predictions = predict_arms(prefix, truth, query)
    assert set(predictions["means"]) == set(ARMS)
    future = query.points + np.asarray([0, 0, 0.05])
    metrics = score_arms(predictions, query, future)
    assert metrics["last_residual"]["rmse_prefix_span"] < 1e-12
    assert (
        metrics["bayesian_shared"]["rmse_prefix_span"]
        < metrics["cut3r_initial_alignment"]["rmse_prefix_span"]
    )
    assert not predictions["bayesian_fallback"]
    for name in ARMS:
        assert predictions["means"][name].shape == (6, 3)
        assert predictions["covariances"][name].shape == (6, 3, 3)


def test_insufficient_update_is_exact_initial_fallback() -> None:
    prefix, truth, query = _prefix()
    selected = prefix.frame <= 2
    prediction = predict_arms(
        Rows(prefix.frame[selected], prefix.identity[selected], prefix.points[selected]),
        truth[selected],
        query,
    )
    assert prediction["bayesian_fallback"]
    for name in ("bayesian_iid", "bayesian_shared"):
        np.testing.assert_array_equal(
            prediction["means"][name], prediction["means"]["cut3r_initial_alignment"]
        )
        np.testing.assert_array_equal(
            prediction["covariances"][name], prediction["covariances"]["cut3r_initial_alignment"]
        )


def test_prefix_future_truth_boundary_and_score_support() -> None:
    prefix, truth, query = _prefix()
    with pytest.raises(ValueError, match="time boundary"):
        predict_arms(Rows(prefix.frame + 1, prefix.identity, prefix.points), truth, query)
    predictions = predict_arms(prefix, truth, query)
    future = query.points.copy()
    future[query.frame == 7] = np.nan
    with pytest.raises(ValueError, match="two supported"):
        score_arms(predictions, query, future)


def test_metrics_use_equal_frame_not_coordinate_pooling() -> None:
    query = Rows(np.asarray([6, 7, 7, 7]), np.arange(4), np.zeros((4, 3)))
    means = {name: np.zeros((4, 3)) for name in ARMS}
    covariance = {name: np.tile(np.eye(3), (4, 1, 1)) for name in ARMS}
    prediction = {
        "means": means,
        "covariances": covariance,
        "normalization_center": np.zeros(3),
        "normalization_span": 1.0,
    }
    truth = np.asarray([[1.0, 0, 0], [3, 0, 0], [3, 0, 0], [3, 0, 0]])
    result = score_arms(prediction, query, truth)
    for row in result.values():
        assert row["rmse_prefix_span"] == 2.0
        assert row["normalized_nees"] == pytest.approx((1 + 9) / 6)


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/science/run_cut3r_bayesian_prefix_dev.py"
    spec = importlib.util.spec_from_file_location("cut3r_prefix_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_seals_all_predictions_before_later_truth(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    points = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    archive_path = tmp_path / "source.zip"

    def member(sequence, dimension, frame, camera):
        return f"{sequence}/{dimension}d/{frame}.txt"

    with zipfile.ZipFile(archive_path, "w") as archive:
        for sequence in runner.SEQUENCES:
            for frame in range(1, 8):
                for dimension in (2, 3):
                    values = points[:, :dimension].copy()
                    if dimension == 3 and frame >= 3:
                        values[:, 2] += 0.05
                    text = "\n".join(",".join(str(float(value)) for value in row) for row in values)
                    archive.writestr(member(sequence, dimension, frame, "cam001"), text)
    monkeypatch.setattr(runner, "ARCHIVE_SHA", runner.sha256(archive_path))
    manifest = {
        "provider_bundle_id": runner.PROVIDER_ID,
        "outputs": [{"sequence": seq, "run": "continuous"} for seq in runner.SEQUENCES],
    }
    base = SimpleNamespace(
        _verify_provider_bundle=lambda *args: manifest,
        _load_run=lambda *args: {},
        _coordinate_member=member,
    )
    pooled = SimpleNamespace(_parse_coordinate_text=parse_coordinate_text)
    monkeypatch.setattr(
        runner, "_load_module", lambda path, name: pooled if "pooled" in name else base
    )
    monkeypatch.setattr(
        runner,
        "_sample",
        lambda run, frame, coords, module: Rows(np.full(3, frame), np.arange(3), points),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **kwargs: "a" * 40 if "rev-parse" in args else "",
    )
    lock = tmp_path / "protocol.json"
    runner.seal(lock, runner.protocol())
    output = tmp_path / "result"
    original_read = zipfile.ZipFile.read
    scored_reads = []

    def checked_read(self, name, *args, **kwargs):
        if "/3d/" in name and name.endswith(("6.txt", "7.txt")):
            barrier = json.loads((output / "prediction-barrier.json").read_text())
            assert len(barrier["cases"]) == 3
            assert all(row["status"] == "ordinary_success" for row in barrier["cases"].values())
            assert barrier["future_3d_opened"] is False
            scored_reads.append(name)
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", checked_read)
    result = runner.run_study(archive_path, tmp_path, output, lock)
    assert len(scored_reads) == 6
    assert result["scored_sequence_count"] == 3
    assert result["complete_denominator"] is True
    assert result["protected_targets_accessed"] is False
    assert not any(
        "/3d/6" in row["member"] or "/3d/7" in row["member"]
        for row in result["opened_coordinate_members"]
        if row["stage"] == "prediction"
    )
    with pytest.raises(FileExistsError):
        runner.run_study(archive_path, tmp_path, output, lock)
