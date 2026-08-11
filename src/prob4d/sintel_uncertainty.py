"""Sequence-held-out uncertainty analysis on the MotionCrafter Sintel benchmark."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._scientific_scalars import require_finite_real
from .experiments import (
    _baseline_sequence,
    _build_alignments,
    _uncertainties,
    _window_truth_gauge,
)
from .fusion import FusedSequence, fuse_windows
from .gauge import FixedLagGaugeSmoother, RelativeGaugeConstraint, SequentialGaugeEstimator
from .io import PredictionBundle, load_prediction_bundle
from .metrics import TruthSequence, uncertainty_diagnostics
from .uncertainty import DepthDisagreementModel, StructuredCovariance, accumulate_disagreement


@dataclass(frozen=True)
class SequenceInputs:
    sequence: str
    prediction_manifest: Path
    ground_truth_hdf5: Path


def _resize_bilinear(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize ``(T,H,W,C)`` values with align-corners bilinear interpolation."""

    if values.shape[1:3] == output_shape:
        return values.copy()
    output_height, output_width = output_shape
    rows = np.linspace(0.0, values.shape[1] - 1, output_height)
    columns = np.linspace(0.0, values.shape[2] - 1, output_width)
    row_low = np.floor(rows).astype(np.int64)
    column_low = np.floor(columns).astype(np.int64)
    row_high = np.minimum(row_low + 1, values.shape[1] - 1)
    column_high = np.minimum(column_low + 1, values.shape[2] - 1)
    row_weight = (rows - row_low)[:, None, None]
    column_weight = (columns - column_low)[None, :, None]
    output = np.empty(
        (values.shape[0], output_height, output_width, values.shape[-1]),
        dtype=np.float32,
    )
    for frame_index, frame in enumerate(values):
        top = (1.0 - column_weight) * frame[row_low[:, None], column_low[None, :]]
        top += column_weight * frame[row_low[:, None], column_high[None, :]]
        bottom = (1.0 - column_weight) * frame[row_high[:, None], column_low[None, :]]
        bottom += column_weight * frame[row_high[:, None], column_high[None, :]]
        output[frame_index] = (1.0 - row_weight) * top + row_weight * bottom
    return output


def _resize_nearest(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[1:3] == output_shape:
        return mask.copy()
    rows = np.rint(np.linspace(0.0, mask.shape[1] - 1, output_shape[0])).astype(np.int64)
    columns = np.rint(np.linspace(0.0, mask.shape[2] - 1, output_shape[1])).astype(np.int64)
    return mask[:, rows[:, None], columns[None, :]]


def _resize_masked_bilinear(
    values: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int],
    *,
    minimum_support: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize values without allowing invalid coordinates to leak into results."""

    if values.ndim != 4 or values.shape[-1] < 1:
        raise ValueError("values must have shape (T, H, W, C)")
    if mask.shape != values.shape[:3]:
        raise ValueError("mask must have shape values.shape[:3]")
    support_threshold = require_finite_real(
        minimum_support,
        name="minimum_support",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    if values.shape[1:3] == output_shape:
        output = np.asarray(values).copy()
        output[~mask] = 0.0
        return output, np.asarray(mask, dtype=bool).copy()

    finite_values = np.where(mask[..., None], values, 0.0)
    numerator = _resize_bilinear(finite_values, output_shape)
    support = _resize_bilinear(mask[..., None].astype(np.float32), output_shape)[..., 0]
    output = np.divide(
        numerator,
        support[..., None],
        out=np.zeros_like(numerator),
        where=support[..., None] > np.finfo(np.float32).eps,
    )
    output_mask = support >= support_threshold
    output[~output_mask] = 0.0
    return output, output_mask


def load_sintel_truth(
    path: Path,
    *,
    output_shape: tuple[int, int] = (320, 640),
    max_depth: float = 70.0,
    minimum_resize_support: float = 0.5,
) -> TruthSequence:
    """Load camera-space Sintel HDF5 truth and return first-pose world coordinates."""

    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("Sintel uncertainty evaluation requires h5py") from error

    with h5py.File(path) as handle:
        points = handle["point_map"][:].astype(np.float32)
        mask = handle["valid_mask"][:].astype(bool)
        poses = handle["camera_pose"][:].astype(np.float64)
    finite = np.isfinite(points).all(axis=-1)
    mask &= finite & (points[..., 2] > 1e-5) & (points[..., 2] < max_depth)
    points = np.where(mask[..., None], points, 0.0)
    points, mask = _resize_masked_bilinear(
        points,
        mask,
        output_shape,
        minimum_support=minimum_resize_support,
    )

    poses = np.linalg.inv(poses[0])[None] @ poses
    world = np.einsum("tij,thwj->thwi", poses[:, :3, :3], points)
    world += poses[:, None, None, :3, 3]
    return TruthSequence(
        frame_indices=np.arange(points.shape[0]),
        point_map=world,
        valid_mask=mask,
    )


def discover_inputs(dataset_directory: Path, results_directory: Path) -> list[SequenceInputs]:
    metadata = dataset_directory / "filename_list.txt"
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    manifests: dict[str, Path] = {}
    for path in results_directory.glob("part*/artifacts/*/*/predictions.json"):
        manifests[path.parent.parent.name] = path
    inputs: list[SequenceInputs] = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        video_text, truth_text = line.split()
        sequence = Path(video_text).parts[0]
        if sequence not in manifests:
            raise FileNotFoundError(f"prediction manifest for {sequence!r} was not found")
        inputs.append(
            SequenceInputs(
                sequence=sequence,
                prediction_manifest=manifests[sequence],
                ground_truth_hdf5=dataset_directory / truth_text,
            )
        )
    return inputs


def held_out_split(
    inputs: list[SequenceInputs],
) -> tuple[list[SequenceInputs], list[SequenceInputs]]:
    """Greedily balance whole scene families between calibration and test."""

    families: dict[str, list[SequenceInputs]] = {}
    for item in inputs:
        family = item.sequence.rsplit("_", maxsplit=1)[0]
        families.setdefault(family, []).append(item)
    calibration: list[SequenceInputs] = []
    test: list[SequenceInputs] = []
    for family_inputs in families.values():
        destination = calibration if len(calibration) <= len(test) else test
        destination.extend(family_inputs)
    return calibration, test


def _sample_active(mask: np.ndarray, maximum: int, generator: np.random.Generator) -> np.ndarray:
    active = np.flatnonzero(mask.reshape(-1))
    if active.size <= maximum:
        return active
    return np.sort(generator.choice(active, size=maximum, replace=False))


def _collect_calibration_sequence(
    item: SequenceInputs,
    model: DepthDisagreementModel,
    *,
    maximum_points_per_window: int,
    seed: int,
) -> tuple[np.ndarray, StructuredCovariance]:
    bundle = load_prediction_bundle(item.prediction_manifest)
    truth = load_sintel_truth(item.ground_truth_hdf5)
    alignments = _build_alignments(bundle.overlap_windows)
    windows = {window.window_id: window for window in bundle.overlap_windows}
    evidence = accumulate_disagreement(windows, alignments)
    truth_positions = {int(frame): index for index, frame in enumerate(truth.frame_indices)}
    errors: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    parallel: list[np.ndarray] = []
    lateral: list[np.ndarray] = []
    generator = np.random.default_rng(seed)

    for window in bundle.overlap_windows:
        true_gauge = _window_truth_gauge(window, truth)
        truth_indices = np.asarray(
            [truth_positions[int(frame)] for frame in window.frame_indices], dtype=np.int64
        )
        active = window.valid_mask & truth.valid_mask[truth_indices]
        selection = _sample_active(active, maximum_points_per_window, generator)
        truth_selected = truth.point_map[truth_indices].reshape(-1, 3)[selection]
        local_truth = true_gauge.inverse().transform_points(truth_selected)
        errors.append(window.point_map.reshape(-1, 3)[selection] - local_truth)
        covariance = model.predict(window, evidence[window.window_id])
        rays.append(covariance.ray_directions.reshape(-1, 3)[selection])
        parallel.append(covariance.parallel_variance.reshape(-1)[selection])
        lateral.append(covariance.lateral_variance.reshape(-1)[selection])

    return np.concatenate(errors), StructuredCovariance(
        ray_directions=np.concatenate(rays),
        parallel_variance=np.concatenate(parallel),
        lateral_variance=np.concatenate(lateral),
    )


def calibrate_model(
    calibration_inputs: list[SequenceInputs],
    *,
    maximum_points_per_window: int,
    seed: int,
) -> tuple[DepthDisagreementModel, dict[str, Any]]:
    model = DepthDisagreementModel()
    errors: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    parallel: list[np.ndarray] = []
    lateral: list[np.ndarray] = []
    per_sequence_counts: dict[str, int] = {}
    for index, item in enumerate(calibration_inputs):
        sequence_errors, covariance = _collect_calibration_sequence(
            item,
            model,
            maximum_points_per_window=maximum_points_per_window,
            seed=seed + index,
        )
        errors.append(sequence_errors)
        rays.append(covariance.ray_directions)
        parallel.append(covariance.parallel_variance)
        lateral.append(covariance.lateral_variance)
        per_sequence_counts[item.sequence] = int(sequence_errors.shape[0])
        print(
            f"calibration samples {item.sequence} ({index + 1}/{len(calibration_inputs)})",
            flush=True,
        )
        gc.collect()
    stacked_covariance = StructuredCovariance(
        ray_directions=np.concatenate(rays),
        parallel_variance=np.concatenate(parallel),
        lateral_variance=np.concatenate(lateral),
    )
    calibrated, report = model.calibrate(
        np.concatenate(errors), stacked_covariance, trim_quantile=0.99
    )
    metadata = {
        "report": asdict(report),
        "model": asdict(calibrated),
        "points_per_sequence": per_sequence_counts,
    }
    return calibrated, metadata


def _fit_scale_translation(
    prediction: FusedSequence,
    truth: TruthSequence,
    frame_limit: int | None,
) -> tuple[float, np.ndarray, list[tuple[int, int]]]:
    common = np.intersect1d(prediction.frame_indices, truth.frame_indices)
    if frame_limit is not None:
        common = common[:frame_limit]
    pairs = [
        (
            int(np.searchsorted(prediction.frame_indices, frame)),
            int(np.searchsorted(truth.frame_indices, frame)),
        )
        for frame in common
    ]
    count = 0
    source_sum = np.zeros(3)
    target_sum = np.zeros(3)
    source_squared_sum = 0.0
    cross_sum = 0.0
    for prediction_index, truth_index in pairs:
        active = prediction.valid_mask[prediction_index] & truth.valid_mask[truth_index]
        source = prediction.point_map[prediction_index][active]
        target = truth.point_map[truth_index][active]
        count += source.shape[0]
        source_sum += source.sum(axis=0)
        target_sum += target.sum(axis=0)
        source_squared_sum += float(np.sum(source * source))
        cross_sum += float(np.sum(source * target))
    if count < 4:
        raise ValueError("prediction and truth have insufficient jointly valid points")
    source_mean = source_sum / count
    target_mean = target_sum / count
    denominator = source_squared_sum - count * float(source_mean @ source_mean)
    numerator = cross_sum - count * float(source_mean @ target_mean)
    scale = max(numerator / max(denominator, 1e-12), np.finfo(np.float64).eps)
    return scale, target_mean - scale * source_mean, pairs


def _sample_scope(
    prediction: FusedSequence,
    truth: TruthSequence,
    pairs: list[tuple[int, int]],
    *,
    overlap_only: bool,
    maximum_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    generator = np.random.default_rng(seed)
    per_frame = max(1, int(np.ceil(maximum_points / max(len(pairs), 1))))
    for prediction_index, truth_index in pairs:
        active = prediction.valid_mask[prediction_index] & truth.valid_mask[truth_index]
        if overlap_only:
            active &= prediction.contributors[prediction_index] > 1
        selection = _sample_active(active, per_frame, generator)
        if selection.size == 0:
            continue
        points.append(prediction.point_map[prediction_index].reshape(-1, 3)[selection])
        targets.append(truth.point_map[truth_index].reshape(-1, 3)[selection])
        covariances.append(
            prediction.point_covariance[prediction_index].reshape(-1, 3, 3)[selection]
        )
    if not points:
        raise ValueError("requested uncertainty scope contains no valid points")
    point_array = np.concatenate(points)
    target_array = np.concatenate(targets)
    covariance_array = np.concatenate(covariances)
    if point_array.shape[0] > maximum_points:
        selection = np.sort(
            generator.choice(point_array.shape[0], size=maximum_points, replace=False)
        )
        point_array = point_array[selection]
        target_array = target_array[selection]
        covariance_array = covariance_array[selection]
    return point_array, target_array, covariance_array, np.linalg.norm(target_array, axis=-1)


def evaluate_prediction_uncertainty(
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    maximum_points: int,
    seed: int,
    frame_limit: int | None = None,
) -> dict[str, Any]:
    scale, translation, pairs = _fit_scale_translation(prediction, truth, frame_limit)
    scopes: dict[str, Any] = {}
    for scope_index, (scope, overlap_only) in enumerate((("all", False), ("overlap", True))):
        try:
            points, targets, covariances, target_norms = _sample_scope(
                prediction,
                truth,
                pairs,
                overlap_only=overlap_only,
                maximum_points=maximum_points,
                seed=seed + scope_index,
            )
        except ValueError:
            scopes[scope] = None
            continue
        aligned_points = scale * points + translation
        errors = aligned_points - targets
        diagnostics = uncertainty_diagnostics(
            errors,
            scale**2 * covariances,
            target_norms,
            uncertainty_normalizers=np.linalg.norm(aligned_points, axis=-1),
        )
        scopes[scope] = diagnostics.to_dict()
    return {
        "fitted_alignment_scale": scale,
        "evaluated_frames": len(pairs),
        "scopes": scopes,
    }


def _fused_methods(
    bundle: PredictionBundle,
    model: DepthDisagreementModel,
) -> list[tuple[str, Callable[[], FusedSequence]]]:
    alignments = _build_alignments(bundle.overlap_windows)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment) for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in bundle.overlap_windows]
    sequential = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(ordered_ids, sequential, constraints)
    uncertainties = _uncertainties(bundle.overlap_windows, alignments, model)
    sequential_gauges = {
        window_id: estimate.global_from_local for window_id, estimate in sequential.items()
    }
    sequential_covariances = {
        window_id: estimate.covariance for window_id, estimate in sequential.items()
    }
    smoothed_gauges = {
        window_id: estimate.global_from_local for window_id, estimate in smoothed.items()
    }
    smoothed_covariances = {
        window_id: estimate.covariance for window_id, estimate in smoothed.items()
    }
    definitions = [
        (
            "disjoint",
            lambda: _baseline_sequence(bundle.disjoint_baseline, model),
        ),
        (
            "latent_linear",
            lambda: _baseline_sequence(bundle.latent_linear_baseline, model),
        ),
        (
            "decoded_uniform",
            lambda: fuse_windows(
                bundle.overlap_windows,
                sequential_gauges,
                uncertainties,
                method="uniform",
                gauge_covariances=sequential_covariances,
            ),
        ),
        (
            "precision",
            lambda: fuse_windows(
                bundle.overlap_windows,
                sequential_gauges,
                uncertainties,
                method="precision",
                gauge_covariances=sequential_covariances,
            ),
        ),
        (
            "ci",
            lambda: fuse_windows(
                bundle.overlap_windows,
                sequential_gauges,
                uncertainties,
                method="covariance_intersection",
                gauge_covariances=sequential_covariances,
            ),
        ),
        (
            "ci_smoothed",
            lambda: fuse_windows(
                bundle.overlap_windows,
                smoothed_gauges,
                uncertainties,
                method="covariance_intersection",
                gauge_covariances=smoothed_covariances,
            ),
        ),
    ]
    return [(key, factory) for key, factory in definitions]


def _aggregate(sequence_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    method_names = list(next(iter(sequence_results.values())))
    output: dict[str, Any] = {}
    for method in method_names:
        output[method] = {}
        for scope in ("all", "overlap"):
            rows = [
                result[method]["scopes"][scope]
                for result in sequence_results.values()
                if result[method]["scopes"][scope] is not None
            ]
            if not rows:
                output[method][scope] = None
                continue
            metrics = [key for key in rows[0] if key != "count"]
            output[method][scope] = {
                "sequences": len(rows),
                "sampled_points": int(sum(row["count"] for row in rows)),
                "minimum_sequence_coverage_95": float(np.min([row["coverage_95"] for row in rows])),
                **{key: float(np.mean([row[key] for row in rows])) for key in metrics},
            }
    return output


def _paired_comparison(
    sequence_results: dict[str, dict[str, Any]],
    first: str,
    second: str,
    scope: str,
    metric: str,
    *,
    seed: int,
) -> dict[str, Any]:
    pairs = [
        (result[first]["scopes"][scope], result[second]["scopes"][scope])
        for result in sequence_results.values()
    ]
    differences = np.asarray(
        [a[metric] - b[metric] for a, b in pairs if a is not None and b is not None]
    )
    generator = np.random.default_rng(seed)
    selections = generator.integers(0, differences.size, size=(20_000, differences.size))
    bootstrap = differences[selections].mean(axis=1)
    return {
        "first": first,
        "second": second,
        "scope": scope,
        "metric": metric,
        "difference_first_minus_second": float(np.mean(differences)),
        "bootstrap_95_interval": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "sequences": int(differences.size),
    }


def _bootstrap_method_metric(
    sequence_results: dict[str, dict[str, Any]],
    method: str,
    scope: str,
    metric: str,
    *,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        [
            result[method]["scopes"][scope][metric]
            for result in sequence_results.values()
            if result[method]["scopes"][scope] is not None
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    selections = generator.integers(0, values.size, size=(20_000, values.size))
    bootstrap = values[selections].mean(axis=1)
    return {
        "method": method,
        "scope": scope,
        "metric": metric,
        "mean": float(np.mean(values)),
        "bootstrap_95_interval": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "sequences": int(values.size),
    }


def run_analysis(
    *,
    dataset_directory: Path,
    results_directory: Path,
    output_directory: Path,
    calibration_points_per_window: int,
    evaluation_points_per_sequence: int,
    seed: int,
    frame_limit: int | None,
    max_calibration_sequences: int | None,
    max_test_sequences: int | None,
) -> Path:
    inputs = discover_inputs(dataset_directory, results_directory)
    calibration_inputs, test_inputs = held_out_split(inputs)
    if max_calibration_sequences is not None:
        calibration_inputs = calibration_inputs[:max_calibration_sequences]
    if max_test_sequences is not None:
        test_inputs = test_inputs[:max_test_sequences]
    calibrated_model, calibration = calibrate_model(
        calibration_inputs,
        maximum_points_per_window=calibration_points_per_window,
        seed=seed,
    )

    sequence_results: dict[str, dict[str, Any]] = {}
    for sequence_index, item in enumerate(test_inputs):
        bundle = load_prediction_bundle(item.prediction_manifest)
        truth = load_sintel_truth(item.ground_truth_hdf5)
        method_results: dict[str, Any] = {}
        for key, factory in _fused_methods(bundle, calibrated_model):
            prediction = factory()
            method_results[key] = evaluate_prediction_uncertainty(
                prediction,
                truth,
                maximum_points=evaluation_points_per_sequence,
                seed=seed + 1000 * sequence_index,
                frame_limit=frame_limit,
            )
            del prediction
            gc.collect()
        sequence_results[item.sequence] = method_results
        print(f"evaluated {item.sequence} ({sequence_index + 1}/{len(test_inputs)})", flush=True)

    aggregate = _aggregate(sequence_results)
    comparisons = [
        _paired_comparison(
            sequence_results,
            "ci",
            "precision",
            "overlap",
            metric,
            seed=seed + index,
        )
        for index, metric in enumerate(
            (
                "coverage_95",
                "coverage_shortfall_95",
                "coverage_calibration_error",
                "gaussian_nll",
                "selective_gain_80",
            )
        )
    ]
    comparisons.extend(
        _paired_comparison(
            sequence_results,
            first,
            second,
            scope,
            "mean_relative_error",
            seed=seed + 20 + index,
        )
        for index, (first, second, scope) in enumerate(
            (
                ("precision", "decoded_uniform", "all"),
                ("precision", "decoded_uniform", "overlap"),
                ("ci", "decoded_uniform", "all"),
                ("ci", "decoded_uniform", "overlap"),
                ("ci", "precision", "all"),
                ("ci", "precision", "overlap"),
            )
        )
    )
    bootstrap_diagnostics = [
        _bootstrap_method_metric(
            sequence_results,
            method,
            scope,
            metric,
            seed=seed + 100 + index,
        )
        for index, (method, scope, metric) in enumerate(
            (method, scope, metric)
            for method in aggregate
            for scope in ("all", "overlap")
            for metric in (
                "coverage_95",
                "coverage_shortfall_95",
                "coverage_calibration_error",
                "uncertainty_error_spearman",
                "selective_gain_80",
            )
            if aggregate[method][scope] is not None
        )
    ]
    payload = {
        "protocol": {
            "split": (
                "whole scene families assigned in official first-seen order to the smaller "
                "partition"
            ),
            "calibration_sequences": [item.sequence for item in calibration_inputs],
            "test_sequences": [item.sequence for item in test_inputs],
            "calibration_points_per_window": calibration_points_per_window,
            "evaluation_points_per_sequence_and_scope": evaluation_points_per_sequence,
            "frame_limit": frame_limit,
            "seed": seed,
            "max_depth_meters": 70.0,
            "resolution": [320, 640],
        },
        "calibration": calibration,
        "aggregate": aggregate,
        "bootstrap_diagnostics": bootstrap_diagnostics,
        "paired_comparisons": comparisons,
        "sequences": sequence_results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "uncertainty_analysis.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output_directory / "uncertainty_analysis.md", payload)
    return output_path


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    columns = [
        "method",
        "scope",
        "coverage_95",
        "minimum_sequence_coverage_95",
        "coverage_shortfall_95",
        "coverage_calibration_error",
        "mean_mahalanobis_squared",
        "gaussian_nll",
        "uncertainty_error_spearman",
        "selective_gain_80",
    ]
    lines = [
        "# Sintel held-out uncertainty analysis",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for method, scopes in payload["aggregate"].items():
        for scope, row in scopes.items():
            if row is None:
                continue
            values = [method, scope] + [row[column] for column in columns[2:]]
            rendered = [
                f"{value:.6g}" if isinstance(value, float) else str(value) for value in values
            ]
            lines.append("| " + " | ".join(rendered) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-points-per-window", type=int, default=20_000)
    parser.add_argument("--evaluation-points-per-sequence", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-calibration-sequences", type=int)
    parser.add_argument("--max-test-sequences", type=int)
    arguments = parser.parse_args(argv)
    path = run_analysis(
        dataset_directory=arguments.dataset_dir,
        results_directory=arguments.results_dir,
        output_directory=arguments.output_dir,
        calibration_points_per_window=arguments.calibration_points_per_window,
        evaluation_points_per_sequence=arguments.evaluation_points_per_sequence,
        seed=arguments.seed,
        frame_limit=arguments.max_frames or None,
        max_calibration_sequences=arguments.max_calibration_sequences,
        max_test_sequences=arguments.max_test_sequences,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
