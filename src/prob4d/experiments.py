"""Reproducible ablation runner for the Prob4D core method."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import WindowAlignment, align_windows, estimate_sim3_robust
from .diagnostics.covariance_support import covariance_support_diagnostic
from .fusion import FusedSequence, fuse_windows
from .gauge import (
    FixedLagGaugeSmoother,
    GaugeEstimate,
    RelativeGaugeConstraint,
    ScaleAnchor,
    SequentialGaugeEstimator,
)
from .io import PredictionBundle, load_prediction_bundle, load_truth
from .metrics import SequenceMetrics, TruthSequence, evaluate_sequence
from .sim3 import Sim3
from .synthetic import SyntheticProblem, make_synthetic_problem
from .uncertainty import (
    CalibrationReport,
    DepthDisagreementModel,
    StructuredCovariance,
    accumulate_disagreement,
)


@dataclass(frozen=True)
class GaugeMetrics:
    log_scale_rmse: float
    rotation_rmse: float
    translation_rmse: float
    mean_normalized_squared_error: float
    mean_rank_normalized_squared_error: float
    mean_covariance_rank: float
    minimum_covariance_rank: int
    support_violation_count: int
    maximum_nullspace_error_norm: float
    all_errors_in_covariance_support: bool


@dataclass(frozen=True)
class AblationRow:
    key: str
    label: str
    sequence_metrics: SequenceMetrics
    gauge_metrics: GaugeMetrics | None
    baseline_source: str = "prob4d"

    def flattened(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "baseline_source": self.baseline_source,
            **self.sequence_metrics.to_dict(),
        }
        gauge = asdict(self.gauge_metrics) if self.gauge_metrics is not None else {}
        result.update({f"gauge_{key}": value for key, value in gauge.items()})
        return result


def _build_alignments(windows) -> list[WindowAlignment]:
    alignments: list[WindowAlignment] = []
    for moving_index, moving in enumerate(windows):
        for reference in windows[:moving_index]:
            if reference.common_frames(moving).size == 0:
                continue
            alignments.append(align_windows(reference, moving, seed=moving_index))
    return alignments


def _calibrate_model(
    problem: SyntheticProblem,
    model: DepthDisagreementModel,
) -> tuple[DepthDisagreementModel, CalibrationReport]:
    alignments = _build_alignments(problem.overlap_windows)
    window_map = {window.window_id: window for window in problem.overlap_windows}
    evidence = accumulate_disagreement(window_map, alignments)
    errors: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    parallel: list[np.ndarray] = []
    lateral: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for window in problem.overlap_windows:
        gauge = problem.true_overlap_gauges[window.window_id]
        truth_local = gauge.inverse().transform_points(
            problem.truth.point_map[window.frame_indices]
        )
        covariance = model.predict(window, evidence[window.window_id])
        errors.append((window.point_map - truth_local).reshape(-1, 3))
        rays.append(covariance.ray_directions.reshape(-1, 3))
        parallel.append(covariance.parallel_variance.reshape(-1))
        lateral.append(covariance.lateral_variance.reshape(-1))
        masks.append(window.valid_mask.reshape(-1))
    stacked_covariance = StructuredCovariance(
        ray_directions=np.concatenate(rays),
        parallel_variance=np.concatenate(parallel),
        lateral_variance=np.concatenate(lateral),
    )
    return model.calibrate(
        np.concatenate(errors),
        stacked_covariance,
        mask=np.concatenate(masks),
    )


def _uncertainties(
    windows,
    alignments: list[WindowAlignment],
    model: DepthDisagreementModel,
) -> dict[str, StructuredCovariance]:
    window_map = {window.window_id: window for window in windows}
    evidence = accumulate_disagreement(window_map, alignments)
    return {
        window.window_id: model.predict(window, evidence[window.window_id]) for window in windows
    }


def _gauge_metrics(
    estimates: dict[str, GaugeEstimate],
    truth: dict[str, Sim3],
) -> GaugeMetrics:
    errors: list[np.ndarray] = []
    diagnostics = []
    for window_id, estimate in estimates.items():
        error = truth[window_id].inverse().compose(estimate.global_from_local).as_vector()
        errors.append(error)
        diagnostics.append(covariance_support_diagnostic(error, estimate.covariance))

    stacked = np.stack(errors)
    normalized_squared_errors = [
        diagnostic.observable_normalized_squared_error for diagnostic in diagnostics
    ]
    rank_normalized_errors = [
        diagnostic.rank_normalized_observable_squared_error
        for diagnostic in diagnostics
        if diagnostic.rank > 0
    ]
    ranks = [diagnostic.rank for diagnostic in diagnostics]
    support_violation_count = sum(
        not diagnostic.support_consistent for diagnostic in diagnostics
    )
    return GaugeMetrics(
        log_scale_rmse=float(np.sqrt(np.mean(stacked[:, 0] ** 2))),
        rotation_rmse=float(np.sqrt(np.mean(np.sum(stacked[:, 1:4] ** 2, axis=1)))),
        translation_rmse=float(np.sqrt(np.mean(np.sum(stacked[:, 4:7] ** 2, axis=1)))),
        mean_normalized_squared_error=float(np.mean(normalized_squared_errors)),
        mean_rank_normalized_squared_error=(
            float(np.mean(rank_normalized_errors)) if rank_normalized_errors else 0.0
        ),
        mean_covariance_rank=float(np.mean(ranks)),
        minimum_covariance_rank=min(ranks),
        support_violation_count=int(support_violation_count),
        maximum_nullspace_error_norm=max(
            diagnostic.nullspace_error_norm for diagnostic in diagnostics
        ),
        all_errors_in_covariance_support=support_violation_count == 0,
    )


def _evaluate(
    key: str,
    label: str,
    sequence: FusedSequence,
    truth: TruthSequence,
    *,
    gauges: dict[str, GaugeEstimate] | None = None,
    true_gauges: dict[str, Sim3] | None = None,
    boundary_frames: list[int] | None = None,
    baseline_source: str = "prob4d",
) -> AblationRow:
    return AblationRow(
        key=key,
        label=label,
        sequence_metrics=evaluate_sequence(
            sequence,
            truth,
            boundary_frames=boundary_frames,
        ),
        gauge_metrics=(
            _gauge_metrics(gauges, true_gauges)
            if gauges is not None and true_gauges is not None
            else None
        ),
        baseline_source=baseline_source,
    )


def run_synthetic_ablation(
    *,
    seed: int = 7,
    num_frames: int = 70,
    height: int = 10,
    width: int = 14,
) -> tuple[list[AblationRow], CalibrationReport]:
    """Run all seven variants on test data calibrated with a separate random seed."""

    calibration_problem = make_synthetic_problem(
        seed=seed + 10_000,
        num_frames=max(50, num_frames // 2),
        height=height,
        width=width,
        overlap=15,
    )
    model, calibration_report = _calibrate_model(calibration_problem, DepthDisagreementModel())
    problem = make_synthetic_problem(
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        overlap=15,
    )

    overlap_alignments = _build_alignments(problem.overlap_windows)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment) for alignment in overlap_alignments
    ]
    uncertainties = _uncertainties(problem.overlap_windows, overlap_alignments, model)
    ordered_ids = [window.window_id for window in problem.overlap_windows]
    sequential = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    sequential_transforms = {
        window_id: estimate.global_from_local for window_id, estimate in sequential.items()
    }
    sequential_covariances = {
        window_id: estimate.covariance for window_id, estimate in sequential.items()
    }
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(ordered_ids, sequential, constraints)
    smoothed_transforms = {
        window_id: estimate.global_from_local for window_id, estimate in smoothed.items()
    }
    smoothed_covariances = {
        window_id: estimate.covariance for window_id, estimate in smoothed.items()
    }
    anchored = FixedLagGaugeSmoother(lag=4).smooth(
        ordered_ids,
        SequentialGaugeEstimator().estimate(
            ordered_ids,
            constraints,
            initial_transform=Sim3(scale=problem.scale_anchors[0].scale),
        ),
        constraints,
        scale_anchors=problem.scale_anchors,
    )
    anchored_transforms = {
        window_id: estimate.global_from_local for window_id, estimate in anchored.items()
    }
    anchored_covariances = {
        window_id: estimate.covariance for window_id, estimate in anchored.items()
    }

    disjoint_uncertainties = {
        window.window_id: model.predict(window) for window in problem.disjoint_windows
    }
    disjoint_sequence = fuse_windows(
        problem.disjoint_windows,
        {window.window_id: Sim3.identity() for window in problem.disjoint_windows},
        disjoint_uncertainties,
        method="uniform",
    )
    # Synthetic proxy only: the real experiment accepts the upstream latent-blend
    # output as an opaque baseline. This proxy crossfades decoded local gauges so
    # the seven-row contract remains testable without loading a diffusion model.
    latent_proxy = fuse_windows(
        problem.overlap_windows,
        {window.window_id: Sim3.identity() for window in problem.overlap_windows},
        uncertainties,
        method="uniform",
    )
    uniform = fuse_windows(
        problem.overlap_windows,
        sequential_transforms,
        uncertainties,
        method="uniform",
        gauge_covariances=sequential_covariances,
    )
    precision = fuse_windows(
        problem.overlap_windows,
        sequential_transforms,
        uncertainties,
        method="precision",
        gauge_covariances=sequential_covariances,
    )
    covariance_intersection = fuse_windows(
        problem.overlap_windows,
        sequential_transforms,
        uncertainties,
        method="covariance_intersection",
        gauge_covariances=sequential_covariances,
    )
    smoothed_ci = fuse_windows(
        problem.overlap_windows,
        smoothed_transforms,
        uncertainties,
        method="covariance_intersection",
        gauge_covariances=smoothed_covariances,
    )
    anchored_ci = fuse_windows(
        problem.overlap_windows,
        anchored_transforms,
        uncertainties,
        method="covariance_intersection",
        gauge_covariances=anchored_covariances,
    )

    rows = [
        _evaluate(
            "disjoint",
            "Disjoint 25-frame windows",
            disjoint_sequence,
            problem.truth,
            boundary_frames=[w.start_frame for w in problem.disjoint_windows[1:]],
            baseline_source="synthetic_disjoint",
        ),
        _evaluate(
            "latent_linear",
            "Latent-space linear overlap blend",
            latent_proxy,
            problem.truth,
            boundary_frames=problem.boundary_frames,
            baseline_source="synthetic_decoded_proxy",
        ),
        _evaluate(
            "decoded_uniform",
            "Decoded Sim(3) alignment + uniform fusion",
            uniform,
            problem.truth,
            gauges=sequential,
            true_gauges=problem.true_overlap_gauges,
            boundary_frames=problem.boundary_frames,
        ),
        _evaluate(
            "precision",
            "Naive precision-weighted fusion",
            precision,
            problem.truth,
            gauges=sequential,
            true_gauges=problem.true_overlap_gauges,
            boundary_frames=problem.boundary_frames,
        ),
        _evaluate(
            "ci",
            "Covariance intersection",
            covariance_intersection,
            problem.truth,
            gauges=sequential,
            true_gauges=problem.true_overlap_gauges,
            boundary_frames=problem.boundary_frames,
        ),
        _evaluate(
            "ci_smoothed",
            "Covariance intersection + fixed-lag gauge smoothing",
            smoothed_ci,
            problem.truth,
            gauges=smoothed,
            true_gauges=problem.true_overlap_gauges,
            boundary_frames=problem.boundary_frames,
        ),
        _evaluate(
            "ci_smoothed_anchored",
            "Smoothed covariance intersection + sparse metric anchors",
            anchored_ci,
            problem.truth,
            gauges=anchored,
            true_gauges=problem.true_overlap_gauges,
            boundary_frames=problem.boundary_frames,
        ),
    ]
    return rows, calibration_report


def _window_truth_gauge(window, truth: TruthSequence) -> Sim3:
    common_frames = np.intersect1d(window.frame_indices, truth.frame_indices)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for frame in common_frames:
        window_index = window.local_index(int(frame))
        truth_index = int(np.searchsorted(truth.frame_indices, frame))
        mask = window.valid_mask[window_index] & truth.valid_mask[truth_index]
        source_parts.append(window.point_map[window_index][mask])
        target_parts.append(truth.point_map[truth_index][mask])
    if not source_parts or sum(part.shape[0] for part in source_parts) < 4:
        raise ValueError(f"window {window.window_id!r} has insufficient overlap with truth")
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    if source.shape[0] > 100_000:
        indices = np.linspace(0, source.shape[0] - 1, 100_000, dtype=int)
        source = source[indices]
        target = target[indices]
    return estimate_sim3_robust(source, target).transform


def _dataset_calibration(
    bundle: PredictionBundle,
    truth: TruthSequence,
    model: DepthDisagreementModel,
) -> tuple[DepthDisagreementModel, CalibrationReport, dict[str, Sim3]]:
    alignments = _build_alignments(bundle.overlap_windows)
    windows = {window.window_id: window for window in bundle.overlap_windows}
    evidence = accumulate_disagreement(windows, alignments)
    true_gauges = {
        window.window_id: _window_truth_gauge(window, truth) for window in bundle.overlap_windows
    }
    errors: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    parallel: list[np.ndarray] = []
    lateral: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    truth_positions = {int(frame): index for index, frame in enumerate(truth.frame_indices)}
    for window in bundle.overlap_windows:
        gauge = true_gauges[window.window_id]
        truth_frames = np.stack(
            [truth.point_map[truth_positions[int(frame)]] for frame in window.frame_indices]
        )
        truth_masks = np.stack(
            [truth.valid_mask[truth_positions[int(frame)]] for frame in window.frame_indices]
        )
        truth_local = gauge.inverse().transform_points(truth_frames)
        covariance = model.predict(window, evidence[window.window_id])
        errors.append((window.point_map - truth_local).reshape(-1, 3))
        rays.append(covariance.ray_directions.reshape(-1, 3))
        parallel.append(covariance.parallel_variance.reshape(-1))
        lateral.append(covariance.lateral_variance.reshape(-1))
        masks.append((window.valid_mask & truth_masks).reshape(-1))
    stacked_covariance = StructuredCovariance(
        np.concatenate(rays), np.concatenate(parallel), np.concatenate(lateral)
    )
    calibrated, report = model.calibrate(
        np.concatenate(errors), stacked_covariance, mask=np.concatenate(masks)
    )
    return calibrated, report, true_gauges


def _baseline_sequence(
    window,
    model: DepthDisagreementModel,
) -> FusedSequence:
    uncertainty = model.predict(window)
    return FusedSequence(
        frame_indices=window.frame_indices,
        point_map=window.point_map,
        valid_mask=window.valid_mask,
        point_covariance=uncertainty.matrices(),
        contributors=np.ones(window.shape, dtype=np.uint16),
        scene_flow=window.scene_flow,
        deform_mask=window.deform_mask,
        flow_covariance=(uncertainty.matrices() if window.scene_flow is not None else None),
    )


def _simulated_scale_anchors(
    windows,
    truth: TruthSequence,
    *,
    every: int,
    standard_deviation: float,
) -> list[ScaleAnchor]:
    if every < 1:
        raise ValueError("metric_anchor_every must be at least one")
    truth_positions = {int(frame): index for index, frame in enumerate(truth.frame_indices)}
    anchors: list[ScaleAnchor] = []
    for window in windows[::every]:
        frame = window.start_frame
        truth_index = truth_positions[frame]
        mask = window.valid_mask[0] & truth.valid_mask[truth_index]
        coordinates = np.argwhere(mask)
        if coordinates.shape[0] < 2:
            continue
        first_row, first_column = coordinates[0]
        distances = np.sum((coordinates - coordinates[0]) ** 2, axis=1)
        second_row, second_column = coordinates[int(np.argmax(distances))]
        first_local = window.point_map[0, first_row, first_column]
        second_local = window.point_map[0, second_row, second_column]
        first_truth = truth.point_map[truth_index, first_row, first_column]
        second_truth = truth.point_map[truth_index, second_row, second_column]
        anchors.append(
            ScaleAnchor.from_metric_pair(
                window.window_id,
                first_local,
                second_local,
                float(np.linalg.norm(first_truth - second_truth)),
                standard_deviation,
            )
        )
    if not anchors or anchors[0].window_id != windows[0].window_id:
        raise ValueError("could not construct a metric scale anchor for the first window")
    return anchors


def run_manifest_ablation(
    *,
    predictions: str | Path,
    truth_path: str | Path,
    calibration_predictions: str | Path,
    calibration_truth_path: str | Path,
    metric_anchor_every: int = 2,
    metric_anchor_standard_deviation: float = 0.02,
) -> tuple[list[AblationRow], CalibrationReport, dict[str, Any]]:
    """Run the seven variants on real MotionCrafter prediction manifests."""

    test_bundle = load_prediction_bundle(predictions)
    truth = load_truth(truth_path)
    calibration_bundle = load_prediction_bundle(calibration_predictions)
    calibration_truth = load_truth(calibration_truth_path)
    model, report, _ = _dataset_calibration(
        calibration_bundle, calibration_truth, DepthDisagreementModel()
    )
    alignments = _build_alignments(test_bundle.overlap_windows)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment) for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in test_bundle.overlap_windows]
    sequential = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(ordered_ids, sequential, constraints)
    anchors = _simulated_scale_anchors(
        test_bundle.overlap_windows,
        truth,
        every=metric_anchor_every,
        standard_deviation=metric_anchor_standard_deviation,
    )
    anchored_initial = SequentialGaugeEstimator().estimate(
        ordered_ids,
        constraints,
        initial_transform=Sim3(scale=anchors[0].scale),
    )
    anchored = FixedLagGaugeSmoother(lag=4).smooth(
        ordered_ids,
        anchored_initial,
        constraints,
        scale_anchors=anchors,
    )
    true_gauges = {
        window.window_id: _window_truth_gauge(window, truth)
        for window in test_bundle.overlap_windows
    }
    uncertainties = _uncertainties(test_bundle.overlap_windows, alignments, model)
    boundaries = [window.start_frame for window in test_bundle.overlap_windows[1:]]
    transforms = {
        "sequential": {key: value.global_from_local for key, value in sequential.items()},
        "smoothed": {key: value.global_from_local for key, value in smoothed.items()},
        "anchored": {key: value.global_from_local for key, value in anchored.items()},
    }
    gauge_covariances = {
        "sequential": {key: value.covariance for key, value in sequential.items()},
        "smoothed": {key: value.covariance for key, value in smoothed.items()},
        "anchored": {key: value.covariance for key, value in anchored.items()},
    }
    sequences = {
        "decoded_uniform": fuse_windows(
            test_bundle.overlap_windows,
            transforms["sequential"],
            uncertainties,
            method="uniform",
            gauge_covariances=gauge_covariances["sequential"],
        ),
        "precision": fuse_windows(
            test_bundle.overlap_windows,
            transforms["sequential"],
            uncertainties,
            method="precision",
            gauge_covariances=gauge_covariances["sequential"],
        ),
        "ci": fuse_windows(
            test_bundle.overlap_windows,
            transforms["sequential"],
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=gauge_covariances["sequential"],
        ),
        "ci_smoothed": fuse_windows(
            test_bundle.overlap_windows,
            transforms["smoothed"],
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=gauge_covariances["smoothed"],
        ),
        "ci_smoothed_anchored": fuse_windows(
            test_bundle.overlap_windows,
            transforms["anchored"],
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=gauge_covariances["anchored"],
        ),
    }
    definitions = [
        (
            "disjoint",
            "Disjoint 25-frame windows",
            _baseline_sequence(test_bundle.disjoint_baseline, model),
            None,
            "upstream_motioncrafter",
        ),
        (
            "latent_linear",
            "Latent-space linear overlap blend",
            _baseline_sequence(test_bundle.latent_linear_baseline, model),
            None,
            "upstream_motioncrafter",
        ),
        (
            "decoded_uniform",
            "Decoded Sim(3) alignment + uniform fusion",
            sequences["decoded_uniform"],
            sequential,
            "prob4d",
        ),
        (
            "precision",
            "Naive precision-weighted fusion",
            sequences["precision"],
            sequential,
            "prob4d",
        ),
        ("ci", "Covariance intersection", sequences["ci"], sequential, "prob4d"),
        (
            "ci_smoothed",
            "Covariance intersection + fixed-lag gauge smoothing",
            sequences["ci_smoothed"],
            smoothed,
            "prob4d",
        ),
        (
            "ci_smoothed_anchored",
            "Smoothed covariance intersection + sparse metric anchors",
            sequences["ci_smoothed_anchored"],
            anchored,
            "prob4d",
        ),
    ]
    rows = [
        _evaluate(
            key,
            label,
            sequence,
            truth,
            gauges=gauges,
            true_gauges=true_gauges,
            boundary_frames=boundaries,
            baseline_source=source,
        )
        for key, label, sequence, gauges, source in definitions
    ]
    metadata = {
        "benchmark": "motioncrafter_manifest",
        "predictions": str(Path(predictions).resolve()),
        "truth": str(Path(truth_path).resolve()),
        "calibration_predictions": str(Path(calibration_predictions).resolve()),
        "calibration_truth": str(Path(calibration_truth_path).resolve()),
        "motioncrafter_commit": test_bundle.metadata.get("motioncrafter_commit"),
        "metric_anchor_source": "simulated_from_ground_truth",
        "metric_anchor_every": metric_anchor_every,
        "metric_anchor_standard_deviation": metric_anchor_standard_deviation,
    }
    return rows, report, metadata


def _write_results(
    output_directory: Path,
    rows: list[AblationRow],
    calibration_report: CalibrationReport,
    metadata: dict[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    flattened = [row.flattened() for row in rows]
    payload = {
        "metadata": metadata,
        "calibration": asdict(calibration_report),
        "rows": flattened,
    }
    (output_directory / "ablation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(dict.fromkeys(key for row in flattened for key in row))
    with (output_directory / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)

    columns = [
        "label",
        "metric_point_rmse",
        "point_rmse",
        "seam_rmse",
        "endpoint_point_rmse",
        "coverage_95",
        "mean_mahalanobis_squared",
        "gauge_mean_rank_normalized_squared_error",
        "gauge_support_violation_count",
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = ["# Prob4D ablation", "", header, divider]
    for row in flattened:
        values = [row.get(column) for column in columns]
        rendered = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in values]
        lines.append("| " + " | ".join(rendered) + " |")
    (output_directory / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic", help="run the correlated synthetic benchmark")
    synthetic.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--num-frames", type=int, default=70)
    synthetic.add_argument("--height", type=int, default=10)
    synthetic.add_argument("--width", type=int, default=14)
    real = subparsers.add_parser("real", help="evaluate MotionCrafter prediction manifests")
    real.add_argument("--predictions", type=Path, required=True)
    real.add_argument("--truth", type=Path, required=True)
    real.add_argument("--calibration-predictions", type=Path, required=True)
    real.add_argument("--calibration-truth", type=Path, required=True)
    real.add_argument("--output-dir", type=Path, required=True)
    real.add_argument("--metric-anchor-every", type=int, default=2)
    real.add_argument("--metric-anchor-std", type=float, default=0.02)
    arguments = parser.parse_args(argv)

    if arguments.command == "synthetic":
        rows, calibration = run_synthetic_ablation(
            seed=arguments.seed,
            num_frames=arguments.num_frames,
            height=arguments.height,
            width=arguments.width,
        )
        _write_results(
            arguments.output_dir,
            rows,
            calibration,
            {
                "benchmark": "correlated_synthetic",
                "seed": arguments.seed,
                "num_frames": arguments.num_frames,
                "height": arguments.height,
                "width": arguments.width,
                "latent_linear_note": (
                    "Synthetic runs use a decoded crossfade proxy. Real runs must provide "
                    "the upstream MotionCrafter latent-blend output."
                ),
            },
        )
        return 0
    if arguments.command == "real":
        rows, calibration, metadata = run_manifest_ablation(
            predictions=arguments.predictions,
            truth_path=arguments.truth,
            calibration_predictions=arguments.calibration_predictions,
            calibration_truth_path=arguments.calibration_truth,
            metric_anchor_every=arguments.metric_anchor_every,
            metric_anchor_standard_deviation=arguments.metric_anchor_std,
        )
        _write_results(arguments.output_dir, rows, calibration, metadata)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
