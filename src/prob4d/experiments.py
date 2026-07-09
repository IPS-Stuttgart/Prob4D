"""Reproducible ablation runner for the Prob4D core method."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import WindowAlignment, align_windows
from .fusion import FusedSequence, fuse_windows
from .gauge import (
    FixedLagGaugeSmoother,
    GaugeEstimate,
    RelativeGaugeConstraint,
    SequentialGaugeEstimator,
)
from .metrics import SequenceMetrics, evaluate_sequence
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
        window.window_id: model.predict(window, evidence[window.window_id])
        for window in windows
    }


def _gauge_metrics(
    estimates: dict[str, GaugeEstimate],
    truth: dict[str, Sim3],
) -> GaugeMetrics:
    errors: list[np.ndarray] = []
    normalized_squared_errors: list[float] = []
    for window_id, estimate in estimates.items():
        error = truth[window_id].inverse().compose(estimate.global_from_local).as_vector()
        errors.append(error)
        covariance = 0.5 * (estimate.covariance + estimate.covariance.T)
        inverse = np.linalg.pinv(covariance, rcond=1e-10)
        normalized_squared_errors.append(float(error @ inverse @ error))
    stacked = np.stack(errors)
    return GaugeMetrics(
        log_scale_rmse=float(np.sqrt(np.mean(stacked[:, 0] ** 2))),
        rotation_rmse=float(np.sqrt(np.mean(np.sum(stacked[:, 1:4] ** 2, axis=1)))),
        translation_rmse=float(np.sqrt(np.mean(np.sum(stacked[:, 4:7] ** 2, axis=1)))),
        mean_normalized_squared_error=float(np.mean(normalized_squared_errors)),
    )


def _evaluate(
    key: str,
    label: str,
    sequence: FusedSequence,
    problem: SyntheticProblem,
    *,
    gauges: dict[str, GaugeEstimate] | None = None,
    boundary_frames: list[int] | None = None,
    baseline_source: str = "prob4d",
) -> AblationRow:
    return AblationRow(
        key=key,
        label=label,
        sequence_metrics=evaluate_sequence(
            sequence,
            problem.truth,
            boundary_frames=boundary_frames or problem.boundary_frames,
        ),
        gauge_metrics=(
            _gauge_metrics(gauges, problem.true_overlap_gauges)
            if gauges is not None
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
    model, calibration_report = _calibrate_model(
        calibration_problem, DepthDisagreementModel()
    )
    problem = make_synthetic_problem(
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        overlap=15,
    )

    overlap_alignments = _build_alignments(problem.overlap_windows)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment)
        for alignment in overlap_alignments
    ]
    uncertainties = _uncertainties(problem.overlap_windows, overlap_alignments, model)
    ordered_ids = [window.window_id for window in problem.overlap_windows]
    sequential = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    sequential_transforms = {
        window_id: estimate.global_from_local for window_id, estimate in sequential.items()
    }
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(
        ordered_ids, sequential, constraints
    )
    smoothed_transforms = {
        window_id: estimate.global_from_local for window_id, estimate in smoothed.items()
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
    )
    precision = fuse_windows(
        problem.overlap_windows,
        sequential_transforms,
        uncertainties,
        method="precision",
    )
    covariance_intersection = fuse_windows(
        problem.overlap_windows,
        sequential_transforms,
        uncertainties,
        method="covariance_intersection",
    )
    smoothed_ci = fuse_windows(
        problem.overlap_windows,
        smoothed_transforms,
        uncertainties,
        method="covariance_intersection",
    )
    anchored_ci = fuse_windows(
        problem.overlap_windows,
        anchored_transforms,
        uncertainties,
        method="covariance_intersection",
    )

    rows = [
        _evaluate(
            "disjoint",
            "Disjoint 25-frame windows",
            disjoint_sequence,
            problem,
            boundary_frames=[w.start_frame for w in problem.disjoint_windows[1:]],
            baseline_source="synthetic_disjoint",
        ),
        _evaluate(
            "latent_linear",
            "Latent-space linear overlap blend",
            latent_proxy,
            problem,
            baseline_source="synthetic_decoded_proxy",
        ),
        _evaluate(
            "decoded_uniform",
            "Decoded Sim(3) alignment + uniform fusion",
            uniform,
            problem,
            gauges=sequential,
        ),
        _evaluate(
            "precision",
            "Naive precision-weighted fusion",
            precision,
            problem,
            gauges=sequential,
        ),
        _evaluate(
            "ci",
            "Covariance intersection",
            covariance_intersection,
            problem,
            gauges=sequential,
        ),
        _evaluate(
            "ci_smoothed",
            "Covariance intersection + fixed-lag gauge smoothing",
            smoothed_ci,
            problem,
            gauges=smoothed,
        ),
        _evaluate(
            "ci_smoothed_anchored",
            "Smoothed covariance intersection + sparse metric anchors",
            anchored_ci,
            problem,
            gauges=anchored,
        ),
    ]
    return rows, calibration_report


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
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = ["# Prob4D ablation", "", header, divider]
    for row in flattened:
        values = [row.get(column) for column in columns]
        rendered = [
            f"{value:.6g}" if isinstance(value, float) else str(value) for value in values
        ]
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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
