#!/usr/bin/env python3
"""Export Prob4D fusion with sparse 3D gauge-anchor measurements."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from prob4d.alignment import estimate_sim3_robust
from prob4d.benchmark import _build_alignments, _write_fused_prediction
from prob4d.fusion import fuse_windows
from prob4d.gauge import (
    FixedLagGaugeSmoother,
    GaugeAnchor,
    GaugeCovarianceCalibration,
    RelativeGaugeConstraint,
    SequentialGaugeEstimator,
)
from prob4d.io import load_prediction_bundle
from prob4d.metrics import TruthSequence
from prob4d.sintel_uncertainty import _resize_bilinear, _resize_nearest, load_sintel_truth
from prob4d.uncertainty import DepthDisagreementModel, accumulate_disagreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    anchor_source = parser.add_mutually_exclusive_group(required=True)
    anchor_source.add_argument(
        "--ground-truth",
        type=Path,
        help="Sintel HDF5 used for the explicitly simulated sensor-anchor protocol.",
    )
    anchor_source.add_argument(
        "--anchor-prediction",
        type=Path,
        help="World-point NPZ from an independent model, such as VGGT.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gauge-calibration", type=Path, required=True)
    parser.add_argument("--max-depth", type=float, default=70.0)
    parser.add_argument("--initialization-points", type=int, default=16)
    parser.add_argument("--anchors-per-window", type=int, default=16)
    parser.add_argument("--measurement-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def load_external_reference(path: Path, output_shape: tuple[int, int]) -> TruthSequence:
    """Load and resize a world-space point prediction for sensor-free anchoring."""

    with np.load(path, allow_pickle=False) as payload:
        points = payload["point_map"].astype(np.float32)
        if "valid_mask" in payload:
            mask = payload["valid_mask"].astype(bool)
        else:
            mask = np.ones(points.shape[:-1], dtype=bool)
    finite = np.isfinite(points).all(axis=-1)
    mask &= finite
    points = np.nan_to_num(points)
    points = _resize_bilinear(points, output_shape)
    mask = _resize_nearest(mask, output_shape)
    return TruthSequence(
        frame_indices=np.arange(points.shape[0]),
        point_map=points,
        valid_mask=mask,
    )


def spatially_spread_indices(mask: np.ndarray, count: int) -> np.ndarray:
    """Select a deterministic center-first farthest-point pixel subset."""

    coordinates = np.argwhere(mask)
    if coordinates.shape[0] < count:
        raise ValueError("not enough valid pixels for requested anchors")
    center = 0.5 * (np.asarray(mask.shape) - 1)
    selected = [int(np.argmin(np.sum((coordinates - center) ** 2, axis=1)))]
    while len(selected) < count:
        chosen = coordinates[np.asarray(selected)]
        squared_distance = np.min(
            np.sum((coordinates[:, None, :] - chosen[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        squared_distance[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(squared_distance)))
    return coordinates[np.asarray(selected)]


def load_calibration(path: Path) -> GaugeCovarianceCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "calibration" in payload:
        return GaugeCovarianceCalibration(**payload["calibration"])
    inflation = payload["inflation"]
    return GaugeCovarianceCalibration(
        scale=inflation["scale"],
        rotation=inflation["rotation"],
        translation=inflation["translation"],
        trim_quantile=payload.get("trim_quantile", 0.99),
        count=len(payload.get("records", [])),
    )


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    if args.initialization_points < 4 or args.anchors_per_window < 4:
        raise ValueError("initialization and per-window anchor counts must be at least four")
    if args.measurement_std < 0:
        raise ValueError("measurement-std must be nonnegative")

    bundle = load_prediction_bundle(args.manifest)
    output_shape = bundle.overlap_windows[0].point_map.shape[1:3]
    if args.ground_truth is not None:
        reference = load_sintel_truth(
            args.ground_truth,
            output_shape=output_shape,
            max_depth=args.max_depth,
        )
        reference_kind = "simulated_ground_truth_sensor"
    else:
        reference = load_external_reference(args.anchor_prediction, output_shape)
        reference_kind = "external_model_prediction"
    reference_positions = {int(frame): index for index, frame in enumerate(reference.frame_indices)}
    generator = np.random.default_rng(args.seed)

    first_window = bundle.overlap_windows[0]
    first_frame = int(first_window.frame_indices[0])
    first_local_index = first_window.local_index(first_frame)
    first_reference_index = reference_positions[first_frame]
    first_active = (
        first_window.valid_mask[first_local_index] & reference.valid_mask[first_reference_index]
    )
    initialization_coordinates = spatially_spread_indices(first_active, args.initialization_points)
    initialization_local = np.stack(
        [
            first_window.point_map[first_local_index, row, column]
            for row, column in initialization_coordinates
        ]
    )
    initialization_global = np.stack(
        [
            reference.point_map[first_reference_index, row, column]
            for row, column in initialization_coordinates
        ]
    )
    initial_registration = estimate_sim3_robust(initialization_local, initialization_global)
    reference_to_first_local = initial_registration.transform.inverse()

    gauge_anchors: list[GaugeAnchor] = []
    anchor_report: list[dict[str, object]] = []
    for window in bundle.overlap_windows[1:]:
        frame = int(window.frame_indices[0])
        local_index = window.local_index(frame)
        reference_index = reference_positions[frame]
        active = window.valid_mask[local_index] & reference.valid_mask[reference_index]
        coordinates = spatially_spread_indices(active, args.anchors_per_window)
        local_points = np.stack(
            [window.point_map[local_index, row, column] for row, column in coordinates]
        )
        global_points = np.stack(
            [
                reference_to_first_local.transform_points(
                    reference.point_map[reference_index, row, column]
                )
                + generator.normal(scale=args.measurement_std, size=3)
                for row, column in coordinates
            ]
        )
        fit = estimate_sim3_robust(local_points, global_points)
        gauge_anchors.append(
            GaugeAnchor(
                window_id=window.window_id,
                global_from_local=fit.transform,
                covariance=fit.covariance,
            )
        )
        anchor_report.append(
            {
                "window_id": window.window_id,
                "frame": frame,
                "coordinates": coordinates.tolist(),
                "residual_rms": fit.residual_rms,
                "inlier_fraction": fit.inlier_fraction,
                "covariance_diagonal": np.diag(fit.covariance).tolist(),
                "covariance_eigenvalues": np.linalg.eigvalsh(fit.covariance).tolist(),
            }
        )

    calibration = load_calibration(args.gauge_calibration)
    alignments = _build_alignments(bundle)
    constraints: list[RelativeGaugeConstraint] = []
    for alignment in alignments:
        constraint = RelativeGaugeConstraint.from_window_alignment(alignment)
        constraints.append(
            RelativeGaugeConstraint(
                reference_id=constraint.reference_id,
                moving_id=constraint.moving_id,
                reference_from_moving=constraint.reference_from_moving,
                covariance=calibration.apply(constraint.covariance),
                residual_rms=constraint.residual_rms,
                num_correspondences=constraint.num_correspondences,
            )
        )
    ordered_ids = [window.window_id for window in bundle.overlap_windows]
    estimates = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(
        ordered_ids,
        estimates,
        constraints,
        gauge_anchors=gauge_anchors,
    )

    windows = {window.window_id: window for window in bundle.overlap_windows}
    evidence = accumulate_disagreement(windows, alignments)
    model = DepthDisagreementModel()
    uncertainties = {
        window_id: model.predict(window, evidence[window_id])
        for window_id, window in windows.items()
    }
    fused = fuse_windows(
        bundle.overlap_windows,
        {window_id: estimate.global_from_local for window_id, estimate in smoothed.items()},
        uncertainties,
        method="uniform",
        gauge_covariances={
            window_id: estimate.covariance for window_id, estimate in smoothed.items()
        },
    )
    _write_fused_prediction(args.output, fused)

    repository = Path(__file__).resolve().parents[1]
    report = {
        "format_version": 1,
        "prob4d_commit": git_commit(repository),
        "manifest": str(args.manifest.resolve()),
        "anchor_source_kind": reference_kind,
        "ground_truth": (
            str(args.ground_truth.resolve()) if args.ground_truth is not None else None
        ),
        "anchor_prediction": (
            str(args.anchor_prediction.resolve()) if args.anchor_prediction is not None else None
        ),
        "output": str(args.output.resolve()),
        "max_depth": args.max_depth,
        "initialization_points": args.initialization_points,
        "anchors_per_window": args.anchors_per_window,
        "measurement_std": args.measurement_std,
        "seed": args.seed,
        "gauge_calibration": str(args.gauge_calibration.resolve()),
        "calibration": asdict(calibration),
        "initialization": {
            "frame": first_frame,
            "coordinates": initialization_coordinates.tolist(),
            "residual_rms": initial_registration.residual_rms,
            "inlier_fraction": initial_registration.inlier_fraction,
        },
        "anchors": anchor_report,
    }
    report_path = args.output.with_suffix(".anchors.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
