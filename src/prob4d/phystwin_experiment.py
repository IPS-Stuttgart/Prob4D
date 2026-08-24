"""Run MotionCrafter experiment zero on a calibrated PhysTwin interaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .alignment import AlignmentResult, estimate_sim3_robust
from .data import PredictionWindow
from .phystwin import (
    CoverResizeCrop,
    PhysTwinCase,
    _load_trusted_legacy_pickle,
    deterministic_subsample,
    directed_nearest_distances,
    nearest_neighbor_indices,
    point_set_metrics,
    sample_vector_field_nearest,
)
from .sim3 import Sim3, so3_log

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ErrorSummary:
    count: int
    rmse_m: float
    mean_m: float
    median_m: float
    p95_m: float

    @classmethod
    def from_vectors(cls, errors: FloatArray) -> ErrorSummary:
        values = np.asarray(errors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("error vectors must have non-empty shape (N, 3)")
        values = values[np.all(np.isfinite(values), axis=1)]
        if values.shape[0] == 0:
            raise ValueError("error vectors contain no finite rows")
        norms = np.linalg.norm(values, axis=1)
        return cls(
            count=int(values.shape[0]),
            rmse_m=float(np.sqrt(np.mean(np.sum(values**2, axis=1)))),
            mean_m=float(np.mean(norms)),
            median_m=float(np.median(norms)),
            p95_m=float(np.quantile(norms, 0.95)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ManualFlowSamples:
    frame_indices: NDArray[np.int64]
    visual_current_world: FloatArray
    visual_flow_world: FloatArray
    truth_current_world: FloatArray
    truth_next_world: FloatArray
    track_indices: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        count = int(np.asarray(self.frame_indices).size)
        shapes = (
            self.visual_current_world.shape,
            self.visual_flow_world.shape,
            self.truth_current_world.shape,
            self.truth_next_world.shape,
        )
        if any(shape != (count, 3) for shape in shapes):
            raise ValueError("manual flow arrays must have shape (N, 3)")
        if self.track_indices is not None and np.asarray(self.track_indices).shape != (count,):
            raise ValueError("track_indices must have shape (N,)")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_prediction_product(
    manifest_path: str | Path,
    product: str,
) -> tuple[PredictionWindow, dict]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "disjoint": "disjoint_baseline",
        "latent_linear": "latent_linear_baseline",
    }
    if product not in fields:
        raise ValueError("prediction product must be disjoint or latent_linear")
    prediction = PredictionWindow.from_npz(
        path.parent / manifest[fields[product]],
        window_id=product,
    )
    return prediction, manifest


def fit_metric_gauge(
    prediction: PredictionWindow,
    truth_points: FloatArray,
    truth_mask: BoolArray,
    *,
    fit_end_frame: int,
    maximum_correspondences: int,
    seed: int,
) -> AlignmentResult:
    if prediction.point_map.shape != truth_points.shape:
        raise ValueError("prediction and metric truth shapes do not match")
    if truth_mask.shape != prediction.valid_mask.shape:
        raise ValueError("metric truth mask shape does not match prediction")
    fit_rows = prediction.frame_indices < fit_end_frame
    active = prediction.valid_mask & truth_mask & fit_rows[:, None, None]
    source = prediction.point_map[active]
    target = truth_points[active]
    if source.shape[0] < 4:
        raise ValueError("training interval has too few metric correspondences")
    if source.shape[0] > maximum_correspondences:
        generator = np.random.default_rng(seed)
        selected = np.sort(
            generator.choice(source.shape[0], maximum_correspondences, replace=False)
        )
        source = source[selected]
        target = target[selected]
    return estimate_sim3_robust(source, target)


def same_view_metrics(
    prediction: PredictionWindow,
    truth_points: FloatArray,
    truth_mask: BoolArray,
    transform: Sim3,
    *,
    fit_end_frame: int,
) -> dict[str, dict[str, float | int]]:
    transformed = transform.transform_points(prediction.point_map)
    active = prediction.valid_mask & truth_mask
    result = {}
    for name, frame_selection in (
        ("fit", prediction.frame_indices < fit_end_frame),
        ("test", prediction.frame_indices >= fit_end_frame),
    ):
        mask = active & frame_selection[:, None, None]
        if not np.any(mask):
            raise ValueError(f"same-view {name} interval has no valid points")
        result[name] = ErrorSummary.from_vectors(transformed[mask] - truth_points[mask]).to_dict()
    return result


def load_manual_flow_samples(
    case: PhysTwinCase,
    prediction: PredictionWindow,
    transform: Sim3,
    crop: CoverResizeCrop,
    manual_tracks_path: str | Path,
    *,
    camera: int,
    occlusion_tolerance_m: float,
) -> ManualFlowSamples:
    if prediction.scene_flow is None or prediction.deform_mask is None:
        raise ValueError("MotionCrafter prediction does not contain scene flow")
    manual_tracks = np.asarray(
        _load_trusted_legacy_pickle(
            Path(manual_tracks_path),
            description="manual tracks",
        ),
        dtype=np.float64,
    )
    if manual_tracks.ndim != 3 or manual_tracks.shape[-1] != 3:
        raise ValueError("manual tracks must have shape (T, N, 3)")
    masks = case.load_processed_masks()
    frame_to_index = {
        int(frame): index for index, frame in enumerate(prediction.frame_indices)
    }
    collected: dict[str, list[np.ndarray]] = {
        "frames": [],
        "tracks": [],
        "visual_current": [],
        "visual_flow": [],
        "truth_current": [],
        "truth_next": [],
    }
    for frame in prediction.frame_indices:
        current_frame = int(frame)
        if current_frame + 1 not in frame_to_index or current_frame + 1 >= manual_tracks.shape[0]:
            continue
        prediction_index = frame_to_index[current_frame]
        current_truth = manual_tracks[current_frame]
        next_truth = manual_tracks[current_frame + 1]
        source_pixels, camera_depth = case.project_world(current_truth, camera)
        model_pixels = crop.source_to_target(source_pixels)
        visual_current, current_active = sample_vector_field_nearest(
            prediction.point_map[prediction_index],
            model_pixels,
            valid_mask=prediction.valid_mask[prediction_index],
        )
        visual_flow, flow_active = sample_vector_field_nearest(
            prediction.scene_flow[prediction_index],
            model_pixels,
            valid_mask=prediction.deform_mask[prediction_index],
        )

        finite_tracks = (
            np.all(np.isfinite(current_truth), axis=1)
            & np.all(np.isfinite(next_truth), axis=1)
            & np.all(np.isfinite(source_pixels), axis=1)
            & np.isfinite(camera_depth)
        )
        safe_pixels = np.where(np.isfinite(source_pixels), source_pixels, 0.0)
        rows = np.rint(safe_pixels[:, 1]).astype(np.int64)
        columns = np.rint(safe_pixels[:, 0]).astype(np.int64)
        in_source = (
            finite_tracks
            & (rows >= 0)
            & (rows < case.source_height)
            & (columns >= 0)
            & (columns < case.source_width)
            & (camera_depth > 0)
        )
        clipped_rows = np.clip(rows, 0, case.source_height - 1)
        clipped_columns = np.clip(columns, 0, case.source_width - 1)
        depth = case.load_depth_m(current_frame, camera)
        object_mask = case.object_mask(masks, current_frame, camera)
        depth_consistent = (
            np.abs(depth[clipped_rows, clipped_columns] - camera_depth)
            <= occlusion_tolerance_m
        )
        active = (
            in_source
            & depth_consistent
            & object_mask[clipped_rows, clipped_columns]
            & current_active
            & flow_active
        )
        if not np.any(active):
            continue
        count = int(np.count_nonzero(active))
        collected["frames"].append(np.full(count, current_frame, dtype=np.int64))
        collected["tracks"].append(np.flatnonzero(active).astype(np.int64))
        collected["visual_current"].append(transform.transform_points(visual_current[active]))
        collected["visual_flow"].append(transform.transform_vectors(visual_flow[active]))
        collected["truth_current"].append(current_truth[active])
        collected["truth_next"].append(next_truth[active])
    if not collected["frames"]:
        raise ValueError("no visible manual tracks overlap the prediction")
    return ManualFlowSamples(
        frame_indices=np.concatenate(collected["frames"]),
        visual_current_world=np.concatenate(collected["visual_current"]),
        visual_flow_world=np.concatenate(collected["visual_flow"]),
        truth_current_world=np.concatenate(collected["truth_current"]),
        truth_next_world=np.concatenate(collected["truth_next"]),
        track_indices=np.concatenate(collected["tracks"]),
    )


def load_physics_trajectory(
    trajectory_path: str | Path | None,
    final_data_path: str | Path | None,
) -> FloatArray | None:
    if trajectory_path is None:
        return None
    if final_data_path is None:
        raise ValueError("physics trajectories require final_data_path")
    trajectory = np.asarray(
        _load_trusted_legacy_pickle(
            Path(trajectory_path),
            description="physics trajectory",
        ),
        dtype=np.float64,
    )
    final_data = _load_trusted_legacy_pickle(
        Path(final_data_path),
        description="PhysTwin final data",
    )
    if not isinstance(final_data, dict):
        raise ValueError("PhysTwin final data must be a dictionary")
    surface_count = int(
        np.asarray(final_data["object_points"]).shape[1]
        + np.asarray(final_data["surface_points"]).shape[0]
    )
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("physics trajectory must have shape (T, N, 3)")
    if trajectory.shape[1] < surface_count:
        raise ValueError("physics trajectory has fewer points than the surface contract")
    return trajectory[:, :surface_count]


def physics_flow_for_samples(
    samples: ManualFlowSamples,
    trajectory: FloatArray,
    *,
    maximum_nodes: int = 6000,
) -> FloatArray:
    result = np.empty_like(samples.visual_flow_world)
    for frame in np.unique(samples.frame_indices):
        active = samples.frame_indices == frame
        current = trajectory[int(frame)]
        following = trajectory[int(frame) + 1]
        if current.shape[0] > maximum_nodes:
            indices = np.linspace(0, current.shape[0] - 1, maximum_nodes, dtype=np.int64)
            current = current[indices]
            following = following[indices]
        nearest, _ = nearest_neighbor_indices(samples.visual_current_world[active], current)
        result[active] = following[nearest] - current[nearest]
    return result


def flow_method_metrics(
    samples: ManualFlowSamples,
    physics_trajectory: FloatArray | None,
    corrected_trajectory: FloatArray | None,
    *,
    fit_end_frame: int,
) -> dict[str, object]:
    training = samples.frame_indices < fit_end_frame
    test = samples.frame_indices >= fit_end_frame
    if not np.any(training) or not np.any(test):
        raise ValueError("manual tracks must cover both fit and test intervals")
    truth_flow = samples.truth_next_world - samples.truth_current_world
    flows: dict[str, FloatArray] = {
        "zero_flow": np.zeros_like(samples.visual_flow_world),
        "visual": samples.visual_flow_world,
    }
    if physics_trajectory is not None:
        flows["physics"] = physics_flow_for_samples(samples, physics_trajectory)
    if corrected_trajectory is not None:
        flows["corrected_physics"] = physics_flow_for_samples(samples, corrected_trajectory)

    calibration = {}
    for name, flow in tuple(flows.items()):
        residual = flow[training] - truth_flow[training]
        calibration[name] = {
            "coordinate_mse_m2": float(np.mean(residual**2)),
            "flow_epe": ErrorSummary.from_vectors(residual).to_dict(),
        }

    for physics_name in ("physics", "corrected_physics"):
        if physics_name not in flows:
            continue
        flows[f"fixed_visual_{physics_name}"] = 0.5 * (
            flows["visual"] + flows[physics_name]
        )
        visual_variance = calibration["visual"]["coordinate_mse_m2"]
        physics_variance = calibration[physics_name]["coordinate_mse_m2"]
        visual_weight = physics_variance / max(
            visual_variance + physics_variance,
            np.finfo(np.float64).eps,
        )
        flows[f"calibrated_visual_{physics_name}"] = (
            visual_weight * flows["visual"]
            + (1.0 - visual_weight) * flows[physics_name]
        )
        calibration[f"calibrated_visual_{physics_name}"] = {
            "visual_weight": float(visual_weight),
            "contract": "single scalar inverse-training-MSE weight",
        }

    test_results = {}
    for name, flow in flows.items():
        flow_error = flow[test] - truth_flow[test]
        endpoint_error = (
            samples.visual_current_world[test] + flow[test] - samples.truth_next_world[test]
        )
        test_results[name] = {
            "flow_epe": ErrorSummary.from_vectors(flow_error).to_dict(),
            "endpoint_error": ErrorSummary.from_vectors(endpoint_error).to_dict(),
        }
    return {
        "fit_sample_count": int(np.count_nonzero(training)),
        "test_sample_count": int(np.count_nonzero(test)),
        "calibration": calibration,
        "test": test_results,
    }


def _aggregate_point_set_rows(rows: list[dict[str, float | int]]) -> dict[str, object]:
    if not rows:
        raise ValueError("held-out evaluation has no point-set rows")
    metric_names = (
        "source_to_target_mean_m",
        "target_to_source_mean_m",
        "symmetric_mean_m",
        "target_to_source_p95_m",
        "target_coverage_10mm",
        "target_coverage_20mm",
        "heldout_only_mean_m",
        "heldout_only_coverage_20mm",
    )
    return {
        "view_frame_count": len(rows),
        **{
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in metric_names
        },
        "mean_source_count": float(np.mean([int(row["source_count"]) for row in rows])),
        "mean_target_count": float(np.mean([int(row["target_count"]) for row in rows])),
        "mean_heldout_only_count": float(
            np.mean([int(row["heldout_only_count"]) for row in rows])
        ),
    }


def heldout_view_metrics(
    case: PhysTwinCase,
    prediction: PredictionWindow,
    input_truth_points: FloatArray,
    input_truth_mask: BoolArray,
    transform: Sim3,
    crop: CoverResizeCrop,
    *,
    fit_end_frame: int,
    heldout_cameras: tuple[int, ...],
    physics_trajectory: FloatArray | None,
    corrected_trajectory: FloatArray | None,
    frame_stride: int,
    maximum_points: int,
    heldout_only_threshold_m: float,
    seed: int,
) -> dict[str, object]:
    masks = case.load_processed_masks()
    sources: dict[str, list[dict[str, float | int]]] = {}
    test_indices = np.flatnonzero(prediction.frame_indices >= fit_end_frame)[::frame_stride]
    for prediction_index in test_indices:
        frame = int(prediction.frame_indices[prediction_index])
        input_active = prediction.valid_mask[prediction_index] & input_truth_mask[prediction_index]
        if not np.any(input_active):
            continue
        visual = deterministic_subsample(
            transform.transform_points(prediction.point_map[prediction_index][input_active]),
            maximum_points,
            seed=seed + frame,
        )
        input_truth = deterministic_subsample(
            input_truth_points[prediction_index][input_truth_mask[prediction_index]],
            maximum_points,
            seed=seed + 10_000 + frame,
        )
        frame_sources: dict[str, FloatArray] = {
            "visual": visual,
            "input_visible_truth": input_truth,
        }
        if physics_trajectory is not None:
            physics = deterministic_subsample(
                physics_trajectory[frame], maximum_points, seed=seed + 20_000 + frame
            )
            frame_sources["physics"] = physics
            frame_sources["visual_union_physics"] = np.concatenate((visual, physics))
        if corrected_trajectory is not None:
            corrected = deterministic_subsample(
                corrected_trajectory[frame],
                maximum_points,
                seed=seed + 30_000 + frame,
            )
            frame_sources["corrected_physics"] = corrected
            frame_sources["visual_union_corrected_physics"] = np.concatenate(
                (visual, corrected)
            )

        for camera in heldout_cameras:
            target_map, target_mask = case.metric_point_map(
                frame,
                camera,
                crop,
                masks=masks,
                object_only=True,
            )
            if not np.any(target_mask):
                continue
            target = deterministic_subsample(
                target_map[target_mask],
                maximum_points,
                seed=seed + 40_000 + 1000 * camera + frame,
            )
            input_distance = directed_nearest_distances(target, input_truth)
            heldout_only = input_distance > heldout_only_threshold_m
            for name, source in frame_sources.items():
                metrics = point_set_metrics(source, target).to_dict()
                target_distance = directed_nearest_distances(target, source)
                selected = target_distance[heldout_only]
                metrics.update(
                    {
                        "heldout_only_count": int(selected.size),
                        "heldout_only_mean_m": (
                            float(np.mean(selected)) if selected.size else 0.0
                        ),
                        "heldout_only_coverage_20mm": (
                            float(np.mean(selected <= 0.02)) if selected.size else 1.0
                        ),
                    }
                )
                sources.setdefault(name, []).append(metrics)
    return {name: _aggregate_point_set_rows(rows) for name, rows in sources.items()}


def evaluate_product(
    case: PhysTwinCase,
    prediction: PredictionWindow,
    manual_tracks_path: str | Path,
    *,
    input_camera: int,
    heldout_cameras: tuple[int, ...],
    fit_end_frame: int,
    physics_trajectory: FloatArray | None,
    corrected_trajectory: FloatArray | None,
    maximum_correspondences: int,
    maximum_heldout_points: int,
    heldout_frame_stride: int,
    seed: int,
) -> dict[str, object]:
    crop = CoverResizeCrop.from_shapes(
        case.source_height,
        case.source_width,
        prediction.shape[1],
        prediction.shape[2],
    )
    truth = case.metric_truth(
        prediction.frame_indices,
        input_camera,
        crop,
        object_only=True,
    )
    gauge = fit_metric_gauge(
        prediction,
        truth.point_map,
        truth.valid_mask,
        fit_end_frame=fit_end_frame,
        maximum_correspondences=maximum_correspondences,
        seed=seed,
    )
    manual_samples = load_manual_flow_samples(
        case,
        prediction,
        gauge.transform,
        crop,
        manual_tracks_path,
        camera=input_camera,
        occlusion_tolerance_m=0.03,
    )
    return {
        "gauge": {
            "scale": gauge.transform.scale,
            "rotation_vector": so3_log(gauge.transform.rotation).tolist(),
            "translation_m": gauge.transform.translation.tolist(),
            "fit_residual_rms_m": gauge.residual_rms,
            "inlier_fraction": gauge.inlier_fraction,
            "correspondence_count": gauge.num_correspondences,
        },
        "same_view_object_geometry": same_view_metrics(
            prediction,
            truth.point_map,
            truth.valid_mask,
            gauge.transform,
            fit_end_frame=fit_end_frame,
        ),
        "manual_track_flow": flow_method_metrics(
            manual_samples,
            physics_trajectory,
            corrected_trajectory,
            fit_end_frame=fit_end_frame,
        ),
        "heldout_views": heldout_view_metrics(
            case,
            prediction,
            truth.point_map,
            truth.valid_mask,
            gauge.transform,
            crop,
            fit_end_frame=fit_end_frame,
            heldout_cameras=heldout_cameras,
            physics_trajectory=physics_trajectory,
            corrected_trajectory=corrected_trajectory,
            frame_stride=heldout_frame_stride,
            maximum_points=maximum_heldout_points,
            heldout_only_threshold_m=0.01,
            seed=seed,
        ),
    }


def run_experiment(
    manifest_path: str | Path,
    case_directory: str | Path,
    output_path: str | Path,
    *,
    input_camera: int,
    heldout_cameras: tuple[int, ...],
    fit_end_frame: int,
    manual_tracks_path: str | Path,
    physics_trajectory_path: str | Path | None,
    corrected_trajectory_path: str | Path | None,
    final_data_path: str | Path | None,
    products: tuple[str, ...],
    maximum_correspondences: int = 100_000,
    maximum_heldout_points: int = 2000,
    heldout_frame_stride: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    case = PhysTwinCase.from_directory(case_directory)
    physics = load_physics_trajectory(physics_trajectory_path, final_data_path)
    corrected = load_physics_trajectory(corrected_trajectory_path, final_data_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    results = {}
    frame_contract = None
    for product in products:
        prediction, _ = load_prediction_product(manifest_path, product)
        current_contract = prediction.frame_indices.tolist()
        if frame_contract is None:
            frame_contract = current_contract
        elif frame_contract != current_contract:
            raise ValueError("prediction products use different source frames")
        results[product] = evaluate_product(
            case,
            prediction,
            manual_tracks_path,
            input_camera=input_camera,
            heldout_cameras=heldout_cameras,
            fit_end_frame=fit_end_frame,
            physics_trajectory=physics,
            corrected_trajectory=corrected,
            maximum_correspondences=maximum_correspondences,
            maximum_heldout_points=maximum_heldout_points,
            heldout_frame_stride=heldout_frame_stride,
            seed=seed,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "status": "experiment-zero out-of-domain MotionCrafter evaluation",
        "case": Path(case_directory).name,
        "input_camera": input_camera,
        "heldout_cameras": list(heldout_cameras),
        "source_frames": frame_contract,
        "fit_end_frame": fit_end_frame,
        "future_labels_used_for_gauge": False,
        "fusion_calibration": "manual 3D tracks before fit_end_frame only",
        "products": results,
        "provenance": {
            "prob4d_commit": git_commit(Path(__file__).resolve().parents[2]),
            "motioncrafter_commit": manifest["motioncrafter_commit"],
            "prediction_manifest": str(Path(manifest_path).resolve()),
            "prediction_manifest_sha256": sha256(manifest_path),
            "manual_tracks_sha256": sha256(manual_tracks_path),
            "physics_trajectory_sha256": (
                sha256(physics_trajectory_path) if physics_trajectory_path else None
            ),
            "corrected_trajectory_sha256": (
                sha256(corrected_trajectory_path) if corrected_trajectory_path else None
            ),
            "final_data_sha256": sha256(final_data_path) if final_data_path else None,
        },
        "claim_boundary": {
            "same_view": "metric RGB-D geometry after train-only global Sim(3)",
            "manual_flow": "sparse visible tracks; fusion weights use training tracks",
            "heldout_views": "object point-set coverage, not dense correspondence truth",
            "uncertainty": "scalar training MSE, not calibrated per-pixel covariance",
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_manifest", type=Path)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-camera", type=int, default=0)
    parser.add_argument("--heldout-camera", type=int, action="append", dest="heldout_cameras")
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--manual-tracks", type=Path)
    parser.add_argument("--physics-trajectory", type=Path)
    parser.add_argument("--corrected-trajectory", type=Path)
    parser.add_argument("--final-data", type=Path)
    parser.add_argument(
        "--product",
        choices=("disjoint", "latent_linear"),
        action="append",
        dest="products",
    )
    parser.add_argument("--maximum-correspondences", type=int, default=100_000)
    parser.add_argument("--maximum-heldout-points", type=int, default=2000)
    parser.add_argument("--heldout-frame-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args(argv)
    manual_tracks = arguments.manual_tracks or arguments.case_directory / "gt_track_3d.pkl"
    final_data = arguments.final_data or arguments.case_directory / "final_data.pkl"
    result = run_experiment(
        arguments.prediction_manifest,
        arguments.case_directory,
        arguments.output,
        input_camera=arguments.input_camera,
        heldout_cameras=tuple(arguments.heldout_cameras or (1, 2)),
        fit_end_frame=arguments.fit_end_frame,
        manual_tracks_path=manual_tracks,
        physics_trajectory_path=arguments.physics_trajectory,
        corrected_trajectory_path=arguments.corrected_trajectory,
        final_data_path=final_data,
        products=tuple(arguments.products or ("disjoint", "latent_linear")),
        maximum_correspondences=arguments.maximum_correspondences,
        maximum_heldout_points=arguments.maximum_heldout_points,
        heldout_frame_stride=arguments.heldout_frame_stride,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
