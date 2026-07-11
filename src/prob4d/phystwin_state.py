"""Evaluate a MotionCrafter endpoint state update followed by PhysTwin rollout."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .phystwin import CoverResizeCrop, PhysTwinCase, nearest_neighbor_indices
from .phystwin_experiment import (
    ErrorSummary,
    fit_metric_gauge,
    git_commit,
    load_manual_flow_samples,
    load_physics_trajectory,
    load_prediction_product,
    sha256,
)
from .sim3 import so3_log

FloatArray = NDArray[np.floating]


def anchored_physics_rollout(
    initial_positions: FloatArray,
    trajectory: FloatArray,
    *,
    endpoint_frame: int,
    output_frame_count: int,
    preserve_endpoint_offset: bool = True,
) -> FloatArray:
    """Attach observed points to nearest simulator nodes and preserve endpoint offsets."""

    initial = np.asarray(initial_positions, dtype=np.float64)
    physics = np.asarray(trajectory, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError("initial_positions must have shape (N, 3)")
    if physics.ndim != 3 or physics.shape[2] != 3:
        raise ValueError("trajectory must have shape (T, M, 3)")
    if not 0 <= endpoint_frame < physics.shape[0]:
        raise ValueError("endpoint_frame lies outside the physics trajectory")
    result = np.full((output_frame_count, initial.shape[0], 3), np.nan, dtype=np.float64)
    active = np.all(np.isfinite(initial), axis=1)
    if not np.any(active):
        return result
    nearest, _ = nearest_neighbor_indices(initial[active], physics[endpoint_frame])
    offsets = initial[active] - physics[endpoint_frame, nearest]
    if not preserve_endpoint_offset:
        offsets = np.zeros_like(offsets)
    stop = min(output_frame_count, physics.shape[0])
    result[:stop, active] = physics[:stop, nearest] + offsets[None, :, :]
    return result


def paired_frame_block_bootstrap(
    method_error_m: FloatArray,
    baseline_error_m: FloatArray,
    frame_indices: NDArray[np.int64],
    *,
    block_length: int = 5,
    repetitions: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    """Circular moving-block bootstrap of a paired sample-weighted mean difference."""

    method = np.asarray(method_error_m, dtype=np.float64)
    baseline = np.asarray(baseline_error_m, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if method.shape != baseline.shape or method.shape != frames.shape or method.ndim != 1:
        raise ValueError("errors and frame_indices must share one-dimensional shape")
    if repetitions < 1 or block_length < 1:
        raise ValueError("repetitions and block_length must be positive")
    unique_frames = np.unique(frames)
    if unique_frames.size < 2:
        raise ValueError("block bootstrap requires at least two test frames")
    differences = method - baseline
    frame_sums = np.asarray(
        [np.sum(differences[frames == frame]) for frame in unique_frames],
        dtype=np.float64,
    )
    frame_counts = np.asarray(
        [np.count_nonzero(frames == frame) for frame in unique_frames],
        dtype=np.float64,
    )
    actual_block = min(block_length, unique_frames.size)
    block_count = int(np.ceil(unique_frames.size / actual_block))
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, unique_frames.size, size=(repetitions, block_count))
    offsets = np.arange(actual_block, dtype=np.int64)
    selected = (starts[:, :, None] + offsets[None, None, :]) % unique_frames.size
    selected = selected.reshape(repetitions, -1)[:, : unique_frames.size]
    bootstrap = np.sum(frame_sums[selected], axis=1) / np.sum(frame_counts[selected], axis=1)
    observed = float(np.mean(differences))
    return {
        "sample_count": int(method.size),
        "frame_count": int(unique_frames.size),
        "block_length_frames": int(actual_block),
        "repetitions": repetitions,
        "method_minus_baseline_mean_m": observed,
        "interval_95_m": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "probability_method_better": float(np.mean(bootstrap < 0.0)),
        "paired_frame_rows": [
            {
                "frame": int(frame),
                "difference_sum_m": float(total),
                "count": int(count),
                "mean_difference_m": float(total / count),
            }
            for frame, total, count in zip(
                unique_frames,
                frame_sums,
                frame_counts,
                strict=True,
            )
        ],
    }


def _summarize_positions(
    errors: FloatArray,
    frames: NDArray[np.int64],
    *,
    fit_end_frame: int,
) -> dict[str, object]:
    result: dict[str, object] = {"overall": ErrorSummary.from_vectors(errors).to_dict()}
    horizons = frames - fit_end_frame + 1
    result["horizons"] = [
        {
            "horizon_frames": int(horizon),
            **ErrorSummary.from_vectors(errors[horizons == horizon]).to_dict(),
        }
        for horizon in np.unique(horizons)
    ]
    buckets = {}
    selections = {
        "early_1_5": (horizons >= 1) & (horizons <= 5),
        "middle_6_15": (horizons >= 6) & (horizons <= 15),
        "late_16_plus": horizons >= 16,
    }
    for name, selection in selections.items():
        if np.any(selection):
            buckets[name] = ErrorSummary.from_vectors(errors[selection]).to_dict()
    result["buckets"] = buckets
    return result


def _endpoint_initializers(
    samples,
    manual_truth: FloatArray,
    *,
    endpoint_frame: int,
    fit_end_frame: int,
) -> dict[str, FloatArray]:
    if samples.track_indices is None:
        raise ValueError("manual samples do not retain track identities")
    track_count = manual_truth.shape[1]
    visual = np.full((track_count, 3), np.nan, dtype=np.float64)
    for track in range(track_count):
        selected = (samples.frame_indices == endpoint_frame) & (
            samples.track_indices == track
        )
        if np.count_nonzero(selected) == 1:
            visual[track] = samples.visual_current_world[selected][0]

    train_bias = np.zeros((track_count, 3), dtype=np.float64)
    for track in range(track_count):
        selected = (samples.frame_indices < fit_end_frame) & (samples.track_indices == track)
        if np.any(selected):
            train_bias[track] = np.mean(
                samples.truth_current_world[selected]
                - samples.visual_current_world[selected],
                axis=0,
            )
        else:
            train_bias[track] = np.nan
    oracle = np.asarray(manual_truth[endpoint_frame], dtype=np.float64)
    return {
        "motioncrafter_endpoint": visual,
        "train_label_bias_corrected_endpoint": visual + train_bias,
        "oracle_truth_endpoint": oracle,
    }


def state_forecast_metrics(
    samples,
    manual_truth: FloatArray,
    physics_trajectory: FloatArray | None,
    corrected_trajectory: FloatArray | None,
    *,
    fit_end_frame: int,
    maximum_frame: int,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Evaluate no-future-visual state forecasts on visible and missing observations."""

    if samples.track_indices is None:
        raise ValueError("manual samples do not retain track identities")
    endpoint_frame = fit_end_frame - 1
    initializers = _endpoint_initializers(
        samples,
        manual_truth,
        endpoint_frame=endpoint_frame,
        fit_end_frame=fit_end_frame,
    )
    output_frame_count = min(manual_truth.shape[0], maximum_frame + 1)
    trajectories: dict[str, FloatArray] = {}
    for initial_name, initial in initializers.items():
        persistence = np.broadcast_to(
            initial[None, :, :],
            (output_frame_count, initial.shape[0], 3),
        ).copy()
        trajectories[f"{initial_name}_persistence"] = persistence
        for physics_name, physics in (
            ("physics", physics_trajectory),
            ("corrected_physics", corrected_trajectory),
        ):
            if physics is not None:
                trajectories[f"{initial_name}_{physics_name}_forecast"] = (
                    anchored_physics_rollout(
                        initial,
                        physics,
                        endpoint_frame=endpoint_frame,
                        output_frame_count=output_frame_count,
                    )
                )
                if initial_name == "motioncrafter_endpoint":
                    trajectories[f"{initial_name}_{physics_name}_association_only"] = (
                        anchored_physics_rollout(
                            initial,
                            physics,
                            endpoint_frame=endpoint_frame,
                            output_frame_count=output_frame_count,
                            preserve_endpoint_offset=False,
                        )
                    )

    visible = (samples.frame_indices >= fit_end_frame) & (
        samples.frame_indices <= maximum_frame
    )
    visible_frames = samples.frame_indices[visible]
    visible_tracks = samples.track_indices[visible]
    visible_truth = samples.truth_current_world[visible]
    visual_errors = samples.visual_current_world[visible] - visible_truth
    visible_results: dict[str, object] = {
        "motioncrafter_per_frame": _summarize_positions(
            visual_errors,
            visible_frames,
            fit_end_frame=fit_end_frame,
        )
    }
    visual_norm = np.linalg.norm(visual_errors, axis=1)
    comparisons: dict[str, object] = {}
    visible_error_norms: dict[str, FloatArray] = {}
    for index, (name, trajectory) in enumerate(trajectories.items()):
        predicted = trajectory[visible_frames, visible_tracks]
        active = np.all(np.isfinite(predicted), axis=1)
        if not np.any(active):
            continue
        errors = predicted[active] - visible_truth[active]
        visible_results[name] = _summarize_positions(
            errors,
            visible_frames[active],
            fit_end_frame=fit_end_frame,
        )
        full_norm = np.full(visible_frames.shape, np.nan, dtype=np.float64)
        full_norm[active] = np.linalg.norm(errors, axis=1)
        visible_error_norms[name] = full_norm
        comparisons[name] = paired_frame_block_bootstrap(
            full_norm[active],
            visual_norm[active],
            visible_frames[active],
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + index,
        )

    state_update_comparisons: dict[str, object] = {}
    for index, physics_name in enumerate(("physics", "corrected_physics")):
        state_name = f"motioncrafter_endpoint_{physics_name}_forecast"
        association_name = f"motioncrafter_endpoint_{physics_name}_association_only"
        if state_name not in visible_error_norms or association_name not in visible_error_norms:
            continue
        state_error = visible_error_norms[state_name]
        association_error = visible_error_norms[association_name]
        active = np.isfinite(state_error) & np.isfinite(association_error)
        state_update_comparisons[physics_name] = paired_frame_block_bootstrap(
            state_error[active],
            association_error[active],
            visible_frames[active],
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + 100 + index,
        )

    all_track_results: dict[str, object] = {}
    frame_grid = np.arange(fit_end_frame, output_frame_count, dtype=np.int64)
    for name, trajectory in trajectories.items():
        prediction = trajectory[frame_grid]
        truth = manual_truth[frame_grid]
        active = np.all(np.isfinite(prediction), axis=2) & np.all(np.isfinite(truth), axis=2)
        if not np.any(active):
            continue
        repeated_frames = np.broadcast_to(frame_grid[:, None], active.shape)[active]
        all_track_results[name] = _summarize_positions(
            prediction[active] - truth[active],
            repeated_frames,
            fit_end_frame=fit_end_frame,
        )

    initializer_errors = {}
    endpoint_truth = manual_truth[endpoint_frame]
    for name, initializer in initializers.items():
        active = np.all(np.isfinite(initializer), axis=1) & np.all(
            np.isfinite(endpoint_truth), axis=1
        )
        initializer_errors[name] = ErrorSummary.from_vectors(
            initializer[active] - endpoint_truth[active]
        ).to_dict()
    return {
        "endpoint_frame": endpoint_frame,
        "initializer_error": initializer_errors,
        "visible_future_tracks": visible_results,
        "paired_block_bootstrap_vs_motioncrafter_per_frame": comparisons,
        "state_update_vs_node_association_only": state_update_comparisons,
        "all_finite_future_tracks_no_future_visual": all_track_results,
    }


def run_state_experiment(
    manifest_path: str | Path,
    case_directory: str | Path,
    output_path: str | Path,
    *,
    product: str,
    input_camera: int,
    fit_end_frame: int,
    manual_tracks_path: str | Path,
    physics_trajectory_path: str | Path | None,
    corrected_trajectory_path: str | Path | None,
    final_data_path: str | Path | None,
    maximum_correspondences: int = 100_000,
    bootstrap_repetitions: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    case = PhysTwinCase.from_directory(case_directory)
    prediction, manifest = load_prediction_product(manifest_path, product)
    crop = CoverResizeCrop.from_shapes(
        case.source_height,
        case.source_width,
        prediction.shape[1],
        prediction.shape[2],
    )
    metric_truth = case.metric_truth(
        prediction.frame_indices,
        input_camera,
        crop,
        object_only=True,
    )
    gauge = fit_metric_gauge(
        prediction,
        metric_truth.point_map,
        metric_truth.valid_mask,
        fit_end_frame=fit_end_frame,
        maximum_correspondences=maximum_correspondences,
        seed=seed,
    )
    samples = load_manual_flow_samples(
        case,
        prediction,
        gauge.transform,
        crop,
        manual_tracks_path,
        camera=input_camera,
        occlusion_tolerance_m=0.03,
    )
    with Path(manual_tracks_path).open("rb") as handle:
        manual_truth = np.asarray(pickle.load(handle), dtype=np.float64)
    physics = load_physics_trajectory(physics_trajectory_path, final_data_path)
    corrected = load_physics_trajectory(corrected_trajectory_path, final_data_path)
    maximum_frame = int(np.max(prediction.frame_indices))
    metrics = state_forecast_metrics(
        samples,
        manual_truth,
        physics,
        corrected,
        fit_end_frame=fit_end_frame,
        maximum_frame=maximum_frame,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=seed,
    )
    prediction_field = {
        "disjoint": "disjoint_baseline",
        "latent_linear": "latent_linear_baseline",
    }[product]
    prediction_path = Path(manifest_path).resolve().parent / manifest[prediction_field]
    result = {
        "schema_version": 1,
        "status": "MotionCrafter endpoint state update followed by PhysTwin rollout",
        "case": Path(case_directory).name,
        "product": product,
        "input_camera": input_camera,
        "source_frames": prediction.frame_indices.tolist(),
        "fit_end_frame": fit_end_frame,
        "future_visual_observations_used_by_forecasts": False,
        "gauge": {
            "scale": gauge.transform.scale,
            "rotation_vector": so3_log(gauge.transform.rotation).tolist(),
            "translation_m": gauge.transform.translation.tolist(),
            "fit_residual_rms_m": gauge.residual_rms,
            "inlier_fraction": gauge.inlier_fraction,
        },
        "state_forecast": metrics,
        "provenance": {
            "prob4d_commit": git_commit(Path(__file__).resolve().parents[2]),
            "motioncrafter_commit": manifest["motioncrafter_commit"],
            "prediction_manifest": str(Path(manifest_path).resolve()),
            "prediction_manifest_sha256": sha256(manifest_path),
            "prediction_sha256": sha256(prediction_path),
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
            "state_association": (
                "manual-track identities select endpoint pixels; MotionCrafter supplies "
                "the endpoint 3D state"
            ),
            "headline_initializer": "motioncrafter_endpoint uses no manual 3D coordinates",
            "label_dependent_controls": (
                "train_label_bias_corrected and oracle_truth methods are controls only"
            ),
            "forecast_information": (
                "future PhysTwin action-conditioned trajectory is known; no future RGB or depth "
                "updates enter forecast methods"
            ),
            "missing_evidence": (
                "all_finite_future_tracks includes tracks absent from camera-visible evaluation"
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_manifest", type=Path)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--product", choices=("disjoint", "latent_linear"), default="disjoint")
    parser.add_argument("--input-camera", type=int, default=0)
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--manual-tracks", type=Path)
    parser.add_argument("--physics-trajectory", type=Path)
    parser.add_argument("--corrected-trajectory", type=Path)
    parser.add_argument("--final-data", type=Path)
    parser.add_argument("--maximum-correspondences", type=int, default=100_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args(argv)
    manual_tracks = arguments.manual_tracks or arguments.case_directory / "gt_track_3d.pkl"
    final_data = arguments.final_data or arguments.case_directory / "final_data.pkl"
    result = run_state_experiment(
        arguments.prediction_manifest,
        arguments.case_directory,
        arguments.output,
        product=arguments.product,
        input_camera=arguments.input_camera,
        fit_end_frame=arguments.fit_end_frame,
        manual_tracks_path=manual_tracks,
        physics_trajectory_path=arguments.physics_trajectory,
        corrected_trajectory_path=arguments.corrected_trajectory,
        final_data_path=final_data,
        maximum_correspondences=arguments.maximum_correspondences,
        bootstrap_repetitions=arguments.bootstrap_repetitions,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
