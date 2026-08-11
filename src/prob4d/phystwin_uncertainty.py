"""Evaluate sampled MotionCrafter uncertainty with PhysTwin flow proposals."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .phystwin import CoverResizeCrop, PhysTwinCase
from .phystwin_experiment import (
    ErrorSummary,
    ManualFlowSamples,
    fit_metric_gauge,
    git_commit,
    load_manual_flow_samples,
    load_physics_trajectory,
    load_prediction_product,
    physics_flow_for_samples,
    same_view_metrics,
    sha256,
)
from .sim3 import so3_log

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class EnsembleManualFlowSamples:
    """Manual-track observations shared by every stochastic prediction."""

    frame_indices: NDArray[np.int64]
    track_indices: NDArray[np.int64]
    visual_current_samples: FloatArray
    visual_flow_samples: FloatArray
    truth_current_world: FloatArray
    truth_next_world: FloatArray

    def __post_init__(self) -> None:
        count = int(np.asarray(self.frame_indices).size)
        if np.asarray(self.track_indices).shape != (count,):
            raise ValueError("track_indices must have shape (N,)")
        if self.visual_current_samples.ndim != 3:
            raise ValueError("visual_current_samples must have shape (K, N, 3)")
        sample_count = self.visual_current_samples.shape[0]
        if sample_count < 2:
            raise ValueError("at least two stochastic predictions are required")
        expected = (sample_count, count, 3)
        if self.visual_current_samples.shape != expected:
            raise ValueError("visual_current_samples must have shape (K, N, 3)")
        if self.visual_flow_samples.shape != expected:
            raise ValueError("visual_flow_samples must have shape (K, N, 3)")
        if self.truth_current_world.shape != (count, 3):
            raise ValueError("truth_current_world must have shape (N, 3)")
        if self.truth_next_world.shape != (count, 3):
            raise ValueError("truth_next_world must have shape (N, 3)")


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean_nis: float
    median_nis: float
    mean_nll: float
    coverage_50: float
    coverage_80: float
    coverage_95: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def intersect_manual_flow_samples(
    sample_sets: list[ManualFlowSamples],
) -> EnsembleManualFlowSamples:
    """Align stochastic samples by persistent manual-track identity."""

    if len(sample_sets) < 2:
        raise ValueError("at least two manual-flow sample sets are required")
    lookup_tables: list[dict[tuple[int, int], int]] = []
    for samples in sample_sets:
        if samples.track_indices is None:
            raise ValueError("manual-flow samples do not contain track identities")
        keys = list(
            zip(
                np.asarray(samples.frame_indices, dtype=np.int64).tolist(),
                np.asarray(samples.track_indices, dtype=np.int64).tolist(),
                strict=True,
            )
        )
        lookup = {key: index for index, key in enumerate(keys)}
        if len(lookup) != len(keys):
            raise ValueError("manual-flow samples contain duplicate frame-track identities")
        lookup_tables.append(lookup)

    common = set(lookup_tables[0])
    for lookup in lookup_tables[1:]:
        common.intersection_update(lookup)
    keys = sorted(common)
    if not keys:
        raise ValueError("stochastic predictions have no shared visible manual tracks")

    indices = [
        np.asarray([lookup[key] for key in keys], dtype=np.int64)
        for lookup in lookup_tables
    ]
    current = np.stack(
        [
            samples.visual_current_world[index]
            for samples, index in zip(sample_sets, indices, strict=True)
        ]
    )
    flow = np.stack(
        [
            samples.visual_flow_world[index]
            for samples, index in zip(sample_sets, indices, strict=True)
        ]
    )
    truth_current = sample_sets[0].truth_current_world[indices[0]]
    truth_next = sample_sets[0].truth_next_world[indices[0]]
    for samples, index in zip(sample_sets[1:], indices[1:], strict=True):
        if not np.allclose(samples.truth_current_world[index], truth_current):
            raise ValueError("stochastic samples disagree on current manual-track truth")
        if not np.allclose(samples.truth_next_world[index], truth_next):
            raise ValueError("stochastic samples disagree on next manual-track truth")
    return EnsembleManualFlowSamples(
        frame_indices=np.asarray([key[0] for key in keys], dtype=np.int64),
        track_indices=np.asarray([key[1] for key in keys], dtype=np.int64),
        visual_current_samples=current,
        visual_flow_samples=flow,
        truth_current_world=truth_current,
        truth_next_world=truth_next,
    )


def sample_covariances(
    samples: FloatArray,
    *,
    shrinkage: float = 0.2,
    variance_floor_m2: float = 6.25e-8,
) -> FloatArray:
    """Return regularized empirical 3D covariance for every sample location."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] != 3:
        raise ValueError("samples must have shape (K, N, 3) with K >= 2")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if variance_floor_m2 <= 0.0:
        raise ValueError("variance_floor_m2 must be positive")
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = np.einsum("kni,knj->nij", centered, centered) / (values.shape[0] - 1)
    isotropic_variance = np.trace(covariance, axis1=1, axis2=2) / 3.0
    identity = np.eye(3, dtype=np.float64)[None, :, :]
    return (
        (1.0 - shrinkage) * covariance
        + shrinkage * isotropic_variance[:, None, None] * identity
        + variance_floor_m2 * identity
    )


def error_second_moment(
    errors: FloatArray,
    *,
    shrinkage: float = 0.2,
    variance_floor_m2: float = 6.25e-8,
) -> FloatArray:
    """Estimate a regularized error second moment, retaining systematic bias."""

    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise ValueError("errors must have shape (N, 3) with N >= 2")
    moment = values.T @ values / values.shape[0]
    isotropic = float(np.trace(moment) / 3.0)
    return (
        (1.0 - shrinkage) * moment
        + shrinkage * isotropic * np.eye(3, dtype=np.float64)
        + variance_floor_m2 * np.eye(3, dtype=np.float64)
    )


def normalized_squared_errors(errors: FloatArray, covariances: FloatArray) -> FloatArray:
    values = np.asarray(errors, dtype=np.float64)
    covariance = np.asarray(covariances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("errors must have shape (N, 3)")
    covariance = np.broadcast_to(covariance, (values.shape[0], 3, 3))
    solved = np.linalg.solve(covariance, values[..., None])[..., 0]
    return np.einsum("ni,ni->n", values, solved)


def calibrate_covariance_scale(errors: FloatArray, covariances: FloatArray) -> float:
    """Scale covariance so preboundary mean NIS equals its three dimensions."""

    scale = float(np.mean(normalized_squared_errors(errors, covariances)) / 3.0)
    return max(scale, np.finfo(np.float64).eps)


def _positive_definite_covariance_batch(
    value: FloatArray,
    *,
    count: int,
    name: str,
) -> FloatArray:
    """Broadcast and validate one symmetric positive-definite covariance batch."""

    raw = np.asarray(value, dtype=np.float64)
    try:
        covariance = np.broadcast_to(raw, (count, 3, 3))
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to shape ({count}, 3, 3)") from error
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (covariance + covariance.swapaxes(1, 2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(1, 2)), 1.0)
    if not np.allclose(
        covariance,
        symmetric,
        atol=1e-12 * scale[:, None, None],
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return symmetric


def _batched_cholesky_solve(cholesky: FloatArray, right_hand_side: FloatArray) -> FloatArray:
    forward = np.linalg.solve(cholesky, right_hand_side)
    return np.linalg.solve(cholesky.swapaxes(1, 2), forward)


def gaussian_product(
    mean_a: FloatArray,
    covariance_a: FloatArray,
    mean_b: FloatArray,
    covariance_b: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fuse independent Gaussian estimates without explicit matrix inversion."""

    first = np.asarray(mean_a, dtype=np.float64)
    second = np.asarray(mean_b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("means must share shape (N, 3)")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("means must be finite")
    cov_a = _positive_definite_covariance_batch(
        covariance_a,
        count=first.shape[0],
        name="covariance_a",
    )
    cov_b = _positive_definite_covariance_batch(
        covariance_b,
        count=first.shape[0],
        name="covariance_b",
    )

    summed = cov_a + cov_b
    cholesky = np.linalg.cholesky(summed)
    innovation = (second - first)[..., None]
    solved_innovation = _batched_cholesky_solve(cholesky, innovation)[..., 0]
    mean = first + np.einsum("nij,nj->ni", cov_a, solved_innovation)

    solved_b = _batched_cholesky_solve(cholesky, cov_b)
    covariance = np.einsum("nij,njk->nik", cov_a, solved_b)
    covariance = 0.5 * (covariance + covariance.swapaxes(1, 2))
    return mean, covariance


def distribution_summary(errors: FloatArray, covariances: FloatArray) -> DistributionSummary:
    covariance = np.broadcast_to(
        np.asarray(covariances, dtype=np.float64),
        (np.asarray(errors).shape[0], 3, 3),
    )
    nis = normalized_squared_errors(errors, covariance)
    _, log_determinant = np.linalg.slogdet(covariance)
    nll = 0.5 * (nis + log_determinant + 3.0 * np.log(2.0 * np.pi))
    return DistributionSummary(
        count=int(nis.size),
        mean_nis=float(np.mean(nis)),
        median_nis=float(np.median(nis)),
        mean_nll=float(np.mean(nll)),
        coverage_50=float(np.mean(nis <= 2.365973884)),
        coverage_80=float(np.mean(nis <= 4.641627676)),
        coverage_95=float(np.mean(nis <= 7.814727903)),
    )


def _method_result(
    flow: FloatArray,
    covariance: FloatArray,
    truth_flow: FloatArray,
    visual_current: FloatArray,
    truth_next: FloatArray,
    test: NDArray[np.bool_],
) -> dict[str, object]:
    flow_error = flow[test] - truth_flow[test]
    endpoint_error = visual_current[test] + flow[test] - truth_next[test]
    selected_covariance = np.broadcast_to(covariance, (flow.shape[0], 3, 3))[test]
    return {
        "flow_epe": ErrorSummary.from_vectors(flow_error).to_dict(),
        "endpoint_error": ErrorSummary.from_vectors(endpoint_error).to_dict(),
        "flow_distribution": distribution_summary(flow_error, selected_covariance).to_dict(),
    }


def ensemble_flow_metrics(
    samples: EnsembleManualFlowSamples,
    physics_trajectory: FloatArray | None,
    corrected_trajectory: FloatArray | None,
    *,
    fit_end_frame: int,
) -> dict[str, object]:
    """Calibrate stochastic visual flow and fuse it with physical flow."""

    training = samples.frame_indices < fit_end_frame
    test = samples.frame_indices >= fit_end_frame
    if np.count_nonzero(training) < 2 or not np.any(test):
        raise ValueError("shared manual tracks must cover fit and test intervals")
    truth_flow = samples.truth_next_world - samples.truth_current_world
    visual_current = np.mean(samples.visual_current_samples, axis=0)
    visual_flow = np.mean(samples.visual_flow_samples, axis=0)
    visual_covariance_raw = sample_covariances(samples.visual_flow_samples)
    visual_errors = visual_flow - truth_flow
    visual_scale = calibrate_covariance_scale(
        visual_errors[training], visual_covariance_raw[training]
    )
    visual_covariance = visual_scale * visual_covariance_raw

    representative = ManualFlowSamples(
        frame_indices=samples.frame_indices,
        visual_current_world=visual_current,
        visual_flow_world=visual_flow,
        truth_current_world=samples.truth_current_world,
        truth_next_world=samples.truth_next_world,
        track_indices=samples.track_indices,
    )
    flows: dict[str, FloatArray] = {"visual_ensemble_mean": visual_flow}
    covariances: dict[str, FloatArray] = {"visual_ensemble_mean": visual_covariance}
    calibration: dict[str, object] = {
        "visual": {
            "sample_count": int(samples.visual_flow_samples.shape[0]),
            "covariance_scale": visual_scale,
            "raw_mean_axis_std_m": float(
                np.mean(np.sqrt(np.trace(visual_covariance_raw, axis1=1, axis2=2) / 3.0))
            ),
            "calibrated_mean_axis_std_m": float(
                np.mean(np.sqrt(np.trace(visual_covariance, axis1=1, axis2=2) / 3.0))
            ),
            "raw_training_distribution": distribution_summary(
                visual_errors[training], visual_covariance_raw[training]
            ).to_dict(),
            "calibrated_training_distribution": distribution_summary(
                visual_errors[training], visual_covariance[training]
            ).to_dict(),
        }
    }

    physical_proposals = {
        "physics": physics_trajectory,
        "corrected_physics": corrected_trajectory,
    }
    for name, trajectory in physical_proposals.items():
        if trajectory is None:
            continue
        physical_flow = physics_flow_for_samples(representative, trajectory)
        physical_error = physical_flow - truth_flow
        physical_covariance = error_second_moment(physical_error[training])
        flows[name] = physical_flow
        covariances[name] = physical_covariance
        calibration[name] = {
            "training_flow_epe": ErrorSummary.from_vectors(physical_error[training]).to_dict(),
            "training_distribution": distribution_summary(
                physical_error[training], physical_covariance
            ).to_dict(),
            "mean_axis_std_m": float(np.sqrt(np.trace(physical_covariance) / 3.0)),
        }

        fixed_name = f"fixed_visual_{name}"
        flows[fixed_name] = 0.5 * (visual_flow + physical_flow)
        covariances[fixed_name] = 0.25 * (visual_covariance + physical_covariance)

        visual_mse = float(np.mean(visual_errors[training] ** 2))
        physical_mse = float(np.mean(physical_error[training] ** 2))
        visual_weight = physical_mse / max(
            visual_mse + physical_mse, np.finfo(np.float64).eps
        )
        scalar_name = f"scalar_visual_{name}"
        flows[scalar_name] = visual_weight * visual_flow + (1.0 - visual_weight) * physical_flow
        covariances[scalar_name] = (
            visual_weight**2 * visual_covariance
            + (1.0 - visual_weight) ** 2 * physical_covariance
        )
        calibration[scalar_name] = {
            "visual_weight": visual_weight,
            "contract": "single inverse-training-MSE scalar",
        }

        raw_name = f"raw_covariance_visual_{name}"
        flows[raw_name], covariances[raw_name] = gaussian_product(
            visual_flow,
            visual_covariance_raw,
            physical_flow,
            physical_covariance,
        )
        fused_name = f"calibrated_covariance_visual_{name}"
        flows[fused_name], covariances[fused_name] = gaussian_product(
            visual_flow,
            visual_covariance,
            physical_flow,
            physical_covariance,
        )

    test_results = {
        name: _method_result(
            flow,
            covariances[name],
            truth_flow,
            visual_current,
            samples.truth_next_world,
            test,
        )
        for name, flow in flows.items()
    }
    return {
        "ensemble_size": int(samples.visual_flow_samples.shape[0]),
        "shared_fit_sample_count": int(np.count_nonzero(training)),
        "shared_test_sample_count": int(np.count_nonzero(test)),
        "calibration": calibration,
        "test": test_results,
    }


def run_ensemble_experiment(
    manifest_paths: list[str | Path],
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
    seed: int = 42,
) -> dict[str, object]:
    if len(manifest_paths) < 2:
        raise ValueError("at least two prediction manifests are required")
    case = PhysTwinCase.from_directory(case_directory)
    physics = load_physics_trajectory(physics_trajectory_path, final_data_path)
    corrected = load_physics_trajectory(corrected_trajectory_path, final_data_path)
    sample_sets: list[ManualFlowSamples] = []
    sample_reports: list[dict[str, object]] = []
    frame_contract: list[int] | None = None
    model_seeds: set[int] = set()

    for index, manifest_path in enumerate(manifest_paths):
        prediction, manifest = load_prediction_product(manifest_path, product)
        prediction_field = {
            "disjoint": "disjoint_baseline",
            "latent_linear": "latent_linear_baseline",
        }[product]
        prediction_path = Path(manifest_path).resolve().parent / manifest[prediction_field]
        config = manifest.get("config", {})
        if config.get("model_type") != "diff":
            raise ValueError("sampled uncertainty requires MotionCrafter diffusion manifests")
        model_seed = int(config["seed"])
        if model_seed in model_seeds:
            raise ValueError("diffusion manifests must use distinct random seeds")
        model_seeds.add(model_seed)
        current_contract = prediction.frame_indices.tolist()
        if frame_contract is None:
            frame_contract = current_contract
        elif frame_contract != current_contract:
            raise ValueError("diffusion manifests use different source frames")

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
            seed=seed + index,
        )
        sample_sets.append(
            load_manual_flow_samples(
                case,
                prediction,
                gauge.transform,
                crop,
                manual_tracks_path,
                camera=input_camera,
                occlusion_tolerance_m=0.03,
            )
        )
        sample_reports.append(
            {
                "manifest": str(Path(manifest_path).resolve()),
                "manifest_sha256": sha256(manifest_path),
                "prediction_sha256": sha256(prediction_path),
                "model_seed": model_seed,
                "gauge": {
                    "scale": gauge.transform.scale,
                    "rotation_vector": so3_log(gauge.transform.rotation).tolist(),
                    "translation_m": gauge.transform.translation.tolist(),
                    "fit_residual_rms_m": gauge.residual_rms,
                    "inlier_fraction": gauge.inlier_fraction,
                },
                "same_view_object_geometry": same_view_metrics(
                    prediction,
                    truth.point_map,
                    truth.valid_mask,
                    gauge.transform,
                    fit_end_frame=fit_end_frame,
                ),
            }
        )

    aligned = intersect_manual_flow_samples(sample_sets)
    metrics = ensemble_flow_metrics(
        aligned,
        physics,
        corrected,
        fit_end_frame=fit_end_frame,
    )
    result = {
        "schema_version": 1,
        "status": "sampled MotionCrafter uncertainty on PhysTwin manual tracks",
        "case": Path(case_directory).name,
        "product": product,
        "input_camera": input_camera,
        "source_frames": frame_contract,
        "fit_end_frame": fit_end_frame,
        "future_labels_used_for_gauge_or_calibration": False,
        "samples": sample_reports,
        "manual_track_flow": metrics,
        "provenance": {
            "prob4d_commit": git_commit(Path(__file__).resolve().parents[2]),
            "motioncrafter_commits": sorted(
                {
                    json.loads(Path(path).read_text(encoding="utf-8"))["motioncrafter_commit"]
                    for path in manifest_paths
                }
            ),
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
            "visual_covariance": "empirical covariance across independently seeded diffusion runs",
            "covariance_calibration": "one global NIS scale fitted before fit_end_frame",
            "physics_covariance": "global error second moment fitted before fit_end_frame",
            "evaluation": "sparse camera-visible manual tracks, not dense flow ground truth",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", required=True, dest="manifests")
    parser.add_argument("--product", choices=("disjoint", "latent_linear"), default="latent_linear")
    parser.add_argument("--input-camera", type=int, default=0)
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--manual-tracks", type=Path)
    parser.add_argument("--physics-trajectory", type=Path)
    parser.add_argument("--corrected-trajectory", type=Path)
    parser.add_argument("--final-data", type=Path)
    parser.add_argument("--maximum-correspondences", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args(argv)
    manual_tracks = arguments.manual_tracks or arguments.case_directory / "gt_track_3d.pkl"
    final_data = arguments.final_data or arguments.case_directory / "final_data.pkl"
    result = run_ensemble_experiment(
        arguments.manifests,
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
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
