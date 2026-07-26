"""Export causally bounded Prob4D windows as ObservationBeliefV1 artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .alignment import WindowAlignment, align_windows
from .gauge import (
    FixedLagGaugeSmoother,
    RelativeGaugeConstraint,
    SequentialGaugeEstimator,
)
from .io import PredictionBundle, load_prediction_bundle
from .observation_contract import (
    ObservationBeliefExportV1,
    file_sha256,
    save_observation_belief_export,
)
from .sim3 import Sim3, so3_log, so3_right_jacobian
from .uncertainty import DepthDisagreementModel, accumulate_disagreement

GAUGE_FACTOR_NAMES = tuple(f"gauge_latent_{index}" for index in range(7))


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _deterministic_covariance_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    if symmetric.shape != (7, 7) or not np.all(np.isfinite(symmetric)):
        raise ValueError("gauge covariance must have finite shape (7, 7)")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    for column in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0
    return eigenvectors * np.sqrt(eigenvalues)[None]


def gauge_covariance_factor(
    values: np.ndarray,
    transform: Sim3,
    gauge_covariance: np.ndarray,
    *,
    include_translation: bool,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Return ``J L`` so shared gauge covariance remains low rank.

    The final axis has rank seven. Observations from the same window share the
    same latent gauge vector, preserving cross-point covariance that would be
    lost if only marginal 3x3 matrices were exported.
    """

    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("gauge factors require three-dimensional values")
    root = _deterministic_covariance_root(gauge_covariance)
    flattened = points.reshape(-1, 3)
    factors = np.empty((len(flattened), 3, 7), dtype=np.float64)
    identity = np.eye(3)
    right_jacobian = so3_right_jacobian(so3_log(transform.rotation))
    for start in range(0, len(flattened), chunk_size):
        stop = min(start + chunk_size, len(flattened))
        chunk = flattened[start:stop]
        scaled_rotated = transform.scale * np.einsum(
            "ij,nj->ni", transform.rotation, chunk
        )
        skew = np.zeros((len(chunk), 3, 3), dtype=np.float64)
        skew[:, 0, 1] = -chunk[:, 2]
        skew[:, 0, 2] = chunk[:, 1]
        skew[:, 1, 0] = chunk[:, 2]
        skew[:, 1, 2] = -chunk[:, 0]
        skew[:, 2, 0] = -chunk[:, 1]
        skew[:, 2, 1] = chunk[:, 0]
        jacobian = np.zeros((len(chunk), 3, 7), dtype=np.float64)
        jacobian[:, :, 0] = scaled_rotated
        jacobian[:, :, 1:4] = -transform.scale * np.einsum(
            "ij,njk,kl->nil",
            transform.rotation,
            skew,
            right_jacobian,
        )
        if include_translation:
            jacobian[:, :, 4:7] = identity
        factors[start:stop] = np.einsum("nij,jk->nik", jacobian, root)
    return factors.reshape(points.shape + (7,))


def _build_alignments(bundle: PredictionBundle) -> list[WindowAlignment]:
    alignments: list[WindowAlignment] = []
    for moving_index, moving in enumerate(bundle.overlap_windows):
        for reference in bundle.overlap_windows[:moving_index]:
            if reference.common_frames(moving).size:
                alignments.append(
                    align_windows(reference, moving, seed=moving_index)
                )
    return alignments


def _gauge_estimates(
    bundle: PredictionBundle,
    *,
    gauge_mode: str,
    fixed_lag: int,
):
    alignments = _build_alignments(bundle)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment)
        for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in bundle.overlap_windows]
    sequential = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    if gauge_mode == "sequential":
        estimates = sequential
    elif gauge_mode == "fixed_lag":
        estimates = FixedLagGaugeSmoother(lag=fixed_lag).smooth(
            ordered_ids,
            sequential,
            constraints,
        )
    else:
        raise ValueError("gauge_mode must be 'sequential' or 'fixed_lag'")
    return alignments, estimates


def _prior_reliability(
    parallel_disagreement: np.ndarray,
    lateral_disagreement: np.ndarray,
    parallel_variance: np.ndarray,
    lateral_variance: np.ndarray,
    overlap_count: np.ndarray,
) -> np.ndarray:
    normalized = (
        parallel_disagreement / np.maximum(parallel_variance, 1e-12)
        + lateral_disagreement / np.maximum(lateral_variance, 1e-12)
    )
    reliability = np.exp(-0.5 * np.minimum(normalized, 50.0))
    reliability = np.where(overlap_count > 0.0, reliability, 1.0)
    return np.clip(reliability, 0.05, 1.0)


def build_prob4d_observation_belief(
    manifest_path: str | Path,
    *,
    case_id: str,
    causal_frame_stop: int,
    pixel_stride: int = 4,
    effective_samples_per_group: float = 64.0,
    gauge_mode: str = "fixed_lag",
    fixed_lag: int = 4,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
) -> ObservationBeliefExportV1:
    """Build a prefix-only artifact from independently decoded overlap windows."""

    if not case_id:
        raise ValueError("case_id must be nonempty")
    if causal_frame_stop < 1:
        raise ValueError("causal_frame_stop must be positive")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    if not np.isfinite(effective_samples_per_group) or (
        effective_samples_per_group <= 0.0
    ):
        raise ValueError("effective_samples_per_group must be positive")
    if fixed_lag < 1:
        raise ValueError("fixed_lag must be positive")

    manifest = Path(manifest_path).resolve()
    bundle = load_prediction_bundle(manifest)
    alignments, estimates = _gauge_estimates(
        bundle,
        gauge_mode=gauge_mode,
        fixed_lag=fixed_lag,
    )
    windows = {window.window_id: window for window in bundle.overlap_windows}
    evidence = accumulate_disagreement(windows, alignments)
    model = uncertainty_model or DepthDisagreementModel()

    eligible_frames = sorted(
        {
            int(frame)
            for window in bundle.overlap_windows
            for frame in window.frame_indices
            if int(frame) < causal_frame_stop
        }
    )
    if not eligible_frames:
        raise ValueError("no overlap-window frame lies before causal_frame_stop")
    frame_to_group = {
        frame: group for group, frame in enumerate(eligible_frames)
    }

    means: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    entity_ids: list[np.ndarray] = []
    view_indices: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    correlation_groups: list[np.ndarray] = []
    factor_groups: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    associations: list[np.ndarray] = []
    local_covariances: list[np.ndarray] = []
    factors: list[np.ndarray] = []

    for window_index, window in enumerate(bundle.overlap_windows):
        estimate = estimates[window.window_id]
        transformed = estimate.global_from_local.transform_points(window.point_map)
        uncertainty = model.predict(window, evidence[window.window_id])
        local_covariance = uncertainty.transformed(
            estimate.global_from_local
        ).matrices()
        factor = gauge_covariance_factor(
            window.point_map,
            estimate.global_from_local,
            estimate.covariance,
            include_translation=True,
        )
        reliability = _prior_reliability(
            evidence[window.window_id].parallel_mean,
            evidence[window.window_id].lateral_mean,
            uncertainty.parallel_variance,
            uncertainty.lateral_variance,
            evidence[window.window_id].count,
        )
        height, width = window.shape[1:]
        sample_mask = np.zeros((height, width), dtype=bool)
        sample_mask[::pixel_stride, ::pixel_stride] = True
        linear_entity = np.arange(height * width, dtype=np.int64).reshape(
            height, width
        )
        for local_index, frame in enumerate(window.frame_indices):
            absolute_frame = int(frame)
            if absolute_frame >= causal_frame_stop:
                continue
            selected = window.valid_mask[local_index] & sample_mask
            if not np.any(selected):
                continue
            count = int(np.sum(selected))
            means.append(transformed[local_index][selected])
            frame_ids.append(np.full(count, absolute_frame, dtype=np.int64))
            entity_ids.append(linear_entity[selected])
            view_indices.append(np.zeros(count, dtype=np.int64))
            window_indices.append(
                np.full(count, window_index, dtype=np.int64)
            )
            correlation_groups.append(
                np.full(
                    count,
                    frame_to_group[absolute_frame],
                    dtype=np.int64,
                )
            )
            factor_groups.append(
                np.full(count, window_index, dtype=np.int64)
            )
            reliabilities.append(reliability[local_index][selected])
            associations.append(np.ones(count, dtype=np.float64))
            local_covariances.append(local_covariance[local_index][selected])
            factors.append(factor[local_index][selected])

    if not means:
        raise ValueError("no valid sampled observation remains in the causal prefix")
    mean_array = np.concatenate(means)
    frame_array = np.concatenate(frame_ids)
    entity_array = np.concatenate(entity_ids)
    view_array = np.concatenate(view_indices)
    window_array = np.concatenate(window_indices)
    correlation_array = np.concatenate(correlation_groups)
    factor_group_array = np.concatenate(factor_groups)
    reliability_array = np.concatenate(reliabilities)
    association_array = np.concatenate(associations)
    local_covariance_array = np.concatenate(local_covariances)
    factor_array = np.concatenate(factors)

    group_ids = np.unique(correlation_array)
    group_prior = np.empty(len(group_ids), dtype=np.float64)
    group_weight = np.empty(len(group_ids), dtype=np.float64)
    for position, group_id in enumerate(group_ids):
        selected = correlation_array == group_id
        group_prior[position] = float(
            np.quantile(reliability_array[selected], 0.25)
        )
        unique_entities = len(np.unique(entity_array[selected]))
        effective = min(effective_samples_per_group, float(unique_entities))
        group_weight[position] = min(
            1.0,
            effective / float(np.sum(selected)),
        )

    revision = source_revision or _git_revision()
    return ObservationBeliefExportV1(
        case_id=case_id,
        stream_id="prob4d:overlap-window-points",
        causal_frame_stop=causal_frame_stop,
        view_names=(view_name,),
        window_names=tuple(
            window.window_id for window in bundle.overlap_windows
        ),
        factor_names=GAUGE_FACTOR_NAMES,
        source_repository="FlorianPfaff/Prob4D",
        source_revision=revision,
        source_artifact_sha256=file_sha256(manifest),
        declared_frame_ids=np.asarray(eligible_frames, dtype=np.int64),
        mean_xyz_m=mean_array,
        frame_ids=frame_array,
        entity_ids=entity_array,
        view_indices=view_array,
        window_indices=window_array,
        correlation_group_ids=correlation_array,
        factor_group_ids=factor_group_array,
        prior_reliability=reliability_array,
        association_probability=association_array,
        local_covariance_m2=local_covariance_array,
        low_rank_factor_m=factor_array,
        group_ids=group_ids,
        group_prior_nominal_probability=group_prior,
        group_composite_weight=group_weight,
        metadata={
            "causal_information_boundary": {
                "maximum_frame_read": max(eligible_frames),
                "causal_frame_stop_exclusive": causal_frame_stop,
                "future_frames_read": 0,
            },
            "gauge_mode": gauge_mode,
            "fixed_lag": fixed_lag if gauge_mode == "fixed_lag" else None,
            "pixel_stride": pixel_stride,
            "effective_samples_per_group": effective_samples_per_group,
            "group_definition": "absolute source frame across overlap windows",
            "factor_definition": "shared seven-dimensional gauge latent per window",
            "prior_reliability_definition": (
                "overlap disagreement only; independent of downstream physical residual"
            ),
            "motioncrafter_commit": bundle.metadata.get(
                "motioncrafter_commit"
            ),
            "prediction_manifest_format_version": bundle.metadata.get(
                "format_version"
            ),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--causal-frame-stop", type=int, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--effective-samples-per-group", type=float, default=64.0)
    parser.add_argument(
        "--gauge-mode",
        choices=("sequential", "fixed_lag"),
        default="fixed_lag",
    )
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument("--view-name", default="camera0")
    parser.add_argument("--source-revision")
    args = parser.parse_args(argv)

    artifact = build_prob4d_observation_belief(
        args.predictions_manifest,
        case_id=args.case_id,
        causal_frame_stop=args.causal_frame_stop,
        pixel_stride=args.pixel_stride,
        effective_samples_per_group=args.effective_samples_per_group,
        gauge_mode=args.gauge_mode,
        fixed_lag=args.fixed_lag,
        view_name=args.view_name,
        source_revision=args.source_revision,
    )
    save_observation_belief_export(args.output_npz, artifact)
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "case_id": artifact.case_id,
                "causal_frame_stop": artifact.causal_frame_stop,
                "frame_count": len(artifact.declared_frame_ids),
                "observation_count": len(artifact.mean_xyz_m),
                "group_count": len(artifact.group_ids),
                "factor_rank": len(artifact.factor_names),
                "output": str(args.output_npz.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
