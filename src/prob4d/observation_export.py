"""Export source-causal Prob4D windows as ObservationBeliefV1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

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


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _deterministic_covariance_root(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    symmetric = 0.5 * (matrix + matrix.T)
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
    """Return ``J L`` so shared gauge covariance remains low rank."""

    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("gauge factors require three-dimensional values")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
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
        factors[start:stop] = np.einsum(
            "nij,jk->nik", jacobian, root
        )
    return factors.reshape(points.shape + (7,))


def _window_records(bundle: PredictionBundle) -> dict[str, dict[str, Any]]:
    records = bundle.metadata.get("overlap_windows")
    if not isinstance(records, list) or not records:
        raise ValueError("prediction manifest has no overlap-window lineage")
    result: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError(
                "overlap-window lineage record must be an object"
            )
        window_id = str(raw.get("window_id", ""))
        if not window_id or window_id in result:
            raise ValueError(
                "overlap-window lineage IDs must be unique"
            )
        if (
            "start_frame" not in raw
            or "stop_frame" not in raw
            or "path" not in raw
        ):
            raise ValueError("overlap-window lineage is incomplete")
        start = int(raw["start_frame"])
        stop = int(raw["stop_frame"])
        if start < 0 or stop <= start:
            raise ValueError(
                "overlap-window source interval is invalid"
            )
        result[window_id] = {
            "window_id": window_id,
            "path": str(raw["path"]),
            "start_frame": start,
            "stop_frame": stop,
        }
    loaded_ids = {window.window_id for window in bundle.overlap_windows}
    if loaded_ids != set(result):
        raise ValueError(
            "manifest and loaded overlap-window identities differ"
        )
    return result


def admit_source_causal_windows(
    bundle: PredictionBundle,
    *,
    causal_frame_stop: int,
) -> tuple[
    PredictionBundle,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    """Admit only windows whose complete source RGB interval is before cutoff."""

    if causal_frame_stop < 1:
        raise ValueError("causal_frame_stop must be positive")
    records = _window_records(bundle)
    admitted = tuple(
        records[window.window_id]
        for window in bundle.overlap_windows
        if records[window.window_id]["stop_frame"] <= causal_frame_stop
    )
    rejected = tuple(
        records[window.window_id]
        for window in bundle.overlap_windows
        if records[window.window_id]["stop_frame"] > causal_frame_stop
    )
    admitted_ids = {record["window_id"] for record in admitted}
    windows = [
        window
        for window in bundle.overlap_windows
        if window.window_id in admitted_ids
    ]
    if not windows:
        raise ValueError(
            "no complete MotionCrafter source window lies before "
            "causal_frame_stop"
        )
    for window in windows:
        if np.any(np.asarray(window.frame_indices) >= causal_frame_stop):
            raise ValueError(
                "admitted source window emits a frame at the cutoff"
            )
    return (
        PredictionBundle(
            manifest_path=bundle.manifest_path,
            overlap_windows=windows,
            disjoint_baseline=bundle.disjoint_baseline,
            latent_linear_baseline=bundle.latent_linear_baseline,
            metadata=bundle.metadata,
        ),
        admitted,
        rejected,
    )


def _source_bundle_digest(
    manifest: Path,
    records: tuple[dict[str, Any], ...],
) -> tuple[str, dict[str, str]]:
    file_hashes = {"predictions.json": file_sha256(manifest)}
    for record in records:
        relative = str(record["path"])
        file_hashes[relative] = file_sha256(manifest.parent / relative)
    digest = hashlib.sha256()
    for name, value in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest(), file_hashes


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
    sequential = SequentialGaugeEstimator().estimate(
        ordered_ids, constraints
    )
    if gauge_mode == "sequential":
        estimates = sequential
    elif gauge_mode == "fixed_lag":
        estimates = FixedLagGaugeSmoother(lag=fixed_lag).smooth(
            ordered_ids,
            sequential,
            constraints,
        )
    else:
        raise ValueError(
            "gauge_mode must be 'sequential' or 'fixed_lag'"
        )
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
    reliability_calibration_id: str | None = None,
    allow_uncalibrated_reliability: bool = False,
    uncertainty_model: DepthDisagreementModel | None = None,
) -> ObservationBeliefExportV1:
    """Build an artifact from whole source windows strictly before the cutoff."""

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
    if reliability_calibration_id is None:
        if not allow_uncalibrated_reliability:
            raise ValueError(
                "a source reliability calibration ID is required; use "
                "allow_uncalibrated_reliability only for an explicit "
                "diagnostic"
            )
    else:
        _validate_sha256(
            reliability_calibration_id,
            name="reliability_calibration_id",
        )

    manifest = Path(manifest_path).resolve()
    raw_bundle = load_prediction_bundle(manifest)
    bundle, admitted_records, rejected_records = (
        admit_source_causal_windows(
            raw_bundle,
            causal_frame_stop=causal_frame_stop,
        )
    )
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
        }
    )
    if not eligible_frames or max(eligible_frames) >= causal_frame_stop:
        raise ValueError(
            "admitted window frames violate causal_frame_stop"
        )
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
        transformed = estimate.global_from_local.transform_points(
            window.point_map
        )
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
        linear_entity = np.arange(
            height * width, dtype=np.int64
        ).reshape(height, width)
        for local_index, frame in enumerate(window.frame_indices):
            absolute_frame = int(frame)
            if absolute_frame >= causal_frame_stop:
                raise ValueError(
                    "admitted window crossed the causal cutoff"
                )
            selected = window.valid_mask[local_index] & sample_mask
            if not np.any(selected):
                continue
            count = int(np.sum(selected))
            means.append(transformed[local_index][selected])
            frame_ids.append(
                np.full(count, absolute_frame, dtype=np.int64)
            )
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
            local_covariances.append(
                local_covariance[local_index][selected]
            )
            factors.append(factor[local_index][selected])

    if not means:
        raise ValueError(
            "no valid sampled observation remains in the causal prefix"
        )
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
        effective = min(
            effective_samples_per_group, float(unique_entities)
        )
        group_weight[position] = min(
            1.0,
            effective / float(np.sum(selected)),
        )

    revision = source_revision or _git_revision()
    if revision == "unknown":
        raise ValueError(
            "Prob4D source revision is unavailable; provide "
            "source_revision"
        )
    source_digest, source_file_hashes = _source_bundle_digest(
        manifest,
        admitted_records,
    )
    maximum_source_frame = max(
        record["stop_frame"] - 1 for record in admitted_records
    )
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
        source_artifact_sha256=source_digest,
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
                "source_window_admission_rule": (
                    "window.stop_frame <= causal_frame_stop"
                ),
                "maximum_source_frame_read": maximum_source_frame,
                "causal_frame_stop_exclusive": causal_frame_stop,
                "future_frames_read": 0,
                "admitted_windows": list(admitted_records),
                "rejected_future_dependent_windows": list(
                    rejected_records
                ),
            },
            "source_file_sha256": source_file_hashes,
            "gauge_mode": gauge_mode,
            "fixed_lag": (
                fixed_lag if gauge_mode == "fixed_lag" else None
            ),
            "pixel_stride": pixel_stride,
            "effective_samples_per_group": effective_samples_per_group,
            "group_definition": (
                "absolute source frame across admitted overlap windows"
            ),
            "factor_definition": (
                "shared seven-dimensional gauge latent per admitted window"
            ),
            "association_probability_definition": (
                "dense MotionCrafter pixel identity; separate from "
                "reliability"
            ),
            "prior_reliability_definition": (
                "overlap disagreement only; independent of physical residual"
            ),
            "reliability_calibration_id": reliability_calibration_id,
            "reliability_calibration_status": (
                "source-locked"
                if reliability_calibration_id is not None
                else "explicit-uncalibrated-diagnostic"
            ),
            "motioncrafter_commit": bundle.metadata.get(
                "motioncrafter_commit"
            ),
            "prediction_manifest_format_version": bundle.metadata.get(
                "format_version"
            ),
            "temporal_lineage": bundle.metadata.get("temporal_lineage"),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--causal-frame-stop", type=int, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument(
        "--effective-samples-per-group",
        type=float,
        default=64.0,
    )
    parser.add_argument(
        "--gauge-mode",
        choices=("sequential", "fixed_lag"),
        default="fixed_lag",
    )
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument("--view-name", default="camera0")
    parser.add_argument("--source-revision")
    parser.add_argument("--reliability-calibration-id")
    parser.add_argument(
        "--allow-uncalibrated-reliability",
        action="store_true",
        help=(
            "permit the feeder-only heuristic for an explicitly labelled "
            "diagnostic; production artifacts should provide a "
            "source-locked calibration ID"
        ),
    )
    args = parser.parse_args(argv)

    artifact = build_prob4d_observation_belief(
        args.predictions_manifest,
        case_id=args.case_id,
        causal_frame_stop=args.causal_frame_stop,
        pixel_stride=args.pixel_stride,
        effective_samples_per_group=(
            args.effective_samples_per_group
        ),
        gauge_mode=args.gauge_mode,
        fixed_lag=args.fixed_lag,
        view_name=args.view_name,
        source_revision=args.source_revision,
        reliability_calibration_id=args.reliability_calibration_id,
        allow_uncalibrated_reliability=(
            args.allow_uncalibrated_reliability
        ),
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
