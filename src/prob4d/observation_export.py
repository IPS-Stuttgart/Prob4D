"""Export causally sealed, metric Prob4D observations for Bayesian consumers."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ._causal_observation_source import (
    CAUSAL_SOURCE_LINEAGE_SCHEMA_VERSION,
    CausalOverlapSelection,
    SelectedOverlapWindow,
    select_causal_overlap_windows,
)
from ._metric_gauge_anchor import (
    METRIC_GAUGE_ANCHOR_SCHEMA,
    METRIC_GAUGE_ANCHOR_VERSION,
    MetricGaugeAnchor,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
)
from .alignment import WindowAlignment, align_windows
from .data import PredictionWindow
from .gauge import (
    FixedLagGaugeSmoother,
    GaugeEstimate,
    RelativeGaugeConstraint,
    SequentialGaugeEstimator,
)
from .observation_contract import (
    ObservationBeliefExportV1,
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


def _validated_source_revision(value: str | None) -> str:
    revision = value or _git_revision()
    if revision == "unknown" or len(revision) not in {40, 64}:
        raise ValueError(
            "source_revision must be an exact 40- or 64-character Git commit; "
            "pass --source-revision when the checkout revision is unavailable"
        )
    if any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("source_revision must be a lowercase hexadecimal Git commit")
    return revision


def _deterministic_covariance_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    if symmetric.shape != (7, 7) or not np.all(np.isfinite(symmetric)):
        raise ValueError("gauge covariance must have finite shape (7, 7)")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("gauge covariance must be positive semidefinite")
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
    """Return ``J L`` so rows from one window share its gauge uncertainty."""

    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("gauge factors require three-dimensional values")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    root = _deterministic_covariance_root(gauge_covariance)
    flattened = points.reshape(-1, 3)
    factors = np.empty((len(flattened), 3, 7), dtype=np.float64)
    right_jacobian = so3_right_jacobian(so3_log(transform.rotation))
    identity = np.eye(3, dtype=np.float64)
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


def _build_alignments(windows: Sequence[PredictionWindow]) -> list[WindowAlignment]:
    alignments: list[WindowAlignment] = []
    for moving_index, moving in enumerate(windows):
        for reference_index, reference in enumerate(windows[:moving_index]):
            if reference.common_frames(moving).size:
                seed = 10_000 * moving_index + reference_index
                alignments.append(align_windows(reference, moving, seed=seed))
    return alignments


def _gauge_estimates(
    windows: Sequence[PredictionWindow],
    *,
    gauge_mode: str,
    fixed_lag: int,
    metric_anchor: MetricGaugeAnchor,
) -> tuple[list[WindowAlignment], dict[str, GaugeEstimate]]:
    if gauge_mode not in {"sequential", "fixed_lag"}:
        raise ValueError("gauge_mode must be 'sequential' or 'fixed_lag'")
    alignments = _build_alignments(windows)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment)
        for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in windows]
    sequential = SequentialGaugeEstimator().estimate(
        ordered_ids,
        constraints,
        initial_transform=metric_anchor.global_from_local,
        initial_covariance=metric_anchor.covariance,
    )
    if gauge_mode == "sequential" or len(ordered_ids) == 1:
        estimates = sequential
    elif gauge_mode == "fixed_lag":
        if fixed_lag < 2:
            raise ValueError("fixed_lag must be at least two")
        estimates = FixedLagGaugeSmoother(lag=fixed_lag).smooth(
            ordered_ids,
            sequential,
            constraints,
        )
    else:  # pragma: no cover - guarded above
        raise ValueError("gauge_mode must be 'sequential' or 'fixed_lag'")
    return alignments, estimates


def _prior_reliability(
    parallel_disagreement: np.ndarray,
    lateral_disagreement: np.ndarray,
    parallel_variance: np.ndarray,
    lateral_variance: np.ndarray,
    overlap_count: np.ndarray,
    *,
    minimum: float,
) -> np.ndarray:
    """Return source-only row reliability, independent of physical innovation."""

    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum prior reliability must lie in (0, 1]")
    normalized = (
        parallel_disagreement / np.maximum(parallel_variance, 1e-12)
        + lateral_disagreement / np.maximum(lateral_variance, 1e-12)
    )
    reliability = np.exp(-0.5 * np.minimum(normalized, 50.0))
    reliability = np.where(overlap_count > 0.0, reliability, 1.0)
    return np.clip(reliability, minimum, 1.0)


def _row_covariance(
    rays_local: np.ndarray,
    parallel_variance: np.ndarray,
    lateral_variance: np.ndarray,
    transform: Sim3,
) -> np.ndarray:
    rays_world = transform.rotate_directions(rays_local)
    parallel = transform.scale**2 * np.asarray(parallel_variance, dtype=np.float64)
    lateral = transform.scale**2 * np.asarray(lateral_variance, dtype=np.float64)
    outer = np.einsum("ni,nj->nij", rays_world, rays_world)
    identity = np.eye(3, dtype=np.float64)
    return lateral[:, None, None] * identity + (
        parallel - lateral
    )[:, None, None] * outer


def _build_prob4d_observation_belief(
    selection: CausalOverlapSelection,
    *,
    case_id: str,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchor,
    pixel_stride: int = 4,
    effective_samples_per_group: float = 64.0,
    minimum_prior_reliability: float = 0.05,
    gauge_mode: str = "fixed_lag",
    fixed_lag: int = 4,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
) -> ObservationBeliefExportV1:
    """Build an artifact without opening or using post-cutoff prediction payloads."""

    if not case_id or not view_name:
        raise ValueError("case_id and view_name must be nonempty")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    if not np.isfinite(effective_samples_per_group) or (
        effective_samples_per_group <= 0.0
    ):
        raise ValueError("effective_samples_per_group must be positive")

    revision = _validated_source_revision(source_revision)
    windows = selection.predictions
    alignments, estimates = _gauge_estimates(
        windows,
        gauge_mode=gauge_mode,
        fixed_lag=fixed_lag,
        metric_anchor=metric_anchor,
    )
    window_map = {window.window_id: window for window in windows}
    evidence = accumulate_disagreement(window_map, alignments)
    model = uncertainty_model or DepthDisagreementModel()

    eligible_frames = sorted(
        {int(frame) for window in windows for frame in window.frame_indices}
    )
    if not eligible_frames or eligible_frames[-1] >= causal_frame_stop:
        raise RuntimeError("causal overlap selection produced an invalid frame set")
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

    for window_index, window in enumerate(windows):
        estimate = estimates[window.window_id]
        disagreement = evidence[window.window_id]
        uncertainty = model.predict(window, disagreement)
        reliability = _prior_reliability(
            disagreement.parallel_mean,
            disagreement.lateral_mean,
            uncertainty.parallel_variance,
            uncertainty.lateral_variance,
            disagreement.count,
            minimum=minimum_prior_reliability,
        )
        rays = window.rays()
        height, width = window.shape[1:]
        sample_mask = np.zeros((height, width), dtype=bool)
        sample_mask[::pixel_stride, ::pixel_stride] = True
        linear_entity = np.arange(height * width, dtype=np.int64).reshape(
            height, width
        )
        for local_index, frame in enumerate(window.frame_indices):
            absolute_frame = int(frame)
            selected = window.valid_mask[local_index] & sample_mask
            if not np.any(selected):
                continue
            count = int(np.count_nonzero(selected))
            local_points = window.point_map[local_index][selected]
            means.append(estimate.global_from_local.transform_points(local_points))
            frame_ids.append(np.full(count, absolute_frame, dtype=np.int64))
            entity_ids.append(linear_entity[selected])
            view_indices.append(np.zeros(count, dtype=np.int64))
            window_indices.append(np.full(count, window_index, dtype=np.int64))
            correlation_groups.append(
                np.full(
                    count,
                    frame_to_group[absolute_frame],
                    dtype=np.int64,
                )
            )
            factor_groups.append(np.full(count, window_index, dtype=np.int64))
            reliabilities.append(reliability[local_index][selected])
            associations.append(np.ones(count, dtype=np.float64))
            local_covariances.append(
                _row_covariance(
                    rays[local_index][selected],
                    uncertainty.parallel_variance[local_index][selected],
                    uncertainty.lateral_variance[local_index][selected],
                    estimate.global_from_local,
                )
            )
            factors.append(
                gauge_covariance_factor(
                    local_points,
                    estimate.global_from_local,
                    estimate.covariance,
                    include_translation=True,
                )
            )

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
    # No independently calibrated nominal/outlier prior is available here. Use
    # the neutral value instead of applying overlap reliability a second time.
    group_prior = np.ones(len(group_ids), dtype=np.float64)
    group_weight = np.empty(len(group_ids), dtype=np.float64)
    for position, group_id in enumerate(group_ids):
        selected = correlation_array == group_id
        unique_entities = len(np.unique(entity_array[selected]))
        effective = min(effective_samples_per_group, float(unique_entities))
        group_weight[position] = min(
            1.0,
            effective / float(np.count_nonzero(selected)),
        )

    metadata = {
        "metric_coordinates": True,
        "metric_units": "m",
        "coordinate_frame": metric_anchor.coordinate_frame,
        "metric_gauge_anchor": {
            "artifact_id": metric_anchor.artifact_id,
            "window_id": metric_anchor.window_id,
            "source_kind": metric_anchor.source_kind,
            "source_artifact_sha256": metric_anchor.source_artifact_sha256,
        },
        "causal_source_lineage": selection.artifact_lineage_metadata(
            causal_frame_stop=causal_frame_stop
        ),
        "gauge_mode": gauge_mode,
        "fixed_lag": fixed_lag if gauge_mode == "fixed_lag" else None,
        "pixel_stride": pixel_stride,
        "effective_samples_per_group": effective_samples_per_group,
        "minimum_prior_reliability": minimum_prior_reliability,
        "uncertainty_model": asdict(model),
        "group_definition": "absolute source frame across overlap windows",
        "factor_definition": "shared seven-dimensional gauge latent per window",
        "factor_group_semantics": (
            "per-window gauge marginals are explicit nuisance factors; schema v1 "
            "does not represent joint cross-window gauge covariance, so remaining "
            "dependence is capped by composite weights"
        ),
        "joint_cross_window_gauge_covariance_represented": False,
        "association_probability_definition": (
            "same decoded pixel identity within one independently decoded window; "
            "not downstream physical-node association"
        ),
        "prior_reliability_definition": (
            "overlap disagreement only; independent of downstream physical innovation"
        ),
        "group_prior_nominal_probability_definition": (
            "neutral one; no independently calibrated group nominal prior supplied"
        ),
        "motioncrafter_commit": selection.manifest["motioncrafter_commit"],
        "prediction_manifest_format_version": selection.manifest["format_version"],
    }
    return ObservationBeliefExportV1(
        case_id=case_id,
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=causal_frame_stop,
        view_names=(view_name,),
        window_names=tuple(window.window_id for window in windows),
        factor_names=GAUGE_FACTOR_NAMES,
        source_repository="FlorianPfaff/Prob4D",
        source_revision=revision,
        source_artifact_sha256=selection.source_artifact_sha256,
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
        metadata=metadata,
    )


def build_prob4d_observation_belief(
    manifest_path: str | Path,
    *,
    case_id: str,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchor,
    pixel_stride: int = 4,
    effective_samples_per_group: float = 64.0,
    minimum_prior_reliability: float = 0.05,
    gauge_mode: str = "fixed_lag",
    fixed_lag: int = 4,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
) -> ObservationBeliefExportV1:
    """Select a causal source prefix and export a portable observation belief."""

    selection = select_causal_overlap_windows(
        manifest_path,
        causal_frame_stop=causal_frame_stop,
        metric_anchor=metric_anchor,
    )
    return _build_prob4d_observation_belief(
        selection,
        case_id=case_id,
        causal_frame_stop=causal_frame_stop,
        metric_anchor=metric_anchor,
        pixel_stride=pixel_stride,
        effective_samples_per_group=effective_samples_per_group,
        minimum_prior_reliability=minimum_prior_reliability,
        gauge_mode=gauge_mode,
        fixed_lag=fixed_lag,
        view_name=view_name,
        source_revision=source_revision,
        uncertainty_model=uncertainty_model,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--causal-frame-stop", type=int, required=True)
    parser.add_argument("--metric-gauge-anchor", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--effective-samples-per-group", type=float, default=64.0)
    parser.add_argument("--minimum-prior-reliability", type=float, default=0.05)
    parser.add_argument(
        "--gauge-mode",
        choices=("sequential", "fixed_lag"),
        default="fixed_lag",
    )
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument("--view-name", default="camera0")
    parser.add_argument("--source-revision")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)

    anchor = load_metric_gauge_anchor(args.metric_gauge_anchor)
    selection = select_causal_overlap_windows(
        args.predictions_manifest,
        causal_frame_stop=args.causal_frame_stop,
        metric_anchor=anchor,
    )
    artifact = _build_prob4d_observation_belief(
        selection,
        case_id=args.case_id,
        causal_frame_stop=args.causal_frame_stop,
        metric_anchor=anchor,
        pixel_stride=args.pixel_stride,
        effective_samples_per_group=args.effective_samples_per_group,
        minimum_prior_reliability=args.minimum_prior_reliability,
        gauge_mode=args.gauge_mode,
        fixed_lag=args.fixed_lag,
        view_name=args.view_name,
        source_revision=args.source_revision,
    )
    save_observation_belief_export(args.output_npz, artifact)
    summary = {
        **selection.run_summary(causal_frame_stop=args.causal_frame_stop),
        **artifact.summary(),
        "metric_gauge_anchor_id": anchor.artifact_id,
        "output": str(args.output_npz.resolve()),
    }
    if args.summary_json is not None:
        _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAUSAL_SOURCE_LINEAGE_SCHEMA_VERSION",
    "GAUGE_FACTOR_NAMES",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "CausalOverlapSelection",
    "MetricGaugeAnchor",
    "SelectedOverlapWindow",
    "build_prob4d_observation_belief",
    "gauge_covariance_factor",
    "load_metric_gauge_anchor",
    "save_metric_gauge_anchor",
    "select_causal_overlap_windows",
]
