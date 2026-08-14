"""Export causally sealed, metric Prob4D observations for Bayesian consumers."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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
from .composition_jacobian import (
    compose_jacobians_for_mode,
    current_composition_jacobian_mode,
)
from .covariance_root import (
    covariance_root_for_mode,
    current_covariance_root_mode,
)
from .data import PredictionWindow
from .export_numerics import (
    ExportNumericsPolicy,
    resolve_export_numerics_policy,
)
from .gauge import RelativeGaugeConstraint, SequentialGaugeEstimator
from .marginalized_gauge import MarginalizedFixedLagGaugeSmoother
from .observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from .observation_factors import sim3_point_jacobian
from .sim3 import Sim3
from .uncertainty import DepthDisagreementModel, accumulate_disagreement

# Retained for source compatibility. Production artifacts use dynamically sized
# ``joint_gauge_latent_*`` names because one latent vector now carries the joint
# cross-window covariance.
GAUGE_FACTOR_NAMES = tuple(f"gauge_latent_{index}" for index in range(7))
GROUP_COMPOSITE_WEIGHT_SEMANTICS = "final-per-row-effective-sample-cap-v1"
GAUGE_RANK_REDUCTION_METRIC = "sim3-observation-displacement-v1"
SamplingMode = Literal["fixed_grid", "information_stratified"]
SAMPLING_MODES: tuple[SamplingMode, ...] = (
    "fixed_grid",
    "information_stratified",
)


@dataclass(frozen=True)
class JointGaugePosterior:
    """Ordered gauge means and one covariance over all seven-dimensional gauges."""

    window_ids: tuple[str, ...]
    estimates: Mapping[str, Sim3]
    joint_covariance: np.ndarray
    mode: str
    cross_window_covariance_preserved: bool
    parent_window_ids: tuple[str | None, ...] = ()
    selected_alignment_indices: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.window_ids or len(set(self.window_ids)) != len(self.window_ids):
            raise ValueError("joint gauge posterior requires unique window IDs")
        if set(self.estimates) != set(self.window_ids):
            raise ValueError("joint gauge posterior estimates do not match window IDs")
        dimension = 7 * len(self.window_ids)
        covariance = np.asarray(self.joint_covariance, dtype=np.float64)
        if covariance.shape != (dimension, dimension):
            raise ValueError("joint gauge covariance has changed shape")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("joint gauge covariance must be finite")
        symmetric = 0.5 * (covariance + covariance.T)
        if np.min(np.linalg.eigvalsh(symmetric)) < -1e-9:
            raise ValueError("joint gauge covariance must be positive semidefinite")
        parent_ids = self.parent_window_ids or tuple(None for _ in self.window_ids)
        alignment_indices = self.selected_alignment_indices or tuple(
            None for _ in self.window_ids
        )
        if len(parent_ids) != len(self.window_ids) or len(alignment_indices) != len(
            self.window_ids
        ):
            raise ValueError("joint gauge posterior lineage changed length")
        if not self.mode:
            raise ValueError("joint gauge posterior mode must be nonempty")
        object.__setattr__(self, "joint_covariance", symmetric)
        object.__setattr__(self, "parent_window_ids", tuple(parent_ids))
        object.__setattr__(
            self,
            "selected_alignment_indices",
            tuple(alignment_indices),
        )


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


def _validated_sampling_mode(value: str) -> SamplingMode:
    if value not in SAMPLING_MODES:
        raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}")
    return value  # type: ignore[return-value]


def observation_sample_mask(
    valid_mask: np.ndarray,
    points_local: np.ndarray,
    parallel_variance: np.ndarray,
    lateral_variance: np.ndarray,
    reliability: np.ndarray,
    *,
    pixel_stride: int,
    sampling_mode: SamplingMode,
) -> np.ndarray:
    """Select at most one deterministic, high-information row per spatial tile."""

    valid = np.asarray(valid_mask, dtype=bool)
    points = np.asarray(points_local, dtype=np.float64)
    parallel = np.asarray(parallel_variance, dtype=np.float64)
    lateral = np.asarray(lateral_variance, dtype=np.float64)
    reliability_array = np.asarray(reliability, dtype=np.float64)
    if valid.ndim != 2 or points.shape != valid.shape + (3,):
        raise ValueError("sampling inputs must contain one H x W point map")
    if (
        parallel.shape != valid.shape
        or lateral.shape != valid.shape
        or reliability_array.shape != valid.shape
    ):
        raise ValueError("sampling variance and reliability shapes must match valid_mask")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    mode = _validated_sampling_mode(sampling_mode)
    if not np.all(np.isfinite(points[valid])):
        raise ValueError("valid sampling points must be finite")
    if not np.all(np.isfinite(parallel[valid])) or not np.all(
        np.isfinite(lateral[valid])
    ):
        raise ValueError("valid sampling variances must be finite")
    if np.any(parallel[valid] <= 0.0) or np.any(lateral[valid] <= 0.0):
        raise ValueError("valid sampling variances must be positive")
    if not np.all(np.isfinite(reliability_array[valid])) or np.any(
        reliability_array[valid] < 0.0
    ):
        raise ValueError("valid sampling reliability must be finite and nonnegative")

    if mode == "fixed_grid":
        selected = np.zeros(valid.shape, dtype=bool)
        selected[::pixel_stride, ::pixel_stride] = True
        return selected & valid

    selected = np.zeros(valid.shape, dtype=bool)
    if not np.any(valid):
        return selected
    depth = np.linalg.norm(points, axis=-1)
    depth_values = depth[valid]
    depth_scale = max(
        float(np.median(depth_values)),
        np.finfo(np.float64).tiny,
    )
    total_variance = parallel + 2.0 * lateral
    variance_scale = max(
        float(np.median(total_variance[valid])),
        np.finfo(np.float64).tiny,
    )
    safe_total_variance = np.where(valid, total_variance, 1.0)
    score = (
        reliability_array
        * (1.0 + np.square(depth / depth_scale))
        * variance_scale
        / safe_total_variance
    )
    score = np.where(valid, score, -np.inf)
    height, width = valid.shape
    for row_start in range(0, height, pixel_stride):
        row_stop = min(row_start + pixel_stride, height)
        for column_start in range(0, width, pixel_stride):
            column_stop = min(column_start + pixel_stride, width)
            tile_score = score[row_start:row_stop, column_start:column_stop]
            flat_index = int(np.argmax(tile_score))
            best_score = float(tile_score.reshape(-1)[flat_index])
            if not np.isfinite(best_score):
                continue
            row_offset, column_offset = np.unravel_index(
                flat_index,
                tile_score.shape,
            )
            selected[row_start + row_offset, column_start + column_offset] = True
    return selected


def _window_reference_radius(
    window: PredictionWindow,
    transform: Sim3,
    *,
    max_samples: int = 4_096,
) -> float:
    """Return a deterministic robust metric radius for Sim(3) normalization."""

    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    valid_indices = np.flatnonzero(window.valid_mask.reshape(-1))
    if not len(valid_indices):
        return float(transform.scale)
    if len(valid_indices) > max_samples:
        positions = np.linspace(
            0,
            len(valid_indices) - 1,
            max_samples,
            dtype=np.int64,
        )
        valid_indices = valid_indices[positions]
    points = window.point_map.reshape(-1, 3)[valid_indices]
    radii = transform.scale * np.linalg.norm(points, axis=1)
    positive = radii[radii > np.finfo(np.float64).eps * transform.scale]
    if not len(positive):
        return float(transform.scale)
    return float(np.median(positive))


def _joint_gauge_coordinate_normalizer(
    windows: Sequence[PredictionWindow],
    posterior: JointGaugePosterior,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Map mixed-unit Sim(3) coordinates to representative point displacement."""

    radii = tuple(
        _window_reference_radius(window, posterior.estimates[window.window_id])
        for window in windows
    )
    normalizer = np.concatenate(
        [
            np.asarray([radius, radius, radius, radius, 1.0, 1.0, 1.0])
            for radius in radii
        ]
    )
    return normalizer, radii


def deterministic_covariance_root(
    covariance: np.ndarray,
    *,
    max_rank: int | None = None,
    relative_eigenvalue_floor: float = 1e-12,
    coordinate_normalizer: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return a deterministic root using the compatibility-local mode.

    New provider entry points inject an :class:`ExportNumericsPolicy` directly.
    This wrapper preserves the historical context-manager surface without any
    import-time replacement of functions in this module.
    """

    return covariance_root_for_mode(
        current_covariance_root_mode(),
        covariance,
        max_rank=max_rank,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
        coordinate_normalizer=coordinate_normalizer,
    )


def _deterministic_covariance_root(covariance: np.ndarray) -> np.ndarray:
    """Backward-compatible square root for one seven-dimensional gauge."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (7, 7):
        raise ValueError("gauge covariance must have finite shape (7, 7)")
    root, _ = deterministic_covariance_root(
        matrix,
        max_rank=7,
        relative_eigenvalue_floor=0.0,
    )
    if root.shape[1] < 7:
        root = np.pad(root, ((0, 0), (0, 7 - root.shape[1])))
    return root


def joint_gauge_factor(
    values: np.ndarray,
    transform: Sim3,
    joint_root_block: np.ndarray,
    *,
    include_translation: bool = True,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Map one gauge block of a joint covariance root into observation space."""

    points = np.asarray(values, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("joint gauge factors require three-dimensional values")
    block = np.asarray(joint_root_block, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != 7:
        raise ValueError("joint_root_block must have shape (7, R)")
    if not np.all(np.isfinite(block)):
        raise ValueError("joint_root_block must be finite")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    flattened = points.reshape(-1, 3)
    factors = np.empty((len(flattened), 3, block.shape[1]), dtype=np.float64)
    for start in range(0, len(flattened), chunk_size):
        stop = min(start + chunk_size, len(flattened))
        jacobian = sim3_point_jacobian(transform, flattened[start:stop])
        if not include_translation:
            jacobian[:, :, 4:7] = 0.0
        factors[start:stop] = np.einsum(
            "nij,jr->nir",
            jacobian,
            block,
            optimize=True,
        )
    return factors.reshape(points.shape + (block.shape[1],))


def gauge_covariance_factor(
    values: np.ndarray,
    transform: Sim3,
    gauge_covariance: np.ndarray,
    *,
    include_translation: bool,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Return ``J L`` so rows from one window share its gauge uncertainty."""

    root = _deterministic_covariance_root(gauge_covariance)
    return joint_gauge_factor(
        values,
        transform,
        root,
        include_translation=include_translation,
        chunk_size=chunk_size,
    )


def _build_alignments(windows: Sequence[PredictionWindow]) -> list[WindowAlignment]:
    alignments: list[WindowAlignment] = []
    for moving_index, moving in enumerate(windows):
        for reference_index, reference in enumerate(windows[:moving_index]):
            if reference.common_frames(moving).size:
                seed = 10_000 * moving_index + reference_index
                alignments.append(align_windows(reference, moving, seed=seed))
    return alignments


def _compose_jacobians(
    parent: Sim3,
    relative: Sim3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return composition derivatives using the compatibility-local mode."""

    return compose_jacobians_for_mode(
        current_composition_jacobian_mode(),
        parent,
        relative,
    )


def estimate_joint_gauge_tree(
    windows: Sequence[PredictionWindow],
    alignments: Sequence[WindowAlignment],
    *,
    initial_transform: Sim3,
    initial_covariance: np.ndarray,
    numerics_policy: ExportNumericsPolicy | None = None,
) -> JointGaugePosterior:
    """Propagate one causal spanning tree into a full joint gauge covariance."""

    numerics = resolve_export_numerics_policy(numerics_policy)
    if not windows:
        raise ValueError("joint gauge estimation requires at least one window")
    window_ids = tuple(window.window_id for window in windows)
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("window IDs must be unique")
    position = {window_id: index for index, window_id in enumerate(window_ids)}
    first_covariance = np.asarray(initial_covariance, dtype=np.float64)
    if first_covariance.shape != (7, 7) or not np.all(np.isfinite(first_covariance)):
        raise ValueError("initial gauge covariance must have finite shape (7, 7)")
    first_covariance = 0.5 * (first_covariance + first_covariance.T)
    if np.min(np.linalg.eigvalsh(first_covariance)) < -1e-12:
        raise ValueError("initial gauge covariance must be positive semidefinite")

    dimension = 7 * len(windows)
    joint = np.zeros((dimension, dimension), dtype=np.float64)
    joint[:7, :7] = first_covariance
    estimates: dict[str, Sim3] = {window_ids[0]: initial_transform}
    parent_ids: list[str | None] = [None]
    alignment_indices: list[int | None] = [None]

    for child_index, child_id in enumerate(window_ids[1:], start=1):
        candidates = [
            (index, alignment)
            for index, alignment in enumerate(alignments)
            if alignment.moving_id == child_id
            and alignment.reference_id in estimates
        ]
        if not candidates:
            raise ValueError(
                f"window {child_id!r} has no causal overlap with an earlier window"
            )
        selected_index, selected = min(
            candidates,
            key=lambda item: (
                -item[1].result.num_correspondences,
                item[1].result.residual_rms,
                position[item[1].reference_id],
            ),
        )
        parent_id = selected.reference_id
        parent_index = position[parent_id]
        parent = estimates[parent_id]
        relative = selected.result.transform
        child = parent.compose(relative)
        parent_jacobian, relative_jacobian = numerics.compose_jacobians(
            parent, relative
        )
        parent_slice = slice(7 * parent_index, 7 * (parent_index + 1))
        child_slice = slice(7 * child_index, 7 * (child_index + 1))
        for previous_index in range(child_index):
            previous_slice = slice(7 * previous_index, 7 * (previous_index + 1))
            cross = parent_jacobian @ joint[parent_slice, previous_slice]
            joint[child_slice, previous_slice] = cross
            joint[previous_slice, child_slice] = cross.T
        relative_covariance = np.asarray(selected.result.covariance, dtype=np.float64)
        child_covariance = (
            parent_jacobian
            @ joint[parent_slice, parent_slice]
            @ parent_jacobian.T
            + relative_jacobian @ relative_covariance @ relative_jacobian.T
        )
        joint[child_slice, child_slice] = 0.5 * (
            child_covariance + child_covariance.T
        )
        estimates[child_id] = child
        parent_ids.append(parent_id)
        alignment_indices.append(selected_index)

    symmetric = 0.5 * (joint + joint.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.min(eigenvalues) < -1e-7:
        raise ValueError("propagated joint gauge covariance is not positive semidefinite")
    joint = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return JointGaugePosterior(
        window_ids=window_ids,
        estimates=estimates,
        joint_covariance=joint,
        mode="sequential_joint_spanning_tree_v1",
        cross_window_covariance_preserved=True,
        parent_window_ids=tuple(parent_ids),
        selected_alignment_indices=tuple(alignment_indices),
    )


def _block_diagonal(values: Sequence[np.ndarray]) -> np.ndarray:
    dimension = sum(value.shape[0] for value in values)
    result = np.zeros((dimension, dimension), dtype=np.float64)
    offset = 0
    for value in values:
        width = value.shape[0]
        result[offset : offset + width, offset : offset + width] = value
        offset += width
    return result


def _fixed_lag_marginal_posterior(
    windows: Sequence[PredictionWindow],
    alignments: Sequence[WindowAlignment],
    *,
    fixed_lag: int,
    metric_anchor: MetricGaugeAnchor,
) -> JointGaugePosterior:
    """Return fixed-lag marginals with a Schur-marginalized boundary prior."""

    if fixed_lag < 2:
        raise ValueError("fixed_lag must be at least two")
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
    estimates = MarginalizedFixedLagGaugeSmoother(lag=fixed_lag).smooth(
        ordered_ids,
        sequential,
        constraints,
    )
    covariance = _block_diagonal(
        [np.asarray(estimates[window_id].covariance) for window_id in ordered_ids]
    )
    return JointGaugePosterior(
        window_ids=tuple(ordered_ids),
        estimates={
            window_id: estimates[window_id].global_from_local
            for window_id in ordered_ids
        },
        joint_covariance=covariance,
        mode="fixed_lag_schur_boundary_block_diagonal_v2",
        cross_window_covariance_preserved=False,
    )


def _gauge_posterior(
    windows: Sequence[PredictionWindow],
    *,
    gauge_mode: str,
    fixed_lag: int,
    metric_anchor: MetricGaugeAnchor,
    allow_approximate_fixed_lag_covariance: bool,
    numerics_policy: ExportNumericsPolicy,
) -> tuple[list[WindowAlignment], JointGaugePosterior]:
    if gauge_mode not in {"sequential", "fixed_lag"}:
        raise ValueError("gauge_mode must be 'sequential' or 'fixed_lag'")
    if gauge_mode == "fixed_lag" and not allow_approximate_fixed_lag_covariance:
        raise ValueError(
            "fixed_lag preserves its moving boundary prior but not historical "
            "cross-window covariance; pass allow_approximate_fixed_lag_covariance=True "
            "only for an explicitly labelled reconstruction ablation"
        )
    alignments = _build_alignments(windows)
    if gauge_mode == "sequential":
        posterior = estimate_joint_gauge_tree(
            windows,
            alignments,
            initial_transform=metric_anchor.global_from_local,
            initial_covariance=metric_anchor.covariance,
            numerics_policy=numerics_policy,
        )
    else:
        posterior = _fixed_lag_marginal_posterior(
            windows,
            alignments,
            fixed_lag=fixed_lag,
            metric_anchor=metric_anchor,
        )
    return alignments, posterior


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
    gauge_mode: str = "sequential",
    fixed_lag: int = 4,
    allow_approximate_fixed_lag_covariance: bool = False,
    max_gauge_rank: int | None = 64,
    minimum_retained_gauge_trace: float = 0.999,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
    sampling_mode: SamplingMode = "fixed_grid",
    numerics_policy: ExportNumericsPolicy | None = None,
) -> ObservationBeliefExportV1:
    """Build an artifact without opening or using post-cutoff prediction payloads."""

    numerics = resolve_export_numerics_policy(numerics_policy)
    if not case_id or not view_name:
        raise ValueError("case_id and view_name must be nonempty")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    if not np.isfinite(effective_samples_per_group) or (
        effective_samples_per_group <= 0.0
    ):
        raise ValueError("effective_samples_per_group must be positive")
    if not 0.0 < minimum_retained_gauge_trace <= 1.0:
        raise ValueError("minimum_retained_gauge_trace must lie in (0, 1]")
    sampling_mode = _validated_sampling_mode(sampling_mode)

    revision = _validated_source_revision(source_revision)
    windows = selection.predictions
    alignments, posterior = _gauge_posterior(
        windows,
        gauge_mode=gauge_mode,
        fixed_lag=fixed_lag,
        metric_anchor=metric_anchor,
        allow_approximate_fixed_lag_covariance=(
            allow_approximate_fixed_lag_covariance
        ),
        numerics_policy=numerics,
    )
    gauge_coordinate_normalizer, gauge_reference_radii = (
        _joint_gauge_coordinate_normalizer(windows, posterior)
    )
    joint_root, retained_trace_fraction = numerics.covariance_root(
        posterior.joint_covariance,
        max_rank=max_gauge_rank,
        coordinate_normalizer=gauge_coordinate_normalizer,
    )
    if retained_trace_fraction + 1e-12 < minimum_retained_gauge_trace:
        raise ValueError(
            "gauge covariance rank cap retains only "
            f"{retained_trace_fraction:.6f} of normalized observation-displacement "
            "covariance trace; increase "
            "max_gauge_rank or lower minimum_retained_gauge_trace explicitly"
        )
    factor_rank = joint_root.shape[1]
    factor_names = tuple(
        f"joint_gauge_latent_{index:04d}" for index in range(factor_rank)
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
        transform = posterior.estimates[window.window_id]
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
        root_block = joint_root[
            7 * window_index : 7 * (window_index + 1), :
        ]
        height, width = window.shape[1:]
        linear_entity = np.arange(height * width, dtype=np.int64).reshape(
            height, width
        )
        for local_index, frame in enumerate(window.frame_indices):
            absolute_frame = int(frame)
            selected = observation_sample_mask(
                window.valid_mask[local_index],
                window.point_map[local_index],
                uncertainty.parallel_variance[local_index],
                uncertainty.lateral_variance[local_index],
                reliability[local_index],
                pixel_stride=pixel_stride,
                sampling_mode=sampling_mode,
            )
            if not np.any(selected):
                continue
            count = int(np.count_nonzero(selected))
            local_points = window.point_map[local_index][selected]
            means.append(transform.transform_points(local_points))
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
            # Every row shares one joint latent vector. Cross-window covariance is
            # encoded by the corresponding nonzero block of that vector.
            factor_groups.append(np.zeros(count, dtype=np.int64))
            reliabilities.append(reliability[local_index][selected])
            associations.append(np.ones(count, dtype=np.float64))
            local_covariances.append(
                _row_covariance(
                    rays[local_index][selected],
                    uncertainty.parallel_variance[local_index][selected],
                    uncertainty.lateral_variance[local_index][selected],
                    transform,
                )
            )
            factors.append(
                joint_gauge_factor(
                    local_points,
                    transform,
                    root_block,
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
    group_statistics: list[dict[str, int | float]] = []
    for position, group_id in enumerate(group_ids):
        selected = correlation_array == group_id
        raw_row_count = int(np.count_nonzero(selected))
        unique_entity_count = int(len(np.unique(entity_array[selected])))
        effective = min(
            effective_samples_per_group,
            float(unique_entity_count),
        )
        per_row_weight = min(1.0, effective / float(raw_row_count))
        group_weight[position] = per_row_weight
        group_statistics.append(
            {
                "group_id": int(group_id),
                "raw_row_count": raw_row_count,
                "unique_entity_count": unique_entity_count,
                "effective_sample_count": float(effective),
                "per_row_composite_weight": float(per_row_weight),
            }
        )

    selected_alignment_indices = {
        value
        for value in posterior.selected_alignment_indices
        if value is not None
    }
    alignment_records = [
        {
            "index": index,
            "reference_id": alignment.reference_id,
            "moving_id": alignment.moving_id,
            "common_frames": [int(value) for value in alignment.common_frames],
            "residual_rms": float(alignment.result.residual_rms),
            "num_correspondences": int(alignment.result.num_correspondences),
            "covariance_method": alignment.result.covariance_method,
            "selected_for_joint_tree": index in selected_alignment_indices,
        }
        for index, alignment in enumerate(alignments)
    ]
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
        "gauge_posterior": {
            "model": posterior.mode,
            "window_count": len(posterior.window_ids),
            "full_dimension": int(posterior.joint_covariance.shape[0]),
            "exported_factor_rank": factor_rank,
            "retained_covariance_trace_fraction": retained_trace_fraction,
            "retained_normalized_covariance_trace_fraction": (
                retained_trace_fraction
            ),
            "rank_reduction_metric": GAUGE_RANK_REDUCTION_METRIC,
            "rank_reduction_reference_radius_m_by_window": {
                window_id: radius
                for window_id, radius in zip(
                    posterior.window_ids,
                    gauge_reference_radii,
                    strict=True,
                )
            },
            "minimum_retained_gauge_trace": minimum_retained_gauge_trace,
            "max_gauge_rank": max_gauge_rank,
            "cross_window_covariance_preserved": (
                posterior.cross_window_covariance_preserved
            ),
            "parent_window_ids": list(posterior.parent_window_ids),
            "alignments": alignment_records,
            "fixed_lag_boundary_covariance_is_approximate": (
                gauge_mode == "fixed_lag"
            ),
            "fixed_lag_boundary_prior": (
                "schur_complement_v1" if gauge_mode == "fixed_lag" else None
            ),
        },
        "pixel_stride": pixel_stride,
        "sampling_mode": sampling_mode,
        "sampling_semantics": (
            "one deterministic maximum-information valid row per pixel_stride tile"
            if sampling_mode == "information_stratified"
            else "legacy fixed upper-left grid sample"
        ),
        "effective_samples_per_group": effective_samples_per_group,
        "group_composite_weight_semantics": GROUP_COMPOSITE_WEIGHT_SEMANTICS,
        "group_statistics": group_statistics,
        "minimum_prior_reliability": minimum_prior_reliability,
        "uncertainty_model": asdict(model),
        "group_definition": "absolute source frame across overlap windows",
        "factor_definition": "one shared joint gauge latent vector",
        "factor_group_semantics": (
            "all rows use one factor group; each window contributes its block of "
            "the same joint gauge covariance root"
        ),
        "joint_cross_window_gauge_covariance_represented": (
            posterior.cross_window_covariance_preserved
        ),
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
        window_names=posterior.window_ids,
        factor_names=factor_names,
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
    gauge_mode: str = "sequential",
    fixed_lag: int = 4,
    allow_approximate_fixed_lag_covariance: bool = False,
    max_gauge_rank: int | None = 64,
    minimum_retained_gauge_trace: float = 0.999,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
    sampling_mode: SamplingMode = "fixed_grid",
    numerics_policy: ExportNumericsPolicy | None = None,
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
        allow_approximate_fixed_lag_covariance=(
            allow_approximate_fixed_lag_covariance
        ),
        max_gauge_rank=max_gauge_rank,
        minimum_retained_gauge_trace=minimum_retained_gauge_trace,
        view_name=view_name,
        source_revision=source_revision,
        uncertainty_model=uncertainty_model,
        sampling_mode=sampling_mode,
        numerics_policy=numerics_policy,
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
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default="fixed_grid",
        help=(
            "fixed_grid preserves frozen artifacts; information_stratified chooses "
            "one deterministic high-information valid row per stride tile"
        ),
    )
    parser.add_argument("--effective-samples-per-group", type=float, default=64.0)
    parser.add_argument("--minimum-prior-reliability", type=float, default=0.05)
    parser.add_argument(
        "--gauge-mode",
        choices=("sequential", "fixed_lag"),
        default="sequential",
        help=(
            "sequential exports a full joint spanning-tree covariance; fixed_lag "
            "is an explicit approximate reconstruction ablation"
        ),
    )
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument(
        "--allow-approximate-fixed-lag-covariance",
        action="store_true",
        help=(
            "acknowledge that legacy fixed-lag covariance treats marginalized "
            "boundary gauges as exact"
        ),
    )
    parser.add_argument("--max-gauge-rank", type=int, default=64)
    parser.add_argument(
        "--minimum-retained-gauge-trace",
        type=float,
        default=0.999,
    )
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
        sampling_mode=args.sampling_mode,
        effective_samples_per_group=args.effective_samples_per_group,
        minimum_prior_reliability=args.minimum_prior_reliability,
        gauge_mode=args.gauge_mode,
        fixed_lag=args.fixed_lag,
        allow_approximate_fixed_lag_covariance=(
            args.allow_approximate_fixed_lag_covariance
        ),
        max_gauge_rank=args.max_gauge_rank,
        minimum_retained_gauge_trace=args.minimum_retained_gauge_trace,
        view_name=args.view_name,
        source_revision=args.source_revision,
    )
    save_observation_belief_export(args.output_npz, artifact)
    summary = {
        **selection.run_summary(causal_frame_stop=args.causal_frame_stop),
        **artifact.summary(),
        "metric_gauge_anchor_id": anchor.artifact_id,
        "gauge_posterior": artifact.metadata["gauge_posterior"],
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
    "GAUGE_RANK_REDUCTION_METRIC",
    "GROUP_COMPOSITE_WEIGHT_SEMANTICS",
    "SAMPLING_MODES",
    "SamplingMode",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "CausalOverlapSelection",
    "JointGaugePosterior",
    "MetricGaugeAnchor",
    "SelectedOverlapWindow",
    "build_prob4d_observation_belief",
    "deterministic_covariance_root",
    "estimate_joint_gauge_tree",
    "gauge_covariance_factor",
    "joint_gauge_factor",
    "observation_sample_mask",
    "load_metric_gauge_anchor",
    "save_metric_gauge_anchor",
    "select_causal_overlap_windows",
]
