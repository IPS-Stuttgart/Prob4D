"""Export a causally sealed Prob4D prefix as a Phys4D observation belief."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import WindowAlignment, align_windows
from .data import PredictionWindow
from .observation_contract import (
    ObservationBeliefExportV1,
    file_sha256,
    save_observation_belief_export,
)
from .observation_factors import sim3_point_jacobian
from .sim3 import Sim3
from .uncertainty import DepthDisagreementModel, accumulate_disagreement

SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
COORDINATE_MODES = ("gauge_relative", "metric_anchored")
_LINEAGE_SCHEMA_VERSION = 1
_LINEAGE_MODEL = "motioncrafter_sliding_window_v1"
_STABLE_CONFIG_KEYS = (
    "model_type",
    "unet_path",
    "vae_path",
    "height",
    "width",
    "window_size",
    "overlap",
    "frame_start",
    "frame_stride",
    "seed",
    "num_inference_steps",
    "guidance_scale",
)


@dataclass(frozen=True)
class CausalPredictionPrefix:
    """Only complete overlap windows whose source frames precede the cutoff."""

    manifest_path: Path
    metadata: Mapping[str, Any]
    windows: tuple[PredictionWindow, ...]
    selected_records: tuple[Mapping[str, Any], ...]
    excluded_records: tuple[Mapping[str, Any], ...]
    payload_sha256: tuple[str, ...]
    source_artifact_sha256: str
    full_manifest_sha256: str
    causal_frame_stop: int
    frame_stride: int


@dataclass(frozen=True)
class MetricGaugeAnchor:
    """Metric prior for the first retained overlap-window gauge."""

    global_from_first_window: Sim3
    covariance: np.ndarray
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7) or not np.all(np.isfinite(covariance)):
            raise ValueError("metric-anchor covariance must have finite shape (7, 7)")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("metric-anchor covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(0.5 * (covariance + covariance.T)) < -1e-12):
            raise ValueError("metric-anchor covariance must be positive semidefinite")
        object.__setattr__(self, "covariance", 0.5 * (covariance + covariance.T))


@dataclass(frozen=True)
class JointGaugeTree:
    """Causal spanning-tree gauge posterior with full cross-window covariance."""

    window_ids: tuple[str, ...]
    estimates: Mapping[str, Sim3]
    joint_covariance: np.ndarray
    parent_window_ids: tuple[str | None, ...]
    selected_alignment_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        dimension = 7 * len(self.window_ids)
        covariance = np.asarray(self.joint_covariance, dtype=np.float64)
        if covariance.shape != (dimension, dimension):
            raise ValueError("joint gauge covariance has changed shape")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("joint gauge covariance must be finite")
        symmetric = 0.5 * (covariance + covariance.T)
        if np.any(np.linalg.eigvalsh(symmetric) < -1e-9):
            raise ValueError("joint gauge covariance must be positive semidefinite")
        object.__setattr__(self, "joint_covariance", symmetric)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
        return "installed-prob4d"
    return result.stdout.strip() or "installed-prob4d"


def _safe_payload_path(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("overlap-window path must be a nonempty string")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("overlap-window path escapes the prediction directory") from error
    return candidate


def _validate_overlap_lineage(manifest: Mapping[str, Any]) -> None:
    lineage = manifest.get("temporal_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError(
            "prediction manifest lacks temporal_lineage; regenerate it with current Prob4D"
        )
    if int(lineage.get("schema_version", -1)) != _LINEAGE_SCHEMA_VERSION:
        raise ValueError("unsupported temporal-lineage schema version")
    if lineage.get("model") != _LINEAGE_MODEL:
        raise ValueError("unsupported temporal-lineage model")
    products = lineage.get("products")
    if not isinstance(products, Mapping):
        raise ValueError("temporal_lineage products must be a mapping")
    overlap = products.get("overlap_windows")
    if not isinstance(overlap, Mapping):
        raise ValueError("temporal_lineage is missing overlap_windows")
    if overlap.get("window_size_source") != "prediction archive frame count":
        raise ValueError("unsupported overlap-window source-lineage rule")
    if int(overlap.get("overlap", -1)) != 0:
        raise ValueError("overlap-window archives must be independently decoded")


def _causal_source_digest(
    manifest: Mapping[str, Any],
    records: tuple[Mapping[str, Any], ...],
    payload_digests: tuple[str, ...],
    *,
    causal_frame_stop: int,
) -> str:
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("prediction manifest config must be a mapping")
    selected_config = {
        key: config[key]
        for key in _STABLE_CONFIG_KEYS
        if key in config
    }
    descriptor = {
        "format_version": int(manifest["format_version"]),
        "motioncrafter_commit": manifest["motioncrafter_commit"],
        "temporal_lineage": manifest["temporal_lineage"],
        "config": selected_config,
        "causal_frame_stop_exclusive": causal_frame_stop,
        "selected_windows": [
            {
                "window_id": str(record["window_id"]),
                "start_frame": int(record["start_frame"]),
                "stop_frame": int(record["stop_frame"]),
                "payload_sha256": payload_digest,
            }
            for record, payload_digest in zip(records, payload_digests, strict=True)
        ],
    }
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest()


def load_causal_prediction_prefix(
    manifest_path: str | Path,
    *,
    causal_frame_stop: int,
) -> CausalPredictionPrefix:
    """Load only complete pre-cutoff overlap windows and never open excluded payloads."""

    if causal_frame_stop < 1:
        raise ValueError("causal_frame_stop must be positive")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    motioncrafter_commit = manifest.get("motioncrafter_commit")
    if not isinstance(motioncrafter_commit, str) or not motioncrafter_commit:
        raise ValueError("prediction manifest must identify the MotionCrafter revision")
    _validate_overlap_lineage(manifest)
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("prediction manifest config must be a mapping")
    frame_stride = int(config.get("frame_stride", 1))
    if frame_stride < 1:
        raise ValueError("prediction frame_stride must be positive")
    raw_records = manifest.get("overlap_windows")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("prediction manifest has no overlap windows")

    selected: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("overlap-window records must be mappings")
        window_id = str(raw.get("window_id", ""))
        relative_path = str(raw.get("path", ""))
        if not window_id or not relative_path:
            raise ValueError("overlap-window ID and path must be nonempty")
        if window_id in seen_ids or relative_path in seen_paths:
            raise ValueError("overlap-window IDs and paths must be unique")
        seen_ids.add(window_id)
        seen_paths.add(relative_path)
        start_frame = int(raw.get("start_frame", -1))
        stop_frame = int(raw.get("stop_frame", -1))
        if start_frame < 0 or stop_frame <= start_frame:
            raise ValueError("overlap-window source bounds are invalid")
        record = {
            "window_id": window_id,
            "path": relative_path,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
        }
        if stop_frame <= causal_frame_stop:
            selected.append(record)
        else:
            excluded.append(record)
    selected.sort(key=lambda item: (int(item["start_frame"]), str(item["window_id"])))
    excluded.sort(key=lambda item: (int(item["start_frame"]), str(item["window_id"])))
    if not selected:
        raise ValueError(
            "no complete overlap window lies strictly before causal_frame_stop"
        )

    root = path.parent.resolve()
    windows: list[PredictionWindow] = []
    payload_digests: list[str] = []
    expected_shape: tuple[int, int] | None = None
    for record in selected:
        payload = _safe_payload_path(root, record["path"])
        payload_digest = file_sha256(payload)
        window = PredictionWindow.from_npz(
            payload,
            start_frame=int(record["start_frame"]),
            window_id=str(record["window_id"]),
        )
        expected_frames = np.arange(
            int(record["start_frame"]),
            int(record["stop_frame"]),
            frame_stride,
            dtype=np.int64,
        )
        if not np.array_equal(window.frame_indices, expected_frames):
            raise ValueError(
                f"window {window.window_id!r} frame IDs disagree with its manifest bounds"
            )
        if int(window.frame_indices[-1]) >= causal_frame_stop:
            raise ValueError("selected overlap window crosses causal_frame_stop")
        spatial_shape = window.shape[1:]
        if expected_shape is None:
            expected_shape = spatial_shape
        elif spatial_shape != expected_shape:
            raise ValueError("causal overlap windows changed spatial resolution")
        windows.append(window)
        payload_digests.append(payload_digest)

    selected_tuple = tuple(selected)
    digest_tuple = tuple(payload_digests)
    return CausalPredictionPrefix(
        manifest_path=path,
        metadata=manifest,
        windows=tuple(windows),
        selected_records=selected_tuple,
        excluded_records=tuple(excluded),
        payload_sha256=digest_tuple,
        source_artifact_sha256=_causal_source_digest(
            manifest,
            selected_tuple,
            digest_tuple,
            causal_frame_stop=causal_frame_stop,
        ),
        full_manifest_sha256=file_sha256(path),
        causal_frame_stop=causal_frame_stop,
        frame_stride=frame_stride,
    )


def load_metric_gauge_anchor(path: str | Path) -> MetricGaugeAnchor:
    """Load ``mean`` (Sim(3) vector) and ``covariance`` from a non-pickled NPZ."""

    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as archive:
        required = {"mean", "covariance"}
        missing = required - set(archive.files)
        extra = set(archive.files) - required
        if missing or extra:
            raise ValueError(
                "metric-anchor arrays changed; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        mean = np.asarray(archive["mean"], dtype=np.float64)
        covariance = np.asarray(archive["covariance"], dtype=np.float64)
    if mean.shape != (7,) or not np.all(np.isfinite(mean)):
        raise ValueError("metric-anchor mean must have finite shape (7,)")
    return MetricGaugeAnchor(
        global_from_first_window=Sim3.from_vector(mean),
        covariance=covariance,
        source_path=source,
        source_sha256=file_sha256(source),
    )


def _build_alignments(windows: tuple[PredictionWindow, ...]) -> list[WindowAlignment]:
    alignments: list[WindowAlignment] = []
    for moving_index, moving in enumerate(windows):
        for reference in windows[:moving_index]:
            if reference.common_frames(moving).size:
                alignments.append(
                    align_windows(reference, moving, seed=moving_index)
                )
    return alignments


def _numerical_jacobian(function, vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    baseline = np.asarray(function(vector), dtype=np.float64)
    jacobian = np.empty((baseline.size, vector.size), dtype=np.float64)
    for index in range(vector.size):
        step = 1e-6 * max(1.0, abs(float(vector[index])))
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            np.asarray(function(plus), dtype=np.float64)
            - np.asarray(function(minus), dtype=np.float64)
        ) / (2.0 * step)
    return jacobian


def _compose_jacobians(
    parent: Sim3,
    relative: Sim3,
) -> tuple[np.ndarray, np.ndarray]:
    parent_vector = parent.as_vector()
    relative_vector = relative.as_vector()
    parent_jacobian = _numerical_jacobian(
        lambda value: Sim3.from_vector(value).compose(relative).as_vector(),
        parent_vector,
    )
    relative_jacobian = _numerical_jacobian(
        lambda value: parent.compose(Sim3.from_vector(value)).as_vector(),
        relative_vector,
    )
    return parent_jacobian, relative_jacobian


def estimate_joint_gauge_tree(
    windows: tuple[PredictionWindow, ...],
    alignments: list[WindowAlignment],
    *,
    initial_transform: Sim3 | None = None,
    initial_covariance: np.ndarray | None = None,
) -> JointGaugeTree:
    """Estimate a causal spanning tree and propagate its full joint covariance."""

    if not windows:
        raise ValueError("joint gauge estimation requires at least one window")
    window_ids = tuple(window.window_id for window in windows)
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("window IDs must be unique")
    position = {window_id: index for index, window_id in enumerate(window_ids)}
    first_transform = initial_transform or Sim3.identity()
    first_covariance = (
        np.zeros((7, 7), dtype=np.float64)
        if initial_covariance is None
        else np.asarray(initial_covariance, dtype=np.float64)
    )
    if first_covariance.shape != (7, 7):
        raise ValueError("initial gauge covariance must have shape (7, 7)")
    first_covariance = 0.5 * (first_covariance + first_covariance.T)
    if np.any(np.linalg.eigvalsh(first_covariance) < -1e-12):
        raise ValueError("initial gauge covariance must be positive semidefinite")

    dimension = 7 * len(windows)
    joint = np.zeros((dimension, dimension), dtype=np.float64)
    joint[:7, :7] = first_covariance
    estimates: dict[str, Sim3] = {window_ids[0]: first_transform}
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
        parent_jacobian, relative_jacobian = _compose_jacobians(
            parent,
            relative,
        )
        parent_slice = slice(7 * parent_index, 7 * (parent_index + 1))
        child_slice = slice(7 * child_index, 7 * (child_index + 1))
        for previous_index in range(child_index):
            previous_slice = slice(7 * previous_index, 7 * (previous_index + 1))
            cross = parent_jacobian @ joint[parent_slice, previous_slice]
            joint[child_slice, previous_slice] = cross
            joint[previous_slice, child_slice] = cross.T
        child_covariance = (
            parent_jacobian
            @ joint[parent_slice, parent_slice]
            @ parent_jacobian.T
            + relative_jacobian
            @ np.asarray(selected.result.covariance, dtype=np.float64)
            @ relative_jacobian.T
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
    return JointGaugeTree(
        window_ids=window_ids,
        estimates=estimates,
        joint_covariance=joint,
        parent_window_ids=tuple(parent_ids),
        selected_alignment_indices=tuple(alignment_indices),
    )


def deterministic_covariance_root(
    covariance: np.ndarray,
    *,
    max_rank: int | None = None,
    relative_eigenvalue_floor: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Return a deterministic PSD square root and retained trace fraction."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance root requires a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance root requires finite values")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive when supplied")
    if not 0.0 <= relative_eigenvalue_floor < 1.0:
        raise ValueError("relative_eigenvalue_floor must lie in [0, 1)")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.min(eigenvalues) < -1e-9:
        raise ValueError("covariance root requires positive semidefinite input")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    keep = eigenvalues > maximum * relative_eigenvalue_floor
    indices = np.flatnonzero(keep)
    if max_rank is not None:
        indices = indices[:max_rank]
    total_trace = float(np.sum(eigenvalues))
    retained_trace = float(np.sum(eigenvalues[indices]))
    retained_fraction = 1.0 if total_trace == 0.0 else retained_trace / total_trace
    selected_vectors = eigenvectors[:, indices].copy()
    for column in range(selected_vectors.shape[1]):
        pivot = int(np.argmax(np.abs(selected_vectors[:, column])))
        if selected_vectors[pivot, column] < 0.0:
            selected_vectors[:, column] *= -1.0
    root = selected_vectors * np.sqrt(eigenvalues[indices])[None]
    return root, retained_fraction


def joint_gauge_factor(
    points_local: np.ndarray,
    transform: Sim3,
    joint_root_block: np.ndarray,
) -> np.ndarray:
    """Map a joint gauge square-root block into observation coordinates."""

    points = np.asarray(points_local, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("joint gauge factors require three-dimensional points")
    block = np.asarray(joint_root_block, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != 7:
        raise ValueError("joint_root_block must have shape (7, R)")
    flattened = points.reshape(-1, 3)
    jacobian = sim3_point_jacobian(transform, flattened)
    factor = np.einsum("nij,jr->nir", jacobian, block, optimize=True)
    return factor.reshape(points.shape + (block.shape[1],))


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


def _group_metadata(
    correlation_group_ids: np.ndarray,
    entity_ids: np.ndarray,
    prior_reliability: np.ndarray,
    *,
    effective_samples_per_group: float,
    group_prior_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(correlation_group_ids, dtype=np.int64)
    entities = np.asarray(entity_ids, dtype=np.int64)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    if groups.shape != entities.shape or groups.shape != reliability.shape:
        raise ValueError("group metadata arrays must have matching shapes")
    if not np.isfinite(effective_samples_per_group) or (
        effective_samples_per_group <= 0.0
    ):
        raise ValueError("effective_samples_per_group must be positive")
    if not 0.0 <= group_prior_quantile <= 1.0:
        raise ValueError("group_prior_quantile must lie in [0, 1]")
    group_ids = np.unique(groups)
    group_prior = np.empty(len(group_ids), dtype=np.float64)
    group_weight = np.empty(len(group_ids), dtype=np.float64)
    for position, group_id in enumerate(group_ids):
        selected = groups == group_id
        group_prior[position] = float(
            np.quantile(reliability[selected], group_prior_quantile)
        )
        unique_entities = len(np.unique(entities[selected]))
        effective = min(effective_samples_per_group, float(unique_entities))
        group_weight[position] = min(
            1.0,
            effective / float(np.sum(selected)),
        )
    return group_ids, group_prior, group_weight


def _load_uncertainty_model(path: str | Path | None) -> DepthDisagreementModel:
    if path is None:
        return DepthDisagreementModel()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("uncertainty-model JSON must contain an object")
    allowed = set(DepthDisagreementModel.__dataclass_fields__)
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"unknown uncertainty-model fields: {sorted(extra)}")
    return DepthDisagreementModel(**{key: float(value) for key, value in payload.items()})


def build_prob4d_observation_belief(
    manifest_path: str | Path,
    *,
    case_id: str,
    causal_frame_stop: int,
    coordinate_mode: str = "gauge_relative",
    metric_anchor: MetricGaugeAnchor | None = None,
    pixel_stride: int = 4,
    effective_samples_per_group: float = 64.0,
    group_prior_quantile: float = 0.25,
    max_gauge_rank: int | None = 64,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
) -> ObservationBeliefExportV1:
    """Build a canonical artifact after prefix restriction and full re-estimation."""

    if not case_id:
        raise ValueError("case_id must be nonempty")
    if coordinate_mode not in COORDINATE_MODES:
        raise ValueError(f"coordinate_mode must be one of {COORDINATE_MODES}")
    if coordinate_mode == "metric_anchored" and metric_anchor is None:
        raise ValueError("metric_anchored export requires a metric gauge anchor")
    if coordinate_mode == "gauge_relative" and metric_anchor is not None:
        raise ValueError("metric gauge anchors require coordinate_mode='metric_anchored'")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    if not view_name:
        raise ValueError("view_name must be nonempty")

    prefix = load_causal_prediction_prefix(
        manifest_path,
        causal_frame_stop=causal_frame_stop,
    )
    alignments = _build_alignments(prefix.windows)
    initial_transform = (
        Sim3.identity()
        if metric_anchor is None
        else metric_anchor.global_from_first_window
    )
    initial_covariance = (
        np.zeros((7, 7), dtype=np.float64)
        if metric_anchor is None
        else metric_anchor.covariance
    )
    gauges = estimate_joint_gauge_tree(
        prefix.windows,
        alignments,
        initial_transform=initial_transform,
        initial_covariance=initial_covariance,
    )
    joint_root, retained_trace_fraction = deterministic_covariance_root(
        gauges.joint_covariance,
        max_rank=max_gauge_rank,
    )
    rank = joint_root.shape[1]
    windows_by_id = {window.window_id: window for window in prefix.windows}
    evidence = accumulate_disagreement(windows_by_id, alignments)
    model = uncertainty_model or DepthDisagreementModel()

    eligible_frames = sorted(
        {
            int(frame)
            for window in prefix.windows
            for frame in window.frame_indices
        }
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

    for window_index, window in enumerate(prefix.windows):
        transform = gauges.estimates[window.window_id]
        transformed = transform.transform_points(window.point_map)
        uncertainty = model.predict(window, evidence[window.window_id])
        conditional_covariance = uncertainty.transformed(transform).matrices()
        root_block = joint_root[
            7 * window_index : 7 * (window_index + 1), :
        ]
        gauge_factor = joint_gauge_factor(
            window.point_map,
            transform,
            root_block,
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
            selected = window.valid_mask[local_index] & sample_mask
            if not np.any(selected):
                continue
            count = int(np.sum(selected))
            means.append(transformed[local_index][selected])
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
            factor_groups.append(np.zeros(count, dtype=np.int64))
            reliabilities.append(reliability[local_index][selected])
            associations.append(np.ones(count, dtype=np.float64))
            local_covariances.append(
                conditional_covariance[local_index][selected]
            )
            factors.append(gauge_factor[local_index][selected])

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
    group_ids, group_prior, group_weight = _group_metadata(
        correlation_array,
        entity_array,
        reliability_array,
        effective_samples_per_group=effective_samples_per_group,
        group_prior_quantile=group_prior_quantile,
    )

    alignment_records = []
    for index, alignment in enumerate(alignments):
        alignment_records.append(
            {
                "index": index,
                "reference_id": alignment.reference_id,
                "moving_id": alignment.moving_id,
                "common_frames": [int(value) for value in alignment.common_frames],
                "residual_rms": float(alignment.result.residual_rms),
                "num_correspondences": int(alignment.result.num_correspondences),
                "covariance_method": alignment.result.covariance_method,
                "selected_for_gauge_tree": index
                in {
                    value
                    for value in gauges.selected_alignment_indices
                    if value is not None
                },
            }
        )
    anchor_metadata: dict[str, Any] | None = None
    if metric_anchor is not None:
        anchor_metadata = {
            "source_sha256": metric_anchor.source_sha256,
            "mean": metric_anchor.global_from_first_window.as_vector().tolist(),
            "covariance": metric_anchor.covariance.tolist(),
        }
    source_revision_value = source_revision or _git_revision()
    selected_window_records = [
        {
            "window_id": str(record["window_id"]),
            "start_frame": int(record["start_frame"]),
            "stop_frame": int(record["stop_frame"]),
            "payload_sha256": payload_sha,
        }
        for record, payload_sha in zip(
            prefix.selected_records,
            prefix.payload_sha256,
            strict=True,
        )
    ]
    stream_suffix = "metric" if coordinate_mode == "metric_anchored" else "gauge-relative"
    return ObservationBeliefExportV1(
        case_id=case_id,
        stream_id=f"prob4d:{stream_suffix}:overlap-window-points",
        causal_frame_stop=causal_frame_stop,
        view_names=(view_name,),
        window_names=gauges.window_ids,
        factor_names=tuple(
            f"joint_gauge_latent_{index:04d}" for index in range(rank)
        ),
        source_repository=SOURCE_REPOSITORY,
        source_revision=source_revision_value,
        source_artifact_sha256=prefix.source_artifact_sha256,
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
            "coordinate_mode": coordinate_mode,
            "coordinate_units": (
                "m" if coordinate_mode == "metric_anchored" else "gauge_unit"
            ),
            "metric_claim_authorized": coordinate_mode == "metric_anchored",
            "metric_gauge_anchor": anchor_metadata,
            "causal_information_boundary": {
                "causal_frame_stop_exclusive": causal_frame_stop,
                "maximum_source_frame_read": max(eligible_frames),
                "selected_complete_window_count": len(prefix.windows),
                "excluded_future_payloads_opened": 0,
                "admissibility_rule": "window_stop_frame <= causal_frame_stop_exclusive",
                "selected_windows": selected_window_records,
            },
            "source_inputs": {
                "causal_source_artifact_sha256": prefix.source_artifact_sha256,
                "motioncrafter_commit": prefix.metadata["motioncrafter_commit"],
                "prediction_manifest_format_version": prefix.metadata["format_version"],
            },
            "gauge_posterior": {
                "model": "causal_spanning_tree_joint_covariance_v1",
                "window_count": len(gauges.window_ids),
                "full_dimension": int(gauges.joint_covariance.shape[0]),
                "exported_factor_rank": rank,
                "retained_covariance_trace_fraction": retained_trace_fraction,
                "max_gauge_rank": max_gauge_rank,
                "cross_window_covariance_preserved": True,
                "parent_window_ids": list(gauges.parent_window_ids),
                "alignments": alignment_records,
            },
            "observation_model": {
                "pixel_stride": pixel_stride,
                "group_definition": "absolute source frame across overlap windows",
                "factor_group_definition": "one shared joint-gauge latent vector",
                "association_probability_definition": "one for retained valid pixels",
                "prior_reliability_definition": (
                    "overlap disagreement only; independent of physical innovation"
                ),
                "effective_samples_per_group": effective_samples_per_group,
                "group_prior_quantile": group_prior_quantile,
                "conditional_covariance_excludes_gauge_uncertainty": True,
                "uncertainty_model": {
                    key: float(getattr(model, key))
                    for key in DepthDisagreementModel.__dataclass_fields__
                },
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--causal-frame-stop-exclusive",
        "--causal-frame-stop",
        dest="causal_frame_stop",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=COORDINATE_MODES,
        default="gauge_relative",
    )
    parser.add_argument("--metric-anchor", type=Path)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--effective-samples-per-group", type=float, default=64.0)
    parser.add_argument("--group-prior-quantile", type=float, default=0.25)
    parser.add_argument("--max-gauge-rank", type=int, default=64)
    parser.add_argument("--view-name", default="camera0")
    parser.add_argument("--source-revision")
    parser.add_argument("--uncertainty-json", type=Path)
    arguments = parser.parse_args(argv)

    anchor = (
        None
        if arguments.metric_anchor is None
        else load_metric_gauge_anchor(arguments.metric_anchor)
    )
    artifact = build_prob4d_observation_belief(
        arguments.predictions_manifest,
        case_id=arguments.case_id,
        causal_frame_stop=arguments.causal_frame_stop,
        coordinate_mode=arguments.coordinate_mode,
        metric_anchor=anchor,
        pixel_stride=arguments.pixel_stride,
        effective_samples_per_group=arguments.effective_samples_per_group,
        group_prior_quantile=arguments.group_prior_quantile,
        max_gauge_rank=arguments.max_gauge_rank,
        view_name=arguments.view_name,
        source_revision=arguments.source_revision,
        uncertainty_model=_load_uncertainty_model(arguments.uncertainty_json),
    )
    save_observation_belief_export(arguments.output_npz, artifact)
    summary = artifact.summary()
    summary["coordinate_mode"] = artifact.metadata["coordinate_mode"]
    summary["output"] = str(arguments.output_npz.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
