"""Target-free fidelity and causal-prefix closure for direct CUT3R point maps."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._atomic_file import atomic_write_text
from ._cut3r_limits import _unproject_world_points, _validate_camera
from ._cut3r_source import (
    _SourceMemberDescriptor,
    _file_descriptor,
    _load_camera,
    _load_npy,
    _record_id,
    _validated_source_directory,
    _verify_source_descriptors,
)
from ._cut3r_window import _close_memmap
from ._strict_json import load_json_object, require_exact_fields

CUT3R_POINTMAP_FIDELITY_SCHEMA: Final = "prob4d.cut3r-pointmap-fidelity"
CUT3R_POINTMAP_FIDELITY_VERSION: Final = 1
CUT3R_POINTMAP_FIDELITY_DOMAIN: Final = "prob4d.cut3r-pointmap-fidelity.v1"
CUT3R_FIDELITY_SOURCE_DOMAIN: Final = "prob4d.cut3r-fidelity-source-bundle.v1"
CUT3R_FIDELITY_SOURCE_LAYOUT: Final = (
    "cut3r-direct-pointmap-depth-conf-camera-fidelity-v1"
)
CUT3R_POINTMAP_FIDELITY_CLAIM_BOUNDARY: Final = (
    "This target-free audit compares CUT3R direct XYZ output with the historical "
    "depth-plus-intrinsics reprojection and tests whether a longer recurrent run "
    "changes the already-emitted prefix. It establishes representation and causal "
    "execution closure only; it does not establish provider accuracy, calibration, "
    "BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the art."
)

_REPORT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "source_layout",
        "confidence_threshold",
        "prefix_frame_count",
        "source_bundles",
        "geometry_equivalence_policy",
        "geometry_fidelity",
        "causal_prefix_closure",
        "geometry_classification",
        "direct_route_ready",
        "depth_reprojection_compatibility_admissible",
        "claim_boundary",
    }
)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_members(
    root: Path,
) -> dict[str, dict[int, Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("CUT3R fidelity root must be an ordinary directory")
    members = {
        "points": _validated_source_directory(root, "points", ".npy"),
        "depth": _validated_source_directory(root, "depth", ".npy"),
        "conf": _validated_source_directory(root, "conf", ".npy"),
        "camera": _validated_source_directory(root, "camera", ".npz"),
    }
    frame_sets = {tuple(group) for group in members.values()}
    if len(frame_sets) != 1:
        raise ValueError(
            "CUT3R fidelity points, depth, confidence, and camera frame sets disagree"
        )
    return members


def _inventory(
    root: Path,
    members: Mapping[str, Mapping[int, Path]],
) -> tuple[tuple[_SourceMemberDescriptor, ...], str, int]:
    descriptors: list[_SourceMemberDescriptor] = []
    for frame_index in sorted(members["points"]):
        for name in ("points", "depth", "conf", "camera"):
            descriptors.append(_file_descriptor(members[name][frame_index], root=root))
    ordered = tuple(descriptors)
    source_bundle_id = _record_id(
        CUT3R_FIDELITY_SOURCE_DOMAIN,
        {
            "layout": CUT3R_FIDELITY_SOURCE_LAYOUT,
            "members": list(ordered),
        },
    )
    return ordered, source_bundle_id, sum(item["byte_count"] for item in ordered)


def _validate_frame_shapes(
    points: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
) -> None:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("CUT3R fidelity point maps must have shape (H, W, 3)")
    if depth.shape != points.shape[:-1]:
        raise ValueError("CUT3R fidelity depth must match the point-map grid")
    if confidence.shape != depth.shape:
        raise ValueError("CUT3R fidelity confidence must match the depth grid")
    for name, value in (
        ("points", points),
        ("depth", depth),
        ("confidence", confidence),
    ):
        if value.dtype.kind not in {"f", "i", "u"}:
            raise ValueError(f"CUT3R fidelity {name} must be a real numeric array")


def _quantile(value: np.ndarray, probability: float) -> float:
    return float(np.quantile(value, probability, method="linear"))


def _geometry_fidelity(
    root: Path,
    members: Mapping[str, Mapping[int, Path]],
    *,
    confidence_threshold: float,
) -> dict[str, object]:
    per_frame: list[dict[str, object]] = []
    error_sum = 0.0
    error_squared_sum = 0.0
    point_count = 0
    maximum_error = 0.0
    support_mismatch_count = 0

    for frame_index in sorted(members["points"]):
        points, _ = _load_npy(members["points"][frame_index], label="direct points")
        depth, _ = _load_npy(members["depth"][frame_index], label="depth")
        confidence, _ = _load_npy(
            members["conf"][frame_index],
            label="confidence",
        )
        try:
            _validate_frame_shapes(points, depth, confidence)
            pose, intrinsics, _ = _load_camera(members["camera"][frame_index])
            _validate_camera(pose, intrinsics)

            points64 = np.asarray(points, dtype=np.float64)
            confidence64 = np.asarray(confidence, dtype=np.float64)
            direct_valid = (
                np.all(np.isfinite(points64), axis=-1)
                & np.isfinite(confidence64)
                & (points64[..., 2] > 0.0)
                & (confidence64 >= confidence_threshold)
            )
            direct_world = (
                np.einsum("ij,hwj->hwi", pose[:3, :3], points64) + pose[:3, 3]
            )
            direct_valid &= np.all(np.isfinite(direct_world), axis=-1)
            reprojected_world, reprojected_valid = _unproject_world_points(
                depth,
                confidence,
                pose,
                intrinsics,
                confidence_threshold=confidence_threshold,
            )
            common = direct_valid & reprojected_valid
            support_mismatch = int(np.count_nonzero(direct_valid ^ reprojected_valid))
            support_mismatch_count += support_mismatch
            count = int(np.count_nonzero(common))
            frame_record: dict[str, object] = {
                "frame_index": frame_index,
                "direct_support_count": int(np.count_nonzero(direct_valid)),
                "reprojected_support_count": int(np.count_nonzero(reprojected_valid)),
                "common_support_count": count,
                "support_mismatch_count": support_mismatch,
            }
            if count:
                error = np.linalg.norm(
                    direct_world[common] - reprojected_world[common],
                    axis=1,
                )
                frame_record.update(
                    mean_error_m=float(np.mean(error)),
                    rms_error_m=float(np.sqrt(np.mean(error**2))),
                    median_error_m=_quantile(error, 0.5),
                    p90_error_m=_quantile(error, 0.9),
                    p95_error_m=_quantile(error, 0.95),
                    maximum_error_m=float(np.max(error)),
                )
                error_sum += float(np.sum(error))
                error_squared_sum += float(np.sum(error**2))
                point_count += count
                maximum_error = max(maximum_error, float(np.max(error)))
            else:
                frame_record.update(
                    mean_error_m=None,
                    rms_error_m=None,
                    median_error_m=None,
                    p90_error_m=None,
                    p95_error_m=None,
                    maximum_error_m=None,
                )
            per_frame.append(frame_record)
        finally:
            _close_memmap(points)
            _close_memmap(depth)
            _close_memmap(confidence)

    if point_count == 0:
        raise ValueError("CUT3R fidelity audit has no common supported direct/reprojected point")
    medians = np.asarray(
        [
            item["median_error_m"]
            for item in per_frame
            if item["median_error_m"] is not None
        ],
        dtype=np.float64,
    )
    frame_p95 = np.asarray(
        [item["p95_error_m"] for item in per_frame if item["p95_error_m"] is not None],
        dtype=np.float64,
    )
    return {
        "evaluated_frame_count": len(per_frame),
        "evaluated_point_count": point_count,
        "support_mismatch_count": support_mismatch_count,
        "point_weighted_mean_error_m": error_sum / point_count,
        "point_weighted_rms_error_m": math.sqrt(error_squared_sum / point_count),
        "maximum_error_m": maximum_error,
        "frame_equal_median_error_m": float(np.median(medians)),
        "maximum_frame_p95_error_m": float(np.max(frame_p95)),
        "per_frame": per_frame,
    }


def _vector_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[int, int, float, float]:
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError("CUT3R prefix point-map shapes changed")
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    first_finite = np.all(np.isfinite(first64), axis=-1)
    second_finite = np.all(np.isfinite(second64), axis=-1)
    finite_mismatch = int(np.count_nonzero(first_finite ^ second_finite))
    common = first_finite & second_finite
    count = int(np.count_nonzero(common))
    if not count:
        return 0, finite_mismatch, 0.0, 0.0
    difference = np.linalg.norm(first64[common] - second64[common], axis=1)
    return count, finite_mismatch, float(np.sum(difference**2)), float(np.max(difference))


def _scalar_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    name: str,
) -> tuple[int, int, float, float]:
    if first.shape != second.shape:
        raise ValueError(f"CUT3R prefix {name} shapes changed")
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    first_finite = np.isfinite(first64)
    second_finite = np.isfinite(second64)
    finite_mismatch = int(np.count_nonzero(first_finite ^ second_finite))
    common = first_finite & second_finite
    count = int(np.count_nonzero(common))
    if not count:
        return 0, finite_mismatch, 0.0, 0.0
    difference = np.abs(first64[common] - second64[common])
    return count, finite_mismatch, float(np.sum(difference**2)), float(np.max(difference))


def _causal_prefix_closure(
    prefix_root: Path,
    extended_root: Path,
    prefix_members: Mapping[str, Mapping[int, Path]],
    extended_members: Mapping[str, Mapping[int, Path]],
    *,
    prefix_frame_count: int,
    confidence_threshold: float,
    point_tolerance_m: float,
    scalar_tolerance: float,
    camera_tolerance: float,
) -> dict[str, object]:
    if len(prefix_members["points"]) != prefix_frame_count:
        raise ValueError("prefix root frame count must equal prefix_frame_count")
    if len(extended_members["points"]) < prefix_frame_count:
        raise ValueError("extended root is shorter than prefix_frame_count")

    point_count = 0
    point_finite_mismatch = 0
    point_squared_sum = 0.0
    point_maximum = 0.0
    depth_count = 0
    depth_finite_mismatch = 0
    depth_squared_sum = 0.0
    depth_maximum = 0.0
    confidence_count = 0
    confidence_finite_mismatch = 0
    confidence_squared_sum = 0.0
    confidence_maximum = 0.0
    camera_maximum = 0.0
    support_mismatch_count = 0
    per_frame: list[dict[str, object]] = []

    for frame_index in range(prefix_frame_count):
        prefix_points, _ = _load_npy(
            prefix_members["points"][frame_index],
            label="prefix direct points",
        )
        extended_points, _ = _load_npy(
            extended_members["points"][frame_index],
            label="extended direct points",
        )
        prefix_depth, _ = _load_npy(
            prefix_members["depth"][frame_index],
            label="prefix depth",
        )
        extended_depth, _ = _load_npy(
            extended_members["depth"][frame_index],
            label="extended depth",
        )
        prefix_confidence, _ = _load_npy(
            prefix_members["conf"][frame_index],
            label="prefix confidence",
        )
        extended_confidence, _ = _load_npy(
            extended_members["conf"][frame_index],
            label="extended confidence",
        )
        try:
            _validate_frame_shapes(prefix_points, prefix_depth, prefix_confidence)
            _validate_frame_shapes(extended_points, extended_depth, extended_confidence)
            p_count, p_mismatch, p_squared, p_maximum = _vector_difference(
                prefix_points,
                extended_points,
            )
            d_count, d_mismatch, d_squared, d_maximum = _scalar_difference(
                prefix_depth,
                extended_depth,
                name="depth",
            )
            c_count, c_mismatch, c_squared, c_maximum = _scalar_difference(
                prefix_confidence,
                extended_confidence,
                name="confidence",
            )
            prefix_pose, prefix_intrinsics, _ = _load_camera(
                prefix_members["camera"][frame_index]
            )
            extended_pose, extended_intrinsics, _ = _load_camera(
                extended_members["camera"][frame_index]
            )
            _validate_camera(prefix_pose, prefix_intrinsics)
            _validate_camera(extended_pose, extended_intrinsics)
            frame_camera_maximum = max(
                float(np.max(np.abs(prefix_pose - extended_pose))),
                float(np.max(np.abs(prefix_intrinsics - extended_intrinsics))),
            )

            prefix_points64 = np.asarray(prefix_points, dtype=np.float64)
            extended_points64 = np.asarray(extended_points, dtype=np.float64)
            prefix_confidence64 = np.asarray(prefix_confidence, dtype=np.float64)
            extended_confidence64 = np.asarray(extended_confidence, dtype=np.float64)
            prefix_support = (
                np.all(np.isfinite(prefix_points64), axis=-1)
                & np.isfinite(prefix_confidence64)
                & (prefix_points64[..., 2] > 0.0)
                & (prefix_confidence64 >= confidence_threshold)
            )
            extended_support = (
                np.all(np.isfinite(extended_points64), axis=-1)
                & np.isfinite(extended_confidence64)
                & (extended_points64[..., 2] > 0.0)
                & (extended_confidence64 >= confidence_threshold)
            )
            frame_support_mismatch = int(
                np.count_nonzero(prefix_support ^ extended_support)
            )

            point_count += p_count
            point_finite_mismatch += p_mismatch
            point_squared_sum += p_squared
            point_maximum = max(point_maximum, p_maximum)
            depth_count += d_count
            depth_finite_mismatch += d_mismatch
            depth_squared_sum += d_squared
            depth_maximum = max(depth_maximum, d_maximum)
            confidence_count += c_count
            confidence_finite_mismatch += c_mismatch
            confidence_squared_sum += c_squared
            confidence_maximum = max(confidence_maximum, c_maximum)
            camera_maximum = max(camera_maximum, frame_camera_maximum)
            support_mismatch_count += frame_support_mismatch
            per_frame.append(
                {
                    "frame_index": frame_index,
                    "point_maximum_difference_m": p_maximum,
                    "depth_maximum_difference": d_maximum,
                    "confidence_maximum_difference": c_maximum,
                    "camera_maximum_difference": frame_camera_maximum,
                    "support_mismatch_count": frame_support_mismatch,
                }
            )
        finally:
            for value in (
                prefix_points,
                extended_points,
                prefix_depth,
                extended_depth,
                prefix_confidence,
                extended_confidence,
            ):
                _close_memmap(value)

    point_rms = math.sqrt(point_squared_sum / point_count) if point_count else None
    depth_rms = math.sqrt(depth_squared_sum / depth_count) if depth_count else None
    confidence_rms = (
        math.sqrt(confidence_squared_sum / confidence_count)
        if confidence_count
        else None
    )
    passed = (
        point_count > 0
        and depth_count > 0
        and confidence_count > 0
        and point_finite_mismatch == 0
        and depth_finite_mismatch == 0
        and confidence_finite_mismatch == 0
        and support_mismatch_count == 0
        and point_maximum <= point_tolerance_m
        and depth_maximum <= scalar_tolerance
        and confidence_maximum <= scalar_tolerance
        and camera_maximum <= camera_tolerance
    )
    return {
        "status": "pass" if passed else "fail",
        "prefix_frame_count": prefix_frame_count,
        "point_common_count": point_count,
        "point_finite_pattern_mismatch_count": point_finite_mismatch,
        "point_rms_difference_m": point_rms,
        "point_maximum_difference_m": point_maximum,
        "depth_common_count": depth_count,
        "depth_finite_pattern_mismatch_count": depth_finite_mismatch,
        "depth_rms_difference": depth_rms,
        "depth_maximum_difference": depth_maximum,
        "confidence_common_count": confidence_count,
        "confidence_finite_pattern_mismatch_count": confidence_finite_mismatch,
        "confidence_rms_difference": confidence_rms,
        "confidence_maximum_difference": confidence_maximum,
        "camera_maximum_difference": camera_maximum,
        "support_mismatch_count": support_mismatch_count,
        "point_tolerance_m": point_tolerance_m,
        "scalar_tolerance": scalar_tolerance,
        "camera_tolerance": camera_tolerance,
        "per_frame": per_frame,
    }


def build_cut3r_pointmap_fidelity_report(
    prefix_root: str | Path,
    extended_root: str | Path,
    *,
    prefix_frame_count: int,
    confidence_threshold: float,
    maximum_rms_error_m: float,
    maximum_frame_p95_error_m: float,
    point_closure_tolerance_m: float = 1e-6,
    scalar_closure_tolerance: float = 1e-6,
    camera_closure_tolerance: float = 1e-7,
) -> dict[str, object]:
    """Build one source-only direct-versus-reprojected and prefix-closure report."""

    prefix_count = _positive_integer(prefix_frame_count, name="prefix_frame_count")
    threshold = _finite_nonnegative(
        confidence_threshold,
        name="confidence_threshold",
    )
    maximum_rms = _finite_nonnegative(
        maximum_rms_error_m,
        name="maximum_rms_error_m",
    )
    maximum_p95 = _finite_nonnegative(
        maximum_frame_p95_error_m,
        name="maximum_frame_p95_error_m",
    )
    point_tolerance = _finite_nonnegative(
        point_closure_tolerance_m,
        name="point_closure_tolerance_m",
    )
    scalar_tolerance = _finite_nonnegative(
        scalar_closure_tolerance,
        name="scalar_closure_tolerance",
    )
    camera_tolerance = _finite_nonnegative(
        camera_closure_tolerance,
        name="camera_closure_tolerance",
    )

    prefix = Path(prefix_root)
    extended = Path(extended_root)
    prefix_members = _validated_members(prefix)
    extended_members = _validated_members(extended)
    prefix_descriptors, prefix_bundle_id, prefix_bytes = _inventory(
        prefix,
        prefix_members,
    )
    extended_descriptors, extended_bundle_id, extended_bytes = _inventory(
        extended,
        extended_members,
    )

    fidelity = _geometry_fidelity(
        prefix,
        prefix_members,
        confidence_threshold=threshold,
    )
    closure = _causal_prefix_closure(
        prefix,
        extended,
        prefix_members,
        extended_members,
        prefix_frame_count=prefix_count,
        confidence_threshold=threshold,
        point_tolerance_m=point_tolerance,
        scalar_tolerance=scalar_tolerance,
        camera_tolerance=camera_tolerance,
    )
    _verify_source_descriptors(prefix, prefix_descriptors)
    _verify_source_descriptors(extended, extended_descriptors)

    equivalent = (
        float(fidelity["point_weighted_rms_error_m"]) <= maximum_rms
        and float(fidelity["maximum_frame_p95_error_m"]) <= maximum_p95
    )
    direct_ready = closure["status"] == "pass"
    identity_record: dict[str, object] = {
        "schema": CUT3R_POINTMAP_FIDELITY_SCHEMA,
        "schema_version": CUT3R_POINTMAP_FIDELITY_VERSION,
        "source_layout": CUT3R_FIDELITY_SOURCE_LAYOUT,
        "confidence_threshold": threshold,
        "prefix_frame_count": prefix_count,
        "source_bundles": {
            "prefix": {
                "source_bundle_id": prefix_bundle_id,
                "member_count": len(prefix_descriptors),
                "total_bytes": prefix_bytes,
            },
            "extended": {
                "source_bundle_id": extended_bundle_id,
                "member_count": len(extended_descriptors),
                "total_bytes": extended_bytes,
            },
        },
        "geometry_equivalence_policy": {
            "maximum_rms_error_m": maximum_rms,
            "maximum_frame_p95_error_m": maximum_p95,
        },
        "geometry_fidelity": fidelity,
        "causal_prefix_closure": closure,
        "geometry_classification": (
            "depth-reprojection-equivalent"
            if equivalent
            else "direct-pointmap-required"
        ),
        "direct_route_ready": direct_ready,
        "depth_reprojection_compatibility_admissible": bool(
            direct_ready and equivalent
        ),
        "claim_boundary": CUT3R_POINTMAP_FIDELITY_CLAIM_BOUNDARY,
    }
    return {
        **identity_record,
        "artifact_id": _record_id(CUT3R_POINTMAP_FIDELITY_DOMAIN, identity_record),
    }


def save_cut3r_pointmap_fidelity_report(
    path: str | Path,
    report: Mapping[str, object],
) -> Path:
    destination = Path(path)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        atomic_write_text(destination, content, overwrite=False)
    except FileExistsError:
        existing = load_cut3r_pointmap_fidelity_report(destination)
        if existing != dict(report):
            raise ValueError("refusing to replace a different CUT3R fidelity report") from None
    return destination


def load_cut3r_pointmap_fidelity_report(path: str | Path) -> dict[str, object]:
    report = load_json_object(path, name="CUT3R point-map fidelity report")
    require_exact_fields(report, _REPORT_FIELDS, name="CUT3R fidelity report")
    if report["schema"] != CUT3R_POINTMAP_FIDELITY_SCHEMA:
        raise ValueError("unsupported CUT3R point-map fidelity schema")
    if report["schema_version"] != CUT3R_POINTMAP_FIDELITY_VERSION:
        raise ValueError("unsupported CUT3R point-map fidelity version")
    identity_record = dict(report)
    supplied_id = identity_record.pop("artifact_id")
    expected_id = _record_id(CUT3R_POINTMAP_FIDELITY_DOMAIN, identity_record)
    if supplied_id != expected_id:
        raise ValueError("CUT3R point-map fidelity artifact ID mismatch")
    return report


def verify_cut3r_pointmap_fidelity_report(
    path: str | Path,
    prefix_root: str | Path,
    extended_root: str | Path,
) -> dict[str, object]:
    report = load_cut3r_pointmap_fidelity_report(path)
    policy = report["geometry_equivalence_policy"]
    closure = report["causal_prefix_closure"]
    if not isinstance(policy, Mapping) or not isinstance(closure, Mapping):
        raise ValueError("CUT3R fidelity policy and closure records must be mappings")
    replay = build_cut3r_pointmap_fidelity_report(
        prefix_root,
        extended_root,
        prefix_frame_count=report["prefix_frame_count"],
        confidence_threshold=report["confidence_threshold"],
        maximum_rms_error_m=policy["maximum_rms_error_m"],
        maximum_frame_p95_error_m=policy["maximum_frame_p95_error_m"],
        point_closure_tolerance_m=closure["point_tolerance_m"],
        scalar_closure_tolerance=closure["scalar_tolerance"],
        camera_closure_tolerance=closure["camera_tolerance"],
    )
    if replay != report:
        raise ValueError("CUT3R point-map fidelity replay mismatch")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction cut3r-fidelity",
        description=(
            "audit direct CUT3R XYZ fidelity and recurrent causal-prefix closure "
            "without target outcomes"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("prefix_root")
    build.add_argument("extended_root")
    build.add_argument("output")
    build.add_argument("--prefix-frame-count", type=int, required=True)
    build.add_argument("--confidence-threshold", type=float, default=1.5)
    build.add_argument("--maximum-rms-error-m", type=float, required=True)
    build.add_argument("--maximum-frame-p95-error-m", type=float, required=True)
    build.add_argument("--point-closure-tolerance-m", type=float, default=1e-6)
    build.add_argument("--scalar-closure-tolerance", type=float, default=1e-6)
    build.add_argument("--camera-closure-tolerance", type=float, default=1e-7)
    build.add_argument("--require-direct-ready", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("report")
    verify.add_argument("prefix_root")
    verify.add_argument("extended_root")
    verify.add_argument("--require-direct-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    if arguments.command == "build":
        report = build_cut3r_pointmap_fidelity_report(
            arguments.prefix_root,
            arguments.extended_root,
            prefix_frame_count=arguments.prefix_frame_count,
            confidence_threshold=arguments.confidence_threshold,
            maximum_rms_error_m=arguments.maximum_rms_error_m,
            maximum_frame_p95_error_m=arguments.maximum_frame_p95_error_m,
            point_closure_tolerance_m=arguments.point_closure_tolerance_m,
            scalar_closure_tolerance=arguments.scalar_closure_tolerance,
            camera_closure_tolerance=arguments.camera_closure_tolerance,
        )
        save_cut3r_pointmap_fidelity_report(arguments.output, report)
    else:
        report = verify_cut3r_pointmap_fidelity_report(
            arguments.report,
            arguments.prefix_root,
            arguments.extended_root,
        )
    print(str(report["artifact_id"]))
    if arguments.require_direct_ready and not report["direct_route_ready"]:
        return 3
    return 0


__all__ = [
    "CUT3R_POINTMAP_FIDELITY_CLAIM_BOUNDARY",
    "CUT3R_POINTMAP_FIDELITY_SCHEMA",
    "CUT3R_POINTMAP_FIDELITY_VERSION",
    "build_cut3r_pointmap_fidelity_report",
    "load_cut3r_pointmap_fidelity_report",
    "main",
    "save_cut3r_pointmap_fidelity_report",
    "verify_cut3r_pointmap_fidelity_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
