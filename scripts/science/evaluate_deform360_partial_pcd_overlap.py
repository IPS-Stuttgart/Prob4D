#!/usr/bin/env python3
"""Evaluate correlation-aware overlap fusion on fixed-identity Deform360 points.

This is a bounded development study for the partial clean Deform360 subset on
``gpuserver6000``. It uses one frozen source session to fit bias, covariance,
and a target-blind disagreement guard, then reports descriptive metrics on one
separate physical object/session. It is not a complete-dataset confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROFILE = "deform360-partial-pcd-overlap-v1"
PROTOCOL_SCHEMA = "prob4d.deform360-partial-pcd-overlap-protocol"
REQUEST_SCHEMA = "prob4d.deform360-partial-pcd-overlap-request"
RESULT_SCHEMA = "prob4d.deform360-partial-pcd-overlap-result"
EXPECTED_DATASET_ROOT = Path("/mnt/lexar4tb/datasets/deform360")
CHI2_3_90 = 6.251388631170325


class StudyError(ValueError):
    """Raised when the registered study contract is violated."""


@dataclass(frozen=True)
class SequenceData:
    positions: np.ndarray
    velocities: np.ndarray
    visibility: np.ndarray
    frame_paths: tuple[str, ...]


@dataclass(frozen=True)
class QueryCases:
    targets: np.ndarray
    forecasts: np.ndarray
    disagreements: np.ndarray
    frame_indices: np.ndarray
    span_m: float
    selected_point_count: int


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: dict[str, Any], field: str) -> str:
    copy = dict(value)
    copy.pop(field, None)
    return hashlib.sha256(_canonical_bytes(copy)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyError(f"{path} must contain an object")
    return value


def validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "schema_version",
        "profile",
        "dataset_root",
        "runner_label",
        "runner_name",
        "calibration_session",
        "evaluation_session",
        "frame_rate_hz",
        "forecast_ages_frames",
        "tip_fraction",
        "minimum_initial_visibility_views",
        "minimum_selected_points",
        "covariance_diagonal_shrinkage",
        "covariance_relative_eigen_floor",
        "guard_quantile",
        "claim_boundary",
        "protocol_id",
    }
    if set(value) != expected:
        raise StudyError("protocol fields differ from the frozen contract")
    if value["schema"] != PROTOCOL_SCHEMA or value["schema_version"] != 1:
        raise StudyError("unsupported protocol schema")
    if value["profile"] != PROFILE:
        raise StudyError("unexpected profile")
    if value["dataset_root"] != str(EXPECTED_DATASET_ROOT):
        raise StudyError("dataset root differs from reviewed partial subset")
    if value["runner_label"] != "gpuserver6000" or value["runner_name"] != "workstation2":
        raise StudyError("runner identity differs from reviewed host")
    for key in ("calibration_session", "evaluation_session"):
        session = value[key]
        if not isinstance(session, str) or session.startswith("/") or ".." in Path(session).parts:
            raise StudyError(f"invalid {key}")
        if not session.endswith("/episode_0000/pcd_clean"):
            raise StudyError(f"{key} must identify one pcd_clean sequence")
    if value["calibration_session"] == value["evaluation_session"]:
        raise StudyError("calibration and evaluation sessions must differ")
    if value["frame_rate_hz"] != 30.0:
        raise StudyError("frame rate changed")
    if value["forecast_ages_frames"] != [2, 4, 6]:
        raise StudyError("forecast ages changed")
    if not 0.0 < float(value["tip_fraction"]) <= 0.5:
        raise StudyError("tip_fraction out of range")
    if value["minimum_initial_visibility_views"] != 2:
        raise StudyError("visibility gate changed")
    if value["minimum_selected_points"] != 24:
        raise StudyError("minimum point support changed")
    if value["covariance_diagonal_shrinkage"] != 0.05:
        raise StudyError("covariance shrinkage changed")
    if value["covariance_relative_eigen_floor"] != 1e-9:
        raise StudyError("covariance floor changed")
    if value["guard_quantile"] != 0.95:
        raise StudyError("guard quantile changed")
    claim = value["claim_boundary"]
    if not isinstance(claim, str) or "development" not in claim.lower():
        raise StudyError("claim boundary must remain development-only")
    if value["protocol_id"] != _content_id(value, "protocol_id"):
        raise StudyError("protocol identity mismatch")
    return value


def validate_request(
    value: dict[str, Any], *, protocol: dict[str, Any], protocol_git_blob_sha: str
) -> dict[str, Any]:
    expected = {
        "schema",
        "schema_version",
        "profile",
        "protocol_path",
        "protocol_git_blob_sha",
        "protocol_id",
        "execution_authorized",
        "numerical_arrays_authorized",
        "evaluation_session_authorized",
        "dataset_mutation_authorized",
        "claim_boundary",
        "request_id",
    }
    if set(value) != expected:
        raise StudyError("request fields differ from the frozen contract")
    validate_protocol(protocol)
    if value["schema"] != REQUEST_SCHEMA or value["schema_version"] != 1:
        raise StudyError("unsupported request schema")
    if value["profile"] != PROFILE:
        raise StudyError("request profile mismatch")
    if value["protocol_path"] != "protocols/deform360-partial-pcd-overlap-v1.json":
        raise StudyError("request protocol path mismatch")
    if value["protocol_git_blob_sha"] != protocol_git_blob_sha:
        raise StudyError("request does not bind the merged protocol blob")
    if value["protocol_id"] != protocol["protocol_id"]:
        raise StudyError("request protocol identity mismatch")
    for key, expected_value in (
        ("execution_authorized", True),
        ("numerical_arrays_authorized", True),
        ("evaluation_session_authorized", True),
        ("dataset_mutation_authorized", False),
    ):
        if value[key] is not expected_value:
            raise StudyError(f"{key} must be {expected_value}")
    if value["claim_boundary"] != protocol["claim_boundary"]:
        raise StudyError("request claim boundary mismatch")
    if value["request_id"] != _content_id(value, "request_id"):
        raise StudyError("request identity mismatch")
    return value


def _ordinary_directory(root: Path, relative: str) -> Path:
    if root != EXPECTED_DATASET_ROOT or root.is_symlink() or not root.is_dir():
        raise StudyError("dataset root must be the reviewed physical directory")
    resolved_root = root.resolve(strict=True)
    if resolved_root != EXPECTED_DATASET_ROOT:
        raise StudyError("dataset root canonical path changed")
    path = root / relative
    if path.is_symlink() or not path.is_dir():
        raise StudyError(f"registered sequence is unavailable: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise StudyError("registered sequence escapes dataset root")
    return resolved


def _load_sequence(path: Path) -> SequenceData:
    files = tuple(sorted(path.glob("[0-9][0-9][0-9][0-9][0-9][0-9].npz")))
    if len(files) < 20:
        raise StudyError(f"too few pcd_clean frames in {path}: {len(files)}")
    expected_names = tuple(f"{index:06d}.npz" for index in range(len(files)))
    if tuple(item.name for item in files) != expected_names:
        raise StudyError(f"pcd_clean frame roster is not contiguous in {path}")

    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    point_count: int | None = None
    for file in files:
        with np.load(file, allow_pickle=False) as archive:
            if set(archive.files) != {
                "pts",
                "colors",
                "vels",
                "camera_indices",
                "visibility_matrix",
            }:
                raise StudyError(f"unexpected pcd_clean members: {file}")
            pts = np.asarray(archive["pts"], dtype=np.float64)
            vels = np.asarray(archive["vels"], dtype=np.float64)
            visible = np.asarray(archive["visibility_matrix"])
        if pts.ndim != 2 or pts.shape[1] != 3 or vels.shape != pts.shape:
            raise StudyError(f"invalid position/velocity shape: {file}")
        if visible.ndim != 2 or visible.shape[0] != pts.shape[0]:
            raise StudyError(f"invalid visibility shape: {file}")
        if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(vels)):
            raise StudyError(f"nonfinite position/velocity: {file}")
        if point_count is None:
            point_count = pts.shape[0]
        elif pts.shape[0] != point_count:
            raise StudyError(f"persistent point count changed: {file}")
        positions.append(pts)
        velocities.append(vels)
        visibility.append(np.count_nonzero(visible, axis=1).astype(np.int64))
    return SequenceData(
        positions=np.stack(positions),
        velocities=np.stack(velocities),
        visibility=np.stack(visibility),
        frame_paths=tuple(str(item) for item in files),
    )


def _canonical_axis(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def _query_cases(
    sequence: SequenceData,
    *,
    ages: Sequence[int],
    frame_rate_hz: float,
    tip_fraction: float,
    minimum_visibility: int,
    minimum_points: int,
) -> QueryCases:
    initial_visible = sequence.visibility[0] >= minimum_visibility
    candidate = np.flatnonzero(initial_visible)
    if candidate.size < minimum_points:
        raise StudyError("insufficient initially visible point support")
    axis = _canonical_axis(sequence.positions[0, candidate])
    center = np.mean(sequence.positions[0, candidate], axis=0)
    projection = (sequence.positions[0, candidate] - center) @ axis
    count = max(minimum_points, int(math.ceil(tip_fraction * candidate.size)))
    if count > candidate.size:
        raise StudyError("tip query requests more points than supported")
    order = np.argsort(projection, kind="mergesort")
    selected = candidate[order[-count:]]
    full_projection = (sequence.positions[0] - center) @ axis
    span = float(np.quantile(full_projection, 0.95) - np.quantile(full_projection, 0.05))
    if not math.isfinite(span) or span <= 1e-6:
        raise StudyError("object span is degenerate")

    maximum_age = max(ages)
    targets: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    disagreements: list[float] = []
    frame_indices: list[int] = []
    for target_index in range(maximum_age, sequence.positions.shape[0]):
        target = np.mean(sequence.positions[target_index, selected], axis=0)
        predictions = []
        for age in ages:
            origin = target_index - age
            projected = (
                sequence.positions[origin, selected]
                + (float(age) / frame_rate_hz) * sequence.velocities[origin, selected]
            )
            predictions.append(np.mean(projected, axis=0))
        values = np.stack(predictions)
        mean = np.mean(values, axis=0)
        disagreement = float(np.max(np.linalg.norm(values - mean, axis=1)) / span)
        targets.append(target)
        forecasts.append(values)
        disagreements.append(disagreement)
        frame_indices.append(target_index)
    return QueryCases(
        targets=np.stack(targets),
        forecasts=np.stack(forecasts),
        disagreements=np.asarray(disagreements, dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        span_m=span,
        selected_point_count=int(selected.size),
    )


def _fit_calibration(
    cases: QueryCases, *, shrinkage: float, relative_floor: float, guard_quantile: float
) -> dict[str, Any]:
    errors = cases.forecasts - cases.targets[:, None, :]
    bias = np.mean(errors, axis=0)
    centered = (errors - bias[None, :, :]).reshape(errors.shape[0], -1)
    empirical = np.cov(centered, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(empirical))
    covariance = (1.0 - shrinkage) * empirical + shrinkage * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    maximum = float(np.max(eigenvalues))
    floor = max(relative_floor * maximum, np.finfo(np.float64).eps)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    covariance = (covariance + covariance.T) * 0.5
    if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance)[0] <= 0.0:
        raise StudyError("calibrated covariance is not positive definite")
    threshold = float(np.quantile(cases.disagreements, guard_quantile, method="higher"))
    return {
        "bias": bias,
        "covariance": covariance,
        "guard_threshold": threshold,
        "source_case_count": int(errors.shape[0]),
        "source_mean_disagreement": float(np.mean(cases.disagreements)),
        "source_max_disagreement": float(np.max(cases.disagreements)),
        "covariance_eigenvalue_min": float(np.linalg.eigvalsh(covariance)[0]),
        "covariance_eigenvalue_max": float(np.linalg.eigvalsh(covariance)[-1]),
    }


def _gls_weight(covariance: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    design = np.kron(np.ones((count, 1)), np.eye(3))
    solved = np.linalg.solve(covariance, design)
    information = design.T @ solved
    query_covariance = np.linalg.inv(information)
    weight = query_covariance @ solved.T
    return weight, (query_covariance + query_covariance.T) * 0.5


def _block_diagonal(covariance: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros_like(covariance)
    for index in range(count):
        block = slice(3 * index, 3 * (index + 1))
        result[block, block] = covariance[block, block]
    return result


def _method_metrics(errors: np.ndarray, covariance: np.ndarray) -> dict[str, float | int]:
    covariance = (covariance + covariance.T) * 0.5
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        raise StudyError("method covariance is not positive definite")
    solved = np.linalg.solve(covariance, errors.T).T
    mahalanobis = np.einsum("ni,ni->n", errors, solved)
    nll = 0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + mahalanobis)
    norm = np.linalg.norm(errors, axis=1)
    return {
        "case_count": int(errors.shape[0]),
        "rmse_m": float(np.sqrt(np.mean(norm**2))),
        "mean_error_m": float(np.mean(norm)),
        "nll_per_dimension": float(np.mean(nll) / 3.0),
        "normalized_nees": float(np.mean(mahalanobis) / 3.0),
        "coverage_90": float(np.mean(mahalanobis <= CHI2_3_90)),
        "predictive_rms_radius_m": float(np.sqrt(np.trace(covariance))),
    }


def evaluate(cases: QueryCases, calibration: dict[str, Any]) -> dict[str, Any]:
    bias = np.asarray(calibration["bias"], dtype=np.float64)
    covariance = np.asarray(calibration["covariance"], dtype=np.float64)
    count = cases.forecasts.shape[1]
    corrected = cases.forecasts - bias[None, :, :]
    stacked = corrected.reshape(corrected.shape[0], -1)

    latest_weight = np.zeros((3, 3 * count), dtype=np.float64)
    latest_weight[:, :3] = np.eye(3)
    uniform_weight = np.tile(np.eye(3) / count, (1, count))
    independent_covariance = _block_diagonal(covariance, count)
    independent_weight, independent_query_covariance = _gls_weight(
        independent_covariance, count
    )
    correlated_weight, correlated_query_covariance = _gls_weight(covariance, count)
    latest_covariance = covariance[:3, :3]
    uniform_covariance = uniform_weight @ covariance @ uniform_weight.T

    means = {
        "latest": stacked @ latest_weight.T,
        "uniform": stacked @ uniform_weight.T,
        "independent_precision": stacked @ independent_weight.T,
        "correlated_gls": stacked @ correlated_weight.T,
    }
    covariances = {
        "latest": latest_covariance,
        "uniform": uniform_covariance,
        "independent_precision": independent_query_covariance,
        "correlated_gls": correlated_query_covariance,
    }
    accepted = cases.disagreements <= float(calibration["guard_threshold"])
    guarded_mean = np.where(accepted[:, None], means["correlated_gls"], means["latest"])
    guarded_errors = guarded_mean - cases.targets
    correlated_errors = means["correlated_gls"] - cases.targets
    latest_errors = means["latest"] - cases.targets

    metrics = {
        name: _method_metrics(means[name] - cases.targets, covariances[name])
        for name in means
    }
    guarded_nll = []
    guarded_nees = []
    for error, use_candidate in zip(guarded_errors, accepted, strict=True):
        route_covariance = correlated_query_covariance if use_candidate else latest_covariance
        sign, logdet = np.linalg.slogdet(route_covariance)
        if sign <= 0:
            raise StudyError("guarded route covariance is not positive definite")
        mahalanobis = float(error @ np.linalg.solve(route_covariance, error))
        guarded_nees.append(mahalanobis)
        guarded_nll.append(
            0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + mahalanobis)
        )
    guarded_norm = np.linalg.norm(guarded_errors, axis=1)
    metrics["guarded_correlated"] = {
        "case_count": int(guarded_errors.shape[0]),
        "rmse_m": float(np.sqrt(np.mean(guarded_norm**2))),
        "mean_error_m": float(np.mean(guarded_norm)),
        "nll_per_dimension": float(np.mean(guarded_nll) / 3.0),
        "normalized_nees": float(np.mean(guarded_nees) / 3.0),
        "coverage_90": float(np.mean(np.asarray(guarded_nees) <= CHI2_3_90)),
        "predictive_rms_radius_m": float(
            np.mean(
                [
                    np.sqrt(
                        np.trace(correlated_query_covariance if flag else latest_covariance)
                    )
                    for flag in accepted
                ]
            )
        ),
    }

    accepted_harm = (
        np.sum(correlated_errors**2, axis=1) > np.sum(latest_errors**2, axis=1)
    ) & accepted
    latest_mse = np.sum(latest_errors**2, axis=1)
    correlated_mse = np.sum(correlated_errors**2, axis=1)
    scales = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scales, scales)
    off_mask = ~np.eye(correlation.shape[0], dtype=bool)

    decision = (
        "development-positive"
        if metrics["guarded_correlated"]["rmse_m"] < metrics["latest"]["rmse_m"]
        and metrics["correlated_gls"]["nll_per_dimension"]
        < metrics["independent_precision"]["nll_per_dimension"]
        else "development-negative"
    )
    return {
        "decision": decision,
        "metrics": metrics,
        "guard": {
            "threshold_normalized_by_span": float(calibration["guard_threshold"]),
            "accepted_count": int(np.sum(accepted)),
            "fallback_count": int(np.sum(~accepted)),
            "acceptance_rate": float(np.mean(accepted)),
            "harmful_accepted_count": int(np.sum(accepted_harm)),
            "harmful_accepted_rate_among_accepted": (
                float(np.sum(accepted_harm) / np.sum(accepted)) if np.any(accepted) else 0.0
            ),
            "worst_accepted_regret_m2": (
                float(np.max((correlated_mse - latest_mse)[accepted]))
                if np.any(accepted)
                else 0.0
            ),
        },
        "dependence_diagnostic": {
            "mean_absolute_off_diagonal_correlation": float(
                np.mean(np.abs(correlation[off_mask]))
            ),
            "maximum_absolute_off_diagonal_correlation": float(
                np.max(np.abs(correlation[off_mask]))
            ),
            "independent_to_correlated_query_covariance_trace_ratio": float(
                np.trace(independent_query_covariance)
                / np.trace(correlated_query_covariance)
            ),
        },
    }


def _write_no_clobber(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def _summary(result: dict[str, Any]) -> str:
    metrics = result["evaluation"]["metrics"]
    lines = [
        "# Deform360 partial persistent-point overlap study",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Method | RMSE (mm) | NLL / dim | normalized NEES | 90% coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "latest",
        "uniform",
        "independent_precision",
        "correlated_gls",
        "guarded_correlated",
    ):
        row = metrics[name]
        lines.append(
            f"| {name} | {1000.0 * row['rmse_m']:.6f} | "
            f"{row['nll_per_dimension']:.6f} | {row['normalized_nees']:.6f} | "
            f"{100.0 * row['coverage_90']:.2f}% |"
        )
    guard = result["evaluation"]["guard"]
    lines.extend(
        [
            "",
            f"Accepted: `{guard['accepted_count']}`; exact latest-window fallbacks: "
            f"`{guard['fallback_count']}`; harmful accepted updates: "
            f"`{guard['harmful_accepted_count']}`.",
            "",
            "This is descriptive development evidence on a partial Deform360 subset. "
            "It is not complete-dataset confirmation or an independent-object claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(
    *,
    protocol: dict[str, Any],
    request_id: str,
    dataset_root: Path,
    repository_revision: str,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol(protocol)
    calibration_path = _ordinary_directory(dataset_root, protocol["calibration_session"])
    evaluation_path = _ordinary_directory(dataset_root, protocol["evaluation_session"])
    calibration_sequence = _load_sequence(calibration_path)
    calibration_cases = _query_cases(
        calibration_sequence,
        ages=protocol["forecast_ages_frames"],
        frame_rate_hz=protocol["frame_rate_hz"],
        tip_fraction=protocol["tip_fraction"],
        minimum_visibility=protocol["minimum_initial_visibility_views"],
        minimum_points=protocol["minimum_selected_points"],
    )
    fitted = _fit_calibration(
        calibration_cases,
        shrinkage=protocol["covariance_diagonal_shrinkage"],
        relative_floor=protocol["covariance_relative_eigen_floor"],
        guard_quantile=protocol["guard_quantile"],
    )
    calibration_public = {
        key: value for key, value in fitted.items() if key not in {"bias", "covariance"}
    }
    calibration_public.update(
        {
            "bias_m": np.asarray(fitted["bias"]).tolist(),
            "covariance_m2": np.asarray(fitted["covariance"]).tolist(),
            "session": protocol["calibration_session"],
            "frame_count": int(calibration_sequence.positions.shape[0]),
            "point_count": int(calibration_sequence.positions.shape[1]),
            "selected_point_count": calibration_cases.selected_point_count,
            "span_m": calibration_cases.span_m,
        }
    )
    calibration_id = hashlib.sha256(_canonical_bytes(calibration_public)).hexdigest()
    calibration_public["calibration_id"] = calibration_id
    _write_no_clobber(
        output_dir / "calibration.json",
        json.dumps(calibration_public, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    evaluation_sequence = _load_sequence(evaluation_path)
    evaluation_cases = _query_cases(
        evaluation_sequence,
        ages=protocol["forecast_ages_frames"],
        frame_rate_hz=protocol["frame_rate_hz"],
        tip_fraction=protocol["tip_fraction"],
        minimum_visibility=protocol["minimum_initial_visibility_views"],
        minimum_points=protocol["minimum_selected_points"],
    )
    evaluation = evaluate(evaluation_cases, fitted)
    evaluation.update(
        {
            "session": protocol["evaluation_session"],
            "frame_count": int(evaluation_sequence.positions.shape[0]),
            "point_count": int(evaluation_sequence.positions.shape[1]),
            "selected_point_count": evaluation_cases.selected_point_count,
            "span_m": evaluation_cases.span_m,
            "frame_index_min": int(evaluation_cases.frame_indices[0]),
            "frame_index_max": int(evaluation_cases.frame_indices[-1]),
        }
    )
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "profile": PROFILE,
        "decision": evaluation["decision"],
        "protocol_id": protocol["protocol_id"],
        "request_id": request_id,
        "repository_revision": repository_revision,
        "dataset_root": str(dataset_root),
        "dataset_status": "partial-clean-development-subset",
        "calibration": calibration_public,
        "evaluation": evaluation,
        "information_boundary": {
            "calibration_opened_before_evaluation": True,
            "evaluation_used_for_tuning": False,
            "dataset_mutated": False,
            "raw_images_opened": False,
            "tactile_arrays_opened": False,
            "robot_arrays_opened": False,
            "only_registered_pcd_clean_arrays_opened": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    _write_no_clobber(
        output_dir / "result.json",
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_no_clobber(output_dir / "summary.md", _summary(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    protocol_parser = subparsers.add_parser("validate-protocol")
    protocol_parser.add_argument("--protocol", type=Path, required=True)
    request_parser = subparsers.add_parser("validate-request")
    request_parser.add_argument("--protocol", type=Path, required=True)
    request_parser.add_argument("--request", type=Path, required=True)
    request_parser.add_argument("--protocol-git-blob-sha", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--request-id", required=True)
    run_parser.add_argument("--dataset-root", type=Path, required=True)
    run_parser.add_argument("--repository-revision", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    protocol = validate_protocol(_load_json(arguments.protocol))
    if arguments.command == "validate-protocol":
        print(json.dumps({"protocol_id": protocol["protocol_id"]}, sort_keys=True))
        return 0
    if arguments.command == "validate-request":
        request = validate_request(
            _load_json(arguments.request),
            protocol=protocol,
            protocol_git_blob_sha=arguments.protocol_git_blob_sha,
        )
        print(
            json.dumps(
                {"protocol_id": protocol["protocol_id"], "request_id": request["request_id"]},
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "run":
        result = run_study(
            protocol=protocol,
            request_id=arguments.request_id,
            dataset_root=arguments.dataset_root,
            repository_revision=arguments.repository_revision,
            output_dir=arguments.output_dir,
        )
        print(json.dumps({"decision": result["decision"], "result_id": result["result_id"]}))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(main())
