#!/usr/bin/env python3
"""Exploratory provider-only CUT3R dense restart-residual diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_SCHEMA = "prob4d.dot-rope-cut3r-dense-residual-protocol"
RESULT_SCHEMA = "prob4d.dot-rope-cut3r-dense-residual-diagnostic"
PROTOCOL_ID = "f5b3140ee1aac7309d1feb8900d07caa19ac6eb54585480e9231239106f2a19e"
PROVIDER_ID = "952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7"
SEQUENCES = ("R01", "R02", "R03")
OVERLAP = (3, 4, 5)
SCORE = (6, 7)
SHRINKAGE = (0.0, 0.25, 0.5, 0.75, 1.0)
MODELS = (
    ("global_sim3", "none", 0),
    ("global_sim3_plus_linear_image_field", "poly", 1),
    ("global_sim3_plus_quadratic_image_field", "poly", 2),
    ("global_sim3_plus_cubic_image_field", "poly", 3),
    ("global_sim3_plus_per_pixel_field", "pixel", 0),
)
EXPECTED_FILES = {
    "R01-continuous.npz": "481acd49211a2081f728345572efe3b3e51dbbf86d64ade7ae970da341f5e5da",
    "R01-window_a.npz": "7fcc13764a76564b2b2e3cd07d9eca5c505e3bab7359f6b9f9c86f8029752ec2",
    "R01-window_b.npz": "e27a08dd7ffe2da5a8bbc5a7c6bcb510df1af1719cbd0d0355b86bb40f52cac8",
    "R02-continuous.npz": "c1b3db6caeb6020622d66537e72dc78d988b479d29547b9b4b81e05e6c2c46b5",
    "R02-window_a.npz": "625417230466d08de7c26a64d985523f8f6d3181699607e94c0bffbe3e967009",
    "R02-window_b.npz": "a4b34d6a8d1883c0c4e9eb6f3d43cbca5c4886a92cddc583f8a860f728997cfb",
    "R03-continuous.npz": "6e3eaf4a1f46691367916bda3ab4901f39fee487aca9106b21fa8398a44f0e8c",
    "R03-window_a.npz": "4ab0624adeb3bf354c0aa74a39618fc76144dda0d69f4a29fc0bad0e42273bb9",
    "R03-window_b.npz": "bee6a080b2ea616f677f92896b5eed5d499f965265d9568b0d9c157aee4b3bf4",
}


def _content_id(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    if protocol_id != PROTOCOL_ID or _content_id(unsigned) != protocol_id:
        raise ValueError("protocol identity changed")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("protocol schema changed")
    if protocol.get("source_sequences") != list(SEQUENCES):
        raise ValueError("source sequence roster changed")
    if protocol.get("reserved_sequences") != "R04-R70":
        raise ValueError("reserved sequence boundary changed")
    if protocol.get("frames") != {
        "overlap_fit_and_selection": list(OVERLAP),
        "provider_only_score": list(SCORE),
    }:
        raise ValueError("frame contract changed")
    if protocol.get("provider") != {
        "artifact_name": "dot-rope-cut3r-sealed-provider-33329701704-1",
        "run_id": 33329701704,
        "bundle_id": PROVIDER_ID,
        "request_id": "83cc26be92364fc7715d692b3bb966cf914fb9f911e0763823f2789480a00cf2",
        "prob4d_revision": "7eb4867e36742d819c514fad21436d4f475b4bed",
    }:
        raise ValueError("provider binding changed")
    boundary = protocol.get("information_boundary", {})
    for key in (
        "normal_view_images_reopened",
        "two_dimensional_markers_opened",
        "three_dimensional_truth_opened",
        "reserved_sequence_payloads_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"information boundary expanded: {key}")
    if boundary.get("provider_predictions_opened") is not True:
        raise ValueError("provider access was not authorized")
    primary = protocol.get("primary_setting", {})
    if (
        primary.get("grid_step_pixels") != 8
        or primary.get("evaluation_confidence_quantile") != 0.5
    ):
        raise ValueError("primary sampling changed")
    if primary.get("shrinkage_grid") != list(SHRINKAGE):
        raise ValueError("shrinkage grid changed")
    if protocol.get("comparators") != [name for name, _, _ in MODELS]:
        raise ValueError("comparator roster changed")
    if protocol.get("sensitivity") != {
        "grid_step_pixels": [4, 8, 16],
        "evaluation_confidence_quantiles": [0.25, 0.5, 0.75],
        "model": "global_sim3_plus_per_pixel_field",
    }:
        raise ValueError("sensitivity grid changed")
    return protocol


def _verify_bundle(bundle: Path) -> dict[str, Any]:
    manifest = _load_json(bundle / "manifest.json")
    if manifest.get("provider_bundle_id") != PROVIDER_ID:
        raise ValueError("provider bundle identity changed")
    if manifest.get("decision") != "sealed-provider-predictions":
        raise ValueError("provider bundle is not sealed")
    if manifest.get("dataset", {}).get("source_sequences") != list(SEQUENCES):
        raise ValueError("provider source sequence roster changed")
    if manifest.get("dataset", {}).get("reserved_sequences") != "R04-R70":
        raise ValueError("provider reserved sequence boundary changed")
    outputs = {
        Path(row["relative_path"]).name: row
        for row in manifest.get("outputs", [])
    }
    if set(outputs) != set(EXPECTED_FILES):
        raise ValueError("provider output roster changed")
    for filename, expected in EXPECTED_FILES.items():
        path = bundle / "runs" / filename
        row = outputs[filename]
        if (
            not path.is_file()
            or _sha256(path) != expected
            or row.get("sha256") != expected
        ):
            raise ValueError(f"provider output changed: {filename}")
    return manifest


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points) @ self.rotation.T) + self.translation


def _weighted_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> Sim3:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
    ):
        raise ValueError("alignment arrays must both have shape (N,3)")
    if weights.shape != (source.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("alignment weights must be positive")
    total = float(np.sum(weights))
    source_mean = np.sum(weights[:, None] * source, axis=0) / total
    target_mean = np.sum(weights[:, None] * target, axis=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered * weights[:, None]).T @ source_centered / total
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    variance = float(
        np.sum(weights * np.sum(source_centered**2, axis=1)) / total
    )
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("alignment source has no extent")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def _robust_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[Sim3, np.ndarray]:
    active = np.ones(source.shape[0], dtype=bool)
    transform: Sim3 | None = None
    for _ in range(12):
        if int(np.count_nonzero(active)) < 16:
            raise ValueError("too few active dense alignment samples")
        transform = _weighted_umeyama(
            source[active],
            target[active],
            weights[active],
        )
        residual = np.linalg.norm(transform.apply(source) - target, axis=1)
        cutoff = float(np.quantile(residual, 0.8))
        updated = residual <= cutoff
        if np.array_equal(updated, active):
            break
        active = updated
    if transform is None:
        raise AssertionError("alignment was not fitted")
    return transform, active


def _load_run(
    bundle: Path,
    sequence: str,
    run: str,
) -> dict[str, np.ndarray]:
    path = bundle / "runs" / f"{sequence}-{run}.npz"
    with np.load(path, allow_pickle=False) as archive:
        value = {name: archive[name].copy() for name in archive.files}
    required = {"points", "confidence", "frames", "original_sizes"}
    if not required.issubset(value):
        raise ValueError(f"provider run is missing arrays: {path.name}")
    if value["points"].shape[:3] != value["confidence"].shape:
        raise ValueError(f"point/confidence shape mismatch: {path.name}")
    return value


def _frame_index(run: dict[str, np.ndarray], frame: int) -> int:
    positions = np.flatnonzero(run["frames"] == frame)
    if positions.size != 1:
        raise ValueError(f"frame {frame} does not occur exactly once")
    return int(positions[0])


def _grid(
    run: dict[str, np.ndarray],
    frame: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = _frame_index(run, frame)
    points = run["points"][index].astype(np.float64)
    confidence = run["confidence"][index].astype(np.float64)
    height, width = points.shape[:2]
    rows, columns = np.mgrid[0:height:step, 0:width:step]
    return (
        points[rows, columns].reshape(-1, 3),
        confidence[rows, columns].reshape(-1),
        columns.reshape(-1).astype(np.int64),
        rows.reshape(-1).astype(np.int64),
    )


def _collect(
    target_run: dict[str, np.ndarray],
    source_run: dict[str, np.ndarray],
    frames: tuple[int, ...],
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for frame in frames:
        target, target_confidence, x, y = _grid(target_run, frame, step)
        source, source_confidence, _, _ = _grid(source_run, frame, step)
        finite = np.all(np.isfinite(target), axis=1) & np.all(
            np.isfinite(source),
            axis=1,
        )
        weight = np.sqrt(target_confidence * source_confidence)
        finite &= np.isfinite(weight) & (weight > 0.0)
        targets.append(target[finite])
        sources.append(source[finite])
        weights.append(weight[finite])
        columns.append(x[finite])
        rows.append(y[finite])
    return tuple(
        np.concatenate(value)
        for value in (targets, sources, weights, columns, rows)
    )  # type: ignore[return-value]


def _select_frame(
    target_run: dict[str, np.ndarray],
    source_run: dict[str, np.ndarray],
    frame: int,
    step: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target, target_confidence, x, y = _grid(target_run, frame, step)
    source, source_confidence, _, _ = _grid(source_run, frame, step)
    weight = np.sqrt(target_confidence * source_confidence)
    finite = np.all(np.isfinite(target), axis=1) & np.all(
        np.isfinite(source),
        axis=1,
    )
    finite &= np.isfinite(weight) & (weight > 0.0)
    cutoff = float(np.quantile(weight[finite], quantile))
    keep = finite & (weight >= cutoff)
    return target[keep], source[keep], weight[keep], x[keep], y[keep]


def _features(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    x_norm = (x.astype(np.float64) - 255.5) / 255.5
    y_norm = (y.astype(np.float64) - 207.5) / 207.5
    values = [np.ones_like(x_norm), x_norm, y_norm]
    if degree >= 2:
        values += [x_norm**2, x_norm * y_norm, y_norm**2]
    if degree >= 3:
        values += [
            x_norm**3,
            x_norm**2 * y_norm,
            x_norm * y_norm**2,
            y_norm**3,
        ]
    return np.column_stack(values)


def _fit_field(
    kind: str,
    degree: int,
    x: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
) -> Any:
    if kind == "none":
        return None
    if kind == "poly":
        design = _features(x, y, degree)
        normalized = weights / float(np.mean(weights))
        normal = design.T @ (normalized[:, None] * design)
        ridge = 1e-5 * np.eye(design.shape[1])
        ridge[0, 0] = 0.0
        right = design.T @ (normalized[:, None] * residual)
        return np.linalg.solve(normal + ridge, right)
    if kind == "pixel":
        key = (y.astype(np.int64) << 16) | x.astype(np.int64)
        unique, inverse = np.unique(key, return_inverse=True)
        weight_sum = np.bincount(inverse, weights=weights)
        residual_sum = np.stack(
            [
                np.bincount(
                    inverse,
                    weights=weights * residual[:, axis],
                )
                for axis in range(3)
            ],
            axis=1,
        )
        return unique, residual_sum / weight_sum[:, None]
    raise ValueError(kind)


def _predict_field(
    kind: str,
    degree: int,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "none":
        return (
            np.zeros((x.size, 3), dtype=np.float64),
            np.ones(x.size, dtype=bool),
        )
    if kind == "poly":
        return (
            _features(x, y, degree) @ model,
            np.ones(x.size, dtype=bool),
        )
    if kind == "pixel":
        unique, values = model
        key = (y.astype(np.int64) << 16) | x.astype(np.int64)
        positions = np.searchsorted(unique, key)
        safe = np.minimum(positions, unique.size - 1)
        supported = (positions < unique.size) & (unique[safe] == key)
        prediction = np.zeros((x.size, 3), dtype=np.float64)
        prediction[supported] = values[positions[supported]]
        return prediction, supported
    raise ValueError(kind)


def _fit_model(
    target_run: dict[str, np.ndarray],
    source_run: dict[str, np.ndarray],
    frames: tuple[int, ...],
    step: int,
    kind: str,
    degree: int,
) -> tuple[Sim3, Any, dict[str, float | int]]:
    target, source, weights, x, y = _collect(
        target_run,
        source_run,
        frames,
        step,
    )
    transform, active = _robust_sim3(source, target, weights)
    global_prediction = transform.apply(source)
    residual = target - global_prediction
    field = _fit_field(
        kind,
        degree,
        x[active],
        y[active],
        residual[active],
        weights[active],
    )
    squared = np.sum(residual**2, axis=1)
    return transform, field, {
        "sample_count": int(source.shape[0]),
        "active_sample_count": int(np.count_nonzero(active)),
        "global_weighted_rmse_provider_units": float(
            np.sqrt(np.average(squared, weights=weights))
        ),
        "active_global_weighted_rmse_provider_units": float(
            np.sqrt(np.average(squared[active], weights=weights[active]))
        ),
    }


def _apply_model(
    transform: Sim3,
    kind: str,
    degree: int,
    field: Any,
    source: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    correction, supported = _predict_field(kind, degree, field, x, y)
    return transform.apply(source) + shrinkage * correction, supported


def _select_shrinkage(
    reference: dict[str, np.ndarray],
    restart: dict[str, np.ndarray],
    step: int,
    quantile: float,
    kind: str,
    degree: int,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    squared_error = {value: 0.0 for value in SHRINKAGE}
    weight_sum = {value: 0.0 for value in SHRINKAGE}
    folds: list[dict[str, Any]] = []
    for held_out in OVERLAP:
        training = tuple(frame for frame in OVERLAP if frame != held_out)
        transform, field, _ = _fit_model(
            reference,
            restart,
            training,
            step,
            kind,
            degree,
        )
        target, source, weights, x, y = _select_frame(
            reference,
            restart,
            held_out,
            step,
            quantile,
        )
        for value in SHRINKAGE:
            prediction, supported = _apply_model(
                transform,
                kind,
                degree,
                field,
                source,
                x,
                y,
                value,
            )
            error_squared = np.sum((prediction - target) ** 2, axis=1)
            squared_error[value] += float(np.sum(weights * error_squared))
            weight_sum[value] += float(np.sum(weights))
            folds.append(
                {
                    "held_out_frame": held_out,
                    "training_frames": list(training),
                    "shrinkage": value,
                    "weighted_rmse_provider_units": float(
                        np.sqrt(
                            np.sum(weights * error_squared) / np.sum(weights)
                        )
                    ),
                    "sample_count": int(source.shape[0]),
                    "field_support_fraction": float(np.mean(supported)),
                }
            )
    scores = {
        value: float(np.sqrt(squared_error[value] / weight_sum[value]))
        for value in SHRINKAGE
    }
    selected = min(scores, key=lambda value: (scores[value], value))
    return (
        selected,
        {f"{value:.2f}": scores[value] for value in SHRINKAGE},
        folds,
    )


def _score_model(
    reference: dict[str, np.ndarray],
    restart: dict[str, np.ndarray],
    third: dict[str, np.ndarray],
    step: int,
    quantile: float,
    kind: str,
    degree: int,
    shrinkage: float,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    restart_to_reference, field, fit = _fit_model(
        reference,
        restart,
        OVERLAP,
        step,
        kind,
        degree,
    )
    third_to_reference, _, third_fit = _fit_model(
        reference,
        third,
        OVERLAP,
        step,
        "none",
        0,
    )
    squared_error = 0.0
    weight_sum = 0.0
    frames: list[dict[str, Any]] = []
    for frame in SCORE:
        third_points, third_confidence, x, y = _grid(third, frame, step)
        restart_points, restart_confidence, _, _ = _grid(
            restart,
            frame,
            step,
        )
        weights = np.sqrt(third_confidence * restart_confidence)
        finite = np.all(np.isfinite(third_points), axis=1) & np.all(
            np.isfinite(restart_points),
            axis=1,
        )
        finite &= np.isfinite(weights) & (weights > 0.0)
        cutoff = float(np.quantile(weights[finite], quantile))
        keep = finite & (weights >= cutoff)
        target = third_to_reference.apply(third_points[keep])
        prediction, supported = _apply_model(
            restart_to_reference,
            kind,
            degree,
            field,
            restart_points[keep],
            x[keep],
            y[keep],
            shrinkage,
        )
        error_squared = np.sum((prediction - target) ** 2, axis=1)
        frame_sse = float(np.sum(weights[keep] * error_squared))
        frame_weight = float(np.sum(weights[keep]))
        squared_error += frame_sse
        weight_sum += frame_weight
        frames.append(
            {
                "frame": frame,
                "weighted_rmse_provider_units": float(
                    np.sqrt(frame_sse / frame_weight)
                ),
                "sample_count": int(np.count_nonzero(keep)),
                "field_support_fraction": float(np.mean(supported)),
            }
        )
    return (
        float(np.sqrt(squared_error / weight_sum)),
        frames,
        {
            "restart_to_reference": fit,
            "third_to_reference": third_fit,
        },
    )


def _reference_span(
    reference: dict[str, np.ndarray],
    step: int,
    quantile: float,
) -> float:
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for frame in OVERLAP:
        value, confidence, _, _ = _grid(reference, frame, step)
        finite = (
            np.all(np.isfinite(value), axis=1)
            & np.isfinite(confidence)
            & (confidence > 0.0)
        )
        cutoff = float(np.quantile(confidence[finite], quantile))
        keep = finite & (confidence >= cutoff)
        points.append(value[keep])
        weights.append(confidence[keep])
    cloud = np.concatenate(points)
    weight = np.concatenate(weights)
    centroid = np.average(cloud, axis=0, weights=weight)
    radius = np.linalg.norm(cloud - centroid, axis=1)
    span = 2.0 * float(np.quantile(radius, 0.95))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("reference span is invalid")
    return span


def _sequence(
    bundle: Path,
    sequence: str,
    step: int,
    quantile: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = _load_run(bundle, sequence, "window_a")
    restart = _load_run(bundle, sequence, "window_b")
    third = _load_run(bundle, sequence, "continuous")
    span = _reference_span(reference, step, quantile)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "sequence": sequence,
        "reference_span_provider_units": span,
        "models": {},
    }
    for name, kind, degree in MODELS:
        selected, cv_scores, folds = _select_shrinkage(
            reference,
            restart,
            step,
            quantile,
            kind,
            degree,
        )
        score, score_frames, fit = _score_model(
            reference,
            restart,
            third,
            step,
            quantile,
            kind,
            degree,
            selected,
        )
        baseline_cv = cv_scores["0.00"]
        row = {
            "sequence": sequence,
            "method": name,
            "selected_shrinkage": selected,
            "overlap_cv_rmse_provider_units": cv_scores[f"{selected:.2f}"],
            "overlap_cv_relative_improvement": (
                baseline_cv - cv_scores[f"{selected:.2f}"]
            )
            / baseline_cv,
            "score_rmse_provider_units": score,
            "score_rmse_fraction_of_reference_span": score / span,
        }
        rows.append(row)
        details["models"][name] = {
            **row,
            "overlap_cv_by_shrinkage": cv_scores,
            "overlap_cv_folds": folds,
            "score_frames": score_frames,
            "fit": fit,
        }
    baseline = next(
        row for row in rows if row["method"] == "global_sim3"
    )
    for row in rows:
        row["score_relative_improvement_over_global"] = (
            baseline["score_rmse_provider_units"]
            - row["score_rmse_provider_units"]
        ) / baseline["score_rmse_provider_units"]
        details["models"][row["method"]][
            "score_relative_improvement_over_global"
        ] = row["score_relative_improvement_over_global"]
    return rows, details


def _sensitivity(bundle: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in (4, 8, 16):
        for quantile in (0.25, 0.5, 0.75):
            for sequence in SEQUENCES:
                reference = _load_run(bundle, sequence, "window_a")
                restart = _load_run(bundle, sequence, "window_b")
                third = _load_run(bundle, sequence, "continuous")
                selected, cv_scores, _ = _select_shrinkage(
                    reference,
                    restart,
                    step,
                    quantile,
                    "pixel",
                    0,
                )
                score, _, _ = _score_model(
                    reference,
                    restart,
                    third,
                    step,
                    quantile,
                    "pixel",
                    0,
                    selected,
                )
                baseline, _, _ = _score_model(
                    reference,
                    restart,
                    third,
                    step,
                    quantile,
                    "none",
                    0,
                    0.0,
                )
                rows.append(
                    {
                        "sequence": sequence,
                        "grid_step_pixels": step,
                        "evaluation_confidence_quantile": quantile,
                        "selected_shrinkage": selected,
                        "overlap_cv_relative_improvement": (
                            cv_scores["0.00"]
                            - cv_scores[f"{selected:.2f}"]
                        )
                        / cv_scores["0.00"],
                        "score_global_rmse_provider_units": baseline,
                        "score_corrected_rmse_provider_units": score,
                        "score_relative_improvement_over_global": (
                            baseline - score
                        )
                        / baseline,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# DOT CUT3R dense restart-residual diagnostic",
        "",
        f"Diagnostic ID: `{result['diagnostic_id']}`",
        "",
        f"Decision: **{result['decision']}**",
        "",
        (
            "This provider-only exploratory diagnostic did not open DOT marker "
            "coordinates or 3-D truth."
        ),
        "",
        (
            "| Sequence | method | shrinkage | overlap-CV improvement | "
            "score RMSE | score improvement |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["primary_rows"]:
        lines.append(
            f"| {row['sequence']} | {row['method']} | "
            f"{row['selected_shrinkage']:.2f} | "
            f"{100.0 * row['overlap_cv_relative_improvement']:+.2f}% | "
            f"{row['score_rmse_provider_units']:.6f} | "
            f"{100.0 * row['score_relative_improvement_over_global']:+.2f}% |"
        )
    diagnostic = result["decision_diagnostic"]
    lines += [
        "",
        (
            "Primary per-pixel field: "
            f"{diagnostic['score_improvement_sequence_count']}/3 score wins, "
            f"{diagnostic['positive_shrinkage_sequence_count']}/3 positive "
            "shrinkage selections, mean relative score improvement "
            f"{100.0 * diagnostic['mean_relative_score_improvement']:+.2f}%."
        ),
        "",
        result["claim_boundary"],
    ]
    return "\n".join(lines) + "\n"


def _evaluate(
    protocol_path: Path,
    bundle: Path,
    output: Path,
    revision: str,
) -> int:
    protocol = _validate_protocol(protocol_path)
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)
    try:
        manifest = _verify_bundle(bundle)
        all_rows: list[dict[str, Any]] = []
        sequences: list[dict[str, Any]] = []
        for sequence in SEQUENCES:
            rows, details = _sequence(bundle, sequence, 8, 0.5)
            all_rows.extend(rows)
            sequences.append(details)
        sensitivity = _sensitivity(bundle)
        primary_rows = [
            row
            for row in all_rows
            if row["method"] == "global_sim3_plus_per_pixel_field"
        ]
        wins = sum(
            row["score_relative_improvement_over_global"] > 0.0
            for row in primary_rows
        )
        positive = sum(
            row["selected_shrinkage"] > 0.0 for row in primary_rows
        )
        mean_improvement = float(
            np.mean(
                [
                    row["score_relative_improvement_over_global"]
                    for row in primary_rows
                ]
            )
        )
        rule = protocol["decision_rule"]
        supported = (
            wins >= rule["minimum_score_improvement_sequences"]
            and positive >= rule["minimum_positive_shrinkage_sequences"]
            and mean_improvement > 0.0
        )
        result = {
            "schema": RESULT_SCHEMA,
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "repository_revision": revision,
            "provider_bundle_id": manifest["provider_bundle_id"],
            "decision": (
                "exploratory-dense-residual-structure-detected"
                if supported
                else "exploratory-dense-residual-structure-not-detected"
            ),
            "decision_diagnostic": {
                "score_improvement_sequence_count": wins,
                "positive_shrinkage_sequence_count": positive,
                "sequence_count": len(primary_rows),
                "mean_relative_score_improvement": mean_improvement,
                "rule": rule,
                "supported": supported,
            },
            "primary_rows": primary_rows,
            "all_method_rows": all_rows,
            "sequences": sequences,
            "sensitivity_rows": sensitivity,
            "information_boundary": protocol["information_boundary"],
            "claim_boundary": protocol["claim_boundary"],
        }
        result["diagnostic_id"] = _content_id(result)
        _write_json(output / "result.json", result)
        _write_csv(output / "method-summary.csv", all_rows)
        _write_csv(output / "sensitivity.csv", sensitivity)
        (output / "summary.md").write_text(
            _summary(result),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "diagnostic_id": result["diagnostic_id"],
                    **result["decision_diagnostic"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        failure = {
            "schema": (
                "prob4d.dot-rope-cut3r-dense-residual-technical-failure"
            ),
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "repository_revision": revision,
            "decision": "technical-failure",
            "failure": (
                f"{type(error).__name__}: {' '.join(str(error).split())}"
            )[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-30:],
            "information_boundary": protocol["information_boundary"],
        }
        failure["technical_result_id"] = _content_id(failure)
        _write_json(output / "technical-failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 3


def _self_test() -> int:
    rng = np.random.default_rng(20260831)
    source = rng.normal(size=(200, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    truth = Sim3(
        1.2,
        rotation,
        np.array([0.3, -0.2, 0.1]),
    )
    target = truth.apply(source)
    target += rng.normal(scale=1e-4, size=target.shape)
    fitted, active = _robust_sim3(
        source,
        target,
        np.ones(source.shape[0]),
    )
    error = np.sqrt(
        np.mean(np.sum((fitted.apply(source) - target) ** 2, axis=1))
    )
    if error > 5e-4 or np.count_nonzero(active) < 150:
        raise AssertionError("Sim3 self-test failed")
    print(json.dumps({"decision": "self-test-passed", "rmse": error}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-protocol")
    validate.add_argument("--protocol", type=Path, required=True)
    sub.add_parser("self-test")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--provider-bundle", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--repository-revision", required=True)
    args = parser.parse_args()
    if args.command == "validate-protocol":
        print(
            json.dumps(
                {
                    "protocol_id": _validate_protocol(args.protocol)[
                        "protocol_id"
                    ]
                }
            )
        )
        return 0
    if args.command == "self-test":
        return _self_test()
    return _evaluate(
        args.protocol,
        args.provider_bundle.resolve(strict=True),
        args.output_dir,
        args.repository_revision,
    )


if __name__ == "__main__":
    raise SystemExit(main())
