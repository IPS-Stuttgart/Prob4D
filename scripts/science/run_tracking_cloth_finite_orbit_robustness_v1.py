#!/usr/bin/env python3
"""Secondary real-trajectory robustness study for an imperfectly estimated SO(2) orbit.

The study keeps the finite-orbit decision rule fixed, estimates its axis from
noisy anchor observations, selects one range threshold on source recordings,
and evaluates acceptance/harm curves on the previously opened Tracking Cloth
v3 target cohort. It is secondary robustness evidence, not a fresh holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "prob4d.tracking-cloth-finite-orbit-robustness.v1"
PROTOCOL_ID = "tracking-cloth-finite-orbit-robustness-v1"
RESULT_SCHEMA = "prob4d.tracking-cloth-finite-orbit-robustness-result.v1"
SOURCE_SCHEMA = "prob4d.tracking-cloth-finite-orbit-robustness-source-seal.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(value))).hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_group_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("protocol must contain one JSON object")
    supplied = value.get("protocol_id")
    if value.get("schema") != SCHEMA or value.get("schema_version") != 1:
        raise ValueError("unknown robustness protocol schema")
    if supplied != PROTOCOL_ID:
        raise ValueError("robustness protocol identity mismatch")
    required_target = value["dataset"]["target_relative_paths"]
    if len(required_target) != 15 or len(set(required_target)) != 15:
        raise ValueError("exact 15-recording target roster is required")
    if value["marker_labels"] != ["1", "20", "5"]:
        raise ValueError("registered marker triplet changed")
    if value["sampling"]["source_frames_per_recording"] != 12:
        raise ValueError("source sampling changed")
    if value["sampling"]["target_frames_per_recording"] != 128:
        raise ValueError("target sampling changed")
    noise = value["estimated_orbit"]["anchor_noise_std_fraction_of_span"]
    if noise != sorted(noise) or noise[0] != 0.0:
        raise ValueError("anchor-noise sweep must be sorted and start at zero")
    if value["source_selection"]["design_noise_fraction"] not in noise:
        raise ValueError("design noise must occur in the registered sweep")
    if value["source_selection"]["threshold_grid_fraction_of_span"] != sorted(
        value["source_selection"]["threshold_grid_fraction_of_span"]
    ):
        raise ValueError("threshold grid must be sorted")
    boundary = value["claim_boundary"]
    if boundary["fresh_target_holdout"] is not False:
        raise ValueError("this secondary study cannot claim a fresh target holdout")
    if boundary["learned_provider"] is not False:
        raise ValueError("this study does not execute a learned provider")
    return value


def _clean(value: str) -> str:
    return " ".join(value.strip().replace("\ufeff", "").split())


def _marker_columns(
    rows: Sequence[Sequence[str]], labels: Sequence[str]
) -> tuple[int, dict[str, tuple[int, int, int]]]:
    axis_row_index = -1
    for index, row in enumerate(rows[:20]):
        cleaned = [_clean(value).upper() for value in row]
        triples = sum(
            cleaned[column : column + 3] == ["X", "Y", "Z"]
            for column in range(max(0, len(cleaned) - 2))
        )
        if triples >= 3:
            axis_row_index = index
            break
    if axis_row_index < 4:
        raise ValueError("could not locate a Motive multirow marker header")
    type_row = rows[axis_row_index - 4]
    label_row = rows[axis_row_index - 3]
    position_row = rows[axis_row_index - 1]
    axis_row = rows[axis_row_index]
    width = max(map(len, (type_row, label_row, position_row, axis_row)))

    def cell(row: Sequence[str], column: int) -> str:
        return _clean(row[column]) if column < len(row) else ""

    found: dict[str, tuple[int, int, int]] = {}
    for column in range(width - 2):
        axes = [cell(axis_row, column + offset).upper() for offset in range(3)]
        if axes != ["X", "Y", "Z"]:
            continue
        types = [cell(type_row, column + offset).lower() for offset in range(3)]
        positions = [cell(position_row, column + offset).lower() for offset in range(3)]
        if "marker" not in types and not any(
            value.startswith("marker") for value in types
        ):
            continue
        if "position" not in positions and not any(
            value.startswith("position") for value in positions
        ):
            continue
        candidates = [cell(label_row, column + offset) for offset in range(3)]
        candidates = [value for value in candidates if value]
        if not candidates:
            continue
        label = candidates[0]
        if label.lower().startswith("marker "):
            label = label.split(maxsplit=1)[1]
        if label in labels:
            if label in found:
                raise ValueError(f"duplicate marker label {label!r}")
            found[label] = (column, column + 1, column + 2)
    missing = [label for label in labels if label not in found]
    if missing:
        raise ValueError(f"missing required marker labels: {missing}")
    return axis_row_index + 1, found


def _parse_recording(
    path: Path, labels: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(
        "r", encoding="utf-8-sig", errors="strict", newline=""
    ) as stream:
        rows = list(csv.reader(stream))
    data_start, columns = _marker_columns(rows, labels)
    frames: list[int] = []
    values = {label: [] for label in labels}
    for row in rows[data_start:]:
        if not row or not _clean(row[0]):
            continue
        try:
            frame = int(float(_clean(row[0])))
        except ValueError:
            continue
        frame_values: dict[str, np.ndarray] = {}
        valid = True
        for label in labels:
            try:
                vector = np.array(
                    [float(_clean(row[column])) for column in columns[label]],
                    dtype=np.float64,
                )
            except (IndexError, ValueError):
                valid = False
                break
            if not np.all(np.isfinite(vector)):
                valid = False
                break
            frame_values[label] = vector
        if valid:
            frames.append(frame)
            for label in labels:
                values[label].append(frame_values[label])
    if not frames:
        raise ValueError(f"no complete marker rows in {path}")
    return np.asarray(frames, dtype=np.int64), {
        label: np.asarray(vectors, dtype=np.float64)
        for label, vectors in values.items()
    }


def _sample_indices(count: int, requested: int) -> np.ndarray:
    if count <= requested:
        return np.arange(count, dtype=np.int64)
    raw = np.linspace(0, count - 1, requested)
    indices = np.unique(np.rint(raw).astype(np.int64))
    if indices.size != requested:
        raise RuntimeError("deterministic frame sampling did not retain requested count")
    return indices


@dataclass(frozen=True)
class GeometryCase:
    group_id: str
    relative_path: str
    frame: int
    anchor_a: np.ndarray
    anchor_b: np.ndarray
    probe: np.ndarray
    center: np.ndarray
    axis: np.ndarray
    radial_1: np.ndarray
    radial_2: np.ndarray
    span: float
    axial_coordinate: float
    radius: float


def _case(
    group_id: str,
    relative_path: str,
    frame: int,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    probe: np.ndarray,
    minimum_radius_fraction: float,
) -> GeometryCase | None:
    displacement = anchor_b - anchor_a
    span = float(np.linalg.norm(displacement))
    if not np.isfinite(span) or span <= 1e-9:
        return None
    axis = displacement / span
    center = 0.5 * (anchor_a + anchor_b)
    offset = probe - center
    axial = float(offset @ axis)
    radial = offset - axial * axis
    radius = float(np.linalg.norm(radial))
    if not np.isfinite(radius) or radius / span < minimum_radius_fraction:
        return None
    radial_1 = radial / radius
    radial_2 = np.cross(axis, radial_1)
    radial_2 /= np.linalg.norm(radial_2)
    return GeometryCase(
        group_id=group_id,
        relative_path=relative_path,
        frame=int(frame),
        anchor_a=np.array(anchor_a, copy=True),
        anchor_b=np.array(anchor_b, copy=True),
        probe=np.array(probe, copy=True),
        center=center,
        axis=axis,
        radial_1=radial_1,
        radial_2=radial_2,
        span=span,
        axial_coordinate=axial,
        radius=radius,
    )


def _recording_cases(
    dataset_root: Path,
    relative_path: str,
    labels: Sequence[str],
    requested_frames: int,
    minimum_radius_fraction: float,
) -> list[GeometryCase]:
    path = dataset_root / relative_path
    if not path.is_file():
        raise ValueError(f"missing registered recording {relative_path}")
    frames, markers = _parse_recording(path, labels)
    indices = _sample_indices(frames.size, requested_frames)
    group_id = _stable_group_id(relative_path)
    cases = [
        _case(
            group_id,
            relative_path,
            int(frames[index]),
            markers[labels[0]][index],
            markers[labels[1]][index],
            markers[labels[2]][index],
            minimum_radius_fraction,
        )
        for index in indices
    ]
    return [value for value in cases if value is not None]


def _all_csv_files(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if len(files) != 120:
        raise ValueError(f"expected 120 official CSV files, found {len(files)}")
    return files


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _source_roster(root: Path, protocol: Mapping[str, Any]) -> list[str]:
    materials = tuple(
        value.lower() for value in protocol["dataset"]["source_materials"]
    )
    categories = tuple(
        value.lower() for value in protocol["dataset"]["source_categories"]
    )
    values = []
    for path in _all_csv_files(root):
        relative = _relative(path, root)
        lowered = relative.lower()
        stem = path.stem.lower()
        if not any(category in lowered for category in categories):
            continue
        if not any(material in stem for material in materials):
            continue
        if "a2" not in stem:
            continue
        values.append(relative)
    expected = protocol["dataset"]["expected_source_recordings"]
    if len(values) != expected:
        raise ValueError(
            f"expected {expected} source recordings, found {len(values)}"
        )
    return sorted(values)


def _estimated_orbit_range(
    case: GeometryCase,
    anchor_noise_fraction: float,
    replicate: int,
    query: str,
    grid_size: int | None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(
        _stable_seed(
            "estimated-orbit",
            case.relative_path,
            case.frame,
            f"{anchor_noise_fraction:.12g}",
            replicate,
        )
    )
    sigma = anchor_noise_fraction * case.span
    anchor_a = case.anchor_a + rng.normal(scale=sigma, size=3)
    anchor_b = case.anchor_b + rng.normal(scale=sigma, size=3)
    displacement = anchor_b - anchor_a
    estimated_span = float(np.linalg.norm(displacement))
    if estimated_span <= 1e-12:
        return math.inf, math.inf, math.inf
    estimated_axis = displacement / estimated_span
    estimated_center = 0.5 * (anchor_a + anchor_b)
    offset = case.probe - estimated_center
    parallel = float(offset @ estimated_axis) * estimated_axis
    cosine = offset - parallel
    sine = np.cross(estimated_axis, cosine)
    query_vector = case.axis if query == "invariant" else case.radial_1
    cosine_coefficient = float(query_vector @ cosine)
    sine_coefficient = float(query_vector @ sine)
    amplitude = float(math.hypot(cosine_coefficient, sine_coefficient))
    if grid_size is None:
        query_range = 2.0 * amplitude
    else:
        phase_rng = np.random.default_rng(
            _stable_seed(
                "orbit-grid-phase",
                case.relative_path,
                case.frame,
                f"{anchor_noise_fraction:.12g}",
                replicate,
                query,
                grid_size,
            )
        )
        phase = float(phase_rng.uniform(0.0, 2.0 * math.pi / grid_size))
        angles = phase + 2.0 * math.pi * np.arange(grid_size) / grid_size
        values = cosine_coefficient * np.cos(angles) + sine_coefficient * np.sin(
            angles
        )
        query_range = float(np.max(values) - np.min(values))
    cosine_error = float(np.clip(case.axis @ estimated_axis, -1.0, 1.0))
    axis_error_degrees = float(np.degrees(np.arccos(abs(cosine_error))))
    pivot_error_fraction = float(
        np.linalg.norm(estimated_center - case.center) / case.span
    )
    return query_range / case.span, axis_error_degrees, pivot_error_fraction


def _hidden_angle(case: GeometryCase, replicate: int, angle_index: int) -> float:
    rng = np.random.default_rng(
        _stable_seed(
            "hidden-gauge-angle",
            case.relative_path,
            case.frame,
            replicate,
            angle_index,
        )
    )
    return float(rng.uniform(0.0, 2.0 * math.pi))


def _case_records(
    cases: Sequence[GeometryCase],
    noise_fraction: float,
    perturbation_replicates: int,
    hidden_angles_per_replicate: int,
    threshold_fraction: float,
    grid_size: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for replicate in range(perturbation_replicates):
            invariant_range, axis_error, pivot_error = _estimated_orbit_range(
                case,
                noise_fraction,
                replicate,
                "invariant",
                grid_size,
            )
            radial_range, _, _ = _estimated_orbit_range(
                case,
                noise_fraction,
                replicate,
                "radial",
                grid_size,
            )
            invariant_accepted = invariant_range <= threshold_fraction
            radial_accepted = radial_range <= threshold_fraction
            for angle_index in range(hidden_angles_per_replicate):
                angle = _hidden_angle(case, replicate, angle_index)
                truth_radial = case.radius * math.cos(angle)
                local_radial = case.radius
                fallback_radial = 0.0
                local_error = local_radial - truth_radial
                fallback_error = fallback_radial - truth_radial
                harmful_local = local_error**2 > fallback_error**2 + 1e-18
                guarded_radial = local_radial if radial_accepted else fallback_radial
                guarded_error = guarded_radial - truth_radial
                harmful_accepted = bool(radial_accepted and harmful_local)
                rows.append(
                    {
                        "group_id": case.group_id,
                        "relative_path": case.relative_path,
                        "frame": case.frame,
                        "noise_fraction": noise_fraction,
                        "axis_error_degrees": axis_error,
                        "pivot_error_fraction": pivot_error,
                        "invariant_range_fraction": invariant_range,
                        "radial_range_fraction": radial_range,
                        "invariant_accepted": invariant_accepted,
                        "radial_accepted": radial_accepted,
                        "harmful_local": harmful_local,
                        "harmful_accepted_radial": harmful_accepted,
                        "local_squared_error_fraction": (local_error / case.span) ** 2,
                        "fallback_squared_error_fraction": (
                            fallback_error / case.span
                        )
                        ** 2,
                        "guarded_squared_error_fraction": (
                            guarded_error / case.span
                        )
                        ** 2,
                    }
                )
    return rows


def _group_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    result = []
    for group_id, values in sorted(grouped.items()):
        count = len(values)
        radial_accepted = sum(bool(row["radial_accepted"]) for row in values)
        harmful_accepted = sum(
            bool(row["harmful_accepted_radial"]) for row in values
        )
        result.append(
            {
                "group_id": group_id,
                "relative_path": values[0]["relative_path"],
                "sample_count": count,
                "invariant_acceptance": float(
                    np.mean([row["invariant_accepted"] for row in values])
                ),
                "radial_rejection": float(
                    1.0 - np.mean([row["radial_accepted"] for row in values])
                ),
                "harmful_local_fraction": float(
                    np.mean([row["harmful_local"] for row in values])
                ),
                "harmful_accepted_radial_fraction_all": harmful_accepted / count,
                "harmful_fraction_among_accepted_radial": (
                    harmful_accepted / radial_accepted if radial_accepted else None
                ),
                "local_rmse_fraction": float(
                    np.sqrt(
                        np.mean(
                            [
                                row["local_squared_error_fraction"]
                                for row in values
                            ]
                        )
                    )
                ),
                "fallback_rmse_fraction": float(
                    np.sqrt(
                        np.mean(
                            [
                                row["fallback_squared_error_fraction"]
                                for row in values
                            ]
                        )
                    )
                ),
                "guarded_rmse_fraction": float(
                    np.sqrt(
                        np.mean(
                            [
                                row["guarded_squared_error_fraction"]
                                for row in values
                            ]
                        )
                    )
                ),
                "median_axis_error_degrees": float(
                    np.median([row["axis_error_degrees"] for row in values])
                ),
                "median_pivot_error_fraction": float(
                    np.median([row["pivot_error_fraction"] for row in values])
                ),
            }
        )
    return result


_METRICS = (
    "invariant_acceptance",
    "radial_rejection",
    "harmful_local_fraction",
    "harmful_accepted_radial_fraction_all",
    "local_rmse_fraction",
    "fallback_rmse_fraction",
    "guarded_rmse_fraction",
    "median_axis_error_degrees",
    "median_pivot_error_fraction",
)


def _equal_group_aggregate(
    groups: Sequence[Mapping[str, Any]], bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    if not groups:
        raise ValueError("cannot aggregate an empty group roster")
    estimates = {
        metric: float(np.mean([float(row[metric]) for row in groups]))
        for metric in _METRICS
    }
    rng = np.random.default_rng(seed)
    samples = {metric: np.empty(bootstrap_replicates) for metric in _METRICS}
    count = len(groups)
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, count, size=count)
        for metric in _METRICS:
            samples[metric][replicate] = np.mean(
                [float(groups[index][metric]) for index in indices]
            )
    intervals = {
        metric: [
            float(np.quantile(samples[metric], 0.025)),
            float(np.quantile(samples[metric], 0.975)),
        ]
        for metric in _METRICS
    }
    estimates["relative_guarded_rmse_reduction_vs_local"] = float(
        1.0
        - estimates["guarded_rmse_fraction"] / estimates["local_rmse_fraction"]
    )
    return {
        "group_count": count,
        "estimates": estimates,
        "recording_bootstrap_95": intervals,
    }


def _select_threshold(
    cases: Sequence[GeometryCase], protocol: Mapping[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    selection = protocol["source_selection"]
    design_noise = float(selection["design_noise_fraction"])
    minimum_invariant = float(selection["minimum_invariant_acceptance"])
    records = []
    for threshold in selection["threshold_grid_fraction_of_span"]:
        rows = _case_records(
            cases,
            design_noise,
            protocol["estimated_orbit"]["perturbation_replicates"],
            protocol["estimated_orbit"]["hidden_angles_per_replicate"],
            float(threshold),
            None,
        )
        groups = _group_metrics(rows)
        aggregate = _equal_group_aggregate(
            groups,
            bootstrap_replicates=1000,
            seed=_stable_seed("source-threshold", threshold),
        )
        estimate = aggregate["estimates"]
        records.append(
            {
                "threshold_fraction_of_span": float(threshold),
                "invariant_acceptance": estimate["invariant_acceptance"],
                "radial_rejection": estimate["radial_rejection"],
                "harmful_accepted_radial_fraction_all": estimate[
                    "harmful_accepted_radial_fraction_all"
                ],
                "guarded_rmse_fraction": estimate["guarded_rmse_fraction"],
            }
        )
    feasible = [
        row for row in records if row["invariant_acceptance"] >= minimum_invariant
    ]
    pool = feasible if feasible else records
    selected = min(
        pool,
        key=lambda row: (
            row["harmful_accepted_radial_fraction_all"],
            -row["radial_rejection"],
            row["threshold_fraction_of_span"],
        ),
    )
    return float(selected["threshold_fraction_of_span"]), records


def _method_name(grid_size: int | None) -> str:
    return "closed_form" if grid_size is None else f"orbit_grid_{grid_size}"


def _evaluate(
    protocol: Mapping[str, Any], dataset_root: Path, output: Path, revision: str
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)
    labels = protocol["marker_labels"]
    minimum_radius = float(
        protocol["geometry"]["minimum_radius_fraction_of_span"]
    )
    source_paths = _source_roster(dataset_root, protocol)
    source_cases = []
    for relative in source_paths:
        source_cases.extend(
            _recording_cases(
                dataset_root,
                relative,
                labels,
                protocol["sampling"]["source_frames_per_recording"],
                minimum_radius,
            )
        )
    selected_threshold, source_grid = _select_threshold(source_cases, protocol)
    source_seal: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "source_relative_paths": source_paths,
        "source_group_ids": [_stable_group_id(path) for path in source_paths],
        "source_group_count": len(source_paths),
        "source_case_count": len(source_cases),
        "selected_threshold_fraction_of_span": selected_threshold,
        "threshold_selection_records": source_grid,
        "target_recording_headers_opened": False,
        "target_trajectory_values_opened": False,
    }
    source_seal["source_seal_id"] = _content_id(source_seal)
    _write_json(output / "source_seal.json", source_seal)

    target_paths = protocol["dataset"]["target_relative_paths"]
    target_cases_by_group: dict[str, list[GeometryCase]] = {}
    for relative in target_paths:
        cases = _recording_cases(
            dataset_root,
            relative,
            labels,
            protocol["sampling"]["target_frames_per_recording"],
            minimum_radius,
        )
        target_cases_by_group[_stable_group_id(relative)] = cases
    target_case_count = sum(map(len, target_cases_by_group.values()))
    if target_case_count < protocol["sampling"]["minimum_target_cases"]:
        raise ValueError("target geometry support fell below the registered minimum")

    curves = []
    group_rows = []
    methods = [None, *protocol["estimated_orbit"]["orbit_grid_sizes"]]
    for grid_size in methods:
        method = _method_name(grid_size)
        for noise_fraction in protocol["estimated_orbit"][
            "anchor_noise_std_fraction_of_span"
        ]:
            all_rows = []
            for cases in target_cases_by_group.values():
                all_rows.extend(
                    _case_records(
                        cases,
                        float(noise_fraction),
                        protocol["estimated_orbit"]["perturbation_replicates"],
                        protocol["estimated_orbit"]["hidden_angles_per_replicate"],
                        selected_threshold,
                        grid_size,
                    )
                )
            groups = _group_metrics(all_rows)
            for row in groups:
                group_rows.append(
                    {
                        "method": method,
                        "noise_fraction": float(noise_fraction),
                        **row,
                    }
                )
            aggregate = _equal_group_aggregate(
                groups,
                bootstrap_replicates=protocol["inference"][
                    "recording_bootstrap_replicates"
                ],
                seed=_stable_seed("target-bootstrap", method, noise_fraction),
            )
            curves.append(
                {
                    "method": method,
                    "noise_fraction": float(noise_fraction),
                    **aggregate,
                }
            )

    lookup = {(row["method"], row["noise_fraction"]): row for row in curves}
    criteria = {}
    for name, spec in protocol["registered_secondary_criteria"].items():
        row = lookup[(spec["method"], float(spec["noise_fraction"]))]
        measured = float(row["estimates"][spec["metric"]])
        operator = spec["operator"]
        threshold = float(spec["threshold"])
        passed = measured >= threshold if operator == ">=" else measured <= threshold
        criteria[name] = {
            "measured": measured,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "evidence_kind": protocol["evidence_kind"],
        "source_seal_id": source_seal["source_seal_id"],
        "selected_threshold_fraction_of_span": selected_threshold,
        "cohort": {
            "source_groups": len(source_paths),
            "source_cases": len(source_cases),
            "target_groups": len(target_paths),
            "target_cases": target_case_count,
            "target_group_ids": [_stable_group_id(path) for path in target_paths],
        },
        "curves": curves,
        "criteria": criteria,
        "all_registered_secondary_criteria_passed": all(
            row["passed"] for row in criteria.values()
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _content_id(result)
    _write_json(output / "result.json", result)
    _write_curve_csv(output / "robustness_curve.csv", curves)
    _write_group_csv(output / "group_metrics.csv", group_rows)
    (output / "summary.md").write_text(
        _summary(result), encoding="utf-8", newline="\n"
    )
    return result


def _write_curve_csv(
    path: Path, curves: Sequence[Mapping[str, Any]]
) -> None:
    metrics = list(_METRICS) + ["relative_guarded_rmse_reduction_vs_local"]
    fields = ["method", "noise_fraction", "group_count", *metrics]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in curves:
            writer.writerow(
                {
                    "method": row["method"],
                    "noise_fraction": row["noise_fraction"],
                    "group_count": row["group_count"],
                    **{metric: row["estimates"][metric] for metric in metrics},
                }
            )


def _write_group_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# Tracking Cloth estimated-orbit robustness",
        "",
        f"Result ID: `{result['result_id']}`",
        "",
        (
            "Secondary post-hoc robustness analysis on the previously opened v3 "
            "target cohort; not a fresh independent holdout."
        ),
        "",
        (
            "Source-selected range threshold: "
            f"{100.0 * result['selected_threshold_fraction_of_span']:.2f}% "
            "of anchor span."
        ),
        "",
        "| method | anchor noise / span | median axis error | invariant accept | radial reject | harmful accepted | guarded RMSE / span | RMSE reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["curves"]:
        if row["method"] not in {"closed_form", "orbit_grid_16"}:
            continue
        estimate = row["estimates"]
        lines.append(
            f"| {row['method']} | {100.0 * row['noise_fraction']:.2f}% | "
            f"{estimate['median_axis_error_degrees']:.3f}° | "
            f"{estimate['invariant_acceptance']:.4f} | "
            f"{estimate['radial_rejection']:.4f} | "
            f"{estimate['harmful_accepted_radial_fraction_all']:.4f} | "
            f"{estimate['guarded_rmse_fraction']:.4f} | "
            f"{100.0 * estimate['relative_guarded_rmse_reduction_vs_local']:+.2f}% |"
        )
    lines += ["", "Registered secondary criteria:", ""]
    for name, row in result["criteria"].items():
        lines.append(
            f"- `{name}`: {row['measured']:.6f} {row['operator']} "
            f"{row['threshold']:.6f} — **{'pass' if row['passed'] else 'fail'}**"
        )
    lines += [
        "",
        (
            "The gate uses noisy anchor-derived axes and a threshold selected on "
            "source recordings only. The hidden gauge, query construction, and "
            "noise model remain controlled."
        ),
    ]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-protocol")
    validate.add_argument("--protocol", type=Path, required=True)
    self_test = sub.add_parser("self-test")
    self_test.add_argument("--protocol", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--repository-revision", required=True)
    return parser


def _self_test(protocol: Mapping[str, Any]) -> None:
    case = _case(
        "synthetic",
        "synthetic.csv",
        0,
        np.array([-0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.2, 0.1]),
        0.01,
    )
    if case is None:
        raise AssertionError("synthetic geometry was rejected")
    exact_invariant, axis_error, pivot_error = _estimated_orbit_range(
        case, 0.0, 0, "invariant", None
    )
    exact_radial, _, _ = _estimated_orbit_range(
        case, 0.0, 0, "radial", None
    )
    if abs(exact_invariant) > 1e-12:
        raise AssertionError("true-axis invariant query varied")
    if abs(exact_radial - 0.4) > 1e-12:
        raise AssertionError("true-axis radial range is incorrect")
    if axis_error > 1e-12 or pivot_error > 1e-12:
        raise AssertionError("zero anchor noise changed the true line")
    rows = _case_records([case], 0.0, 2, 8, 0.01, None)
    metrics = _group_metrics(rows)[0]
    if metrics["invariant_acceptance"] != 1.0:
        raise AssertionError("invariant query was not accepted")
    if metrics["radial_rejection"] != 1.0:
        raise AssertionError("radial query was not rejected")
    if not 0.45 <= metrics["harmful_local_fraction"] <= 0.8:
        raise AssertionError("hidden-angle local harm control is implausible")
    if protocol["marker_labels"] != ["1", "20", "5"]:
        raise AssertionError("self-test protocol triplet changed")


def main() -> int:
    args = _parser().parse_args()
    protocol = _load_protocol(args.protocol)
    if args.command == "validate-protocol":
        print(json.dumps({"protocol_id": protocol["protocol_id"]}))
        return 0
    if args.command == "self-test":
        _self_test(protocol)
        print(json.dumps({"decision": "self-test-passed"}))
        return 0
    try:
        result = _evaluate(
            protocol,
            args.dataset_root.resolve(strict=True),
            args.output_dir,
            args.repository_revision,
        )
    except Exception as error:
        if not args.output_dir.exists():
            args.output_dir.mkdir(parents=True)
        failure = {
            "schema": "prob4d.tracking-cloth-finite-orbit-robustness-technical-failure.v1",
            "protocol_id": protocol["protocol_id"],
            "failure": f"{type(error).__name__}: {' '.join(str(error).split())}"[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-30:],
        }
        failure["failure_id"] = _content_id(failure)
        _write_json(args.output_dir / "technical-failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 3
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "all_registered_secondary_criteria_passed": result[
                    "all_registered_secondary_criteria_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
