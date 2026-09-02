#!/usr/bin/env python3
"""Build a retrospective sealed DEFORM DLO4/DLO5 benchmark panel.

The adapter fits covariance parameters on the public training trajectories and
constructs predictions from only the current prefix plus registered future
endpoint motion.  It emits a challenge-owned truth tree and three provider-owned
submission trees, then invokes the official separated evaluator.  The result is
retrospective diagnostic evidence: the public evaluation trajectories were
already open before this protocol was registered.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
PROTOCOL_SCHEMA: Final = "prob4d.information-contract-deform-adapter-protocol"
PROTOCOL_ID: Final = "prob4d-information-contract-deform-dlo45-retrospective-v1"
DLOS: Final = ("DLO4", "DLO5")
FRAME_COUNT: Final = 500
NODE_COUNT: Final = 12
INTERNAL: Final = slice(2, 10)
ACTION_NODES: Final = np.asarray([0, 1, 10, 11], dtype=np.int64)
_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class WindowSpec:
    prefix_frames: int
    horizon_frames: int
    stride_frames: int
    decay: float


@dataclass(frozen=True)
class SourceCovariance:
    local_m2: FloatArray
    shared_m2: FloatArray
    source_window_count: int
    source_row_count: int


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deterministic_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.kind == "O":
                raise ValueError(f"object array is forbidden: {name}")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stream.getvalue())


def _load_protocol(path: Path) -> tuple[dict[str, Any], WindowSpec]:
    value = _read_json(path)
    if (
        value.get("schema_name") != PROTOCOL_SCHEMA
        or value.get("schema_version") != 1
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "retrospective-diagnostic-frozen"
    ):
        raise ValueError("unexpected DEFORM adapter protocol identity")
    information = value.get("information_order")
    dataset = value.get("dataset")
    window = value.get("window")
    prediction = value.get("prediction")
    covariance = value.get("covariance")
    aggregation = value.get("aggregation")
    if not all(
        isinstance(item, dict)
        for item in (information, dataset, window, prediction, covariance, aggregation)
    ):
        raise ValueError("protocol sections must be objects")
    assert isinstance(information, dict)
    assert isinstance(dataset, dict)
    assert isinstance(window, dict)
    assert isinstance(prediction, dict)
    assert isinstance(covariance, dict)
    assert isinstance(aggregation, dict)
    expected_dataset = {
        "repository": "roahmlab/DEFORM",
        "commit": "b73b8b8ecc033caefa693fab7898741d4e6dbeff",
        "dlos": ["DLO4", "DLO5"],
        "source_partition": "train",
        "held_partition": "eval",
        "source_trajectories_per_dlo": 56,
        "held_trajectories_per_dlo": 14,
        "frames_per_trajectory": FRAME_COUNT,
        "nodes_per_frame": NODE_COUNT,
    }
    if dataset != expected_dataset:
        raise ValueError("frozen dataset contract changed")
    if (
        information.get("claim_class") != "retrospective-diagnostic"
        or information.get("prospective_claim_eligible") is not False
        or prediction.get("uses_held_internal_future") is not False
        or prediction.get("target_tuning") is not False
        or prediction.get("target_retries") is not False
        or aggregation.get("group_id") != "one complete physical trajectory"
        or aggregation.get("windows_as_independent_units") is not False
        or value.get("benchmark_tasks")
        != ["forecast", "calibration", "dependence", "communication"]
    ):
        raise ValueError("frozen information-order or aggregation contract changed")
    spec = WindowSpec(
        prefix_frames=int(window["prefix_frames"]),
        horizon_frames=int(window["forecast_horizon_frames"]),
        stride_frames=int(window["stride_frames"]),
        decay=float(window["decay"]),
    )
    if (
        spec.prefix_frames != 5
        or spec.horizon_frames != 25
        or spec.stride_frames != 25
        or not math.isclose(spec.decay, 0.85, rel_tol=0.0, abs_tol=0.0)
        or window.get("internal_node_slice") != [2, 10]
        or window.get("known_action_nodes") != ACTION_NODES.tolist()
        or float(covariance.get("jitter_m2", -1.0)) != 1e-10
    ):
        raise ValueError("frozen window or covariance contract changed")
    return value, spec


def _trajectory_paths(root: Path, dlo: str, split: str) -> tuple[Path, ...]:
    expected = 56 if split == "train" else 14
    paths = tuple(sorted((root / dlo / split).glob("*.pkl")))
    if len(paths) != expected:
        raise ValueError(f"{dlo}/{split}: expected {expected} trajectories, got {len(paths)}")
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise ValueError(f"{dlo}/{split}: missing or empty trajectory")
    return paths


def _load_trajectory(path: Path) -> FloatArray:
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (FRAME_COUNT, 3, NODE_COUNT) or not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: expected finite {(FRAME_COUNT, 3, NODE_COUNT)}")
    nodes = np.transpose(array, (0, 2, 1)).copy()
    nodes[:, :, 2] = np.clip(nodes[:, :, 2], 2e-3 + 1e-6, 10000.0)
    return nodes


def _window_starts(spec: WindowSpec) -> tuple[int, ...]:
    starts = tuple(
        range(
            spec.prefix_frames - 1,
            FRAME_COUNT - spec.horizon_frames,
            spec.stride_frames,
        )
    )
    if len(starts) != 19:
        raise ValueError(f"frozen window contract must yield 19 decisions, got {len(starts)}")
    return starts


def _anchor_means(nodes: FloatArray) -> tuple[FloatArray, FloatArray]:
    left = np.mean(nodes[..., :2, :], axis=-2)
    right = np.mean(nodes[..., -2:, :], axis=-2)
    return left, right


def _prediction_and_truth(
    trajectory: FloatArray,
    current: int,
    spec: WindowSpec,
) -> tuple[FloatArray, FloatArray]:
    prefix = trajectory[current + 1 - spec.prefix_frames : current + 1]
    future_action = trajectory[
        current + 1 : current + 1 + spec.horizon_frames,
        ACTION_NODES,
        :,
    ]
    if prefix.shape != (spec.prefix_frames, NODE_COUNT, 3):
        raise ValueError("prefix shape changed")
    if future_action.shape != (spec.horizon_frames, 4, 3):
        raise ValueError("future action shape changed")

    current_nodes = prefix[-1]
    previous_nodes = prefix[-2]
    current_left, current_right = _anchor_means(current_nodes[None, ...])
    previous_left, previous_right = _anchor_means(previous_nodes[None, ...])
    current_left = current_left[0]
    current_right = current_right[0]
    previous_left = previous_left[0]
    previous_right = previous_right[0]
    future_left = np.mean(future_action[:, :2], axis=1)
    future_right = np.mean(future_action[:, 2:], axis=1)
    length_scale = max(float(np.linalg.norm(current_right - current_left)), 1e-6)

    internal_count = NODE_COUNT - 4
    weights = np.linspace(
        1.0 / (internal_count + 1),
        internal_count / (internal_count + 1),
        internal_count,
        dtype=np.float64,
    )[:, None]
    current_line = (1.0 - weights) * current_left + weights * current_right
    previous_line = (1.0 - weights) * previous_left + weights * previous_right
    current_internal = current_nodes[INTERNAL]
    previous_internal = previous_nodes[INTERNAL]
    shape_velocity = (
        (current_internal - current_line) - (previous_internal - previous_line)
    ) / length_scale

    left_displacement = future_left - current_left
    right_displacement = future_right - current_right
    blend = weights[None, :, :]
    anchor_displacement = (
        (1.0 - blend) * left_displacement[:, None, :]
        + blend * right_displacement[:, None, :]
    )
    steps = np.arange(1, spec.horizon_frames + 1, dtype=np.float64)
    velocity_factor = (
        spec.decay * (1.0 - np.power(spec.decay, steps)) / (1.0 - spec.decay)
    )
    prediction = (
        current_internal[None, ...]
        + anchor_displacement
        + velocity_factor[:, None, None] * shape_velocity[None, ...] * length_scale
    )
    truth = trajectory[
        current + 1 : current + 1 + spec.horizon_frames,
        INTERNAL,
        :,
    ].copy()
    return prediction.reshape(-1, 3), truth.reshape(-1, 3)


def _regularized_covariance(values: FloatArray, jitter: float) -> FloatArray:
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("covariance input must have shape (N, 3), N >= 2")
    covariance = np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += jitter * np.eye(3, dtype=np.float64)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("source covariance is not finite")
    np.linalg.cholesky(covariance)
    return covariance


def _fit_source_covariance(
    paths: tuple[Path, ...],
    spec: WindowSpec,
    *,
    jitter: float,
) -> SourceCovariance:
    common: list[FloatArray] = []
    conditional: list[FloatArray] = []
    for path in paths:
        trajectory = _load_trajectory(path)
        for current in _window_starts(spec):
            prediction, truth = _prediction_and_truth(trajectory, current, spec)
            residual = truth - prediction
            translation = np.mean(residual, axis=0)
            common.append(translation)
            conditional.append(residual - translation[None, :])
    common_array = np.asarray(common, dtype=np.float64)
    conditional_array = np.concatenate(conditional, axis=0)
    return SourceCovariance(
        local_m2=_regularized_covariance(conditional_array, jitter),
        shared_m2=_regularized_covariance(common_array, jitter),
        source_window_count=len(common),
        source_row_count=len(conditional_array),
    )


def _payload_keys(case: dict[str, Any]) -> tuple[str, str]:
    path_keys = [
        key
        for key, value in case.items()
        if isinstance(value, str) and value.endswith(".npz")
    ]
    if len(path_keys) != 1:
        raise ValueError(f"template case must expose one NPZ path, got {path_keys}")
    path_key = path_keys[0]
    preferred = [
        key
        for key, value in case.items()
        if isinstance(value, str)
        and _HEX.fullmatch(value)
        and key.startswith(path_key.removesuffix("_path").removesuffix("_payload"))
    ]
    all_hashes = [
        key
        for key, value in case.items()
        if isinstance(value, str) and _HEX.fullmatch(value) and "sha256" in key
    ]
    candidates = preferred or all_hashes
    if len(candidates) != 1:
        raise ValueError(f"template case must expose one payload digest, got {candidates}")
    return path_key, candidates[0]


def _replace_recursive(
    value: Any,
    *,
    challenge_id: str | None = None,
    challenge_sha256: str | None = None,
    challenge_payload_sha256: str | None = None,
    submission_id: str | None = None,
) -> Any:
    if isinstance(value, list):
        return [
            _replace_recursive(
                item,
                challenge_id=challenge_id,
                challenge_sha256=challenge_sha256,
                challenge_payload_sha256=challenge_payload_sha256,
                submission_id=submission_id,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        lower = key.lower()
        if challenge_id is not None and key == "challenge_id":
            result[key] = challenge_id
        elif (
            challenge_sha256 is not None
            and "challenge" in lower
            and "sha256" in lower
            and "payload" not in lower
        ):
            result[key] = challenge_sha256
        elif (
            challenge_payload_sha256 is not None
            and "challenge" in lower
            and "payload" in lower
            and "sha256" in lower
        ):
            result[key] = challenge_payload_sha256
        elif submission_id is not None and key in {
            "submission_id",
            "provider_id",
            "system_id",
        }:
            result[key] = submission_id
        else:
            result[key] = _replace_recursive(
                item,
                challenge_id=challenge_id,
                challenge_sha256=challenge_sha256,
                challenge_payload_sha256=challenge_payload_sha256,
                submission_id=submission_id,
            )
    return result


def _case_from_template(
    template: dict[str, Any],
    *,
    case_id: str,
    group_id: str,
    relative_payload: str,
    payload_sha256: str,
    tasks: list[str] | None,
    metadata: dict[str, Any],
    challenge_payload_sha256: str | None = None,
) -> dict[str, Any]:
    case = copy.deepcopy(template)
    path_key, hash_key = _payload_keys(case)
    case[path_key] = relative_payload
    case[hash_key] = payload_sha256
    if "case_id" in case:
        case["case_id"] = case_id
    if "group_id" in case:
        case["group_id"] = group_id
    if tasks is not None and "tasks" in case:
        case["tasks"] = tasks
    if "metadata" in case:
        case["metadata"] = metadata
    return _replace_recursive(
        case,
        challenge_payload_sha256=challenge_payload_sha256,
    )


def _set_top_level_identity(
    manifest: dict[str, Any],
    *,
    challenge_id: str | None = None,
    submission_id: str | None = None,
) -> dict[str, Any]:
    result = _replace_recursive(
        copy.deepcopy(manifest),
        challenge_id=challenge_id,
        submission_id=submission_id,
    )
    for key in tuple(result):
        if challenge_id is not None and key in {"suite_id", "challenge_id"}:
            result[key] = challenge_id
        if submission_id is not None and key in {
            "submission_id",
            "provider_id",
            "system_id",
        }:
            result[key] = submission_id
    return result


def _write_submission_payload(
    path: Path,
    prediction: FloatArray,
    covariance: SourceCovariance,
    *,
    mode: str,
) -> None:
    sample_count = len(prediction)
    local = np.repeat(covariance.local_m2[None, :, :], sample_count, axis=0)
    shared_root = np.linalg.cholesky(covariance.shared_m2)
    factor = np.repeat(shared_root[None, :, :], sample_count, axis=0)
    if mode == "full-source-fitted-low-rank":
        pass
    elif mode == "same-mean-marginal-matched-diagonal":
        marginal = np.diag(covariance.local_m2) + np.diag(covariance.shared_m2)
        local = np.repeat(np.diag(marginal)[None, :, :], sample_count, axis=0)
        factor = np.empty((sample_count, 3, 0), dtype=np.float64)
    elif mode == "same-mean-overconfident-scale-0.01":
        local = 0.01 * local
        factor = 0.1 * factor
    else:
        raise ValueError(f"unknown covariance control {mode}")
    _deterministic_npz(
        path,
        {
            "conditional_covariance_m2": local,
            "prediction_mean_xyz_m": prediction,
            "shared_factor_m": factor,
        },
    )


def _challenge_and_submissions(
    *,
    dataset_root: Path,
    output_root: Path,
    protocol: dict[str, Any],
    spec: WindowSpec,
    source_covariances: dict[str, SourceCovariance],
) -> tuple[Path, dict[str, Path]]:
    template_root = output_root / "template"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "prob4d.information_contract_sealed",
            "smoke",
            str(template_root),
        ],
        check=True,
    )
    template_challenge_path = template_root / "challenge" / "challenge.json"
    template_submission_path = template_root / "submission" / "submission.json"
    challenge_template = _read_json(template_challenge_path)
    submission_template = _read_json(template_submission_path)
    if not isinstance(challenge_template.get("cases"), list) or not challenge_template["cases"]:
        raise ValueError("sealed smoke challenge has no cases")
    if not isinstance(submission_template.get("cases"), list) or not submission_template["cases"]:
        raise ValueError("sealed smoke submission has no cases")
    challenge_case_template = challenge_template["cases"][0]
    submission_case_template = submission_template["cases"][0]
    if not isinstance(challenge_case_template, dict) or not isinstance(
        submission_case_template, dict
    ):
        raise ValueError("sealed smoke case templates must be objects")

    challenge_id = PROTOCOL_ID + "-challenge"
    challenge = _set_top_level_identity(
        challenge_template,
        challenge_id=challenge_id,
    )
    challenge["cases"] = []
    if "claim_boundary" in challenge:
        challenge["claim_boundary"] = str(protocol["claim_boundary"])
    challenge_directory = output_root / "challenge"
    challenge_payloads = challenge_directory / "cases"
    modes = tuple(str(item) for item in protocol["covariance"]["controls"])
    submissions: dict[str, dict[str, Any]] = {}
    submission_paths: dict[str, Path] = {}
    for mode in modes:
        submission_id = f"{PROTOCOL_ID}-{mode}"
        manifest = _set_top_level_identity(
            submission_template,
            challenge_id=challenge_id,
            submission_id=submission_id,
        )
        manifest["cases"] = []
        if "claim_boundary" in manifest:
            manifest["claim_boundary"] = str(protocol["claim_boundary"])
        submissions[mode] = manifest
        submission_paths[mode] = output_root / f"submission-{mode}" / "submission.json"

    tasks = list(protocol["benchmark_tasks"])
    seen_case_ids: set[str] = set()
    for dlo in DLOS:
        covariance = source_covariances[dlo]
        for trajectory_path in _trajectory_paths(dataset_root, dlo, "eval"):
            trajectory = _load_trajectory(trajectory_path)
            trajectory_id = f"{dlo}/{trajectory_path.stem}"
            for window_index, current in enumerate(_window_starts(spec)):
                case_id = f"{dlo.lower()}-{trajectory_path.stem}-w{window_index:02d}"
                if case_id in seen_case_ids:
                    raise ValueError(f"duplicate generated case ID {case_id}")
                seen_case_ids.add(case_id)
                prediction, truth = _prediction_and_truth(trajectory, current, spec)
                metadata = {
                    "adapter_protocol_id": PROTOCOL_ID,
                    "dataset": "DEFORM",
                    "dlo": dlo,
                    "trajectory": trajectory_path.name,
                    "window_index": window_index,
                    "current_frame": current,
                    "held_start_frame": current + 1,
                    "held_stop_frame_exclusive": current + 1 + spec.horizon_frames,
                    "classification": "retrospective public-data diagnostic",
                }
                challenge_payload = challenge_payloads / f"{case_id}.npz"
                _deterministic_npz(challenge_payload, {"truth_xyz_m": truth})
                challenge_digest = _sha256_file(challenge_payload)
                challenge_case = _case_from_template(
                    challenge_case_template,
                    case_id=case_id,
                    group_id=trajectory_id,
                    relative_payload=f"cases/{case_id}.npz",
                    payload_sha256=challenge_digest,
                    tasks=tasks,
                    metadata=metadata,
                )
                challenge["cases"].append(challenge_case)

                for mode in modes:
                    submission_directory = submission_paths[mode].parent
                    submission_payload = submission_directory / "cases" / f"{case_id}.npz"
                    _write_submission_payload(
                        submission_payload,
                        prediction,
                        covariance,
                        mode=mode,
                    )
                    submission_case = _case_from_template(
                        submission_case_template,
                        case_id=case_id,
                        group_id=trajectory_id,
                        relative_payload=f"cases/{case_id}.npz",
                        payload_sha256=_sha256_file(submission_payload),
                        tasks=None,
                        metadata={**metadata, "covariance_control": mode},
                        challenge_payload_sha256=challenge_digest,
                    )
                    submissions[mode]["cases"].append(submission_case)

    if len(challenge["cases"]) != 532:
        raise ValueError(f"expected 532 held windows, got {len(challenge['cases'])}")
    challenge_path = challenge_directory / "challenge.json"
    _write_json(challenge_path, challenge)
    challenge_sha256 = _sha256_file(challenge_path)
    for mode, manifest in submissions.items():
        updated = _replace_recursive(
            manifest,
            challenge_id=challenge_id,
            challenge_sha256=challenge_sha256,
            submission_id=f"{PROTOCOL_ID}-{mode}",
        )
        _write_json(submission_paths[mode], updated)
    shutil.rmtree(template_root)
    return challenge_path, submission_paths


def _metric(result: dict[str, Any], name: str) -> float:
    aggregate = result.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("benchmark result lacks aggregate")
    equal_group = aggregate.get("equal_group_mean")
    if not isinstance(equal_group, dict) or name not in equal_group:
        raise ValueError(f"benchmark result lacks equal-group metric {name}")
    return float(equal_group[name])


def _evaluate(
    challenge_path: Path,
    submissions: dict[str, Path],
    output_root: Path,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for mode, submission_path in submissions.items():
        result_path = output_root / f"result-{mode}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "prob4d.information_contract_sealed",
                "evaluate",
                str(challenge_path),
                str(submission_path),
                str(result_path),
            ],
            check=True,
        )
        result = _read_json(result_path)
        aggregate = result.get("aggregate")
        if not isinstance(aggregate, dict):
            raise ValueError("benchmark result lacks aggregate")
        contract = aggregate.get("contract")
        if not isinstance(contract, dict) or contract.get("all_cases_pass") is not True:
            raise ValueError(f"benchmark contract failed for {mode}")
        if aggregate.get("case_count") != 532 or aggregate.get("independent_group_count") != 28:
            raise ValueError(f"unexpected statistical-unit counts for {mode}")
        information = result.get("information_order")
        if (
            not isinstance(information, dict)
            or information.get("claim_class") != "retrospective-diagnostic"
            or information.get("prospective_claim_eligible") is not False
        ):
            raise ValueError("retrospective information order was not preserved")
        results[mode] = result

    full = results["full-source-fitted-low-rank"]
    diagonal = results["same-mean-marginal-matched-diagonal"]
    overconfident = results["same-mean-overconfident-scale-0.01"]
    full_rmse = _metric(full, "forecast_rmse_m")
    diagonal_rmse = _metric(diagonal, "forecast_rmse_m")
    overconfident_rmse = _metric(overconfident, "forecast_rmse_m")
    if not (
        math.isclose(full_rmse, diagonal_rmse, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(full_rmse, overconfident_rmse, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("same-mean covariance controls changed point accuracy")
    comparison = {
        "schema_name": "prob4d.information-contract-deform-adapter-result",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "classification": "retrospective public-data adapter validation",
        "case_count": 532,
        "independent_trajectory_count": 28,
        "same_prediction_mean_and_forecast_rmse": True,
        "equal_trajectory_forecast_rmse_m": full_rmse,
        "systems": {
            mode: {
                "gaussian_nll_per_dimension": _metric(
                    result, "gaussian_nll_per_dimension"
                ),
                "normalized_nees": _metric(result, "normalized_nees"),
                "marginal_coverage": _metric(result, "marginal_coverage"),
                "dependence_nll_gain_per_dimension": _metric(
                    result, "dependence_nll_gain_per_dimension"
                ),
                "dense_to_submitted_covariance_ratio": _metric(
                    result, "dense_to_submitted_covariance_ratio"
                ),
            }
            for mode, result in results.items()
        },
        "accuracy_cannot_rank_same_mean_controls": True,
        "probabilistic_metrics_distinguish_same_mean_controls": len(
            {
                round(_metric(result, "gaussian_nll_per_dimension"), 12)
                for result in results.values()
            }
        )
        > 1,
        "claim_boundary": (
            "The adapter demonstrates deterministic public-data ingestion, source-only "
            "covariance construction, truth/submission separation, trajectory-level "
            "aggregation, and a same-mean probabilistic ranking distinction. It is "
            "retrospective and is not a prospective provider leaderboard result."
        ),
    }
    _write_json(output_root / "comparison.json", comparison)
    return comparison


def _source_record(
    protocol_path: Path,
    covariances: dict[str, SourceCovariance],
    output_root: Path,
) -> None:
    record = {
        "schema_name": "prob4d.information-contract-deform-source-covariance",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(protocol_path),
        "dlos": {
            dlo: {
                "source_window_count": value.source_window_count,
                "source_row_count": value.source_row_count,
                "conditional_covariance_m2": value.local_m2.tolist(),
                "shared_translation_covariance_m2": value.shared_m2.tolist(),
            }
            for dlo, value in sorted(covariances.items())
        },
        "target_outcomes_used_for_covariance_fit": False,
    }
    _write_json(output_root / "source-covariance.json", record)


def _write_hash_manifest(output_root: Path) -> None:
    lines: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{_sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(
    protocol_path: Path,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol, spec = _load_protocol(protocol_path)
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True)
    covariances = {
        dlo: _fit_source_covariance(
            _trajectory_paths(dataset_root, dlo, "train"),
            spec,
            jitter=float(protocol["covariance"]["jitter_m2"]),
        )
        for dlo in DLOS
    }
    _source_record(protocol_path, covariances, output_root)
    challenge, submissions = _challenge_and_submissions(
        dataset_root=dataset_root,
        output_root=output_root,
        protocol=protocol,
        spec=spec,
        source_covariances=covariances,
    )
    result = _evaluate(challenge, submissions, output_root)
    _write_hash_manifest(output_root)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build(
        args.protocol.resolve(),
        args.dataset_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
