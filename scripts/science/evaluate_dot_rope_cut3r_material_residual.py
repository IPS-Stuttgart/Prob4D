#!/usr/bin/env python3
"""Source-only DOT test of a persistent material-point CUT3R restart residual."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import traceback
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.dot_rope_cut3r_study import Sim3, content_id, robust_fit_sim3

PROTOCOL_SCHEMA = "prob4d.dot-rope-cut3r-material-residual-protocol"
RESULT_SCHEMA = "prob4d.dot-rope-cut3r-material-residual-evaluation"
TECHNICAL_SCHEMA = "prob4d.dot-rope-cut3r-material-residual-technical-result"
BASE_BLOB = "612c8ae61b0a64d464256a11992b46c486c88012"
POOLED_BLOB = "6195e70997f0e9582251c08772b1e423a3062ad6"
PROTOCOL_ID = "d81b0d02dcff3a83c7c6d6b7af34f51d026a0d11319e2327ba627a72ddaa1487"
PROVIDER_ID = "952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7"
PREDECESSOR_ID = "ec86014b945b5f0d7e0d1fc8e38e5976acfb5efc7ef92fc816714c1cc5b60e09"
SEQUENCES = ("R01", "R02", "R03")
FRAMES = tuple(range(1, 8))
OVERLAP = (3, 4, 5)
SCORE = (6, 7)
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
_HEX = re.compile(r"[0-9a-f]+")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-protocol")
    validate.add_argument("--protocol", type=Path, required=True)
    commands.add_parser("self-test")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--provider-bundle", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--repository-revision", required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(value)}\0".encode() + value,
        usedforsecurity=False,
    ).hexdigest()


def _load_module(filename: str, blob: str, module_name: str) -> Any:
    path = Path(__file__).with_name(filename)
    if _git_blob_sha1(path.read_bytes()) != blob:
        raise RuntimeError(f"registered source changed: {filename}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol(path: Path) -> dict[str, Any]:
    value = _json(path)
    unsigned = dict(value)
    protocol_id = unsigned.pop("protocol_id", None)
    if protocol_id != PROTOCOL_ID or content_id(unsigned) != protocol_id:
        raise ValueError("material-residual protocol identity changed")
    fixed = {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "dataset_doi": "10.13021/ORC2020/XXLVXM",
        "archive": "R01-10.zip",
        "archive_byte_count": 1705947395,
        "archive_md5": "ca546ff5f22c0279123ccb18509858ee",
        "camera": "cam001",
        "source_sequences": list(SEQUENCES),
        "reserved_sequences": "R04-R70",
        "frames": list(FRAMES),
        "windows": {
            "continuous": list(FRAMES),
            "window_a": [1, 2, 3, 4, 5],
            "window_b": [3, 4, 5, 6, 7],
        },
    }
    for name, expected in fixed.items():
        if value.get(name) != expected:
            raise ValueError(f"protocol field changed: {name}")
    if value.get("coordinate_sampling") != {
        "coordinate_columns": [0, 1],
        "coordinate_mode": "pixel-zero-based",
        "preprocessing_transform": (
            "cut3r-long-edge-resize-512-center-crop-multiple-of-16-pixel-centers"
        ),
    }:
        raise ValueError("coordinate sampling changed")
    if value.get("provider") != {
        "artifact_name": "dot-rope-cut3r-sealed-provider-33329701704-1",
        "bundle_id": PROVIDER_ID,
        "prob4d_revision": "7eb4867e36742d819c514fad21436d4f475b4bed",
        "request_id": "83cc26be92364fc7715d692b3bb966cf914fb9f911e0763823f2789480a00cf2",
        "run_id": 33329701704,
    }:
        raise ValueError("provider binding changed")
    if value.get("predecessor", {}).get("evaluation_id") != PREDECESSOR_ID:
        raise ValueError("predecessor result changed")
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation contract is missing")
    expected_eval = {
        "correction_model": (
            "global-sim3-plus-persistent-marker-mean-residual-provider-space-v1"
        ),
        "metric_fit_a_frames": [1, 2],
        "metric_fit_continuous_frames": [1, 2],
        "metric_oracle_b_frames": [6, 7],
        "missing_marker_fallback": "zero-residual",
        "overlap_frames": list(OVERLAP),
        "score_frames": list(SCORE),
        "shrinkage_grid": list(GRID),
        "shrinkage_selection": (
            "leave-one-overlap-frame-out-provider-agreement-rmse-refit-sim3-v1"
        ),
        "sim3_fit": "proper-umeyama-80-percent-iterative-trim",
        "tie_tolerance": 1e-12,
    }
    if evaluation != expected_eval:
        raise ValueError("evaluation contract changed")
    boundary = value.get("information_boundary", {})
    required_false = (
        "bayesian_phystwin_executed",
        "causal4d_executed",
        "correction_fit_uses_ground_truth",
        "normal_view_images_reopened",
        "reserved_payloads_opened",
        "target_outcomes_used",
    )
    if any(boundary.get(name) is not False for name in required_false):
        raise ValueError("information boundary expanded")
    required_true = (
        "provider_predictions_opened",
        "source_2d_markers_opened",
        "source_3d_truth_opened",
    )
    if any(boundary.get(name) is not True for name in required_true):
        raise ValueError("required source access was not authorized")
    return value


def _provider_modules(
    protocol: Mapping[str, Any],
    bundle: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    base = _load_module(
        "run_dot_rope_cut3r_native_provider.py",
        BASE_BLOB,
        "dot_base",
    )
    pooled = _load_module(
        "evaluate_dot_rope_cut3r_pooled.py",
        POOLED_BLOB,
        "dot_pooled",
    )
    source_protocol = Path(protocol["source_protocol"]["path"])
    if _git_blob_sha1(source_protocol.read_bytes()) != protocol["source_protocol"][
        "git_blob_sha1"
    ]:
        raise ValueError("source protocol Git blob changed")
    base_protocol = base._load_protocol(source_protocol)
    manifest = base._verify_provider_bundle(bundle, base_protocol)
    if manifest["provider_bundle_id"] != PROVIDER_ID:
        raise ValueError("provider bundle changed")
    if manifest["request_id"] != protocol["provider"]["request_id"]:
        raise ValueError("provider request changed")
    if manifest["prob4d_revision"] != protocol["provider"]["prob4d_revision"]:
        raise ValueError("provider revision changed")
    base._ORIGINAL_LOAD_RUN = base._load_run
    pooled._ACTIVE_COORDINATE_COLUMNS = (0, 1)
    pooled._ACTIVE_COORDINATE_MODE = "pixel-zero-based"
    pooled._MARKER_DIAGNOSTICS.clear()
    pooled._COLLECTION_DIAGNOSTICS.clear()
    return base, pooled, manifest


def _common(
    pooled: Any,
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    frame: int,
    payload: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_2d, points_3d = payload
    a, _, indices_a = pooled._sample_markers(
        run_a,
        frame,
        points_2d,
        points_3d,
    )
    b, _, indices_b = pooled._sample_markers(
        run_b,
        frame,
        points_2d,
        points_3d,
    )
    marker_ids, positions_a, positions_b = np.intersect1d(
        indices_a,
        indices_b,
        assume_unique=True,
        return_indices=True,
    )
    if marker_ids.size < 3:
        raise ValueError(f"frame {frame} has fewer than three common markers")
    return a[positions_a], b[positions_b], marker_ids.astype(np.int64)


def _collect_truth(
    pooled: Any,
    run: Mapping[str, Any],
    payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    providers: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    marker_ids: list[np.ndarray] = []
    for frame in frames:
        points_2d, points_3d = payloads[int(frame)]
        provider, truth, marker = pooled._sample_markers(
            run,
            int(frame),
            points_2d,
            points_3d,
        )
        if provider.shape[0] < 3:
            raise ValueError(f"frame {frame} has fewer than three markers")
        providers.append(provider)
        truths.append(truth)
        marker_ids.append(marker.astype(np.int64))
    return (
        np.concatenate(providers),
        np.concatenate(truths),
        np.concatenate(marker_ids),
    )


def _fit(
    samples: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    frames: Sequence[int],
) -> tuple[Sim3, dict[int, np.ndarray], dict[str, Any]]:
    target = np.concatenate([samples[int(frame)][0] for frame in frames])
    source = np.concatenate([samples[int(frame)][1] for frame in frames])
    marker_ids = np.concatenate([samples[int(frame)][2] for frame in frames])
    transform, residual_norms = robust_fit_sim3(source, target)
    residual = target - transform.apply(source)
    corrections = {
        int(marker): np.mean(residual[marker_ids == marker], axis=0)
        for marker in np.unique(marker_ids)
    }
    corrected = transform.apply(source) + np.stack(
        [corrections[int(marker)] for marker in marker_ids]
    )
    return transform, corrections, {
        "frames": [int(frame) for frame in frames],
        "sample_count": int(source.shape[0]),
        "global_overlap_rmse_provider_units": float(
            np.sqrt(np.mean(residual_norms**2))
        ),
        "corrected_overlap_rmse_provider_units": float(
            np.sqrt(np.mean(np.sum((corrected - target) ** 2, axis=1)))
        ),
        "marker_counts": {
            str(int(marker)): int(np.count_nonzero(marker_ids == marker))
            for marker in np.unique(marker_ids)
        },
    }


def _apply(
    transform: Sim3,
    corrections: Mapping[int, np.ndarray],
    source: np.ndarray,
    marker_ids: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, int]:
    added = np.zeros_like(source, dtype=np.float64)
    supported = 0
    for row, marker in enumerate(marker_ids):
        if int(marker) in corrections:
            added[row] = shrinkage * corrections[int(marker)]
            supported += 1
    return transform.apply(source) + added, supported


def _select(
    samples: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    frames: Sequence[int],
    grid: Sequence[float],
    tolerance: float,
) -> tuple[float, dict[str, Any]]:
    squared_error = {float(value): 0.0 for value in grid}
    counts = {float(value): 0 for value in grid}
    folds: list[dict[str, Any]] = []
    for held_out in frames:
        training = [frame for frame in frames if frame != held_out]
        transform, corrections, fit = _fit(samples, training)
        target, source, marker_ids = samples[int(held_out)]
        for value in grid:
            prediction, supported = _apply(
                transform,
                corrections,
                source,
                marker_ids,
                float(value),
            )
            error = prediction - target
            fold_sse = float(np.sum(error**2))
            squared_error[float(value)] += fold_sse
            counts[float(value)] += int(source.shape[0])
            folds.append(
                {
                    "held_out_frame": int(held_out),
                    "training_frames": [int(frame) for frame in training],
                    "shrinkage": float(value),
                    "rmse_provider_units": float(
                        np.sqrt(fold_sse / source.shape[0])
                    ),
                    "sample_count": int(source.shape[0]),
                    "supported_sample_count": int(supported),
                    "training_global_overlap_rmse_provider_units": fit[
                        "global_overlap_rmse_provider_units"
                    ],
                }
            )
    scores = {
        value: float(np.sqrt(squared_error[value] / counts[value]))
        for value in squared_error
    }
    minimum = min(scores.values())
    selected = min(
        value
        for value, score in scores.items()
        if score <= minimum + tolerance * max(1.0, abs(minimum))
    )
    return selected, {
        "selected_shrinkage": selected,
        "aggregate_rmse_provider_units": {
            f"{value:.2f}": scores[value] for value in sorted(scores)
        },
        "folds": folds,
    }


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
    )


def _sequence(
    sequence: str,
    pooled: Any,
    runs: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del protocol
    samples = {
        frame: _common(
            pooled,
            runs["window_a"],
            runs["window_b"],
            frame,
            payloads[frame],
        )
        for frame in OVERLAP
    }
    selected, selection = _select(samples, OVERLAP, GRID, 1e-12)
    relative, corrections, fit = _fit(samples, OVERLAP)

    a, a_truth, _ = _collect_truth(
        pooled,
        runs["window_a"],
        payloads,
        (1, 2),
    )
    continuous_fit, continuous_truth, _ = _collect_truth(
        pooled,
        runs["continuous"],
        payloads,
        (1, 2),
    )
    b_fit, b_truth, _ = _collect_truth(
        pooled,
        runs["window_b"],
        payloads,
        SCORE,
    )
    a_to_truth, _ = robust_fit_sim3(a, a_truth)
    continuous_to_truth, _ = robust_fit_sim3(
        continuous_fit,
        continuous_truth,
    )
    b_to_truth, _ = robust_fit_sim3(b_fit, b_truth)

    b, truth, marker_ids = _collect_truth(
        pooled,
        runs["window_b"],
        payloads,
        SCORE,
    )
    continuous, continuous_score_truth, _ = _collect_truth(
        pooled,
        runs["continuous"],
        payloads,
        SCORE,
    )
    corrected, supported = _apply(
        relative,
        corrections,
        b,
        marker_ids,
        selected,
    )
    predictions = {
        "identity_stitch": (a_to_truth.apply(b), truth),
        "global_sim3_stitch": (
            a_to_truth.apply(relative.apply(b)),
            truth,
        ),
        "persistent_material_residual_stitch": (
            a_to_truth.apply(corrected),
            truth,
        ),
        "continuous": (
            continuous_to_truth.apply(continuous),
            continuous_score_truth,
        ),
        "oracle_window": (b_to_truth.apply(b), truth),
    }
    truth_cloud = np.concatenate([payload[1] for payload in payloads.values()])
    centered = truth_cloud - np.mean(truth_cloud, axis=0)
    truth_span = 2.0 * float(np.max(np.linalg.norm(centered, axis=1)))
    metrics = {
        method: {
            "rmse": _rmse(*values),
            "sample_count": int(values[0].shape[0]),
        }
        for method, values in predictions.items()
    }
    for value in metrics.values():
        value["rmse_fraction_of_truth_span"] = value["rmse"] / truth_span
    baseline = min(
        metrics["identity_stitch"]["rmse_fraction_of_truth_span"],
        metrics["global_sim3_stitch"]["rmse_fraction_of_truth_span"],
    )
    material = metrics["persistent_material_residual_stitch"][
        "rmse_fraction_of_truth_span"
    ]
    correction_rows = [
        {
            "sequence": sequence,
            "marker_index": int(marker),
            "residual_x_provider_units": float(vector[0]),
            "residual_y_provider_units": float(vector[1]),
            "residual_z_provider_units": float(vector[2]),
            "residual_norm_provider_units": float(np.linalg.norm(vector)),
            "selected_shrinkage": selected,
        }
        for marker, vector in sorted(corrections.items())
    ]
    return {
        "sequence": sequence,
        "truth_span": truth_span,
        "method_metrics": metrics,
        "selected_shrinkage": selected,
        "selection": selection,
        "fit": fit,
        "score_marker_count": int(b.shape[0]),
        "score_supported_by_overlap_residual_count": supported,
        "score_unique_marker_indices": [
            int(value) for value in np.unique(marker_ids)
        ],
        "correction_marker_count": len(corrections),
        "mean_correction_norm_provider_units": float(
            np.mean([np.linalg.norm(value) for value in corrections.values()])
        ),
        "best_global_baseline_rmse_fraction_of_truth_span": baseline,
        "material_improvement_fraction_of_truth_span": baseline - material,
        "material_wins": bool(material < baseline),
    }, correction_rows


def _aggregate(
    sequences: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    methods = (
        "identity_stitch",
        "global_sim3_stitch",
        "persistent_material_residual_stitch",
        "continuous",
        "oracle_window",
    )
    aggregate = []
    for method in methods:
        values = [
            float(row["method_metrics"][method]["rmse_fraction_of_truth_span"])
            for row in sequences
        ]
        aggregate.append(
            {
                "method": method,
                "sequence_count": len(values),
                "mean_rmse_fraction_of_truth_span": float(np.mean(values)),
                "median_rmse_fraction_of_truth_span": float(np.median(values)),
                "maximum_rmse_fraction_of_truth_span": float(np.max(values)),
            }
        )
    aggregate.sort(key=lambda row: row["mean_rmse_fraction_of_truth_span"])
    improvements = [
        float(row["material_improvement_fraction_of_truth_span"])
        for row in sequences
    ]
    wins = sum(bool(row["material_wins"]) for row in sequences)
    positive = sum(
        float(row["selected_shrinkage"]) > 0.0 for row in sequences
    )
    mean_improvement = float(np.mean(improvements))
    rule = protocol["decision_rule"]
    supported = (
        mean_improvement > 0.0
        and wins >= rule["minimum_sequence_wins"]
        and positive >= rule["minimum_positive_shrinkage_sequences"]
    )
    diagnostic = {
        "mean_improvement_fraction_of_truth_span": mean_improvement,
        "sequence_win_count": wins,
        "positive_shrinkage_sequence_count": positive,
        "sequence_count": len(sequences),
        "rule": rule,
        "supported": supported,
    }
    decision = (
        "source-persistent-material-residual-supported"
        if supported
        else "source-persistent-material-residual-not-supported"
    )
    return aggregate, diagnostic, decision


def _dataset(root: Path, protocol: Mapping[str, Any]) -> Path:
    if not root.is_symlink():
        raise ValueError("dataset root must be the registered symlink")
    resolved = root.resolve(strict=True)
    if resolved != Path("/mnt/seagate10tb/florianpfaff/datasets/dot-rope"):
        raise ValueError("dataset root resolves to an unexpected location")
    archive = resolved / protocol["archive"]
    if (
        not archive.is_file()
        or archive.stat().st_size != protocol["archive_byte_count"]
    ):
        raise ValueError("registered archive identity changed")
    return archive


def _evaluate(args: argparse.Namespace) -> int:
    protocol = _protocol(args.protocol)
    revision = _hex(
        args.repository_revision,
        name="repository_revision",
        length=40,
    )
    output = args.output_dir
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)
    try:
        archive_path = _dataset(args.dataset_root, protocol)
        bundle = args.provider_bundle.resolve(strict=True)
        base, pooled, manifest = _provider_modules(protocol, bundle)
        records = {sequence: {} for sequence in SEQUENCES}
        for record in manifest["outputs"]:
            records[str(record["sequence"])][str(record["run"])] = record
        opened: list[dict[str, Any]] = []
        sequence_results: list[dict[str, Any]] = []
        correction_rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for sequence in SEQUENCES:
                payloads = {}
                for frame in FRAMES:
                    member_2d = base._coordinate_member(
                        sequence,
                        2,
                        frame,
                        protocol["camera"],
                    )
                    member_3d = base._coordinate_member(
                        sequence,
                        3,
                        frame,
                        protocol["camera"],
                    )
                    if member_2d not in names or member_3d not in names:
                        raise ValueError("registered marker member is missing")
                    raw_2d = archive.read(member_2d)
                    raw_3d = archive.read(member_3d)
                    for kind, member, raw in (
                        ("2d", member_2d, raw_2d),
                        ("3d", member_3d, raw_3d),
                    ):
                        opened.append(
                            {
                                "sequence": sequence,
                                "frame": frame,
                                "kind": kind,
                                "member": member,
                                "byte_count": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        )
                    payloads[frame] = (
                        pooled._parse_coordinate_text(raw_2d.decode(), 2),
                        pooled._parse_coordinate_text(raw_3d.decode(), 3),
                    )
                runs = {
                    name: pooled._load_run_with_metadata(
                        base,
                        bundle,
                        records[sequence][name],
                    )
                    for name in ("continuous", "window_a", "window_b")
                }
                result, rows = _sequence(
                    sequence,
                    pooled,
                    runs,
                    payloads,
                    protocol,
                )
                sequence_results.append(result)
                correction_rows.extend(rows)
        aggregate, diagnostic, decision = _aggregate(
            sequence_results,
            protocol,
        )
        support = sorted(
            pooled._MARKER_DIAGNOSTICS.values(),
            key=lambda row: (row["sequence"], row["run"], row["frame"]),
        )
        result = {
            "schema": RESULT_SCHEMA,
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "repository_revision": revision,
            "provider_bundle_id": manifest["provider_bundle_id"],
            "provider_run_id": protocol["provider"]["run_id"],
            "predecessor": protocol["predecessor"],
            "decision": decision,
            "decision_diagnostic": diagnostic,
            "sequences": sequence_results,
            "aggregate_methods": aggregate,
            "correction_rows": correction_rows,
            "marker_support": support,
            "opened_members": opened,
            "information_boundary": {
                **protocol["information_boundary"],
                "opened_sequences": list(SEQUENCES),
                "reserved_sequences": "R04-R70",
                "opened_member_count": len(opened),
                "provider_stage_normal_view_images_were_previously_opened": True,
                "means_compared_on_frozen_score_frames": True,
            },
            "claim_boundary": protocol["claim_boundary"],
        }
        result["evaluation_id"] = content_id(result)
        _write(output / "result.json", result)
        _write(
            output / "opened-members.json",
            {
                "archive": protocol["archive"],
                "members": opened,
                "reserved_sequences": "R04-R70",
            },
        )
        _write_csv(output / "sequence-metrics.csv", sequence_results)
        _write_corrections(output / "correction-models.csv", correction_rows)
        _write_summary(output / "summary.md", result)
        print(
            json.dumps(
                {
                    "decision": decision,
                    "evaluation_id": result["evaluation_id"],
                    "mean_improvement_fraction_of_truth_span": diagnostic[
                        "mean_improvement_fraction_of_truth_span"
                    ],
                    "sequence_win_count": diagnostic["sequence_win_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        failure = {
            "schema": TECHNICAL_SCHEMA,
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "repository_revision": revision,
            "decision": "technical-failure",
            "failure": (
                f"{type(error).__name__}: {' '.join(str(error).split())}"
            )[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-25:],
            "information_boundary": protocol["information_boundary"],
            "claim_boundary": protocol["claim_boundary"],
        }
        failure["technical_result_id"] = content_id(failure)
        _write(output / "technical-failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 3


def _write_csv(
    path: Path,
    sequences: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "sequence",
        "selected_shrinkage",
        "identity_stitch_rmse_fraction_of_truth_span",
        "global_sim3_stitch_rmse_fraction_of_truth_span",
        "persistent_material_residual_stitch_rmse_fraction_of_truth_span",
        "continuous_rmse_fraction_of_truth_span",
        "oracle_window_rmse_fraction_of_truth_span",
        "material_improvement_fraction_of_truth_span",
        "material_wins",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sequences:
            metrics = row["method_metrics"]
            writer.writerow(
                {
                    "sequence": row["sequence"],
                    "selected_shrinkage": row["selected_shrinkage"],
                    **{
                        f"{name}_rmse_fraction_of_truth_span": metrics[name][
                            "rmse_fraction_of_truth_span"
                        ]
                        for name in (
                            "identity_stitch",
                            "global_sim3_stitch",
                            "persistent_material_residual_stitch",
                            "continuous",
                            "oracle_window",
                        )
                    },
                    "material_improvement_fraction_of_truth_span": row[
                        "material_improvement_fraction_of_truth_span"
                    ],
                    "material_wins": row["material_wins"],
                }
            )


def _write_corrections(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# DOT rope CUT3R persistent material-residual source result",
        "",
        f"Evaluation ID: `{result['evaluation_id']}`",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "Shrinkage was selected only from restart-window provider agreement on overlap frames 3--5; 3-D truth did not enter the correction fit or selection.",
        "",
        "| Sequence | shrinkage | identity | global Sim(3) | global + material | continuous | oracle | improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["sequences"]:
        metrics = row["method_metrics"]
        lines.append(
            "| {sequence} | {selected_shrinkage:.2f} | {identity:.6f} | "
            "{global_value:.6f} | {material:.6f} | {continuous:.6f} | "
            "{oracle:.6f} | {improvement:+.6f} |".format(
                sequence=row["sequence"],
                selected_shrinkage=row["selected_shrinkage"],
                identity=metrics["identity_stitch"][
                    "rmse_fraction_of_truth_span"
                ],
                global_value=metrics["global_sim3_stitch"][
                    "rmse_fraction_of_truth_span"
                ],
                material=metrics["persistent_material_residual_stitch"][
                    "rmse_fraction_of_truth_span"
                ],
                continuous=metrics["continuous"][
                    "rmse_fraction_of_truth_span"
                ],
                oracle=metrics["oracle_window"][
                    "rmse_fraction_of_truth_span"
                ],
                improvement=row[
                    "material_improvement_fraction_of_truth_span"
                ],
            )
        )
    diagnostic = result["decision_diagnostic"]
    lines += ["", "Aggregate mean RMSE / truth span:", ""]
    lines += [
        f"- `{row['method']}`: {row['mean_rmse_fraction_of_truth_span']:.6f}"
        for row in result["aggregate_methods"]
    ]
    lines += [
        "",
        (
            f"Material wins: {diagnostic['sequence_win_count']}/3; positive "
            f"shrinkage: {diagnostic['positive_shrinkage_sequence_count']}/3; "
            "mean improvement: "
            f"{diagnostic['mean_improvement_fraction_of_truth_span']:+.6f} "
            "of truth span."
        ),
        "",
        (
            "Source-development evidence only. R04--R70 remained unopened. No "
            "BayesianPhysTwin or Causal4D outcome was executed."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> int:
    rng = np.random.default_rng(20260831)
    base = rng.normal(size=(8, 3))
    local = rng.normal(scale=0.08, size=(8, 3))
    local -= np.mean(local, axis=0)
    marker_ids = np.arange(8, dtype=np.int64)
    samples = {}
    for frame in OVERLAP:
        source = (
            base
            + rng.normal(scale=0.005, size=base.shape)
            + 0.01 * (frame - 4)
        )
        samples[frame] = (source + local, source, marker_ids)
    selected, selection = _select(samples, OVERLAP, GRID, 1e-12)
    if selected < 0.75:
        raise AssertionError(f"unexpected synthetic shrinkage {selected}")
    scores = selection["aggregate_rmse_provider_units"]
    if not scores["1.00"] < scores["0.00"]:
        raise AssertionError(
            "persistent correction did not improve synthetic agreement"
        )
    print(
        json.dumps(
            {
                "decision": "self-test-passed",
                "selected_shrinkage": selected,
            }
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-protocol":
        print(json.dumps({"protocol_id": _protocol(args.protocol)["protocol_id"]}))
        return 0
    if args.command == "self-test":
        return _self_test()
    if args.command == "evaluate":
        return _evaluate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
