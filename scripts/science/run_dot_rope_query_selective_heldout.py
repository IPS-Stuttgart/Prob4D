#!/usr/bin/env python3
"""Run the frozen DOT R11--R30 query-selective CUT3R experiment.

The control plane requires a strong-positive R04--R10 prerequisite. CUT3R point
maps are sealed before any target marker access. A separate factor-seal command
then reads only 2-D marker locations on the overlap frames, constructs the
rank-deficient factors and all query predictions, and content-addresses those
predictions. The final command opens 3-D marker outcomes only after that seal.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import traceback
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from prob4d.dot_rope_cut3r_study import (
    content_id,
    normalized_gaussian_score,
    robust_fit_sim3,
)
from prob4d.observable_gauge import (
    GaugeGaussianPosterior,
    estimate_observable_sim3_factor,
)
from prob4d.query_observability import (
    QueryObservabilityGate,
    evaluate_query_observability,
    point_position_query_jacobian,
)
from prob4d.sim3 import Sim3

PROTOCOL_SCHEMA = "prob4d.dot-rope-query-selective-heldout-protocol"
REQUEST_SCHEMA = "prob4d.dot-rope-query-selective-heldout-request"
PROVIDER_SCHEMA = "prob4d.dot-rope-query-selective-provider-bundle"
SEAL_SCHEMA = "prob4d.dot-rope-query-selective-prediction-seal"
RESULT_SCHEMA = "prob4d.dot-rope-query-selective-heldout-result"
FAILURE_SCHEMA = "prob4d.dot-rope-query-selective-heldout-failure"
SCHEMA_VERSION = 1

TARGET_SEQUENCES = [f"R{index:02d}" for index in range(11, 31)]
RESERVED_SEQUENCES = "R31-R70"
ARCHIVES = {
    "R11-20.zip": {
        "md5": "23ce3e7067465d3edabe20b4c7cfa388",
        "sequences": [f"R{index:02d}" for index in range(11, 21)],
    },
    "R21-30.zip": {
        "md5": "8aee77f79d1aff6e1f3fd21886b251a0",
        "sequences": [f"R{index:02d}" for index in range(21, 31)],
    },
}
CAMERA = "cam001"
FRAMES = list(range(1, 8))
BASE_PROVIDER_BLOB = "612c8ae61b0a64d464256a11992b46c486c88012"
POOLED_EVALUATOR_BLOB = "6195e70997f0e9582251c08772b1e423a3062ad6"
PREREQUISITE_PROTOCOL_ID = (
    "a83258295d5ecabd95017a775f334173bb48141918832fb1a065a1dff66d16ba"
)
PREREQUISITE_DECISION = "heldout-strong-positive"
SOURCE_CALIBRATION_ID = (
    "943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7"
)
SELECTED_DEPENDENCE_ALPHA = 0.85
METHODS = (
    "physical_fallback",
    "full_rank_only",
    "observable_subspace_unconditional",
    "query_aware",
    "invalid_full_rank_completion",
)
QUERIES = ("centerline_centroid", "off_axis_probe")
CHI2_3_90 = 6.251388631170325


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--protocol-git-blob-sha", required=True)

    predict = commands.add_parser("predict")
    _execution_arguments(predict)
    predict.add_argument("--dataset-root", type=Path, required=True)
    predict.add_argument("--cut3r-checkout", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--runtime-receipt", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)

    seal = commands.add_parser("seal")
    _execution_arguments(seal)
    seal.add_argument("--dataset-root", type=Path, required=True)
    seal.add_argument("--provider-bundle", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)

    evaluate = commands.add_parser("evaluate")
    _execution_arguments(evaluate)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--provider-bundle", type=Path, required=True)
    evaluate.add_argument("--prediction-seal", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--prob4d-revision", required=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported query-selective protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _hex(protocol_id, name="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("query-selective protocol identity mismatch")
    expected_archives = [
        {
            "name": name,
            "md5": value["md5"],
            "sequences": value["sequences"],
        }
        for name, value in ARCHIVES.items()
    ]
    if protocol.get("archives") != expected_archives:
        raise ValueError("target archive roster changed")
    if protocol.get("target_sequences") != TARGET_SEQUENCES:
        raise ValueError("target sequence roster changed")
    if protocol.get("reserved_sequences") != RESERVED_SEQUENCES:
        raise ValueError("reserved sequence boundary changed")
    if protocol.get("camera") != CAMERA or protocol.get("frames") != FRAMES:
        raise ValueError("camera or frame roster changed")
    if protocol.get("prerequisite", {}).get("protocol_id") != PREREQUISITE_PROTOCOL_ID:
        raise ValueError("prerequisite protocol changed")
    if (
        protocol.get("prerequisite", {}).get("required_decision")
        != PREREQUISITE_DECISION
    ):
        raise ValueError("prerequisite decision changed")
    if (
        protocol.get("prerequisite", {}).get("source_calibration_id")
        != SOURCE_CALIBRATION_ID
    ):
        raise ValueError("source calibration identity changed")
    if (
        float(protocol.get("prerequisite", {}).get("selected_dependence_alpha"))
        != SELECTED_DEPENDENCE_ALPHA
    ):
        raise ValueError("source-selected dependence alpha changed")
    factor = protocol.get("factor") or {}
    if float(factor.get("rank_threshold", math.nan)) != 0.01:
        raise ValueError("rank threshold changed")
    if factor.get("methods") != list(METHODS):
        raise ValueError("method roster changed")
    if set(factor.get("queries") or {}) != set(QUERIES):
        raise ValueError("query roster changed")
    boundary = protocol.get("information_boundary") or {}
    required_false = (
        "r04_r10_confirmation_reused_for_tuning",
        "target_side_retuning_allowed",
        "r31_r70_payloads_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
        "dataset_modified",
    )
    if any(boundary.get(name) is not False for name in required_false):
        raise ValueError("information boundary changed")
    if boundary.get("provider_predictions_sealed_before_any_r11_r30_marker_access") is not True:
        raise ValueError("provider/marker custody changed")
    if boundary.get("factor_and_query_decisions_sealed_before_3d_marker_access") is not True:
        raise ValueError("prediction/outcome custody changed")
    return protocol


def validate_request(
    request_path: Path,
    protocol_path: Path,
    protocol_git_blob_sha: str,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    request = _read_json(request_path)
    expected = {
        "bayesian_phystwin_executed",
        "causal4d_executed",
        "claim_boundary",
        "marker_2d_factor_seal_authorized",
        "marker_3d_scoring_authorized",
        "normal_view_prediction_authorized",
        "post_open_tuning_authorized",
        "prerequisite",
        "protocol_git_blob_sha",
        "protocol_path",
        "request_id",
        "reserved_sequences",
        "schema",
        "schema_version",
        "target_sequences",
    }
    if set(request) != expected:
        raise ValueError("query-selective request fields changed")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported query-selective request schema")
    _hex(protocol_git_blob_sha, name="protocol Git blob", length=40)
    if request["protocol_path"] != protocol_path.as_posix():
        raise ValueError("request protocol path changed")
    if request["protocol_git_blob_sha"] != protocol_git_blob_sha:
        raise ValueError("request does not bind the reviewed protocol blob")
    if request["target_sequences"] != TARGET_SEQUENCES:
        raise ValueError("request target roster changed")
    if request["reserved_sequences"] != RESERVED_SEQUENCES:
        raise ValueError("request reserve changed")
    for name in (
        "normal_view_prediction_authorized",
        "marker_2d_factor_seal_authorized",
        "marker_3d_scoring_authorized",
    ):
        if request[name] is not True:
            raise ValueError(f"{name} must be explicitly authorized")
    for name in (
        "post_open_tuning_authorized",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    ):
        if request[name] is not False:
            raise ValueError(f"{name} exceeds the frozen boundary")
    prerequisite = request["prerequisite"]
    expected_prerequisite = {
        "artifact_digest",
        "artifact_id",
        "artifact_name",
        "decision",
        "evaluation_id",
        "marker_support_id",
        "protocol_id",
        "run_id",
        "source_calibration_id",
    }
    if not isinstance(prerequisite, dict) or set(prerequisite) != expected_prerequisite:
        raise ValueError("prerequisite binding fields changed")
    if prerequisite["protocol_id"] != PREREQUISITE_PROTOCOL_ID:
        raise ValueError("request prerequisite protocol changed")
    if prerequisite["decision"] != PREREQUISITE_DECISION:
        raise ValueError("request prerequisite did not pass the strong-positive gate")
    if prerequisite["source_calibration_id"] != SOURCE_CALIBRATION_ID:
        raise ValueError("request prerequisite source calibration changed")
    for name in ("run_id", "artifact_id"):
        if not isinstance(prerequisite[name], int) or prerequisite[name] <= 0:
            raise ValueError(f"prerequisite {name} must be a positive integer")
    if not isinstance(prerequisite["artifact_name"], str) or not prerequisite["artifact_name"]:
        raise ValueError("prerequisite artifact_name must be nonempty")
    digest = prerequisite["artifact_digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("prerequisite artifact_digest must be a GitHub SHA-256 digest")
    _hex(digest.removeprefix("sha256:"), name="artifact digest", length=64)
    _hex(prerequisite["evaluation_id"], name="evaluation_id", length=64)
    _hex(prerequisite["marker_support_id"], name="marker_support_id", length=64)
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("query-selective request identity mismatch")
    return {
        "request_id": request_id,
        "protocol_id": protocol["protocol_id"],
        "prerequisite": prerequisite,
    }


def _require_execution_identity(request_id: str, revision: str) -> None:
    _hex(request_id, name="request_id", length=64)
    _hex(revision, name="Prob4D revision", length=40)


def _load_script(filename: str, name: str, expected_blob: str) -> Any:
    path = Path(__file__).with_name(filename)
    source = path.read_bytes()
    if _git_blob_sha1(source) != expected_blob:
        raise RuntimeError(f"{filename} source bytes changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider_protocol(
    protocol: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "prob4d.dot-rope-cut3r-native-provider-protocol",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "dataset_doi": protocol["dataset_doi"],
        "archive": archive["name"],
        "camera": protocol["camera"],
        "frames": protocol["frames"],
        "source_sequences": archive["sequences"],
        "reserved_sequences": protocol["reserved_sequences"],
        "windows": protocol["windows"],
        "provider": protocol["provider"],
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }


def predict(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    runs_root = output / "runs"
    runs_root.mkdir()
    base = _load_script(
        "run_dot_rope_cut3r_native_provider.py",
        "dot_query_selective_provider",
        BASE_PROVIDER_BLOB,
    )
    input_records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    runtime_ids: set[str] = set()
    checkpoint_hashes: set[str] = set()
    part_ids: list[str] = []
    for archive in protocol["archives"]:
        adapted = _provider_protocol(protocol, archive)
        base._load_protocol = lambda _path, value=adapted: value
        part = output / f"part-{archive['name'].removesuffix('.zip').lower()}"
        status = int(
            base.predict(
                argparse.Namespace(
                    protocol=args.protocol,
                    request_id=args.request_id,
                    prob4d_revision=args.prob4d_revision,
                    dataset_root=args.dataset_root,
                    cut3r_checkout=args.cut3r_checkout,
                    checkpoint=args.checkpoint,
                    runtime_receipt=args.runtime_receipt,
                    output_dir=part,
                )
            )
        )
        if status != 0:
            raise RuntimeError(f"provider part {archive['name']} returned {status}")
        manifest = _read_json(part / "manifest.json")
        part_ids.append(str(manifest["provider_bundle_id"]))
        runtime_ids.add(str(manifest["runtime_artifact_id"]))
        checkpoint_hashes.add(str(manifest["checkpoint_sha256"]))
        for record in manifest["inputs"]:
            input_records.append({"archive": archive["name"], **record})
        for record in manifest["outputs"]:
            relative = PurePosixPath(str(record["relative_path"]))
            source = part.joinpath(*relative.parts)
            target_relative = Path("runs") / source.name
            target = output / target_relative
            if target.exists():
                raise ValueError("duplicate provider output path")
            shutil.copy2(source, target)
            output_records.append(
                {
                    **record,
                    "archive": archive["name"],
                    "relative_path": target_relative.as_posix(),
                    "sha256": _sha256(target),
                    "byte_count": int(target.stat().st_size),
                }
            )
        shutil.rmtree(part)
    if len(runtime_ids) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("provider archive parts used different frozen runtimes")
    if {record["sequence"] for record in output_records} != set(TARGET_SEQUENCES):
        raise ValueError("combined provider bundle does not cover every target sequence")
    manifest = {
        "schema": PROVIDER_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": args.request_id,
        "protocol_id": protocol["protocol_id"],
        "prob4d_revision": args.prob4d_revision,
        "runtime_artifact_id": next(iter(runtime_ids)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "part_provider_bundle_ids": part_ids,
        "dataset": {
            "doi": protocol["dataset_doi"],
            "archives": protocol["archives"],
            "target_sequences": TARGET_SEQUENCES,
            "reserved_sequences": RESERVED_SEQUENCES,
        },
        "inputs": input_records,
        "outputs": output_records,
        "information_boundary": {
            "normal_view_images_opened": True,
            "two_dimensional_markers_opened": False,
            "three_dimensional_markers_opened": False,
            "provider_residuals_opened": False,
            "r31_r70_payloads_opened": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
        "decision": "sealed-r11-r30-provider-predictions",
    }
    manifest["provider_bundle_id"] = content_id(manifest)
    _write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "decision": manifest["decision"],
                "provider_bundle_id": manifest["provider_bundle_id"],
                "prediction_runs": len(output_records),
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_provider_bundle(
    bundle: Path,
    protocol: Mapping[str, Any],
    request_id: str,
    revision: str,
) -> dict[str, Any]:
    manifest = _read_json(bundle / "manifest.json")
    if manifest.get("schema") != PROVIDER_SCHEMA or manifest.get("schema_version") != 1:
        raise ValueError("provider bundle schema changed")
    unsigned = dict(manifest)
    bundle_id = unsigned.pop("provider_bundle_id", None)
    _hex(bundle_id, name="provider_bundle_id", length=64)
    if content_id(unsigned) != bundle_id:
        raise ValueError("provider bundle identity mismatch")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("provider bundle protocol changed")
    if manifest.get("request_id") != request_id:
        raise ValueError("provider bundle request changed")
    if manifest.get("prob4d_revision") != revision:
        raise ValueError("provider bundle revision changed")
    if manifest.get("decision") != "sealed-r11-r30-provider-predictions":
        raise ValueError("provider bundle is not sealed")
    boundary = manifest.get("information_boundary") or {}
    if boundary.get("two_dimensional_markers_opened") is not False:
        raise ValueError("provider stage opened 2-D markers")
    if boundary.get("three_dimensional_markers_opened") is not False:
        raise ValueError("provider stage opened 3-D markers")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("provider outputs are unavailable")
    for record in outputs:
        relative = PurePosixPath(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("provider output path is unsafe")
        path = bundle.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(bundle.resolve(strict=True))
        if _sha256(path) != record["sha256"]:
            raise ValueError("provider output hash changed")
        if path.stat().st_size != record["byte_count"]:
            raise ValueError("provider output size changed")
    return manifest


def _archive_for_sequence(protocol: Mapping[str, Any], sequence: str) -> str:
    matches = [
        str(archive["name"])
        for archive in protocol["archives"]
        if sequence in archive["sequences"]
    ]
    if len(matches) != 1:
        raise ValueError(f"sequence {sequence} does not map to one archive")
    return matches[0]


def _coordinate_member(sequence: str, dimension: int, frame: int) -> str:
    return f"{sequence}/coordinates/{dimension}d/frame{frame:06d}_{CAMERA}.txt"


def _load_run(bundle: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    relative = PurePosixPath(str(record["relative_path"]))
    path = bundle.joinpath(*relative.parts)
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    required = {"points", "confidence", "poses", "intrinsics", "frames", "original_sizes"}
    if set(result) != required:
        raise ValueError("provider run fields changed")
    return result


def _sample_provider_2d(
    pooled: Any,
    run: Mapping[str, np.ndarray],
    frame: int,
    coordinates_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frames = np.asarray(run["frames"], dtype=np.int64)
    matches = np.flatnonzero(frames == frame)
    if matches.size != 1:
        raise ValueError("provider run does not contain the requested frame exactly once")
    index = int(matches[0])
    points = np.asarray(run["points"][index], dtype=np.float64)
    confidence = np.asarray(run["confidence"][index], dtype=np.float64)
    original_width, original_height = (int(value) for value in run["original_sizes"][index])
    raw = np.asarray(coordinates_2d, dtype=np.float64)
    pixels = pooled._to_original_pixel_coordinates(
        raw,
        mode="pixel-zero-based",
        width=original_width,
        height=original_height,
    )
    mapped, original_valid, transform = pooled.cut3r_output_coordinates(
        pixels,
        original_width=original_width,
        original_height=original_height,
        output_width=int(points.shape[1]),
        output_height=int(points.shape[0]),
    )
    sampled_points, valid_points = pooled.bilinear_sample(points, mapped)
    sampled_confidence, valid_confidence = pooled.bilinear_sample(
        confidence[..., None],
        mapped,
    )
    valid = (
        original_valid
        & valid_points
        & valid_confidence
        & np.isfinite(sampled_points).all(axis=1)
        & np.isfinite(sampled_confidence[:, 0])
        & (sampled_confidence[:, 0] > 0.0)
    )
    indices = np.flatnonzero(valid)
    diagnostic = {
        "frame": int(frame),
        "coordinate_rows": int(raw.shape[0]),
        "valid_provider_markers": int(indices.size),
        "original_image_size": [original_width, original_height],
        "provider_point_map_size": [int(points.shape[1]), int(points.shape[0])],
        "transform": transform,
    }
    return sampled_points[valid], indices, diagnostic


def _collect_overlap_factor_points(
    pooled: Any,
    run_a: Mapping[str, np.ndarray],
    run_b: Mapping[str, np.ndarray],
    coordinates_by_frame: Mapping[int, np.ndarray],
    frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for frame in frames:
        coordinates = coordinates_by_frame[int(frame)]
        a_points, a_indices, a_diagnostic = _sample_provider_2d(
            pooled,
            run_a,
            int(frame),
            coordinates,
        )
        b_points, b_indices, b_diagnostic = _sample_provider_2d(
            pooled,
            run_b,
            int(frame),
            coordinates,
        )
        common, a_positions, b_positions = np.intersect1d(
            a_indices,
            b_indices,
            assume_unique=True,
            return_indices=True,
        )
        diagnostics.append(
            {
                "frame": int(frame),
                "common_markers": int(common.size),
                "window_a": a_diagnostic,
                "window_b": b_diagnostic,
            }
        )
        if common.size:
            sources.append(b_points[b_positions])
            targets.append(a_points[a_positions])
    if not sources:
        raise ValueError("overlap has no common provider markers")
    return np.concatenate(sources), np.concatenate(targets), diagnostics


def _deterministic_normal(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    tangent = np.asarray(right[0], dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(tangent) > 1e-12)
    if nonzero.size and tangent[int(nonzero[0])] < 0.0:
        tangent *= -1.0
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ tangent)))]
    normal = np.cross(tangent, reference)
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("cannot construct an off-axis query normal")
    return normal / norm


def _fuse_information(
    factor: Any,
    prior_mean_local: np.ndarray,
    prior_covariance_local: np.ndarray,
    information: np.ndarray,
) -> GaugeGaussianPosterior:
    prior_information = np.linalg.solve(prior_covariance_local, np.eye(7))
    posterior_information = prior_information + information
    posterior_covariance = np.linalg.solve(posterior_information, np.eye(7))
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    posterior_mean = posterior_covariance @ (prior_information @ prior_mean_local)
    return GaugeGaussianPosterior(
        chart=factor.chart,
        mean_local=posterior_mean,
        covariance_local=posterior_covariance,
    )


def _prediction(
    factor: Any,
    query_source: np.ndarray,
    posterior: GaugeGaussianPosterior,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = point_position_query_jacobian(factor, query_source)
    mean = np.asarray(
        posterior.mean_transform.transform_points(query_source),
        dtype=np.float64,
    )
    covariance = jacobian @ posterior.covariance_local @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise ValueError("projected query covariance is not positive definite")
    return mean, covariance


def _report_payload(report: Any) -> dict[str, Any]:
    return {
        "factor_rank": int(report.factor_rank),
        "query_dimension": int(report.query_dimension),
        "direct_observability_fraction": float(report.direct_observability_fraction),
        "nullspace_sensitivity_fraction": float(report.nullspace_sensitivity_fraction),
        "metric_variance_reduction_fraction": float(
            report.metric_variance_reduction_fraction
        ),
        "worst_supported_variance_ratio": float(
            report.worst_supported_variance_ratio
        ),
        "gauge_invariant_query": bool(report.gauge_invariant_query),
    }


def _prediction_record(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    accepted: bool,
    exact_fallback: bool,
) -> dict[str, Any]:
    return {
        "mean": [float(value) for value in mean],
        "covariance": [[float(value) for value in row] for row in covariance],
        "accepted": bool(accepted),
        "exact_fallback": bool(exact_fallback),
    }


def seal(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    bundle = args.provider_bundle.expanduser().resolve(strict=True)
    manifest = _verify_provider_bundle(
        bundle,
        protocol,
        args.request_id,
        args.prob4d_revision,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    pooled = _load_script(
        "evaluate_dot_rope_cut3r_pooled.py",
        "dot_query_selective_marker_mapping",
        POOLED_EVALUATOR_BLOB,
    )
    pooled._ACTIVE_COORDINATE_COLUMNS = (0, 1)
    pooled._ACTIVE_COORDINATE_MODE = "pixel-zero-based"
    records_by_sequence: dict[str, dict[str, Mapping[str, Any]]] = {
        sequence: {} for sequence in TARGET_SEQUENCES
    }
    for record in manifest["outputs"]:
        records_by_sequence[str(record["sequence"])][str(record["run"])] = record
    prior_std = np.asarray(
        protocol["factor"]["prior_standard_deviations_local"],
        dtype=np.float64,
    )
    prior_covariance = np.diag(prior_std**2)
    gate = QueryObservabilityGate(**protocol["factor"]["query_gate"])
    sequence_records: list[dict[str, Any]] = []
    opened_2d_members: list[dict[str, Any]] = []
    root = args.dataset_root.expanduser().resolve(strict=True)
    for archive in protocol["archives"]:
        archive_path = (root / archive["name"]).resolve(strict=True)
        archive_path.relative_to(root)
        with zipfile.ZipFile(archive_path, "r") as source_archive:
            names = set(source_archive.namelist())
            for sequence in archive["sequences"]:
                item: dict[str, Any] = {
                    "sequence": sequence,
                    "archive": archive["name"],
                    "support": "supported",
                }
                try:
                    coordinates_by_frame: dict[int, np.ndarray] = {}
                    for frame in protocol["factor"]["overlap_frames"]:
                        member = _coordinate_member(sequence, 2, int(frame))
                        if member not in names:
                            raise ValueError("registered 2-D marker member is missing")
                        raw = source_archive.read(member)
                        opened_2d_members.append(
                            {
                                "sequence": sequence,
                                "archive": archive["name"],
                                "frame": int(frame),
                                "member": member,
                                "byte_count": len(raw),
                                "sha256": _sha256_bytes(raw),
                            }
                        )
                        coordinates_by_frame[int(frame)] = pooled._parse_coordinate_text(
                            raw.decode("utf-8"),
                            2,
                        )
                    runs = {
                        name: _load_run(bundle, records_by_sequence[sequence][name])
                        for name in ("window_a", "window_b")
                    }
                    source, target, diagnostics = _collect_overlap_factor_points(
                        pooled,
                        runs["window_a"],
                        runs["window_b"],
                        coordinates_by_frame,
                        protocol["factor"]["overlap_frames"],
                    )
                    nonempty = sum(row["common_markers"] > 0 for row in diagnostics)
                    if source.shape[0] < int(
                        protocol["marker_sampling"]["minimum_overlap_common_markers"]
                    ):
                        raise ValueError("overlap support is below the frozen minimum")
                    if nonempty < int(
                        protocol["marker_sampling"]["minimum_overlap_nonempty_frames"]
                    ):
                        raise ValueError("overlap spans too few nonempty frames")
                    factor = estimate_observable_sim3_factor(
                        source,
                        target,
                        rank_threshold=float(protocol["factor"]["rank_threshold"]),
                    )
                    prior_mean_local = factor.chart.to_local(Sim3.identity())
                    fallback = GaugeGaussianPosterior(
                        chart=factor.chart,
                        mean_local=prior_mean_local,
                        covariance_local=prior_covariance,
                    )
                    candidate = factor.fuse_transform_gaussian(
                        Sim3.identity(),
                        prior_covariance,
                    )
                    observable_eigenvalues = np.linalg.eigvalsh(
                        factor.observable_information
                    )
                    nullspace_precision = float(np.mean(observable_eigenvalues)) * float(
                        protocol["factor"]["invalid_nullspace_precision_ratio"]
                    )
                    invalid_information = factor.information_matrix + (
                        nullspace_precision
                        * factor.nullspace_basis
                        @ factor.nullspace_basis.T
                    )
                    invalid = _fuse_information(
                        factor,
                        prior_mean_local,
                        prior_covariance,
                        invalid_information,
                    )
                    centroid = np.asarray(factor.chart.source_centroid, dtype=np.float64)
                    normal = _deterministic_normal(source)
                    query_points = {
                        "centerline_centroid": centroid,
                        "off_axis_probe": centroid + factor.chart.cloud_scale * normal,
                    }
                    provider_span = 2.0 * float(
                        np.max(np.linalg.norm(source - centroid, axis=1))
                    )
                    if provider_span <= np.finfo(np.float64).eps:
                        raise ValueError("provider overlap span is degenerate")
                    queries: dict[str, Any] = {}
                    for query_name, query_source in query_points.items():
                        jacobian = point_position_query_jacobian(factor, query_source)
                        report = evaluate_query_observability(
                            factor,
                            prior_covariance_local=prior_covariance,
                            query_jacobian_local=jacobian,
                        )
                        decision = gate.evaluate(report)
                        fallback_mean, fallback_covariance = _prediction(
                            factor,
                            query_source,
                            fallback,
                        )
                        candidate_mean, candidate_covariance = _prediction(
                            factor,
                            query_source,
                            candidate,
                        )
                        invalid_mean, invalid_covariance = _prediction(
                            factor,
                            query_source,
                            invalid,
                        )
                        query_aware_mean = (
                            candidate_mean if decision.admitted else fallback_mean
                        )
                        query_aware_covariance = (
                            candidate_covariance
                            if decision.admitted
                            else fallback_covariance
                        )
                        full_rank_mean = (
                            candidate_mean if factor.rank == 7 else fallback_mean
                        )
                        full_rank_covariance = (
                            candidate_covariance
                            if factor.rank == 7
                            else fallback_covariance
                        )
                        predictions = {
                            "physical_fallback": _prediction_record(
                                fallback_mean,
                                fallback_covariance,
                                accepted=False,
                                exact_fallback=True,
                            ),
                            "full_rank_only": _prediction_record(
                                full_rank_mean,
                                full_rank_covariance,
                                accepted=factor.rank == 7,
                                exact_fallback=factor.rank != 7,
                            ),
                            "observable_subspace_unconditional": _prediction_record(
                                candidate_mean,
                                candidate_covariance,
                                accepted=True,
                                exact_fallback=False,
                            ),
                            "query_aware": _prediction_record(
                                query_aware_mean,
                                query_aware_covariance,
                                accepted=decision.admitted,
                                exact_fallback=not decision.admitted,
                            ),
                            "invalid_full_rank_completion": _prediction_record(
                                invalid_mean,
                                invalid_covariance,
                                accepted=True,
                                exact_fallback=False,
                            ),
                        }
                        if not decision.admitted:
                            query_aware = predictions["query_aware"]
                            fallback_record = predictions["physical_fallback"]
                            if query_aware["mean"] != fallback_record["mean"]:
                                raise ValueError("rejected query changed the fallback mean")
                            if query_aware["covariance"] != fallback_record["covariance"]:
                                raise ValueError(
                                    "rejected query changed the fallback covariance"
                                )
                        queries[query_name] = {
                            "query_source": [float(value) for value in query_source],
                            "observability": _report_payload(report),
                            "admitted": decision.admitted,
                            "reason_codes": list(decision.reason_codes),
                            "predictions": predictions,
                        }
                    item.update(
                        {
                            "factor": {
                                "rank": factor.rank,
                                "normalized_geometry_spectrum": [
                                    float(value)
                                    for value in factor.normalized_geometry_spectrum
                                ],
                                "residual_rms": float(factor.residual_rms),
                                "inlier_fraction": float(factor.inlier_fraction),
                                "num_correspondences": factor.num_correspondences,
                                "covariance_method": factor.covariance_method,
                            },
                            "provider_span": provider_span,
                            "overlap_diagnostics": diagnostics,
                            "queries": queries,
                        }
                    )
                except Exception as error:
                    item["support"] = "support-negative"
                    item["failure"] = (
                        f"{type(error).__name__}: {' '.join(str(error).split())}"
                    )[:1000]
                sequence_records.append(item)
    supported = [row["sequence"] for row in sequence_records if row["support"] == "supported"]
    seal_value: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": args.request_id,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": args.prob4d_revision,
        "provider_bundle_id": manifest["provider_bundle_id"],
        "target_sequences": TARGET_SEQUENCES,
        "reserved_sequences": RESERVED_SEQUENCES,
        "supported_sequences": supported,
        "sequence_records": sequence_records,
        "opened_2d_members": opened_2d_members,
        "information_boundary": {
            "sealed_provider_predictions_opened": True,
            "two_dimensional_markers_opened": True,
            "three_dimensional_markers_opened": False,
            "predictions_and_admission_complete": True,
            "post_seal_tuning_authorized": False,
            "r31_r70_payloads_opened": False,
        },
        "decision": "predictions-sealed-before-3d-outcomes",
        "claim_boundary": protocol["claim_boundary"],
    }
    seal_value["seal_id"] = content_id(seal_value)
    _write_json(output / "prediction-seal.json", seal_value)
    print(
        json.dumps(
            {
                "decision": seal_value["decision"],
                "seal_id": seal_value["seal_id"],
                "supported_sequences": len(supported),
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_seal(
    path: Path,
    protocol: Mapping[str, Any],
    provider_bundle_id: str,
    request_id: str,
    revision: str,
) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != SEAL_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("prediction seal schema changed")
    unsigned = dict(value)
    seal_id = unsigned.pop("seal_id", None)
    _hex(seal_id, name="seal_id", length=64)
    if content_id(unsigned) != seal_id:
        raise ValueError("prediction seal identity mismatch")
    if value.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("prediction seal protocol changed")
    if value.get("provider_bundle_id") != provider_bundle_id:
        raise ValueError("prediction seal provider changed")
    if value.get("request_id") != request_id:
        raise ValueError("prediction seal request changed")
    if value.get("repository_revision") != revision:
        raise ValueError("prediction seal revision changed")
    boundary = value.get("information_boundary") or {}
    if boundary.get("three_dimensional_markers_opened") is not False:
        raise ValueError("prediction seal opened 3-D marker outcomes")
    if boundary.get("predictions_and_admission_complete") is not True:
        raise ValueError("prediction seal is incomplete")
    return value


def _collect_provider_truth(
    pooled: Any,
    run: Mapping[str, np.ndarray],
    payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    providers: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for frame in frames:
        coordinates_2d, coordinates_3d = payloads[int(frame)]
        provider, indices, diagnostic = _sample_provider_2d(
            pooled,
            run,
            int(frame),
            coordinates_2d,
        )
        count = min(coordinates_2d.shape[0], coordinates_3d.shape[0])
        truth = np.asarray(coordinates_3d[:count], dtype=np.float64)
        finite_truth = np.isfinite(truth).all(axis=1)
        keep = np.asarray([index for index in indices if index < count and finite_truth[index]])
        positions = np.searchsorted(indices, keep)
        if keep.size:
            providers.append(provider[positions])
            truths.append(truth[keep])
        diagnostics.append({**diagnostic, "valid_provider_truth": int(keep.size)})
    total = sum(row["valid_provider_truth"] for row in diagnostics)
    if total < minimum:
        raise ValueError("metric-fit support is below the frozen minimum")
    return np.concatenate(providers), np.concatenate(truths), diagnostics


def _mean_ci(values: Sequence[float], *, seed: int, replicates: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replicates = int(protocol["statistics"]["paired_bootstrap_replicates"])
    base_seed = int(protocol["statistics"]["paired_bootstrap_seed"])
    aggregate: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for query_index, query in enumerate(QUERIES):
        aggregate[query] = {}
        comparisons[query] = {}
        query_rows = [row for row in rows if row["query"] == query]
        for method_index, method in enumerate(METHODS):
            selected = [row for row in query_rows if row["method"] == method]
            if len(selected) != len({row["sequence"] for row in query_rows}):
                raise ValueError("every scored sequence must contribute every method")
            aggregate[query][method] = {
                "sequence_count": len(selected),
                "mean_rmse_fraction_of_provider_span": float(
                    np.mean([row["rmse_fraction_of_provider_span"] for row in selected])
                ),
                "mean_normalized_nll_per_dimension": float(
                    np.mean([row["normalized_nll_per_dimension"] for row in selected])
                ),
                "mean_normalized_nees": float(
                    np.mean([row["normalized_nees"] for row in selected])
                ),
                "covered_90_count": int(sum(row["covered_90"] for row in selected)),
                "covered_95_count": int(sum(row["covered_95"] for row in selected)),
                "harmful_sequence_count": int(sum(row["harmful_vs_fallback"] for row in selected)),
                "accepted_sequence_count": int(sum(row["accepted"] for row in selected)),
                "exact_fallback_sequence_count": int(
                    sum(row["exact_fallback"] for row in selected)
                ),
            }
            if method == "physical_fallback":
                continue
            fallback_by_sequence = {
                row["sequence"]: row
                for row in query_rows
                if row["method"] == "physical_fallback"
            }
            rmse_improvements = [
                fallback_by_sequence[row["sequence"]][
                    "rmse_fraction_of_provider_span"
                ]
                - row["rmse_fraction_of_provider_span"]
                for row in selected
            ]
            nll_improvements = [
                fallback_by_sequence[row["sequence"]][
                    "normalized_nll_per_dimension"
                ]
                - row["normalized_nll_per_dimension"]
                for row in selected
            ]
            comparisons[query][method] = {
                "fallback_minus_method_rmse": _mean_ci(
                    rmse_improvements,
                    seed=base_seed + 100 * query_index + 2 * method_index,
                    replicates=replicates,
                ),
                "fallback_minus_method_nll": _mean_ci(
                    nll_improvements,
                    seed=base_seed + 100 * query_index + 2 * method_index + 1,
                    replicates=replicates,
                ),
                "sequence_wins_rmse": int(sum(value > 0.0 for value in rmse_improvements)),
                "sequence_wins_nll": int(sum(value > 0.0 for value in nll_improvements)),
            }
    return aggregate, comparisons


def _classification(
    protocol: Mapping[str, Any],
    seal_value: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    statistics = protocol["statistics"]
    supported = len(rows) // (len(METHODS) * len(QUERIES))
    seal_rows = [
        row for row in seal_value["sequence_records"] if row["support"] == "supported"
    ]
    centroid_admitted = sum(
        row["queries"]["centerline_centroid"]["admitted"] for row in seal_rows
    )
    off_axis_rejected = sum(
        not row["queries"]["off_axis_probe"]["admitted"] for row in seal_rows
    )
    rejected_query_aware = [
        prediction
        for row in seal_rows
        for query in QUERIES
        if not row["queries"][query]["admitted"]
        for prediction in [row["queries"][query]["predictions"]["query_aware"]]
    ]
    centroid = comparisons["centerline_centroid"]["query_aware"]
    unconditional_off_axis = comparisons["off_axis_probe"][
        "observable_subspace_unconditional"
    ]
    unconditional_aggregate = aggregate["off_axis_probe"][
        "observable_subspace_unconditional"
    ]
    checks = {
        "minimum_supported_sequences": supported
        >= int(statistics["minimum_supported_sequences"]),
        "minimum_centroid_acceptance_fraction": (
            centroid_admitted / max(len(seal_rows), 1)
            >= float(statistics["minimum_centroid_acceptance_fraction"])
        ),
        "minimum_off_axis_rejection_fraction": (
            off_axis_rejected / max(len(seal_rows), 1)
            >= float(statistics["minimum_off_axis_rejection_fraction"])
        ),
        "exact_fallback_for_every_rejection": all(
            prediction["exact_fallback"] for prediction in rejected_query_aware
        ),
        "centroid_rmse_lower_bound_positive": centroid[
            "fallback_minus_method_rmse"
        ]["lower_95"]
        > 0.0,
        "centroid_nll_lower_bound_positive": centroid[
            "fallback_minus_method_nll"
        ]["lower_95"]
        > 0.0,
    }
    selective_pass = all(checks.values())
    negative_control = (
        unconditional_off_axis["fallback_minus_method_nll"]["mean"] < 0.0
        or unconditional_aggregate["harmful_sequence_count"]
        / max(unconditional_aggregate["sequence_count"], 1)
        >= 0.1
    )
    checks["unconditional_off_axis_negative_control"] = negative_control
    if selective_pass and negative_control:
        return "query-selective-strong-positive", checks
    if selective_pass:
        return "query-selective-bounded-positive", checks
    return "query-selective-mixed-negative-or-insufficient-support", checks


def evaluate(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    bundle = args.provider_bundle.expanduser().resolve(strict=True)
    manifest = _verify_provider_bundle(
        bundle,
        protocol,
        args.request_id,
        args.prob4d_revision,
    )
    seal_value = _verify_seal(
        args.prediction_seal,
        protocol,
        manifest["provider_bundle_id"],
        args.request_id,
        args.prob4d_revision,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    pooled = _load_script(
        "evaluate_dot_rope_cut3r_pooled.py",
        "dot_query_selective_outcome_mapping",
        POOLED_EVALUATOR_BLOB,
    )
    pooled._ACTIVE_COORDINATE_COLUMNS = (0, 1)
    pooled._ACTIVE_COORDINATE_MODE = "pixel-zero-based"
    records_by_sequence: dict[str, dict[str, Mapping[str, Any]]] = {
        sequence: {} for sequence in TARGET_SEQUENCES
    }
    for record in manifest["outputs"]:
        records_by_sequence[str(record["sequence"])][str(record["run"])] = record
    seal_by_sequence = {row["sequence"]: row for row in seal_value["sequence_records"]}
    rows: list[dict[str, Any]] = []
    opened_members: list[dict[str, Any]] = []
    scoring_failures: list[dict[str, str]] = []
    root = args.dataset_root.expanduser().resolve(strict=True)
    for sequence in seal_value["supported_sequences"]:
        archive_name = _archive_for_sequence(protocol, sequence)
        archive_path = (root / archive_name).resolve(strict=True)
        archive_path.relative_to(root)
        try:
            with zipfile.ZipFile(archive_path, "r") as source_archive:
                names = set(source_archive.namelist())
                fit_frames = sorted(
                    set(protocol["evaluation"]["metric_fit_a_frames"])
                    | set(protocol["evaluation"]["metric_fit_b_frames"])
                )
                payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for frame in fit_frames:
                    member_2d = _coordinate_member(sequence, 2, int(frame))
                    member_3d = _coordinate_member(sequence, 3, int(frame))
                    if member_2d not in names or member_3d not in names:
                        raise ValueError("registered metric-fit marker member is missing")
                    raw_2d = source_archive.read(member_2d)
                    raw_3d = source_archive.read(member_3d)
                    opened_members.extend(
                        [
                            {
                                "sequence": sequence,
                                "archive": archive_name,
                                "frame": int(frame),
                                "kind": "2d",
                                "member": member_2d,
                                "byte_count": len(raw_2d),
                                "sha256": _sha256_bytes(raw_2d),
                            },
                            {
                                "sequence": sequence,
                                "archive": archive_name,
                                "frame": int(frame),
                                "kind": "3d",
                                "member": member_3d,
                                "byte_count": len(raw_3d),
                                "sha256": _sha256_bytes(raw_3d),
                            },
                        ]
                    )
                    payloads[int(frame)] = (
                        pooled._parse_coordinate_text(raw_2d.decode("utf-8"), 2),
                        pooled._parse_coordinate_text(raw_3d.decode("utf-8"), 3),
                    )
            run_a = _load_run(bundle, records_by_sequence[sequence]["window_a"])
            run_b = _load_run(bundle, records_by_sequence[sequence]["window_b"])
            minimum = int(protocol["marker_sampling"]["minimum_metric_fit_markers"])
            a_source, a_truth, _ = _collect_provider_truth(
                pooled,
                run_a,
                payloads,
                protocol["evaluation"]["metric_fit_a_frames"],
                minimum,
            )
            b_source, b_truth, _ = _collect_provider_truth(
                pooled,
                run_b,
                payloads,
                protocol["evaluation"]["metric_fit_b_frames"],
                minimum,
            )
            a_to_truth, _ = robust_fit_sim3(a_source, a_truth)
            b_to_truth, _ = robust_fit_sim3(b_source, b_truth)
            true_relative = a_to_truth.inverse().compose(b_to_truth)
            sequence_seal = seal_by_sequence[sequence]
            for query in QUERIES:
                query_record = sequence_seal["queries"][query]
                query_source = np.asarray(query_record["query_source"], dtype=np.float64)
                truth = np.asarray(
                    true_relative.transform_points(query_source),
                    dtype=np.float64,
                )
                fallback_error = None
                pending: list[dict[str, Any]] = []
                for method in METHODS:
                    prediction = query_record["predictions"][method]
                    mean = np.asarray(prediction["mean"], dtype=np.float64)
                    covariance = np.asarray(prediction["covariance"], dtype=np.float64)
                    score = normalized_gaussian_score(
                        truth,
                        mean,
                        covariance,
                        span=float(sequence_seal["provider_span"]),
                        observation_noise_fraction=float(
                            protocol["evaluation"]["observation_noise_fraction"]
                        ),
                    )
                    error = float(score["mean_error_fraction_of_span"])
                    nees = float(score["mahalanobis"]) / 3.0
                    row = {
                        "sequence": sequence,
                        "query": query,
                        "method": method,
                        "rmse_fraction_of_provider_span": error,
                        "normalized_nees": nees,
                        "covered_90": bool(3.0 * nees <= CHI2_3_90),
                        "accepted": bool(prediction["accepted"]),
                        "exact_fallback": bool(prediction["exact_fallback"]),
                        **score,
                    }
                    if method == "physical_fallback":
                        fallback_error = error
                    pending.append(row)
                if fallback_error is None:
                    raise AssertionError("fallback row is missing")
                for row in pending:
                    row["harmful_vs_fallback"] = bool(
                        row["rmse_fraction_of_provider_span"] > fallback_error + 1e-12
                    )
                    rows.append(row)
        except Exception as error:
            scoring_failures.append(
                {
                    "sequence": sequence,
                    "failure": (
                        f"{type(error).__name__}: {' '.join(str(error).split())}"
                    )[:1000],
                }
            )
    scored_sequences = sorted({row["sequence"] for row in rows})
    aggregate, comparisons = _aggregate_rows(rows, protocol)
    classification, checks = _classification(
        protocol,
        seal_value,
        rows,
        aggregate,
        comparisons,
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": args.request_id,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": args.prob4d_revision,
        "provider_bundle_id": manifest["provider_bundle_id"],
        "prediction_seal_id": seal_value["seal_id"],
        "decision": classification,
        "target_sequences": TARGET_SEQUENCES,
        "reserved_sequences": RESERVED_SEQUENCES,
        "supported_sequences": seal_value["supported_sequences"],
        "scored_sequences": scored_sequences,
        "scoring_failures": scoring_failures,
        "method_rows": rows,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "registered_checks": checks,
        "opened_marker_members": opened_members,
        "information_boundary": {
            "provider_predictions_sealed_before_marker_access": True,
            "factor_predictions_sealed_before_3d_marker_access": True,
            "two_dimensional_markers_opened_for_factor_localization": True,
            "three_dimensional_markers_opened_only_for_final_scoring": True,
            "post_open_tuning_performed": False,
            "r31_r70_payloads_opened": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    _write_json(output / "result.json", result)
    lines = [
        "# DOT R11-R30 query-selective learned-provider result",
        "",
        f"Decision: **{classification}**",
        "",
        f"Result ID: `{result['result_id']}`",
        f"Prediction seal ID: `{seal_value['seal_id']}`",
        "",
        f"Supported sequences: {len(seal_value['supported_sequences'])}/20",
        f"Scored sequences: {len(scored_sequences)}/20",
        "",
        "## Registered checks",
        "",
    ]
    lines.extend(f"- {name}: `{passed}`" for name, passed in checks.items())
    lines.extend(["", "## Primary centroid comparison", ""])
    primary = comparisons["centerline_centroid"]["query_aware"]
    lines.append(
        "- fallback minus query-aware RMSE/span: "
        f"{primary['fallback_minus_method_rmse']['mean']:.6f} "
        f"[{primary['fallback_minus_method_rmse']['lower_95']:.6f}, "
        f"{primary['fallback_minus_method_rmse']['upper_95']:.6f}]"
    )
    lines.append(
        "- fallback minus query-aware normalized NLL/dim: "
        f"{primary['fallback_minus_method_nll']['mean']:.6f} "
        f"[{primary['fallback_minus_method_nll']['lower_95']:.6f}, "
        f"{primary['fallback_minus_method_nll']['upper_95']:.6f}]"
    )
    lines.extend(
        [
            "",
            "R31-R70 remained unopened. No target-side tuning, BayesianPhysTwin, "
            "or Causal4D execution was performed.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": classification,
                "result_id": result["result_id"],
                "scored_sequences": len(scored_sequences),
            },
            sort_keys=True,
        )
    )
    return 0


def _technical_failure(
    output: Path,
    *,
    request_id: str,
    revision: str,
    protocol_id: str,
    stage: str,
    error: Exception,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    failure: dict[str, Any] = {
        "schema": FAILURE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "protocol_id": protocol_id,
        "repository_revision": revision,
        "decision": "technical-failure",
        "stage": stage,
        "failure": f"{type(error).__name__}: {' '.join(str(error).split())}"[:2000],
        "traceback_tail": traceback.format_exc().splitlines()[-20:],
        "information_boundary": {
            "post_failure_retuning_authorized": False,
            "r31_r70_payloads_opened": False,
        },
    }
    failure["result_id"] = content_id(failure)
    _write_json(output / "failure.json", failure)
    print(json.dumps({"decision": failure["decision"], "result_id": failure["result_id"]}))
    return 3


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-request":
        print(
            json.dumps(
                validate_request(
                    args.request,
                    args.protocol,
                    args.protocol_git_blob_sha,
                ),
                sort_keys=True,
            )
        )
        return 0
    protocol = _load_protocol(args.protocol)
    try:
        if args.command == "predict":
            return predict(args)
        if args.command == "seal":
            return seal(args)
        if args.command == "evaluate":
            return evaluate(args)
    except Exception as error:
        return _technical_failure(
            args.output_dir,
            request_id=args.request_id,
            revision=args.prob4d_revision,
            protocol_id=protocol["protocol_id"],
            stage=args.command,
            error=error,
        )
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
