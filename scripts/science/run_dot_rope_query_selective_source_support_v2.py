#!/usr/bin/env python3
"""Run the clean DOT R11-R20 source/support qualification for query-selective v2.

This stage is deliberately source-only. It seals marker-free CUT3R predictions
for R11-R20 before opening any source markers. It then selects one support
geometry using support counts and the frozen rank-six factor mechanism only.
It never opens R21-R70 and never computes a reconstruction error or proper
score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import traceback
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.dot_rope_cut3r_study import content_id
from prob4d.observable_gauge import estimate_observable_sim3_factor

PROTOCOL_SCHEMA = "prob4d.dot-rope-query-selective-source-support-protocol"
REQUEST_SCHEMA = "prob4d.dot-rope-query-selective-source-support-request"
RESULT_SCHEMA = "prob4d.dot-rope-query-selective-source-support-result"
FAILURE_SCHEMA = "prob4d.dot-rope-query-selective-source-support-failure"
SCHEMA_VERSION = 2

SOURCE_ARCHIVE = "R11-20.zip"
SOURCE_ARCHIVE_MD5 = "23ce3e7067465d3edabe20b4c7cfa388"
SOURCE_SEQUENCES = [f"R{index:02d}" for index in range(11, 21)]
CONFIRMATION_SEQUENCES = [f"R{index:02d}" for index in range(21, 31)]
RESERVED_SEQUENCES = "R31-R70"
SOURCE_CLOSED_BOUNDARY = "R21-R70"
CAMERA = "cam001"
FRAMES = list(range(1, 8))

BASE_PROVIDER_BLOB = "612c8ae61b0a64d464256a11992b46c486c88012"
MARKER_AUDIT_BLOB = "95c3039493fec8201dccc7b3608ccfe92c4fa1c2"
QUERY_V1_BLOB = "ac2751e327aafcf9c49ce9ffcfe8bb0e7eae916e"

R04_RESULT_ID = "531272d64eee29f26321030b236db9bd6ae3aadb7e16a1f492ac8140afc801ee"
R04_SUPPORT_ID = "0a03bccbe1545b7d58819dacffb8f384332cde06e7e8a362532bd0b038aec63d"
R04_PROVIDER_ID = "57dc11d9e39258a2f620d67e39f1176cafe74252173ead8cb4ba2f76083499ec"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--protocol-git-blob-sha", required=True)

    for name in ("runtime-smoke", "predict"):
        command = commands.add_parser(name)
        _execution_arguments(command)
        command.add_argument("--cut3r-checkout", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--runtime-receipt", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "predict":
            command.add_argument("--dataset-root", type=Path, required=True)

    qualify = commands.add_parser("qualify")
    _execution_arguments(qualify)
    qualify.add_argument("--dataset-root", type=Path, required=True)
    qualify.add_argument("--provider-bundle", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
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


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    return value


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


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported source-support protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _hex(protocol_id, name="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("source-support protocol identity mismatch")
    if protocol.get("source_archive") != {
        "name": SOURCE_ARCHIVE,
        "md5": SOURCE_ARCHIVE_MD5,
    }:
        raise ValueError("source archive changed")
    if protocol.get("source_sequences") != SOURCE_SEQUENCES:
        raise ValueError("source sequence roster changed")
    if protocol.get("confirmation_sequences") != CONFIRMATION_SEQUENCES:
        raise ValueError("confirmation sequence roster changed")
    if protocol.get("reserved_sequences") != RESERVED_SEQUENCES:
        raise ValueError("reserve changed")
    if protocol.get("camera") != CAMERA or protocol.get("frames") != FRAMES:
        raise ValueError("camera or frame roster changed")
    diagnostic = protocol.get("r04_r10_diagnostic") or {}
    if diagnostic.get("decision") != "heldout-support-negative":
        raise ValueError("R04-R10 diagnostic decision changed")
    if diagnostic.get("result_id") != R04_RESULT_ID:
        raise ValueError("R04-R10 result identity changed")
    if diagnostic.get("marker_support_id") != R04_SUPPORT_ID:
        raise ValueError("R04-R10 support identity changed")
    if diagnostic.get("provider_bundle_id") != R04_PROVIDER_ID:
        raise ValueError("R04-R10 provider identity changed")
    boundary = protocol.get("information_boundary") or {}
    false_fields = (
        "r04_r10_reused_for_tuning",
        "source_support_selection_uses_performance_error",
        "r21_r30_payloads_opened",
        "r31_r70_payloads_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
        "dataset_modified",
        "target_side_retuning_allowed",
    )
    if any(boundary.get(name) is not False for name in false_fields):
        raise ValueError("source information boundary changed")
    if boundary.get("source_provider_sealed_before_source_marker_access") is not True:
        raise ValueError("source provider/marker custody changed")
    selection = protocol.get("support_selection") or {}
    if float(selection.get("rank_threshold", math.nan)) != 0.01:
        raise ValueError("rank threshold changed")
    if int(selection.get("expected_factor_rank", -1)) != 6:
        raise ValueError("expected factor rank changed")
    if selection.get("outcome_metrics_used_for_selection") is not False:
        raise ValueError("outcome-based source selection is forbidden")
    return protocol


def validate_request(
    request_path: Path,
    protocol_path: Path,
    protocol_git_blob_sha: str,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    request = _read_json(request_path)
    expected = {
        "schema",
        "schema_version",
        "request_id",
        "protocol_path",
        "protocol_git_blob_sha",
        "r04_r10_diagnostic",
        "source_sequences",
        "confirmation_sequences",
        "reserved_sequences",
        "normal_view_source_prediction_authorized",
        "source_marker_support_qualification_authorized",
        "confirmation_prediction_authorized",
        "confirmation_marker_access_authorized",
        "post_source_tuning_authorized",
        "bayesian_phystwin_executed",
        "causal4d_executed",
        "claim_boundary",
    }
    if set(request) != expected:
        raise ValueError("source-support request fields changed")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported source-support request schema")
    _hex(protocol_git_blob_sha, name="protocol Git blob", length=40)
    if request["protocol_path"] != protocol_path.as_posix():
        raise ValueError("request protocol path changed")
    if request["protocol_git_blob_sha"] != protocol_git_blob_sha:
        raise ValueError("request does not bind the reviewed protocol blob")
    if request["source_sequences"] != SOURCE_SEQUENCES:
        raise ValueError("request source roster changed")
    if request["confirmation_sequences"] != CONFIRMATION_SEQUENCES:
        raise ValueError("request confirmation roster changed")
    if request["reserved_sequences"] != RESERVED_SEQUENCES:
        raise ValueError("request reserve changed")
    if request["r04_r10_diagnostic"] != {
        "decision": "heldout-support-negative",
        "result_id": R04_RESULT_ID,
        "marker_support_id": R04_SUPPORT_ID,
        "provider_bundle_id": R04_PROVIDER_ID,
    }:
        raise ValueError("request R04-R10 diagnostic binding changed")
    for name in (
        "normal_view_source_prediction_authorized",
        "source_marker_support_qualification_authorized",
    ):
        if request[name] is not True:
            raise ValueError(f"{name} must be explicitly authorized")
    for name in (
        "confirmation_prediction_authorized",
        "confirmation_marker_access_authorized",
        "post_source_tuning_authorized",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    ):
        if request[name] is not False:
            raise ValueError(f"{name} exceeds the source-only boundary")
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("source-support request identity mismatch")
    return {
        "request_id": request_id,
        "protocol_id": protocol["protocol_id"],
        "source_sequences": SOURCE_SEQUENCES,
    }


def _require_execution_identity(request_id: str, revision: str) -> None:
    _hex(request_id, name="request_id", length=64)
    _hex(revision, name="Prob4D revision", length=40)


def _adapted_provider_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "prob4d.dot-rope-cut3r-native-provider-protocol",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "dataset_doi": protocol["dataset_doi"],
        "archive": SOURCE_ARCHIVE,
        "camera": CAMERA,
        "frames": FRAMES,
        "source_sequences": SOURCE_SEQUENCES,
        "reserved_sequences": SOURCE_CLOSED_BOUNDARY,
        "windows": protocol["windows"],
        "provider": protocol["provider"],
        "information_boundary": {
            "evaluation_stage_reads": "R11-R20 markers only after source provider seal",
            "evaluation_stage_requires_sealed_provider_bundle": True,
            "provider_stage_forbids": [
                "2-D marker payloads",
                "3-D marker payloads",
                "R21-R70 payloads",
                "provider residuals",
                "BayesianPhysTwin outcomes",
                "Causal4D outcomes",
            ],
            "provider_stage_reads": "normal-view JPEG payloads from R11-R20 only",
            "target_payloads_opened": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _run_base_provider(args: argparse.Namespace, command: str) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    base = _load_script(
        "run_dot_rope_cut3r_native_provider.py",
        "dot_r11_r20_source_provider",
        BASE_PROVIDER_BLOB,
    )
    adapted = _adapted_provider_protocol(protocol)
    base._load_protocol = lambda _path: adapted
    if command == "runtime-smoke":
        return int(base.runtime_smoke(args))
    if command == "predict":
        return int(base.predict(args))
    raise AssertionError(command)


def _positive_condition_ratio(information: np.ndarray) -> float:
    eigenvalues = np.asarray(np.linalg.eigvalsh(information), dtype=np.float64)
    largest = float(np.max(eigenvalues))
    if not math.isfinite(largest) or largest <= 0.0:
        return 0.0
    positive = eigenvalues[eigenvalues > largest * 1e-12]
    if positive.size == 0:
        return 0.0
    return float(np.min(positive) / largest)


def _select_candidate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("no source support candidates were evaluated")
    ordered = sorted(
        summaries,
        key=lambda value: (
            -int(value["supported_sequences"]),
            -int(value["rank_six_sequences"]),
            -float(value["worst_normalized_support_margin"]),
            -float(value["worst_observable_condition_ratio"]),
            int(value["selected_frame_count"]),
            str(value["candidate_id"]),
        ),
    )
    return ordered[0]


def qualify(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)

    audit = _load_script(
        "audit_dot_rope_marker_support.py",
        "dot_r11_r20_support_helpers",
        MARKER_AUDIT_BLOB,
    )
    audit.ARCHIVE = SOURCE_ARCHIVE
    audit.SEQUENCES = SOURCE_SEQUENCES
    audit.RESERVED = SOURCE_CLOSED_BOUNDARY
    audit.FRAMES = FRAMES
    audit.CAMERA = CAMERA

    query = _load_script(
        "run_dot_rope_query_selective_heldout.py",
        "dot_query_selective_v1_helpers",
        QUERY_V1_BLOB,
    )
    pooled = query._load_script(
        "evaluate_dot_rope_cut3r_pooled.py",
        "dot_r11_r20_marker_mapping",
        query.POOLED_EVALUATOR_BLOB,
    )
    pooled._ACTIVE_COORDINATE_COLUMNS = (0, 1)
    pooled._ACTIVE_COORDINATE_MODE = "pixel-zero-based"

    bundle = args.provider_bundle.expanduser().resolve(strict=True)
    provider_manifest = _read_json(bundle / "manifest.json")
    provider_id = provider_manifest.get("provider_bundle_id")
    _hex(provider_id, name="provider_bundle_id", length=64)
    manifest, provider_paths = audit._verify_provider(
        bundle,
        {"provider_bundle_id": provider_id},
    )
    if manifest.get("request_id") != args.request_id:
        raise ValueError("source provider request identity changed")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("source provider protocol identity changed")
    if manifest.get("prob4d_revision") != args.prob4d_revision:
        raise ValueError("source provider revision changed")

    runs = {
        sequence: {
            run_name: audit._load_run(path)
            for run_name, path in sequence_paths.items()
        }
        for sequence, sequence_paths in provider_paths.items()
    }

    root = args.dataset_root.expanduser().resolve(strict=True)
    archive_path = (root / SOURCE_ARCHIVE).resolve(strict=True)
    archive_path.relative_to(root)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("DOT R11-R20 source archive is unavailable")

    support_by_sequence: dict[str, dict[int, dict[str, int]]] = {
        sequence: {} for sequence in SOURCE_SEQUENCES
    }
    coordinates_by_sequence: dict[str, dict[int, np.ndarray]] = {
        sequence: {} for sequence in SOURCE_SEQUENCES
    }
    opened_members: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path, "r") as source_archive:
        names = set(source_archive.namelist())
        for sequence in SOURCE_SEQUENCES:
            for frame in FRAMES:
                member_2d = audit._coordinate_member(sequence, 2, frame)
                member_3d = audit._coordinate_member(sequence, 3, frame)
                if member_2d not in names or member_3d not in names:
                    raise ValueError("registered R11-R20 marker payload is missing")
                audit._safe_member(member_2d)
                audit._safe_member(member_3d)
                raw_2d = source_archive.read(member_2d)
                raw_3d = source_archive.read(member_3d)
                opened_members.extend(
                    [
                        {
                            "sequence": sequence,
                            "frame": frame,
                            "kind": "2d",
                            "member": member_2d,
                            "byte_count": len(raw_2d),
                            "sha256": _sha256_bytes(raw_2d),
                        },
                        {
                            "sequence": sequence,
                            "frame": frame,
                            "kind": "3d",
                            "member": member_3d,
                            "byte_count": len(raw_3d),
                            "sha256": _sha256_bytes(raw_3d),
                        },
                    ]
                )
                rows_2d = audit._numeric_rows(raw_2d.decode("utf-8"))
                rows_3d = audit._numeric_rows(raw_3d.decode("utf-8"))
                count = min(len(rows_2d), len(rows_3d))
                if count == 0 or any(len(row) < 2 for row in rows_2d[:count]):
                    coordinates = np.empty((0, 2), dtype=np.float64)
                else:
                    coordinates = np.asarray(
                        [[row[0], row[1]] for row in rows_2d[:count]],
                        dtype=np.float64,
                    )
                if count == 0 or any(len(row) < 3 for row in rows_3d[:count]):
                    truth = np.empty((0, 3), dtype=np.float64)
                else:
                    truth = np.asarray(
                        [row[-3:] for row in rows_3d[:count]],
                        dtype=np.float64,
                    )
                coordinates_by_sequence[sequence][frame] = coordinates
                support_by_sequence[sequence][frame] = audit._support(
                    runs,
                    sequence=sequence,
                    frame=frame,
                    coordinates=coordinates,
                    truth=truth,
                )

    selection = protocol["support_selection"]
    minimum_fit = int(selection["minimum_metric_fit_markers"])
    minimum_overlap = int(selection["minimum_overlap_common_markers"])
    minimum_nonempty = int(selection["minimum_overlap_nonempty_frames"])
    expected_rank = int(selection["expected_factor_rank"])
    rank_threshold = float(selection["rank_threshold"])

    candidate_summaries: list[dict[str, Any]] = []
    for fit_profile in selection["fit_frame_profiles"]:
        for overlap_group in selection["overlap_groups"]:
            candidate_id = f"{fit_profile['id']}__{overlap_group['id']}"
            per_sequence: list[dict[str, Any]] = []
            margins: list[float] = []
            condition_ratios: list[float] = []
            supported_sequences = 0
            rank_six_sequences = 0
            for sequence in SOURCE_SEQUENCES:
                support = support_by_sequence[sequence]
                fit_a_total = sum(
                    max(0, support[int(frame)]["window_a"])
                    for frame in fit_profile["metric_fit_a_frames"]
                )
                fit_b_total = sum(
                    max(0, support[int(frame)]["window_b"])
                    for frame in fit_profile["metric_fit_b_frames"]
                )
                overlap_values = [
                    max(0, support[int(frame)]["window_common"])
                    for frame in overlap_group["frames"]
                ]
                overlap_total = int(sum(overlap_values))
                overlap_nonempty = int(sum(value > 0 for value in overlap_values))
                margin = min(
                    fit_a_total / minimum_fit,
                    fit_b_total / minimum_fit,
                    overlap_total / minimum_overlap,
                )
                margins.append(float(margin))

                factor_rank = 0
                condition_ratio = 0.0
                factor_error: str | None = None
                try:
                    source, target, _ = query._collect_overlap_factor_points(
                        pooled,
                        runs[sequence]["window_a"],
                        runs[sequence]["window_b"],
                        coordinates_by_sequence[sequence],
                        overlap_group["frames"],
                    )
                    factor = estimate_observable_sim3_factor(
                        source,
                        target,
                        rank_threshold=rank_threshold,
                    )
                    factor_rank = int(factor.rank)
                    condition_ratio = _positive_condition_ratio(
                        factor.observable_information
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    factor_error = f"{type(error).__name__}: {error}"
                if factor_rank == expected_rank:
                    rank_six_sequences += 1
                    condition_ratios.append(condition_ratio)

                count_support = (
                    fit_a_total >= minimum_fit
                    and fit_b_total >= minimum_fit
                    and overlap_total >= minimum_overlap
                    and overlap_nonempty >= minimum_nonempty
                )
                supported = count_support and factor_rank == expected_rank
                supported_sequences += int(supported)
                per_sequence.append(
                    {
                        "sequence": sequence,
                        "supported": bool(supported),
                        "fit_a_total": int(fit_a_total),
                        "fit_b_total": int(fit_b_total),
                        "overlap_common_total": overlap_total,
                        "overlap_nonempty_frames": overlap_nonempty,
                        "normalized_support_margin": float(margin),
                        "factor_rank": factor_rank,
                        "observable_condition_ratio": condition_ratio,
                        "factor_error": factor_error,
                    }
                )

            candidate_summaries.append(
                {
                    "candidate_id": candidate_id,
                    "fit_profile": fit_profile,
                    "overlap_group": overlap_group,
                    "supported_sequences": supported_sequences,
                    "rank_six_sequences": rank_six_sequences,
                    "worst_normalized_support_margin": min(margins, default=0.0),
                    "worst_observable_condition_ratio": min(
                        condition_ratios,
                        default=0.0,
                    ),
                    "selected_frame_count": (
                        len(fit_profile["metric_fit_a_frames"])
                        + len(fit_profile["metric_fit_b_frames"])
                        + len(overlap_group["frames"])
                    ),
                    "per_sequence": per_sequence,
                }
            )

    selected = _select_candidate_summaries(candidate_summaries)
    promotion_minimum = int(
        selection["minimum_source_supported_sequences_for_promotion"]
    )
    decision = (
        "source-support-qualified"
        if int(selected["supported_sequences"]) >= promotion_minimum
        else "source-support-negative"
    )

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "request_id": args.request_id,
        "protocol_id": protocol["protocol_id"],
        "prob4d_revision": args.prob4d_revision,
        "provider_bundle_id": provider_id,
        "source_archive": protocol["source_archive"],
        "source_sequences": SOURCE_SEQUENCES,
        "confirmation_sequences": CONFIRMATION_SEQUENCES,
        "reserved_sequences": RESERVED_SEQUENCES,
        "selection_objective": selection["selection_objective"],
        "candidate_summaries": candidate_summaries,
        "selected_support_rule": {
            "candidate_id": selected["candidate_id"],
            "metric_fit_a_frames": selected["fit_profile"]["metric_fit_a_frames"],
            "metric_fit_b_frames": selected["fit_profile"]["metric_fit_b_frames"],
            "overlap_frames": selected["overlap_group"]["frames"],
            "marker_group_construction": selection["marker_group_construction"],
            "minimum_metric_fit_markers": minimum_fit,
            "minimum_overlap_common_markers": minimum_overlap,
            "minimum_overlap_nonempty_frames": minimum_nonempty,
            "expected_factor_rank": expected_rank,
            "rank_threshold": rank_threshold,
            "supported_source_sequences": selected["supported_sequences"],
        },
        "downstream_frozen_config": protocol["downstream_frozen_config"],
        "confirmation_statistics": protocol["confirmation_statistics"],
        "opened_source_marker_members": opened_members,
        "information_boundary": {
            "r04_r10_payloads_opened_by_this_stage": False,
            "source_normal_view_opened_by_provider_stage": True,
            "source_2d_markers_opened_after_provider_seal": True,
            "source_3d_markers_opened_after_provider_seal": True,
            "source_reconstruction_error_computed": False,
            "source_proper_score_computed": False,
            "source_outcome_used_for_selection": False,
            "r21_r30_payloads_opened": False,
            "r31_r70_payloads_opened": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "decision": decision,
                "result_id": result["result_id"],
                "selected_candidate": selected["candidate_id"],
                "supported_source_sequences": selected["supported_sequences"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "validate-request":
            value = validate_request(
                args.request,
                args.protocol,
                args.protocol_git_blob_sha,
            )
            print(json.dumps(value, sort_keys=True))
            return 0
        if args.command == "runtime-smoke":
            return _run_base_provider(args, "runtime-smoke")
        if args.command == "predict":
            return _run_base_provider(args, "predict")
        if args.command == "qualify":
            return qualify(args)
    except Exception as error:
        if args.command == "qualify" and getattr(args, "output", None):
            failure = {
                "schema": FAILURE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "decision": "technical-failure",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _write_json(args.output, failure)
        raise
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
