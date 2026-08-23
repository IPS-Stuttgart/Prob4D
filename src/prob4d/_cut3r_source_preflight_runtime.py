"""Build the retained CUT3R source-comparison preflight report."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Any, Final, cast

from ._cut3r_source_preflight_common import (
    SOURCE_ROLES,
    _confined_directory,
    _confined_regular_file,
    _file_sha256,
    _load_json,
    _record_id,
    _stat_identity,
    validate_request,
)
from ._cut3r_source_preflight_environment import (
    _candidate_reference_files,
    _cut3r_surface,
    _ffprobe,
    _repository_revision,
    _sanitize_text,
)
from ._cut3r_source_preflight_freeze import (
    _load_comparison_lock,
    _validate_source_freeze,
)

REPORT_SCHEMA: Final = "prob4d.cut3r-deform360-source-comparison-preflight"
REPORT_VERSION: Final = 1


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve(strict=True)
    request = validate_request(args.request.resolve(strict=True), repository=repository)
    freeze_path = _confined_regular_file(
        repository, cast(str, request["source_freeze_path"]), name="source-freeze lock"
    )
    spec_path = _confined_regular_file(
        repository, cast(str, request["comparison_spec_path"]), name="comparison spec"
    )
    lock_path = _confined_regular_file(
        repository, cast(str, request["comparison_lock_path"]), name="comparison lock"
    )
    freeze = _load_json(freeze_path, name="source-freeze lock")
    spec = _load_json(spec_path, name="comparison specification")
    lock = _load_comparison_lock(lock_path, spec)
    contract = _validate_source_freeze(freeze, request=request, spec=spec, lock=lock)

    if args.processed_root.is_symlink():
        raise ValueError("processed source root must not be a symbolic link")
    processed_root = args.processed_root.resolve(strict=True)
    if not processed_root.is_dir():
        raise ValueError("processed source root must be an ordinary directory")
    if args.cut3r_checkout.is_symlink():
        raise ValueError("CUT3R checkout must not be a symbolic link")
    checkout = args.cut3r_checkout.resolve(strict=True)
    if not checkout.is_dir() or not (checkout / ".git").is_dir():
        raise ValueError("CUT3R checkout must be an ordinary Git checkout")
    checkpoint = args.checkpoint
    if checkpoint.is_symlink():
        raise ValueError("CUT3R checkpoint must not be a symbolic link")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("CUT3R checkpoint must be an ordinary file")

    redactions = {
        os.fspath(repository): "<PROB4D_REPOSITORY>",
        os.fspath(processed_root): "<DEFORM360_PROCESSED_ROOT>",
        os.fspath(checkout): "<CUT3R_CHECKOUT>",
        os.fspath(checkpoint): "<CUT3R_CHECKPOINT>",
    }
    cases: list[dict[str, object]] = []
    failures: list[str] = []
    reference_cache: dict[str, list[dict[str, object]]] = {}
    for descriptor in cast(list[dict[str, object]], contract["descriptors"]):
        case_id = cast(str, descriptor["case_id"])
        relative_video = cast(str, descriptor["relative_video_path"])
        expected_sha = cast(str, descriptor["video_sha256"])
        expected_bytes = cast(int, descriptor["video_byte_count"])
        try:
            path = _confined_regular_file(
                processed_root,
                relative_video,
                name=f"source video {case_id}",
            )
            before = path.stat()
            measured_sha = _file_sha256(path)
            measured_bytes = int(path.stat().st_size)
            if measured_sha != expected_sha or measured_bytes != expected_bytes:
                raise ValueError(f"source video identity changed: {relative_video}")

            probe = _ffprobe(path)
            verified_sidecars: list[dict[str, object]] = []
            for sidecar_name, raw_sidecar in cast(
                Mapping[str, Mapping[str, object]],
                descriptor["sidecars"],
            ).items():
                relative_sidecar = cast(str, raw_sidecar["relative_path"])
                sidecar_path = _confined_regular_file(
                    processed_root,
                    relative_sidecar,
                    name=f"source sidecar {case_id}/{sidecar_name}",
                )
                sidecar_sha = _file_sha256(sidecar_path)
                sidecar_bytes = int(sidecar_path.stat().st_size)
                if (
                    sidecar_sha != raw_sidecar["sha256"]
                    or sidecar_bytes != raw_sidecar["byte_count"]
                ):
                    raise ValueError(f"source sidecar identity changed: {relative_sidecar}")
                verified_sidecars.append(
                    {
                        "name": sidecar_name,
                        "relative_path": relative_sidecar,
                        "sha256": sidecar_sha,
                        "byte_count": sidecar_bytes,
                    }
                )
            after = path.stat()
            if _stat_identity(before) != _stat_identity(after):
                raise ValueError("video changed during preflight inspection")
            relative_episode = cast(str, descriptor["relative_episode_path"])
            if relative_episode not in reference_cache:
                episode = _confined_directory(
                    processed_root,
                    relative_episode,
                    name=f"source episode for {case_id}",
                )
                reference_cache[relative_episode] = _candidate_reference_files(
                    episode, root=processed_root
                )
        except (OSError, ValueError) as error:
            failures.append(f"{case_id}: {_sanitize_text(str(error), redactions)}")
            continue
        if probe.get("available") is not True or probe.get("status") != 0 or "stream" not in probe:
            failures.append(f"ffprobe could not inspect source video: {relative_video}")
        case_record = dict(descriptor)
        case_record.pop("sidecars")
        cases.append(
            {
                **case_record,
                "video_sha256": measured_sha,
                "video_byte_count": measured_bytes,
                "video_probe": probe,
                "verified_sidecars": sorted(
                    verified_sidecars,
                    key=lambda item: cast(str, item["name"]),
                ),
                "candidate_reference_files": reference_cache[
                    cast(str, descriptor["relative_episode_path"])
                ],
            }
        )
    cases.sort(key=lambda item: cast(str, item["case_id"]))
    group_ids = sorted({cast(str, item["group_id"]) for item in cases})
    role_case_counts = {
        role: sum(1 for item in cases if item["role"] == role) for role in SOURCE_ROLES
    }
    role_group_counts = {
        role: len(
            {
                cast(str, item["group_id"])
                for item in cases
                if item["role"] == role
            }
        )
        for role in SOURCE_ROLES
    }

    try:
        cut3r = _cut3r_surface(checkout, checkpoint)
    except (OSError, ValueError) as error:
        cut3r = {"inspection_status": "technical-failure"}
        failures.append(
            "CUT3R provider inspection failed: "
            + _sanitize_text(str(error), redactions)
        )
    else:
        cut3r["inspection_status"] = "completed"
        if cut3r["checkout_revision_status"] != 0:
            failures.append("CUT3R checkout revision could not be read")
        if cut3r["checkout_revision"] != contract["provider_revision"]:
            failures.append("CUT3R checkout differs from the frozen provider revision")
        if cut3r["origin_repository"] != contract["provider_repository"]:
            failures.append("CUT3R checkout origin differs from the frozen provider repository")
        if cut3r["tracked_worktree_clean"] is not True:
            failures.append("CUT3R checkout has tracked local modifications")
        if cut3r["checkpoint_filename"] != contract["checkpoint_filename"]:
            failures.append("CUT3R checkpoint filename differs from the source freeze")
        if cut3r["checkpoint_sha256"] != contract["checkpoint_sha256"]:
            failures.append("CUT3R checkpoint digest differs from the source freeze")
        if cut3r["checkpoint_byte_count"] != contract["checkpoint_byte_count"]:
            failures.append("CUT3R checkpoint byte count differs from the source freeze")
        if cut3r["demo_relative_path"] is None:
            failures.append("CUT3R tracked demo.py was not uniquely resolved")
        if cut3r["demo_help_status"] != 0:
            failures.append("CUT3R tracked demo.py --help failed")
        if cut3r["dependency_probe_status"] != 0:
            failures.append("CUT3R Python dependency probe failed")

    if len(cases) != cast(int, request["expected_case_count"]):
        failures.append(
            f"resolved case count {len(cases)} differs from expected "
            f"{request['expected_case_count']}"
        )
    if len(group_ids) != cast(int, request["source_group_count"]):
        failures.append(
            f"resolved group count {len(group_ids)} differs from expected "
            f"{request['source_group_count']}"
        )

    decision = (
        "source-comparison-preflight-ready" if not failures else "technical-preflight-failure"
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_VERSION,
        "preflight_request_id": request["preflight_request_id"],
        "repository_revision": _repository_revision(repository),
        "source_freeze_id": contract["source_freeze_id"],
        "comparison_spec_sha256": contract["comparison_spec_sha256"],
        "comparison_lock_id": contract["comparison_lock_id"],
        "prob4d_distribution": {
            "revision": contract["prob4d_revision"],
            "filename": contract["prob4d_distribution_filename"],
            "sha256": contract["prob4d_distribution_sha256"],
            "byte_count": contract["prob4d_distribution_byte_count"],
        },
        "decision": decision,
        "resolved_case_count": len(cases),
        "resolved_group_count": len(group_ids),
        "role_case_counts": role_case_counts,
        "role_group_counts": role_group_counts,
        "cases": cases,
        "cut3r": cut3r,
        "failures": failures,
        "source_rgb_frames_decoded": False,
        "cut3r_inference_executed": False,
        "source_input_video_bytes_hashed": True,
        "source_input_sidecars_hashed": True,
        "source_prediction_payloads_opened": False,
        "source_residuals_or_truth_opened": False,
        "candidate_reference_file_contents_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "comparison_execution_authorized": False,
        "claim_boundary": request["claim_boundary"],
    }
    report["artifact_id"] = _record_id(report)
    return report
