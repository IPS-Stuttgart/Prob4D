#!/usr/bin/env python3
"""Build an outcome-blind retained CUT3R source-comparison preflight.

The preflight resolves and verifies the forty frozen source videos, records video
container metadata, inspects the exact CUT3R checkout and callable demo surface,
and enumerates candidate source-reference file names without opening their
contents. It never decodes RGB frames, executes CUT3R inference, reads source
truth values, or touches confirmation/target groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

REQUEST_SCHEMA: Final = (
    "prob4d.cut3r-deform360-source-comparison-preflight-request"
)
REQUEST_VERSION: Final = 1
REPORT_SCHEMA: Final = "prob4d.cut3r-deform360-source-comparison-preflight"
REPORT_VERSION: Final = 1
FALSE_FIELDS: Final = (
    "source_rgb_frames_decoded",
    "source_prediction_payloads_opened",
    "source_residuals_or_truth_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "comparison_execution_authorized",
)
GROUP_KEYS: Final = (
    "group_id",
    "source_group_id",
    "object_session_id",
    "physical_object_session_id",
    "episode_id",
)
ROLE_KEYS: Final = ("role", "source_role", "split", "partition")
CASE_KEYS: Final = ("case_id", "stream_id", "source_case_id")
VIEW_KEYS: Final = ("view_id", "camera_id", "camera_name", "stream_name")
PATH_KEYS: Final = ("path", "relative_path", "video_path", "video_relative_path")
SHA_KEYS: Final = ("sha256", "video_sha256", "source_video_sha256")
BYTE_KEYS: Final = ("byte_count", "bytes", "video_byte_count")
REFERENCE_TOKENS: Final = (
    "depth",
    "point",
    "xyz",
    "track",
    "trajectory",
    "robot",
    "pose",
    "keypoint",
    "flow",
    "mask",
    "geometry",
    "reference",
    "truth",
)
REFERENCE_SUFFIXES: Final = {
    ".json",
    ".npy",
    ".npz",
    ".ply",
    ".pcd",
    ".csv",
    ".txt",
    ".png",
    ".exr",
    ".h5",
    ".hdf5",
}


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name} {path}: {error}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _literal_string(value, name=name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _safe_relative(value: object, *, name: str) -> str:
    text = _literal_string(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a confined relative path")
    return path.as_posix()


def validate_request(path: Path, *, repository: Path) -> dict[str, Any]:
    request = _load_json(path, name="source-comparison preflight request")
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != REQUEST_VERSION:
        raise ValueError("unsupported source-comparison preflight request")
    if request.get("issue_number") != 49:
        raise ValueError("preflight request is not bound to issue 49")
    _sha256(request.get("preflight_request_id"), name="preflight_request_id")
    if request.get("source_group_count") != 10:
        raise ValueError("preflight source group count changed")
    if request.get("forbidden_target_group_count") != 12:
        raise ValueError("preflight forbidden target count changed")
    if request.get("expected_case_count") != 40:
        raise ValueError("preflight expected case count changed")
    if any(request.get(name) is not False for name in FALSE_FIELDS):
        raise ValueError("preflight request exceeds its outcome-blind boundary")
    for name in ("source_freeze_path", "comparison_spec_path", "comparison_lock_path"):
        relative = _safe_relative(request.get(name), name=name)
        if not relative.startswith("protocols/locks/") or not relative.endswith(".json"):
            raise ValueError(f"{name} must name a repository lock JSON")
        candidate = repository.joinpath(*PurePosixPath(relative).parts)
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"required merged lock is missing: {relative}")
        request[name] = relative
    _literal_string(request.get("claim_boundary"), name="claim_boundary")
    return request


def _first_string(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if type(value) is str and value:
            return value
    return None


def _first_integer(mapping: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if type(value) is int and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _descriptor_from_mapping(mapping: Mapping[str, Any]) -> tuple[str, str, int] | None:
    path = _first_string(mapping, PATH_KEYS)
    if path is None or not path.lower().endswith(".mp4"):
        return None
    digest = _first_string(mapping, SHA_KEYS)
    byte_count = _first_integer(mapping, BYTE_KEYS)
    if digest is None or byte_count is None:
        return None
    return _safe_relative(path, name="source video path"), _sha256(
        digest, name="source video sha256"
    ), byte_count


def _context_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    value = _first_string(mapping, keys)
    return None if value is None else value.strip()


def _collect_video_descriptors(value: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []

    def visit(node: object, context: dict[str, str]) -> None:
        if type(node) is dict:
            mapping = cast(dict[str, Any], node)
            next_context = dict(context)
            for target, keys in (
                ("group_id", GROUP_KEYS),
                ("role", ROLE_KEYS),
                ("case_id", CASE_KEYS),
                ("view_id", VIEW_KEYS),
            ):
                candidate = _context_value(mapping, keys)
                if candidate is not None:
                    next_context[target] = candidate
            descriptor = _descriptor_from_mapping(mapping)
            if descriptor is not None:
                path, digest, byte_count = descriptor
                found.append(
                    {
                        **next_context,
                        "relative_video_path": path,
                        "video_sha256": digest,
                        "video_byte_count": byte_count,
                    }
                )
            for key, child in mapping.items():
                if type(child) is str and child.lower().endswith(".mp4"):
                    prefix = key.removesuffix("_path").removesuffix("_relative_path")
                    digest = mapping.get(f"{prefix}_sha256", mapping.get("sha256"))
                    byte_count = mapping.get(
                        f"{prefix}_byte_count", mapping.get("byte_count")
                    )
                    if type(digest) is str and type(byte_count) is int:
                        found.append(
                            {
                                **next_context,
                                "relative_video_path": _safe_relative(
                                    child, name="source video path"
                                ),
                                "video_sha256": _sha256(
                                    digest, name="source video sha256"
                                ),
                                "video_byte_count": byte_count,
                            }
                        )
                visit(child, next_context)
        elif type(node) is list:
            for child in cast(list[object], node):
                visit(child, context)

    visit(value, {})
    deduplicated: dict[tuple[str, str], dict[str, object]] = {}
    for record in found:
        key = (
            cast(str, record["relative_video_path"]),
            cast(str, record["video_sha256"]),
        )
        previous = deduplicated.get(key)
        if previous is None or len(record) > len(previous):
            deduplicated[key] = record
    return [deduplicated[key] for key in sorted(deduplicated)]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(arguments: Sequence[str], *, cwd: Path, timeout: int = 60) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, str(error)
    return int(completed.returncode), (completed.stdout + completed.stderr)


def _ffprobe(path: Path) -> dict[str, object]:
    executable = shutil_which("ffprobe")
    if executable is None:
        return {"available": False}
    status, output = _run_text(
        (
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            os.fspath(path),
        ),
        cwd=path.parent,
    )
    if status != 0:
        return {"available": True, "status": status, "error": output[-1000:]}
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {"available": True, "status": status, "error": "invalid ffprobe JSON"}
    streams = payload.get("streams", [])
    return {
        "available": True,
        "status": status,
        "stream": streams[0] if type(streams) is list and streams else {},
    }


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _derive_identity(record: Mapping[str, object]) -> tuple[str, str, str, str]:
    path = PurePosixPath(cast(str, record["relative_video_path"]))
    text = path.as_posix()
    group = cast(str | None, record.get("group_id"))
    if not group:
        match = re.search(r"(\d{3}-[^/]+).*?(?:episode[-_]?0*(\d+))", text, re.I)
        group = (
            f"{match.group(1)}-episode-{int(match.group(2)):04d}"
            if match
            else "unknown-group"
        )
    view = cast(str | None, record.get("view_id"))
    if not view:
        view = path.parent.name
    case = cast(str | None, record.get("case_id"))
    if not case:
        case = f"{group}|{view}"
    role = cast(str | None, record.get("role")) or "unknown-role"
    return group, role, case, view


def _candidate_reference_files(video: Path, *, root: Path) -> list[dict[str, object]]:
    anchors: list[Path] = []
    current = video.parent
    for _ in range(4):
        try:
            current.relative_to(root)
        except ValueError:
            break
        anchors.append(current)
        if current == root:
            break
        current = current.parent
    candidates: dict[str, dict[str, object]] = {}
    for anchor in anchors:
        count = 0
        for path in sorted(anchor.rglob("*")):
            if count >= 1000:
                break
            count += 1
            if path.is_symlink() or not path.is_file() or path == video:
                continue
            lower = path.name.lower()
            if path.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            if not any(token in lower for token in REFERENCE_TOKENS):
                continue
            relative = path.relative_to(root).as_posix()
            candidates[relative] = {
                "relative_path": relative,
                "byte_count": int(path.stat().st_size),
                "suffix": path.suffix.lower(),
                "content_opened": False,
            }
    return [candidates[key] for key in sorted(candidates)]


def _cut3r_surface(checkout: Path, checkpoint: Path) -> dict[str, object]:
    revision_status, revision = _run_text(
        ("git", "rev-parse", "HEAD"), cwd=checkout
    )
    demo_candidates = [checkout / "demo.py"]
    if not demo_candidates[0].is_file():
        demo_candidates = sorted(checkout.glob("**/demo.py"))
    demo = demo_candidates[0] if len(demo_candidates) == 1 else None
    help_status = 127
    help_text = "demo.py not uniquely resolved"
    if demo is not None:
        help_status, help_text = _run_text(
            (sys.executable, os.fspath(demo), "--help"),
            cwd=checkout,
            timeout=120,
        )
    import_status, import_text = _run_text(
        (
            sys.executable,
            "-c",
            (
                "import json; import cv2, numpy, torch; "
                "print(json.dumps({'cv2': cv2.__version__, "
                "'numpy': numpy.__version__, 'torch': torch.__version__, "
                "'cuda_available': bool(torch.cuda.is_available())}, sort_keys=True))"
            ),
        ),
        cwd=checkout,
        timeout=60,
    )
    versions: object = {"error": import_text[-2000:]}
    if import_status == 0:
        try:
            versions = json.loads(import_text.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            versions = {"error": "dependency probe returned invalid JSON"}
    return {
        "checkout_revision_status": revision_status,
        "checkout_revision": revision.strip() if revision_status == 0 else None,
        "demo_relative_path": (
            demo.relative_to(checkout).as_posix() if demo is not None else None
        ),
        "demo_sha256": _file_sha256(demo) if demo is not None else None,
        "demo_help_status": help_status,
        "demo_help": help_text[-20000:],
        "dependency_probe_status": import_status,
        "dependency_versions": versions,
        "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_byte_count": int(checkpoint.stat().st_size),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    request = validate_request(args.request.resolve(), repository=repository)
    freeze = _load_json(
        repository / cast(str, request["source_freeze_path"]),
        name="source-freeze lock",
    )
    spec = _load_json(
        repository / cast(str, request["comparison_spec_path"]),
        name="comparison specification",
    )
    lock = _load_json(
        repository / cast(str, request["comparison_lock_path"]),
        name="comparison lock",
    )
    processed_root = args.processed_root.resolve()
    checkout = args.cut3r_checkout.resolve()
    checkpoint = args.checkpoint.resolve()
    if not processed_root.is_dir() or processed_root.is_symlink():
        raise ValueError("processed source root must be an ordinary directory")
    if not (checkout / ".git").is_dir() or checkout.is_symlink():
        raise ValueError("CUT3R checkout must be an ordinary Git checkout")
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("CUT3R checkpoint must be an ordinary file")

    descriptors = _collect_video_descriptors(freeze)
    cases: list[dict[str, object]] = []
    failures: list[str] = []
    for descriptor in descriptors:
        relative = cast(str, descriptor["relative_video_path"])
        path = processed_root.joinpath(*PurePosixPath(relative).parts).resolve()
        try:
            path.relative_to(processed_root)
        except ValueError:
            failures.append(f"video path escapes processed root: {relative}")
            continue
        group, role, case, view = _derive_identity(descriptor)
        if path.is_symlink() or not path.is_file():
            failures.append(f"source video is missing: {relative}")
            continue
        measured_bytes = int(path.stat().st_size)
        expected_bytes = cast(int, descriptor["video_byte_count"])
        measured_sha = _file_sha256(path)
        expected_sha = cast(str, descriptor["video_sha256"])
        if measured_bytes != expected_bytes or measured_sha != expected_sha:
            failures.append(f"source video identity changed: {relative}")
            continue
        cases.append(
            {
                "group_id": group,
                "role": role,
                "case_id": case,
                "view_id": view,
                "relative_video_path": relative,
                "video_sha256": measured_sha,
                "video_byte_count": measured_bytes,
                "video_probe": _ffprobe(path),
                "candidate_reference_files": _candidate_reference_files(
                    path, root=processed_root
                ),
            }
        )
    cases.sort(key=lambda item: (str(item["group_id"]), str(item["case_id"])))
    group_ids = sorted({str(item["group_id"]) for item in cases})
    role_counts: dict[str, int] = {}
    for item in cases:
        role = str(item["role"])
        role_counts[role] = role_counts.get(role, 0) + 1

    cut3r = _cut3r_surface(checkout, checkpoint)
    expected_cases = cast(int, request["expected_case_count"])
    decision = "source-comparison-preflight-ready"
    if len(cases) != expected_cases:
        failures.append(
            f"resolved case count {len(cases)} differs from expected {expected_cases}"
        )
    if len(group_ids) != cast(int, request["source_group_count"]):
        failures.append(
            f"resolved group count {len(group_ids)} differs from expected 10"
        )
    if cut3r["checkout_revision_status"] != 0:
        failures.append("CUT3R checkout revision could not be read")
    if cut3r["demo_relative_path"] is None:
        failures.append("CUT3R demo.py was not uniquely resolved")
    if cut3r["demo_help_status"] != 0:
        failures.append("CUT3R demo.py --help failed")
    if cut3r["dependency_probe_status"] != 0:
        failures.append("CUT3R Python dependency probe failed")
    if failures:
        decision = "technical-preflight-failure"

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_VERSION,
        "preflight_request_id": request["preflight_request_id"],
        "repository_revision": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_freeze_artifact_id": freeze.get("artifact_id"),
        "comparison_spec_id": spec.get("artifact_id", spec.get("lock_id")),
        "comparison_lock_id": lock.get("lock_id", lock.get("artifact_id")),
        "decision": decision,
        "resolved_case_count": len(cases),
        "resolved_group_count": len(group_ids),
        "role_case_counts": role_counts,
        "cases": cases,
        "cut3r": cut3r,
        "failures": failures,
        "source_rgb_frames_decoded": False,
        "cut3r_inference_executed": False,
        "source_prediction_payloads_opened": False,
        "source_residuals_or_truth_opened": False,
        "candidate_reference_file_contents_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "comparison_execution_authorized": False,
        "claim_boundary": request["claim_boundary"],
    }
    identity = dict(report)
    report["artifact_id"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_id": report["artifact_id"],
                "decision": report["decision"],
                "resolved_case_count": report["resolved_case_count"],
                "resolved_group_count": report["resolved_group_count"],
                "failure_count": len(report["failures"]),
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "source-comparison-preflight-ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
