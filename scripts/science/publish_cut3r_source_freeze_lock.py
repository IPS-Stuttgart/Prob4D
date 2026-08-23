#!/usr/bin/env python3
"""Publish exact retained CUT3R source-freeze locks from a workflow artifact.

The command discovers a nonexpired Actions artifact for the exact source request,
verifies the uploaded checksum manifest and target-closed decision, then copies
only the retained JSON bytes into their preregistered repository paths. It never
executes CUT3R or opens source residuals, truth, confirmation, or target payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

REQUEST_SCHEMA: Final = "prob4d.cut3r-source-freeze-publication-request"
REQUEST_VERSION: Final = 1
RECEIPT_SCHEMA: Final = "prob4d.cut3r-source-freeze-publication-receipt"
RECEIPT_VERSION: Final = 1
EXPECTED_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "publication_request_id",
        "source_request_id",
        "artifact_name_prefix",
        "required_decision",
        "source_group_count",
        "forbidden_target_group_count",
        "source_rgb_frames_decoded",
        "source_residuals_or_truth_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "output_paths",
        "claim_boundary",
    }
)
OUTPUT_NAMES: Final = {
    "source_freeze": "cut3r-deform360-source-freeze.json",
    "comparison_spec": "cut3r-comparison-spec.json",
    "comparison_lock": "cut3r-comparison-lock.json",
    "comparison_summary": "cut3r-comparison-summary.json",
    "execution_summary": "execution-summary.json",
}
FALSE_EXECUTION_FIELDS: Final = (
    "source_rgb_frames_decoded",
    "source_prediction_payloads_opened",
    "source_residuals_or_truth_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "comparison_execution_authorized",
)
FALSE_FREEZE_BOUNDARY_FIELDS: Final = (
    "source_rgb_frames_decoded",
    "source_prediction_payloads_opened",
    "source_residuals_or_truth_opened",
    "source_future_geometry_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "downstream_physical_innovations_opened",
)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load {name} from {path}: {error}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(dict[str, Any], value)


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _sha256_string(value: object, *, name: str) -> str:
    result = _literal_string(value, name=name)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _safe_repository_path(value: object, *, name: str) -> str:
    text = _literal_string(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must be a POSIX repository path")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a confined repository path")
    if path.parts[:2] != ("protocols", "locks") or path.suffix != ".json":
        raise ValueError(f"{name} must be a JSON path under protocols/locks")
    return path.as_posix()


def validate_request(path: Path) -> dict[str, Any]:
    request = _load_json(path, name="source-freeze publication request")
    if set(request) != EXPECTED_REQUEST_FIELDS:
        missing = sorted(EXPECTED_REQUEST_FIELDS - set(request))
        extra = sorted(set(request) - EXPECTED_REQUEST_FIELDS)
        raise ValueError(
            f"publication request fields changed; missing={missing}, extra={extra}"
        )
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != REQUEST_VERSION:
        raise ValueError("unsupported source-freeze publication request")
    publication_id = _sha256_string(
        request["publication_request_id"], name="publication_request_id"
    )
    source_id = _sha256_string(request["source_request_id"], name="source_request_id")
    if publication_id != source_id:
        raise ValueError("publication request must bind the exact source request ID")
    prefix = _literal_string(request["artifact_name_prefix"], name="artifact_name_prefix")
    if not prefix.startswith(f"cut3r-source-freeze-v2-{source_id[:12]}-"):
        raise ValueError("artifact prefix does not bind the source request ID")
    if request["required_decision"] != "source-support-freeze-ready":
        raise ValueError("publication request must require the support-positive decision")
    if request["source_group_count"] != 10:
        raise ValueError("publication request source group count changed")
    if request["forbidden_target_group_count"] != 12:
        raise ValueError("publication request forbidden target count changed")
    for name in (
        "source_rgb_frames_decoded",
        "source_residuals_or_truth_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
    ):
        if request[name] is not False:
            raise ValueError("publication request exceeds its target-closed boundary")
    output_paths = request["output_paths"]
    if type(output_paths) is not dict or set(output_paths) != set(OUTPUT_NAMES):
        raise ValueError("publication output path inventory changed")
    normalized = {
        key: _safe_repository_path(value, name=f"output_paths.{key}")
        for key, value in cast(dict[str, object], output_paths).items()
    }
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("publication output paths must be unique")
    request["output_paths"] = normalized
    _literal_string(request["claim_boundary"], name="claim_boundary")
    return request


def _request_json(url: str, *, token: str, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "prob4d-source-freeze-publication-v1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"GitHub request failed for {url}: {error}") from error


def _list_artifacts(
    *, api_url: str, repository: str, token: str, prefix: str
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{api_url}/repos/{repository}/actions/artifacts?per_page=100&page={page}"
        payload = json.loads(
            _request_json(
                url,
                token=token,
                accept="application/vnd.github+json",
            )
        )
        artifacts = payload.get("artifacts")
        if type(artifacts) is not list:
            raise RuntimeError("GitHub artifact listing changed shape")
        for artifact in artifacts:
            if type(artifact) is not dict:
                continue
            name = artifact.get("name")
            if (
                type(name) is str
                and name.startswith(prefix)
                and artifact.get("expired") is False
            ):
                matches.append(cast(dict[str, Any], artifact))
        if len(artifacts) < 100:
            break
        page += 1
    matches.sort(
        key=lambda item: (str(item.get("created_at", "")), int(item.get("id", 0))),
        reverse=True,
    )
    return matches


def _safe_extract_zip(payload: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise ValueError("workflow artifact contains an unsafe path")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("workflow artifact contains a symbolic link")
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                with archive.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)


def _verify_checksum_manifests(root: Path) -> None:
    manifests = tuple(sorted(root.rglob("SHA256SUMS")))
    if not manifests:
        raise ValueError("workflow artifact has no SHA256SUMS manifest")
    verified = 0
    for manifest in manifests:
        base = manifest.parent
        lines = manifest.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line:
                continue
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if match is None:
                raise ValueError(f"invalid checksum row {index + 1} in {manifest}")
            expected, relative_text = match.groups()
            relative = PurePosixPath(relative_text)
            if relative.is_absolute() or any(
                part in {"", ".", ".."} for part in relative.parts
            ):
                raise ValueError("checksum manifest contains an unsafe path")
            candidate = base.joinpath(*relative.parts)
            if candidate.name == "SHA256SUMS":
                continue
            if not candidate.is_file() or candidate.is_symlink():
                raise ValueError(f"checksummed artifact member is missing: {relative_text}")
            measured = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if measured != expected:
                raise ValueError(f"artifact checksum mismatch for {relative_text}")
            verified += 1
    if verified == 0:
        raise ValueError("workflow artifact checksum manifest verifies no members")


def _unique_named_file(root: Path, name: str) -> Path:
    matches = tuple(path for path in root.rglob(name) if path.is_file() and not path.is_symlink())
    if len(matches) != 1:
        raise ValueError(f"workflow artifact must contain exactly one {name}; found {len(matches)}")
    return matches[0]


def validate_artifact(
    root: Path,
    *,
    request: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    _verify_checksum_manifests(root)
    files = {
        key: _unique_named_file(root, filename)
        for key, filename in OUTPUT_NAMES.items()
    }
    execution = _load_json(files["execution_summary"], name="execution summary")
    if execution.get("request_id") != request["source_request_id"]:
        raise ValueError("execution summary is bound to a different source request")
    if execution.get("decision") != request["required_decision"]:
        raise ValueError("source-freeze artifact is not support positive")
    if execution.get("source_group_count") != request["source_group_count"]:
        raise ValueError("execution summary source group count changed")
    if execution.get("forbidden_target_group_count") != request[
        "forbidden_target_group_count"
    ]:
        raise ValueError("execution summary forbidden target count changed")
    if any(execution.get(name) is not False for name in FALSE_EXECUTION_FIELDS):
        raise ValueError("execution summary exceeds the target-closed boundary")

    freeze = _load_json(files["source_freeze"], name="source-freeze artifact")
    if freeze.get("decision") != request["required_decision"]:
        raise ValueError("source-freeze artifact decision changed")
    if freeze.get("source_group_count") != request["source_group_count"]:
        raise ValueError("source-freeze artifact group count changed")
    if freeze.get("forbidden_target_group_count") != request[
        "forbidden_target_group_count"
    ]:
        raise ValueError("source-freeze artifact target count changed")
    boundary = freeze.get("information_boundary")
    if type(boundary) is not dict or any(
        cast(dict[str, Any], boundary).get(name) is not False
        for name in FALSE_FREEZE_BOUNDARY_FIELDS
    ):
        raise ValueError("source-freeze information boundary was exceeded")

    for key in ("comparison_spec", "comparison_lock", "comparison_summary"):
        _load_json(files[key], name=key.replace("_", " "))
    return files, execution


def _publish_exact_files(
    *,
    repository_root: Path,
    request: Mapping[str, Any],
    files: Mapping[str, Path],
    artifact: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    published: dict[str, dict[str, object]] = {}
    output_paths = cast(dict[str, str], request["output_paths"])
    for key, relative in output_paths.items():
        destination = repository_root.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != files[key].read_bytes():
                raise FileExistsError(f"refusing to replace different lock bytes: {relative}")
        else:
            destination.write_bytes(files[key].read_bytes())
        published[key] = {
            "path": relative,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "byte_count": destination.stat().st_size,
        }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_VERSION,
        "publication_request_id": request["publication_request_id"],
        "source_request_id": request["source_request_id"],
        "source_workflow_artifact_id": int(artifact["id"]),
        "source_workflow_artifact_name": artifact["name"],
        "source_workflow_run_id": int(artifact["workflow_run"]["id"]),
        "source_decision": execution["decision"],
        "source_freeze_artifact_id": execution["freeze_artifact_id"],
        "published": published,
        "source_group_count": request["source_group_count"],
        "forbidden_target_group_count": request["forbidden_target_group_count"],
        "source_rgb_frames_decoded": False,
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "claim_boundary": request["claim_boundary"],
    }
    receipt_path = (
        repository_root
        / "protocols/locks/cut3r_deform360_source_freeze_publication_v2.json"
    )
    encoded = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if receipt_path.exists() and receipt_path.read_bytes() != encoded:
        raise FileExistsError("refusing to replace a different publication receipt")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(encoded)
    return receipt


def discover_and_publish(args: argparse.Namespace) -> int:
    request = validate_request(args.request.resolve())
    token = _literal_string(args.token, name="GitHub token")
    candidates = _list_artifacts(
        api_url=args.api_url.rstrip("/"),
        repository=args.repository_name,
        token=token,
        prefix=request["artifact_name_prefix"],
    )
    if not candidates:
        raise RuntimeError("no nonexpired source-freeze workflow artifact matches the request")

    selected: dict[str, Any] | None = None
    selected_files: dict[str, Path] | None = None
    selected_execution: dict[str, Any] | None = None
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for artifact in candidates:
        artifact_id = int(artifact["id"])
        candidate_root = workspace / f"artifact-{artifact_id}"
        candidate_root.mkdir(parents=True, exist_ok=False)
        try:
            payload = _request_json(
                f"{args.api_url.rstrip('/')}/repos/{args.repository_name}/actions/artifacts/{artifact_id}/zip",
                token=token,
                accept="application/vnd.github+json",
            )
            _safe_extract_zip(payload, candidate_root)
            files, execution = validate_artifact(candidate_root, request=request)
        except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
            failures.append(f"artifact {artifact_id}: {error}")
            continue
        selected = artifact
        selected_files = files
        selected_execution = execution
        break
    if selected is None or selected_files is None or selected_execution is None:
        raise RuntimeError(
            "no candidate artifact passed validation: " + "; ".join(failures)
        )

    receipt = _publish_exact_files(
        repository_root=args.repository_root.resolve(),
        request=request,
        files=selected_files,
        artifact=selected,
        execution=selected_execution,
    )
    if args.github_output is not None:
        branch = f"evidence/cut3r-source-freeze-{str(receipt['source_freeze_artifact_id'])[:12]}"
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"branch={branch}\n")
            output.write(f"artifact_id={receipt['source_workflow_artifact_id']}\n")
            output.write(f"run_id={receipt['source_workflow_run_id']}\n")
            output.write(f"freeze_artifact_id={receipt['source_freeze_artifact_id']}\n")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-request")
    verify.add_argument("--request", type=Path, required=True)

    publish = subparsers.add_parser("discover-and-publish")
    publish.add_argument("--request", type=Path, required=True)
    publish.add_argument("--repository-root", type=Path, required=True)
    publish.add_argument("--repository-name", required=True)
    publish.add_argument("--api-url", required=True)
    publish.add_argument("--token", required=True)
    publish.add_argument("--workspace", type=Path, required=True)
    publish.add_argument("--github-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-request":
        request = validate_request(args.request.resolve())
        print(
            json.dumps(
                {
                    "publication_request_id": request["publication_request_id"],
                    "artifact_name_prefix": request["artifact_name_prefix"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "discover-and-publish":
        return discover_and_publish(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
