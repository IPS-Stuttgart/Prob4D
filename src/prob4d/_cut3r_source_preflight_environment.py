"""Retained filesystem/provider inspection for the CUT3R source preflight."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from ._cut3r_source_preflight_common import (
    _confined_regular_file,
    _file_sha256,
    _revision,
    _stat_identity,
)

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
    return int(completed.returncode), completed.stdout + completed.stderr


def _sanitize_text(text: str, replacements: Mapping[str, str]) -> str:
    sanitized = text
    for source in sorted((item for item in replacements if item), key=len, reverse=True):
        sanitized = sanitized.replace(source, replacements[source])
    return sanitized


def _text_evidence(text: str, replacements: Mapping[str, str]) -> dict[str, object]:
    encoded = _sanitize_text(text, replacements).encode("utf-8", errors="replace")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_count": len(encoded),
        "content_retained": False,
    }


def _ffprobe(path: Path) -> dict[str, object]:
    from shutil import which

    replacements = {
        os.fspath(path): "<SOURCE_VIDEO>",
        os.fspath(path.resolve(strict=True)): "<SOURCE_VIDEO>",
    }
    executable = which("ffprobe")
    if executable is None:
        return {"available": False, "status": 127, "error": "ffprobe-unavailable"}
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
    output_evidence = _text_evidence(output, replacements)
    if status != 0:
        return {
            "available": True,
            "status": status,
            "error": "ffprobe-failed",
            "output_evidence": output_evidence,
        }
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {
            "available": True,
            "status": status,
            "error": "invalid-ffprobe-json",
            "output_evidence": output_evidence,
        }
    streams = payload.get("streams", [])
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        return {
            "available": True,
            "status": status,
            "error": "video-stream-not-unique",
            "output_evidence": output_evidence,
        }
    return {
        "available": True,
        "status": status,
        "stream": streams[0],
        "output_evidence": output_evidence,
    }


def _candidate_reference_files(
    episode: Path,
    *,
    root: Path,
    maximum_entries: int = 5000,
) -> list[dict[str, object]]:
    root_resolved = root.resolve(strict=True)
    episode_resolved = episode.resolve(strict=True)
    episode_resolved.relative_to(root_resolved)
    candidates: dict[str, dict[str, object]] = {}
    visited = 0
    for current, directory_names, file_names in os.walk(episode_resolved, followlinks=False):
        current_path = Path(current)
        directory_names[:] = [
            name for name in sorted(directory_names) if not (current_path / name).is_symlink()
        ]
        for name in sorted(file_names):
            visited += 1
            if visited > maximum_entries:
                raise ValueError(
                    f"source-episode reference inventory exceeds {maximum_entries} entries"
                )
            path = current_path / name
            if path.is_symlink():
                continue
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            lower = path.name.lower()
            if path.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            if not any(token in lower for token in REFERENCE_TOKENS):
                continue
            relative = path.relative_to(root_resolved).as_posix()
            candidates[relative] = {
                "relative_path": relative,
                "byte_count": int(metadata.st_size),
                "suffix": path.suffix.lower(),
                "content_opened": False,
            }
    return [candidates[key] for key in sorted(candidates)]


def _github_repository_from_remote(value: str) -> str | None:
    text = value.strip()
    for prefix in ("https://github.com/", "ssh://git@github.com/", "git@github.com:"):
        if text.startswith(prefix):
            repository = text.removeprefix(prefix).removesuffix(".git").strip("/")
            return repository if repository.count("/") == 1 else None
    return None


def _tracked_demo(checkout: Path) -> tuple[Path | None, int, dict[str, object]]:
    status, output = _run_text(
        ("git", "ls-files", "--", "demo.py", ":(glob)**/demo.py"),
        cwd=checkout,
    )
    replacements = {os.fspath(checkout): "<CUT3R_CHECKOUT>"}
    if status != 0:
        return (
            None,
            status,
            {
                "tracked_candidate_count": 0,
                "output_evidence": _text_evidence(output, replacements),
            },
        )
    relative_paths = sorted({line.strip() for line in output.splitlines() if line.strip()})
    if len(relative_paths) != 1:
        return (
            None,
            status,
            {
                "tracked_candidate_count": len(relative_paths),
                "output_evidence": _text_evidence(output, replacements),
            },
        )
    try:
        demo = _confined_regular_file(
            checkout,
            relative_paths[0],
            name="tracked CUT3R demo.py",
        )
    except ValueError as error:
        return (
            None,
            1,
            {
                "tracked_candidate_count": 1,
                "output_evidence": _text_evidence(str(error), replacements),
            },
        )
    return (
        demo,
        status,
        {
            "tracked_candidate_count": 1,
            "output_evidence": _text_evidence(output, replacements),
        },
    )


def _cut3r_surface(
    checkout: Path,
    checkpoint: Path,
    *,
    expected_repository: str | None = None,
    expected_revision: str | None = None,
    expected_checkpoint_filename: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_byte_count: int | None = None,
) -> dict[str, object]:
    replacements = {
        os.fspath(checkout): "<CUT3R_CHECKOUT>",
        os.fspath(checkout.resolve(strict=True)): "<CUT3R_CHECKOUT>",
        os.fspath(checkpoint): "<CUT3R_CHECKPOINT>",
        os.fspath(checkpoint.resolve(strict=True)): "<CUT3R_CHECKPOINT>",
    }
    revision_status, revision_output = _run_text(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
    )
    remote_status, remote_output = _run_text(
        ("git", "remote", "get-url", "origin"),
        cwd=checkout,
    )
    worktree_status, worktree_output = _run_text(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=checkout,
    )
    checkout_revision = revision_output.strip() if revision_status == 0 else None
    origin_repository = (
        _github_repository_from_remote(remote_output) if remote_status == 0 else None
    )
    worktree_clean = worktree_status == 0 and not worktree_output.strip()

    checkpoint_before = checkpoint.stat()
    checkpoint_sha = _file_sha256(checkpoint)
    checkpoint_after = checkpoint.stat()
    if _stat_identity(checkpoint_before) != _stat_identity(checkpoint_after):
        raise ValueError("CUT3R checkpoint changed during provider inspection")
    checkpoint_byte_count = int(checkpoint_before.st_size)

    executable_probe_authorized = (
        expected_repository is not None
        and expected_revision is not None
        and expected_checkpoint_filename is not None
        and expected_checkpoint_sha256 is not None
        and expected_checkpoint_byte_count is not None
        and checkout_revision == expected_revision
        and origin_repository == expected_repository
        and worktree_clean
        and checkpoint.name == expected_checkpoint_filename
        and checkpoint_sha == expected_checkpoint_sha256
        and checkpoint_byte_count == expected_checkpoint_byte_count
    )
    demo: Path | None = None
    demo_resolution_status = 126
    authorization_text = "executable probe blocked by frozen provider identity"
    demo_resolution: dict[str, object] = {
        "tracked_candidate_count": 0,
        "output_evidence": _text_evidence(authorization_text, replacements),
    }
    help_status = 127
    help_text = authorization_text
    import_status = 127
    import_text = authorization_text
    versions: object = {"error": "dependency-probe-not-authorized"}

    if executable_probe_authorized:
        demo, demo_resolution_status, demo_resolution = _tracked_demo(checkout)
        help_status = 127
        help_text = "tracked demo.py not uniquely resolved"
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
        versions = {"error": "dependency-probe-failed"}
        if import_status == 0:
            try:
                versions = json.loads(import_text.splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                versions = {"error": "dependency-probe-invalid-json"}

    return {
        "checkout_revision_status": revision_status,
        "checkout_revision": checkout_revision,
        "origin_status": remote_status,
        "origin_repository": origin_repository,
        "worktree_status": worktree_status,
        "worktree_clean_including_untracked": worktree_clean,
        "executable_probe_authorized": executable_probe_authorized,
        "demo_resolution_status": demo_resolution_status,
        "demo_resolution": demo_resolution,
        "demo_relative_path": (demo.relative_to(checkout).as_posix() if demo is not None else None),
        "demo_sha256": _file_sha256(demo) if demo is not None else None,
        "demo_help_status": help_status,
        "demo_help_output_evidence": _text_evidence(help_text, replacements),
        "dependency_probe_status": import_status,
        "dependency_versions": versions,
        "dependency_probe_output_evidence": _text_evidence(import_text, replacements),
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_byte_count": checkpoint_byte_count,
    }


def _repository_revision(repository: Path) -> str:
    status, output = _run_text(("git", "rev-parse", "HEAD"), cwd=repository)
    if status != 0:
        raise ValueError("Prob4D repository revision could not be read")
    return _revision(output.strip(), name="Prob4D repository revision")
