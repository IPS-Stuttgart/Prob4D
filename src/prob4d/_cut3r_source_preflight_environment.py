"""Retained filesystem/provider inspection for the CUT3R source preflight."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from ._cut3r_source_preflight_common import (
    _file_sha256,
    _revision,
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


def _ffprobe(path: Path) -> dict[str, object]:
    from shutil import which

    executable = which("ffprobe")
    if executable is None:
        return {"available": False, "status": 127, "error": "ffprobe is unavailable"}
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
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        return {"available": True, "status": status, "error": "video stream was not unique"}
    return {"available": True, "status": status, "stream": streams[0]}


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


def _cut3r_surface(checkout: Path, checkpoint: Path) -> dict[str, object]:
    revision_status, revision = _run_text(("git", "rev-parse", "HEAD"), cwd=checkout)
    remote_status, remote = _run_text(
        ("git", "remote", "get-url", "origin"),
        cwd=checkout,
    )
    worktree_status, worktree = _run_text(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=checkout,
    )
    demo_candidates = [checkout / "demo.py"]
    if not demo_candidates[0].is_file() or demo_candidates[0].is_symlink():
        demo_candidates = sorted(
            path for path in checkout.glob("**/demo.py") if path.is_file() and not path.is_symlink()
        )
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
    checkpoint_sha = _file_sha256(checkpoint)
    return {
        "checkout_revision_status": revision_status,
        "checkout_revision": revision.strip() if revision_status == 0 else None,
        "origin_status": remote_status,
        "origin_url": remote.strip() if remote_status == 0 else None,
        "origin_repository": (
            _github_repository_from_remote(remote) if remote_status == 0 else None
        ),
        "tracked_worktree_status": worktree_status,
        "tracked_worktree_clean": worktree_status == 0 and not worktree.strip(),
        "demo_relative_path": (demo.relative_to(checkout).as_posix() if demo is not None else None),
        "demo_sha256": _file_sha256(demo) if demo is not None else None,
        "demo_help_status": help_status,
        "demo_help": help_text[-20000:],
        "dependency_probe_status": import_status,
        "dependency_versions": versions,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_byte_count": int(checkpoint.stat().st_size),
    }


def _repository_revision(repository: Path) -> str:
    status, output = _run_text(("git", "rev-parse", "HEAD"), cwd=repository)
    if status != 0:
        raise ValueError("Prob4D repository revision could not be read")
    return _revision(output.strip(), name="Prob4D repository revision")
