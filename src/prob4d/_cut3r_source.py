"""Strict source-tree loading and identities for CUT3R imports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypedDict

import numpy as np

_MIB: Final = 1024**2
_ADAPTER_IMPLEMENTATION_MEMBERS: Final = (
    "_cut3r_cli.py",
    "_cut3r_limits.py",
    "_cut3r_source.py",
    "_cut3r_window.py",
    "cut3r_provider_adapter.py",
)


class _SourceMemberDescriptor(TypedDict):
    """Content descriptor for one exact CUT3R output member."""

    path: str
    sha256: str
    byte_count: int


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_id(domain: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(_MIB), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read CUT3R source member {path.name!r}") from error
    return digest.hexdigest()


def _file_descriptor(path: Path, *, root: Path) -> _SourceMemberDescriptor:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"CUT3R source member {path.name!r} must be an ordinary file")
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("CUT3R source member escapes the declared output root") from error
    return {
        "path": PurePosixPath(*relative.parts).as_posix(),
        "sha256": _file_sha256(path),
        "byte_count": int(path.stat().st_size),
    }


def _validated_source_directory(root: Path, name: str, suffix: str) -> dict[int, Path]:
    directory = root / name
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"CUT3R output requires an ordinary {name!r} directory")
    indexed: dict[int, Path] = {}
    for member in sorted(directory.iterdir()):
        if member.is_symlink() or not member.is_file():
            raise ValueError(f"CUT3R {name!r} directory contains a non-regular member")
        if member.suffix != suffix or len(member.stem) != 6 or not member.stem.isdigit():
            raise ValueError(f"CUT3R {name!r} members must use six-digit {suffix} filenames")
        index = int(member.stem)
        if index in indexed:
            raise ValueError(f"duplicate CUT3R frame index {index} in {name!r}")
        indexed[index] = member
    if not indexed:
        raise ValueError(f"CUT3R {name!r} directory is empty")
    if set(indexed) != set(range(len(indexed))):
        raise ValueError(f"CUT3R {name!r} frame indices must be contiguous from zero")
    return indexed


def _validated_source_members(
    root: Path,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("CUT3R output root must be an ordinary directory")
    depth = _validated_source_directory(root, "depth", ".npy")
    confidence = _validated_source_directory(root, "conf", ".npy")
    camera = _validated_source_directory(root, "camera", ".npz")
    if set(depth) != set(confidence) or set(depth) != set(camera):
        raise ValueError("CUT3R depth, confidence, and camera frame sets disagree")
    return depth, confidence, camera


def _source_tree_byte_count(member_groups: Sequence[Mapping[int, Path]]) -> int:
    total = 0
    for members in member_groups:
        for path in members.values():
            try:
                total += int(path.stat().st_size)
            except OSError as error:
                raise ValueError(f"cannot stat CUT3R source member {path.name!r}") from error
    return total


def _load_npy(
    path: Path,
    *,
    label: str,
) -> tuple[np.ndarray, _SourceMemberDescriptor]:
    root = path.parents[1]
    before = _file_descriptor(path, root=root)
    try:
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load CUT3R {label} member {path.name!r}") from error
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"CUT3R {label} member {path.name!r} must be one NPY array")
    after = _file_descriptor(path, root=root)
    if before != after:
        raise ValueError(f"CUT3R {label} member changed while it was being read")
    return loaded, before


def _load_camera(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, _SourceMemberDescriptor]:
    root = path.parents[1]
    before = _file_descriptor(path, root=root)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"pose", "intrinsics"}:
                raise ValueError("camera archive fields must be exactly pose and intrinsics")
            pose = np.asarray(archive["pose"], dtype=np.float64)
            intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load CUT3R camera member {path.name!r}: {error}") from error
    after = _file_descriptor(path, root=root)
    if before != after:
        raise ValueError("CUT3R camera member changed while it was being read")
    return pose, intrinsics, before


def _verify_source_descriptors(
    root: Path,
    descriptors: Sequence[_SourceMemberDescriptor],
) -> None:
    for descriptor in descriptors:
        relative = PurePosixPath(descriptor["path"])
        candidate = root.joinpath(*relative.parts)
        if _file_descriptor(candidate, root=root) != descriptor:
            raise ValueError("CUT3R source tree changed after canonical loading")


def _adapter_implementation_sha256() -> str:
    """Hash the complete adapter implementation, including internal helpers."""

    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in sorted(_ADAPTER_IMPLEMENTATION_MEMBERS):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(_MIB), b""):
                    digest.update(block)
        except OSError as error:
            raise ValueError(f"cannot read CUT3R adapter implementation member {name!r}") from error
    return digest.hexdigest()
