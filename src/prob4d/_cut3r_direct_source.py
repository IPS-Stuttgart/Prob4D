"""Strict source loading and implementation identity for direct CUT3R point maps."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ._cut3r_source import (
    _MIB,
    _SourceMemberDescriptor,
    _source_tree_byte_count,
    _validated_source_directory,
    _verify_source_descriptors,
)

_DIRECT_ADAPTER_IMPLEMENTATION_MEMBERS: Final = (
    "_cut3r_direct_cli.py",
    "_cut3r_direct_source.py",
    "_cut3r_direct_window.py",
    "_cut3r_limits.py",
    "_cut3r_source.py",
    "_cut3r_window.py",
    "cut3r_direct_provider_adapter.py",
)


def _validated_direct_source_members(
    root: Path,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    """Return matching direct-point, confidence, and camera member maps."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("CUT3R direct output root must be an ordinary directory")
    points = _validated_source_directory(root, "points", ".npy")
    confidence = _validated_source_directory(root, "conf", ".npy")
    camera = _validated_source_directory(root, "camera", ".npz")
    if set(points) != set(confidence) or set(points) != set(camera):
        raise ValueError("CUT3R points, confidence, and camera frame sets disagree")
    return points, confidence, camera


def _direct_source_tree_byte_count(
    points: dict[int, Path],
    confidence: dict[int, Path],
    camera: dict[int, Path],
) -> int:
    return _source_tree_byte_count((points, confidence, camera))


def _verify_direct_source_descriptors(
    root: Path,
    descriptors: tuple[_SourceMemberDescriptor, ...],
) -> None:
    _verify_source_descriptors(root, descriptors)


def _direct_adapter_implementation_sha256() -> str:
    """Hash every module that defines the direct-point-map import semantics."""

    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in sorted(_DIRECT_ADAPTER_IMPLEMENTATION_MEMBERS):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(_MIB), b""):
                    digest.update(block)
        except OSError as error:
            raise ValueError(
                f"cannot read CUT3R direct-adapter implementation member {name!r}"
            ) from error
    return digest.hexdigest()


__all__ = [
    "_direct_adapter_implementation_sha256",
    "_direct_source_tree_byte_count",
    "_validated_direct_source_members",
    "_verify_direct_source_descriptors",
]
