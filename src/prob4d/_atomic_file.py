"""Durable same-directory publication primitives for immutable artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _fsync_directory(path: Path) -> None:
    """Best-effort directory synchronization after changing one directory entry."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def publish_temporary_file(
    temporary: str | Path,
    destination: str | Path,
    *,
    overwrite: bool,
) -> None:
    """Publish one complete same-directory temporary file atomically.

    With ``overwrite=False``, hard-link creation is the decisive no-clobber
    operation. Concurrent writers therefore cannot both publish successfully.
    The caller retains ownership of the temporary file when publication fails.
    """

    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    temporary_path = Path(temporary)
    destination_path = Path(destination)
    if temporary_path.parent.resolve() != destination_path.parent.resolve():
        raise ValueError("temporary and destination must share one directory")

    # Windows requires a writable descriptor for fsync. The temporary file is
    # owned by this publication path, so opening it read/write changes no bytes
    # while preserving the same durability barrier on every supported platform.
    with temporary_path.open("rb+") as stream:
        os.fsync(stream.fileno())
    if overwrite:
        os.replace(temporary_path, destination_path)
    else:
        try:
            os.link(temporary_path, destination_path)
        except FileExistsError:
            raise FileExistsError(destination_path) from None
        temporary_path.unlink()
    _fsync_directory(destination_path.parent)


def atomic_write_bytes(
    path: str | Path,
    content: bytes,
    *,
    overwrite: bool,
) -> None:
    """Publish complete bytes atomically with an optional no-clobber guarantee."""

    if type(content) is not bytes:
        raise TypeError("content must be bytes")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        publish_temporary_file(temporary, destination, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    path: str | Path,
    content: str,
    *,
    overwrite: bool,
) -> None:
    """UTF-8 text wrapper around :func:`atomic_write_bytes`."""

    if type(content) is not str:
        raise TypeError("content must be a string")
    atomic_write_bytes(path, content.encode("utf-8"), overwrite=overwrite)
