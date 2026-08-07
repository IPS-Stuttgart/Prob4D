"""Immutable writer for portable sparse gauge-tree prior artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ._gauge_tree_prior_artifact_json import relative_payload_path
from ._gauge_tree_prior_artifact_schema import (
    factor_arrays,
    manifest_record,
    sha256_file,
)
from .gauge_tree_prior import GaugeTreeSquareRootPriorV1


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    """Atomically publish one closed temporary file without replacing a peer."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise
    except OSError as error:
        raise OSError(f"failed to publish immutable output {destination}") from error


def write_gauge_tree_prior(
    prior: GaugeTreeSquareRootPriorV1,
    manifest_path: str | os.PathLike[str],
    *,
    payload_path: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Write one immutable sparse-prior manifest and checksum-bound payload."""

    if not isinstance(prior, GaugeTreeSquareRootPriorV1):
        raise TypeError("prior must be a GaugeTreeSquareRootPriorV1")
    manifest = Path(manifest_path)
    payload = (
        Path(payload_path)
        if payload_path is not None
        else manifest.with_suffix(".npz")
    )
    if manifest.resolve() == payload.resolve():
        raise ValueError("manifest and payload paths must differ")
    relative_payload = relative_payload_path(manifest, payload)
    existing = [path for path in (manifest, payload) if path.exists()]
    if existing:
        raise FileExistsError(
            "gauge-tree prior output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{prior.prior_id[:16]}"
    temporary_payload = payload.with_name(f".{payload.name}.tmp-{token}")
    temporary_manifest = manifest.with_name(f".{manifest.name}.tmp-{token}")
    try:
        with temporary_payload.open("xb") as stream:
            np.savez_compressed(stream, **factor_arrays(prior))
            stream.flush()
            os.fsync(stream.fileno())
        record = manifest_record(
            prior,
            payload_relative_path=relative_payload,
            payload_sha256=sha256_file(temporary_payload),
        )
        encoded = (
            json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        with temporary_manifest.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        payload_published = False
        try:
            _publish_no_replace(temporary_payload, payload)
            payload_published = True
            _publish_no_replace(temporary_manifest, manifest)
        except BaseException:
            if payload_published and not manifest.exists():
                payload.unlink(missing_ok=True)
            raise
    finally:
        temporary_payload.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return manifest, payload
