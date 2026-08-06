"""Race-safe I/O and CLI for the canonical prediction-provider manifest.

The scientific and portable schema remains ``PredictionProviderManifestV1``.
This module adds serialized, idempotent persistence; stable-byte verification;
and safe generic or MotionCrafter import without introducing a second contract.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from .data import PredictionWindow
from .prediction_provider_manifest import (
    PredictionProviderManifestV1,
    _resolved_member,
    _sha256_file,
    import_motioncrafter_prediction_manifest as _import_motioncrafter_manifest,
    load_prediction_provider_manifest as _load_prediction_provider_manifest,
)

PREDICTION_PROVIDER_IO_SCHEMA: Final = "prob4d.prediction-provider-io"
PREDICTION_PROVIDER_IO_VERSION: Final = 1


def _file_signature(path: Path, *, name: str) -> tuple[int, int, int, int]:
    try:
        information = path.stat()
    except OSError as error:
        raise ValueError(f"cannot stat {name}") from error
    return (
        int(information.st_dev),
        int(information.st_ino),
        int(information.st_size),
        int(information.st_mtime_ns),
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish_new_bytes(path: Path, payload: bytes) -> None:
    """Publish a new path without replacing a racing destination."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                "prediction-provider manifest appeared during publication"
            ) from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_prediction_provider_manifest(
    path: str | Path,
) -> PredictionProviderManifestV1:
    """Load one manifest from a stable, non-symlinked byte snapshot."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("prediction-provider manifest must not be a symbolic link")
    manifest_path = supplied.resolve()
    before = _file_signature(manifest_path, name="prediction-provider manifest")
    sha256 = _sha256_file(manifest_path)
    manifest = _load_prediction_provider_manifest(manifest_path)
    if (
        _file_signature(manifest_path, name="prediction-provider manifest") != before
        or _sha256_file(manifest_path) != sha256
    ):
        raise ValueError("prediction-provider manifest changed while it was loaded")
    return manifest


def save_prediction_provider_manifest(
    path: str | Path,
    manifest: PredictionProviderManifestV1,
) -> Path:
    """Persist one canonical manifest under an exclusive fail-closed lock."""

    if not isinstance(manifest, PredictionProviderManifestV1):
        raise TypeError("manifest must be a PredictionProviderManifestV1")
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("prediction-provider manifest destination is a symbolic link")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = manifest.to_record()
    payload = (
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    lock_path = destination.with_name(f".{destination.name}.lock")
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"prediction-provider manifest writer is already active: {lock_path}"
        ) from error
    try:
        with os.fdopen(lock_descriptor, "w", encoding="ascii") as lock_stream:
            lock_stream.write(f"{os.getpid()}\n")
            lock_stream.flush()
            os.fsync(lock_stream.fileno())
        if destination.is_symlink():
            raise ValueError(
                "prediction-provider manifest destination is a symbolic link"
            )
        if destination.exists():
            existing = load_prediction_provider_manifest(destination)
            if existing.to_record() != record:
                raise ValueError(
                    "refusing to replace a different prediction-provider manifest"
                )
            return destination
        _publish_new_bytes(destination, payload)
        return destination
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(destination.parent)


def verify_prediction_provider_manifest(
    path: str | Path,
    *,
    verify_payloads: bool = True,
    causal_frame_stop: int | None = None,
) -> tuple[PredictionProviderManifestV1, dict[str, object]]:
    """Verify stable manifest and payload snapshots against the canonical contract."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("prediction-provider manifest must not be a symbolic link")
    manifest_path = supplied.resolve()
    manifest_before = _file_signature(
        manifest_path,
        name="prediction-provider manifest",
    )
    manifest_sha256 = _sha256_file(manifest_path)
    manifest = _load_prediction_provider_manifest(manifest_path)
    if (
        _file_signature(manifest_path, name="prediction-provider manifest")
        != manifest_before
        or _sha256_file(manifest_path) != manifest_sha256
    ):
        raise ValueError("prediction-provider manifest changed during verification")

    verified_payloads = 0
    verified_bytes = 0
    if verify_payloads:
        for index, descriptor in enumerate(manifest.payloads):
            member = _resolved_member(
                manifest_path.parent,
                descriptor.path,
                name=f"prediction payload {index} path",
            )
            if not member.is_file():
                raise ValueError(f"prediction payload {descriptor.path!r} is missing")
            member_before = _file_signature(
                member,
                name=f"prediction payload {descriptor.path!r}",
            )
            member_sha256 = _sha256_file(member)
            if member_before[2] != descriptor.byte_count:
                raise ValueError(
                    "prediction payload byte count mismatch for "
                    f"{descriptor.path!r}"
                )
            if member_sha256 != descriptor.sha256:
                raise ValueError(
                    "prediction payload SHA-256 mismatch for "
                    f"{descriptor.path!r}"
                )
            window = PredictionWindow.from_npz(
                member,
                dense_storage_dtype=descriptor.dense_storage_dtype,
            )
            if (
                _file_signature(
                    member,
                    name=f"prediction payload {descriptor.path!r}",
                )
                != member_before
                or _sha256_file(member) != member_sha256
            ):
                raise ValueError(
                    "prediction payload changed during verification: "
                    f"{descriptor.path!r}"
                )
            if window.window_id != descriptor.window_id:
                raise ValueError("prediction payload window identity changed")
            if tuple(int(value) for value in window.frame_indices) != (
                descriptor.output_frame_ids
            ):
                raise ValueError("prediction payload output-frame identities changed")
            if (window.scene_flow is not None) != descriptor.has_scene_flow:
                raise ValueError("prediction payload scene-flow declaration changed")
            if (window.ray_directions is not None) != descriptor.has_ray_directions:
                raise ValueError("prediction payload ray declaration changed")
            if window.dense_storage_dtype != descriptor.dense_storage_dtype:
                raise ValueError("prediction payload dense storage dtype changed")
            verified_payloads += 1
            verified_bytes += member_before[2]

    if (
        _file_signature(manifest_path, name="prediction-provider manifest")
        != manifest_before
        or _sha256_file(manifest_path) != manifest_sha256
    ):
        raise ValueError("prediction-provider manifest changed during verification")
    report = {
        **manifest.summary(causal_frame_stop=causal_frame_stop),
        "io_schema": PREDICTION_PROVIDER_IO_SCHEMA,
        "io_version": PREDICTION_PROVIDER_IO_VERSION,
        "payloads_verified": verify_payloads,
        "verified_payload_count": verified_payloads,
        "verified_payload_bytes": verified_bytes,
    }
    return manifest, report


def import_motioncrafter_prediction_manifest(
    source_manifest_path: str | Path,
    output_path: str | Path,
    *,
    sequence_id: str,
    view_id: str = "camera-0",
) -> PredictionProviderManifestV1:
    """Import MotionCrafter through a stable source snapshot and safe publication."""

    source_supplied = Path(source_manifest_path)
    if source_supplied.is_symlink():
        raise ValueError("MotionCrafter prediction manifest must not be a symbolic link")
    source_path = source_supplied.resolve()
    source_before = _file_signature(
        source_path,
        name="MotionCrafter prediction manifest",
    )
    source_sha256 = _sha256_file(source_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.import.",
        suffix=".json",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _import_motioncrafter_manifest(
            source_path,
            temporary,
            sequence_id=sequence_id,
            view_id=view_id,
        )
        if (
            _file_signature(
                source_path,
                name="MotionCrafter prediction manifest",
            )
            != source_before
            or _sha256_file(source_path) != source_sha256
        ):
            raise ValueError("MotionCrafter prediction manifest changed during import")
        manifest, _ = verify_prediction_provider_manifest(temporary)
        save_prediction_provider_manifest(output, manifest)
        return manifest
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction",
        description="Import and validate canonical provider-neutral predictions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    specification_parser = subparsers.add_parser(
        "import-spec",
        help="import canonical windows from a strict external-provider specification",
    )
    specification_parser.add_argument("specification")
    specification_parser.add_argument("output")

    motioncrafter_parser = subparsers.add_parser(
        "import-motioncrafter",
        help="convert an integrity-bound MotionCrafter bundle",
    )
    motioncrafter_parser.add_argument("source_manifest")
    motioncrafter_parser.add_argument("output")
    motioncrafter_parser.add_argument("--sequence-id", required=True)
    motioncrafter_parser.add_argument("--view-id", default="camera-0")

    validate_parser = subparsers.add_parser(
        "validate",
        help="strictly validate a neutral manifest and its payloads",
    )
    validate_parser.add_argument("manifest")
    validate_parser.add_argument("--metadata-only", action="store_true")
    validate_parser.add_argument("--causal-frame-stop", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    if arguments.command == "import-spec":
        from .prediction_provider_import import (
            import_prediction_provider_specification,
        )

        manifest = import_prediction_provider_specification(
            arguments.specification,
            arguments.output,
        )
        print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
        return 0
    if arguments.command == "import-motioncrafter":
        manifest = import_motioncrafter_prediction_manifest(
            arguments.source_manifest,
            arguments.output,
            sequence_id=arguments.sequence_id,
            view_id=arguments.view_id,
        )
        print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
        return 0
    if arguments.command == "validate":
        _, report = verify_prediction_provider_manifest(
            arguments.manifest,
            verify_payloads=not arguments.metadata_only,
            causal_frame_stop=arguments.causal_frame_stop,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable prediction-provider command")


__all__ = [
    "PREDICTION_PROVIDER_IO_SCHEMA",
    "PREDICTION_PROVIDER_IO_VERSION",
    "import_motioncrafter_prediction_manifest",
    "load_prediction_provider_manifest",
    "main",
    "save_prediction_provider_manifest",
    "verify_prediction_provider_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
