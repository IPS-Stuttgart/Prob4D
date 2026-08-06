#!/usr/bin/env python3
"""Apply the one-shot canonical provider-manifest hardening patch."""

from __future__ import annotations

from pathlib import Path


def _replace_once(source: str, old: str, new: str, *, name: str) -> str:
    if old in source:
        return source.replace(old, new, 1)
    if new in source:
        return source
    raise RuntimeError(f"{name} changed unexpectedly")


def main() -> int:
    path = Path("src/prob4d/prediction_provider_manifest.py")
    source = path.read_text(encoding="utf-8")

    if "def _file_signature(" not in source:
        marker = "\n\ndef _require_boolean(value: object, *, name: str) -> bool:\n"
        if marker not in source:
            raise RuntimeError("file-signature insertion marker is missing")
        helper = '''


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
'''
        source = source.replace(marker, helper + marker, 1)

    save_start = source.index("def save_prediction_provider_manifest(")
    save_stop = source.index("\n\ndef load_prediction_provider_manifest(", save_start)
    save_block = '''def save_prediction_provider_manifest(
    path: str | Path,
    manifest: PredictionProviderManifestV1,
) -> Path:
    """Persist one manifest atomically under a fail-closed writer lock."""

    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("prediction-provider manifest destination is a symbolic link")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = manifest.to_record()
    content = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
        with os.fdopen(lock_descriptor, "w", encoding="utf-8") as lock_stream:
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
        _atomic_write_text(destination, content)
        return destination
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(destination.parent)
'''
    source = source[:save_start] + save_block + source[save_stop:]

    verify_start = source.index("def verify_prediction_provider_manifest(")
    verify_stop = source.index("\n\ndef _motioncrafter_seed_members(", verify_start)
    verify_block = '''def verify_prediction_provider_manifest(
    path: str | Path,
    *,
    verify_payloads: bool = True,
    causal_frame_stop: int | None = None,
) -> tuple[PredictionProviderManifestV1, dict[str, object]]:
    """Validate one stable manifest snapshot and its exact payload bytes."""

    manifest_path = Path(path).resolve()
    manifest_before = _file_signature(
        manifest_path,
        name="prediction-provider manifest",
    )
    manifest_sha256 = _sha256_file(manifest_path)
    manifest = load_prediction_provider_manifest(manifest_path)
    if (
        _file_signature(
            manifest_path,
            name="prediction-provider manifest",
        )
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
            member_after = _file_signature(
                member,
                name=f"prediction payload {descriptor.path!r}",
            )
            if member_after != member_before or _sha256_file(member) != member_sha256:
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
        _file_signature(
            manifest_path,
            name="prediction-provider manifest",
        )
        != manifest_before
        or _sha256_file(manifest_path) != manifest_sha256
    ):
        raise ValueError("prediction-provider manifest changed during verification")
    report = {
        **manifest.summary(causal_frame_stop=causal_frame_stop),
        "payloads_verified": verify_payloads,
        "verified_payload_count": verified_payloads,
        "verified_payload_bytes": verified_bytes,
    }
    return manifest, report
'''
    source = source[:verify_start] + verify_block + source[verify_stop:]

    source = _replace_once(
        source,
        '''    source_path = Path(source_manifest_path).resolve()
    verification = verify_motioncrafter_prediction_manifest(
        source_path,
        verify_hashes=True,
    )
    if verification.get("integrity_bound") is not True:
        raise ValueError("provider-neutral import requires an integrity-bound bundle")
    record = load_json_object(source_path, name="MotionCrafter prediction manifest")
''',
        '''    source_path = Path(source_manifest_path).resolve()
    source_before = _file_signature(
        source_path,
        name="MotionCrafter prediction manifest",
    )
    source_manifest_sha = _sha256_file(source_path)
    verification = verify_motioncrafter_prediction_manifest(
        source_path,
        verify_hashes=True,
    )
    if verification.get("integrity_bound") is not True:
        raise ValueError("provider-neutral import requires an integrity-bound bundle")
    if (
        _file_signature(
            source_path,
            name="MotionCrafter prediction manifest",
        )
        != source_before
        or _sha256_file(source_path) != source_manifest_sha
    ):
        raise ValueError("MotionCrafter prediction manifest changed during import")
    record = load_json_object(source_path, name="MotionCrafter prediction manifest")
    if (
        _file_signature(
            source_path,
            name="MotionCrafter prediction manifest",
        )
        != source_before
        or _sha256_file(source_path) != source_manifest_sha
    ):
        raise ValueError("MotionCrafter prediction manifest changed during import")
''',
        name="MotionCrafter import-start block",
    )
    source = _replace_once(
        source,
        '''        payload_path = _resolved_member(
            source_root,
            relative,
            name=f"MotionCrafter window {window_id!r} path",
        )
        prediction = PredictionWindow.from_npz(payload_path)
        if prediction.window_id != window_id:
''',
        '''        payload_path = _resolved_member(
            source_root,
            relative,
            name=f"MotionCrafter window {window_id!r} path",
        )
        payload_before = _file_signature(
            payload_path,
            name=f"MotionCrafter window {window_id!r} payload",
        )
        payload_sha256 = _sha256_file(payload_path)
        if payload_before[2] != byte_count:
            raise ValueError("MotionCrafter payload byte count changed during import")
        if payload_sha256 != sha256:
            raise ValueError("MotionCrafter payload SHA-256 changed during import")
        prediction = PredictionWindow.from_npz(payload_path)
        if (
            _file_signature(
                payload_path,
                name=f"MotionCrafter window {window_id!r} payload",
            )
            != payload_before
            or _sha256_file(payload_path) != payload_sha256
        ):
            raise ValueError("MotionCrafter payload changed during import")
        if prediction.window_id != window_id:
''',
        name="MotionCrafter payload-load block",
    )
    source = source.replace("    source_manifest_sha = _sha256_file(source_path)\n", "", 1)
    source = _replace_once(
        source,
        '''    save_prediction_provider_manifest(output_path, manifest)
    return manifest
''',
        '''    if (
        _file_signature(
            source_path,
            name="MotionCrafter prediction manifest",
        )
        != source_before
        or _sha256_file(source_path) != source_manifest_sha
    ):
        raise ValueError("MotionCrafter prediction manifest changed during import")
    save_prediction_provider_manifest(output_path, manifest)
    return manifest
''',
        name="MotionCrafter import-save block",
    )

    if '"import-spec"' not in source:
        parser_marker = '''    import_parser = subparsers.add_parser(
        "import-motioncrafter",
        help="convert an integrity-bound MotionCrafter bundle",
    )
'''
        if parser_marker not in source:
            raise RuntimeError("prediction parser insertion marker is missing")
        source = source.replace(
            parser_marker,
            '''    specification_parser = subparsers.add_parser(
        "import-spec",
        help="import canonical windows from a strict external-provider specification",
    )
    specification_parser.add_argument("specification")
    specification_parser.add_argument("output")

'''
            + parser_marker,
            1,
        )
    if 'arguments.command == "import-spec"' not in source:
        main_marker = '''    if arguments.command == "import-motioncrafter":
        manifest = import_motioncrafter_prediction_manifest(
'''
        if main_marker not in source:
            raise RuntimeError("prediction main insertion marker is missing")
        source = source.replace(
            main_marker,
            '''    if arguments.command == "import-spec":
        from .prediction_provider_import import (
            import_prediction_provider_specification,
        )

        manifest = import_prediction_provider_specification(
            arguments.specification,
            arguments.output,
        )
        print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
        return 0
'''
            + main_marker,
            1,
        )

    path.write_text(source, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
