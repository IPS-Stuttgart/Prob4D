"""Integrity primitives for portable MotionCrafter prediction bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from . import data as _data_module
from . import lineage as _lineage_module
from . import motioncrafter as _motioncrafter_module
from .data import PredictionWindow
from .motioncrafter import MotionCrafterRunConfig, validate_motioncrafter_seed_schedule

MOTIONCRAFTER_RUN_SPEC_SCHEMA: Final = "prob4d.motioncrafter-run-spec.v1"
MOTIONCRAFTER_PROGRESS_SCHEMA: Final = "prob4d.motioncrafter-progress.v1"
MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA: Final = (
    "prob4d.motioncrafter-artifact-integrity.v1"
)
MOTIONCRAFTER_PROGRESS_FILENAME: Final = ".motioncrafter-progress.json"
MOTIONCRAFTER_MANIFEST_FILENAME: Final = "predictions.json"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read MotionCrafter artifact {path}") from error
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_git_commit(value: object, *, name: str) -> str:
    commit = str(value)
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"{name} must be a full lowercase Git object ID")
    return commit


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _safe_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    if not value or "\\" in value:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return pure.as_posix()


def resolve_motioncrafter_member(
    root: str | Path,
    relative_path: object,
    *,
    name: str,
) -> Path:
    """Resolve a member while refusing absolute, parent, and symlink escapes."""

    safe = _safe_relative_path(relative_path, name=name)
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / Path(*PurePosixPath(safe).parts)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes the prediction-bundle directory") from error
    return candidate


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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _artifact_descriptor(path: Path, *, root: Path, kind: str) -> dict[str, object]:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("MotionCrafter artifacts must lie inside the output directory") from error
    relative_path = _safe_relative_path(relative.as_posix(), name="artifact path")
    try:
        byte_count = path_resolved.stat().st_size
    except OSError as error:
        raise ValueError(f"cannot stat MotionCrafter artifact {relative_path!r}") from error
    return {
        "path": relative_path,
        "sha256": _sha256_file(path_resolved),
        "bytes": byte_count,
        "kind": str(kind),
    }


def _validate_descriptor(
    descriptor: Mapping[str, object],
    *,
    root: Path,
    name: str,
    verify_hash: bool,
) -> dict[str, object]:
    relative = _safe_relative_path(descriptor.get("path"), name=f"{name} path")
    expected_sha = _require_sha256(descriptor.get("sha256"), name=f"{name} sha256")
    expected_bytes = _require_nonnegative_integer(
        descriptor.get("bytes"),
        name=f"{name} bytes",
    )
    kind = str(descriptor.get("kind", ""))
    if not kind:
        raise ValueError(f"{name} kind must be non-empty")
    path = resolve_motioncrafter_member(root, relative, name=f"{name} path")
    if not path.is_file():
        raise ValueError(f"{name} file {relative!r} is missing")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{name} byte count mismatch for {relative!r}")
    if verify_hash and _sha256_file(path) != expected_sha:
        raise ValueError(f"{name} SHA-256 mismatch for {relative!r}")
    return {
        "path": relative,
        "sha256": expected_sha,
        "bytes": expected_bytes,
        "kind": kind,
    }


def _git_provenance(upstream_root: Path) -> Mapping[str, object]:
    root = upstream_root.resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect MotionCrafter checkout {root}") from error
    _require_git_commit(commit, name="MotionCrafter commit")
    return {
        "commit": commit,
        "clean": not bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_entry_count": len(status.splitlines()),
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("prob4d")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def _producer_files() -> list[dict[str, object]]:
    modules = (
        ("prob4d.data", _data_module),
        ("prob4d.lineage", _lineage_module),
        ("prob4d.motioncrafter", _motioncrafter_module),
        ("prob4d.motioncrafter_integrity", None),
    )
    records: list[dict[str, object]] = []
    for module_name, module in modules:
        module_file = __file__ if module is None else getattr(module, "__file__", None)
        if not module_file:
            raise ValueError(f"cannot locate executed producer module {module_name}")
        path = Path(module_file).resolve()
        if not path.is_file():
            raise ValueError(f"executed producer module {module_name} is not a file")
        records.append(
            {
                "module": module_name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _json_config(config: MotionCrafterRunConfig) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _video_descriptor(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"input video {path} is not a regular file")
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _run_spec(
    config: MotionCrafterRunConfig,
    *,
    video_descriptor: Mapping[str, object],
    upstream_provenance: Mapping[str, object],
    runner_descriptor: Mapping[str, object],
) -> dict[str, object]:
    config_record = _json_config(config)
    for field in ("upstream_root", "video_path", "output_directory", "cache_directory"):
        config_record.pop(field, None)
    producer_files = _producer_files()
    producer_files.append(dict(runner_descriptor))
    return {
        "schema": MOTIONCRAFTER_RUN_SPEC_SCHEMA,
        "producer": {
            "package": "prob4d",
            "version": _package_version(),
            "entrypoint": "prob4d.motioncrafter_safe:main",
            "executed_module_files": producer_files,
        },
        "input_video": dict(video_descriptor),
        "motioncrafter_upstream": dict(upstream_provenance),
        "inference_config": config_record,
    }


def _new_progress(
    run_spec: Mapping[str, object],
    run_spec_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": MOTIONCRAFTER_PROGRESS_SCHEMA,
        "run_spec": dict(run_spec),
        "run_spec_sha256": run_spec_sha256,
        "status": "in_progress",
        "artifacts": {},
    }


def _load_progress(
    path: Path,
    *,
    expected_run_spec_sha256: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read MotionCrafter progress journal {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("MotionCrafter progress journal must contain a JSON object")
    if payload.get("schema") != MOTIONCRAFTER_PROGRESS_SCHEMA:
        raise ValueError("unsupported MotionCrafter progress schema")
    run_spec = payload.get("run_spec")
    if not isinstance(run_spec, Mapping):
        raise ValueError("MotionCrafter progress journal lacks a run_spec mapping")
    recorded_digest = _require_sha256(
        payload.get("run_spec_sha256"),
        name="progress run_spec_sha256",
    )
    _validate_run_spec(run_spec)
    if _sha256_json(run_spec) != recorded_digest:
        raise ValueError("MotionCrafter progress run-spec digest mismatch")
    if recorded_digest != expected_run_spec_sha256:
        raise ValueError("resume request does not match the recorded MotionCrafter run spec")
    if payload.get("status") not in {"in_progress", "complete"}:
        raise ValueError("MotionCrafter progress status is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("MotionCrafter progress artifacts must be a mapping")
    normalized: dict[str, dict[str, object]] = {}
    for key, descriptor in artifacts.items():
        relative = _safe_relative_path(key, name="progress artifact key")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"progress artifact {relative!r} must be a mapping")
        validated = _validate_descriptor(
            descriptor,
            root=path.parent,
            name=f"progress artifact {relative!r}",
            verify_hash=verify_hashes,
        )
        if validated["path"] != relative:
            raise ValueError(f"progress artifact key/path mismatch for {relative!r}")
        normalized[relative] = validated
    payload["artifacts"] = normalized
    return payload


def _prediction_window_payload(window: PredictionWindow) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "window_id": np.asarray(window.window_id),
        "frame_indices": np.asarray(window.frame_indices),
        "point_map": np.asarray(window.point_map, dtype=np.float32),
        "valid_mask": np.asarray(window.valid_mask, dtype=bool),
    }
    if window.scene_flow is not None:
        payload["scene_flow"] = np.asarray(window.scene_flow, dtype=np.float32)
        payload["deform_mask"] = np.asarray(window.deform_mask, dtype=bool)
    if window.ray_directions is not None:
        payload["ray_directions"] = np.asarray(window.ray_directions, dtype=np.float32)
    return payload


def _expected_members(manifest: Mapping[str, Any]) -> dict[str, str]:
    windows = manifest.get("overlap_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("prediction manifest must contain overlap windows")
    expected: dict[str, str] = {}
    window_ids: set[str] = set()
    for index, item in enumerate(windows):
        if not isinstance(item, Mapping):
            raise ValueError(f"overlap window {index} must be a mapping")
        window_id = str(item.get("window_id", ""))
        if not window_id or window_id in window_ids:
            raise ValueError("overlap window IDs must be non-empty and unique")
        window_ids.add(window_id)
        relative = _safe_relative_path(
            item.get("path"),
            name=f"overlap window {window_id!r} path",
        )
        if relative in expected:
            raise ValueError("prediction manifest member paths must be unique")
        expected[relative] = "independently_decoded_overlap_window"

    baselines = (
        (
            manifest.get("disjoint_baseline"),
            "disjoint baseline path",
            "disjoint_baseline",
        ),
        (
            manifest.get("latent_linear_baseline"),
            "latent-linear baseline path",
            "latent_linear_baseline",
        ),
    )
    for value, name, kind in baselines:
        relative = _safe_relative_path(value, name=name)
        if relative in expected:
            raise ValueError("prediction manifest member paths must be unique")
        expected[relative] = kind
    return expected


def _manifest_inference_config(manifest: Mapping[str, Any]) -> dict[str, object]:
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("integrity-bound prediction manifest config must be a mapping")
    portable = dict(config)
    for field in ("upstream_root", "video_path", "output_directory", "cache_directory"):
        portable.pop(field, None)
    return portable


def _validate_run_spec(run_spec: Mapping[str, object]) -> None:
    if run_spec.get("schema") != MOTIONCRAFTER_RUN_SPEC_SCHEMA:
        raise ValueError("unsupported MotionCrafter run-spec schema")
    producer = run_spec.get("producer")
    if not isinstance(producer, Mapping):
        raise ValueError("MotionCrafter run spec lacks producer metadata")
    if producer.get("package") != "prob4d":
        raise ValueError("MotionCrafter run spec has an unexpected producer package")
    if not str(producer.get("version", "")):
        raise ValueError("MotionCrafter run spec lacks a producer version")
    if producer.get("entrypoint") != "prob4d.motioncrafter_safe:main":
        raise ValueError("MotionCrafter run spec has an unexpected producer entrypoint")
    module_files = producer.get("executed_module_files")
    if not isinstance(module_files, list) or not module_files:
        raise ValueError("MotionCrafter run spec lacks executed-module identities")
    modules: set[str] = set()
    for index, descriptor in enumerate(module_files):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"producer module descriptor {index} must be a mapping")
        module_name = str(descriptor.get("module", ""))
        if not module_name or module_name in modules:
            raise ValueError("producer module identities must be non-empty and unique")
        modules.add(module_name)
        _require_sha256(
            descriptor.get("sha256"),
            name=f"producer module {module_name!r} sha256",
        )
        _require_nonnegative_integer(
            descriptor.get("bytes"),
            name=f"producer module {module_name!r} bytes",
        )
    required_modules = {
        "prob4d.data",
        "prob4d.lineage",
        "prob4d.motioncrafter",
        "prob4d.motioncrafter_integrity",
        "prob4d.motioncrafter_runner",
    }
    if modules != required_modules:
        raise ValueError("MotionCrafter run spec has an incomplete producer-module set")

    video = run_spec.get("input_video")
    if not isinstance(video, Mapping):
        raise ValueError("MotionCrafter run spec lacks an input_video descriptor")
    _require_sha256(video.get("sha256"), name="input video sha256")
    _require_nonnegative_integer(video.get("bytes"), name="input video bytes")

    upstream = run_spec.get("motioncrafter_upstream")
    if not isinstance(upstream, Mapping):
        raise ValueError("MotionCrafter run spec lacks upstream provenance")
    _require_git_commit(upstream.get("commit"), name="MotionCrafter upstream commit")
    if not isinstance(upstream.get("clean"), bool):
        raise ValueError("MotionCrafter upstream clean flag must be Boolean")
    _require_sha256(
        upstream.get("status_sha256"),
        name="MotionCrafter upstream status_sha256",
    )
    status_entry_count = _require_nonnegative_integer(
        upstream.get("status_entry_count"),
        name="MotionCrafter upstream status_entry_count",
    )
    if bool(upstream["clean"]) != (status_entry_count == 0):
        raise ValueError("MotionCrafter upstream clean state contradicts its status count")
    if not isinstance(run_spec.get("inference_config"), Mapping):
        raise ValueError("MotionCrafter run spec lacks inference_config")


def verify_motioncrafter_prediction_manifest(
    path: str | Path,
    *,
    verify_hashes: bool = True,
    expected_run_spec_sha256: str | None = None,
) -> dict[str, object]:
    """Validate paths, optional hashes, and run identity for one bundle."""

    manifest_path = Path(path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prediction manifest {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("prediction manifest must contain a JSON object")
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    config = manifest.get("config")
    if "stochastic_seed_schedule" in manifest or (
        isinstance(config, dict) and "seed_policy" in config
    ):
        validate_motioncrafter_seed_schedule(manifest)

    root = manifest_path.parent
    expected_members = _expected_members(manifest)
    expected_paths = list(expected_members)
    for index, relative in enumerate(expected_paths):
        member = resolve_motioncrafter_member(
            root,
            relative,
            name=f"prediction member {index}",
        )
        if not member.is_file():
            raise ValueError(f"prediction member {relative!r} is missing")

    integrity = manifest.get("artifact_integrity")
    if integrity is None:
        if expected_run_spec_sha256 is not None:
            raise ValueError("legacy prediction manifest has no bound run spec")
        return {
            "manifest_path": str(manifest_path),
            "integrity_bound": False,
            "hashes_verified": False,
            "member_count": len(expected_paths),
        }
    if not isinstance(integrity, Mapping):
        raise ValueError("artifact_integrity must be a mapping")
    if integrity.get("schema") != MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA:
        raise ValueError("unsupported MotionCrafter artifact-integrity schema")
    run_spec = integrity.get("run_spec")
    if not isinstance(run_spec, Mapping):
        raise ValueError("artifact_integrity run_spec must be a mapping")
    _validate_run_spec(run_spec)
    upstream = run_spec.get("motioncrafter_upstream")
    inference_config = run_spec.get("inference_config")
    manifest_config = manifest.get("config")
    if not isinstance(upstream, Mapping) or not isinstance(inference_config, Mapping):
        raise ValueError("MotionCrafter run spec is internally inconsistent")
    if not isinstance(manifest_config, Mapping):
        raise ValueError("integrity-bound prediction manifest config must be a mapping")
    if manifest.get("motioncrafter_commit") != upstream.get("commit"):
        raise ValueError("prediction manifest MotionCrafter commit differs from its run spec")
    if _manifest_inference_config(manifest) != dict(inference_config):
        raise ValueError("prediction manifest config differs from its bound run spec")
    if manifest.get("video_path") != manifest_config.get("video_path"):
        raise ValueError("prediction manifest video paths are inconsistent")
    run_spec_sha256 = _require_sha256(
        integrity.get("run_spec_sha256"),
        name="artifact_integrity run_spec_sha256",
    )
    if _sha256_json(run_spec) != run_spec_sha256:
        raise ValueError("MotionCrafter run-spec digest mismatch")
    if (
        expected_run_spec_sha256 is not None
        and run_spec_sha256 != expected_run_spec_sha256
    ):
        raise ValueError("prediction manifest belongs to a different MotionCrafter run spec")

    members = integrity.get("members")
    if not isinstance(members, list):
        raise ValueError("artifact_integrity members must be a list")
    descriptors: dict[str, dict[str, object]] = {}
    for index, descriptor in enumerate(members):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"artifact_integrity member {index} must be a mapping")
        validated = _validate_descriptor(
            descriptor,
            root=root,
            name=f"artifact_integrity member {index}",
            verify_hash=verify_hashes,
        )
        relative = str(validated["path"])
        if relative in descriptors:
            raise ValueError(f"duplicate artifact descriptor for {relative!r}")
        expected_kind = expected_members.get(relative)
        if expected_kind is not None and validated["kind"] != expected_kind:
            raise ValueError(f"artifact kind mismatch for {relative!r}")
        descriptors[relative] = validated
    if set(descriptors) != set(expected_paths):
        missing = sorted(set(expected_paths) - set(descriptors))
        extra = sorted(set(descriptors) - set(expected_paths))
        raise ValueError(
            f"artifact descriptors do not match manifest members; missing={missing}, extra={extra}"
        )
    return {
        "manifest_path": str(manifest_path),
        "integrity_bound": True,
        "hashes_verified": verify_hashes,
        "member_count": len(expected_paths),
        "run_spec_sha256": run_spec_sha256,
    }
