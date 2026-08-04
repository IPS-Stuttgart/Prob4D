"""Shared validation and hashing for covariance calibration artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

GAUGE_COVARIANCE_CALIBRATION_SCHEMA = "prob4d.gauge_covariance_calibration"
GAUGE_COVARIANCE_CALIBRATION_VERSION = 1
POINT_UNCERTAINTY_CALIBRATION_SCHEMA = "prob4d.point_uncertainty_calibration"
POINT_UNCERTAINTY_CALIBRATION_VERSION = 1


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _artifact_id(descriptor: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validate_revision(value: str, *, name: str) -> str:
    revision = str(value)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase 40- or 64-character revision")
    return revision


def _validated_case_ids(values: Sequence[str]) -> tuple[str, ...]:
    case_ids = tuple(str(value) for value in values)
    if not case_ids or any(not value for value in case_ids):
        raise ValueError("calibration_case_ids must contain nonempty identifiers")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("calibration_case_ids must be unique")
    return case_ids


def _validated_resolution(value: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    resolution = tuple(int(item) for item in value)
    if len(resolution) != 2 or any(item <= 0 for item in resolution):
        raise ValueError("image_resolution must contain two positive integers")
    return resolution[0], resolution[1]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validated_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be finite JSON data") from error
    return _freeze_json(normalized)


def _validate_common_provenance(
    *,
    calibration_case_ids: Sequence[str],
    source_repository: str,
    source_revision: str,
    motioncrafter_revision: str,
    model_identifier: str,
    covariance_method: str,
    image_resolution: tuple[int, int] | None,
    window_size: int | None,
    window_overlap: int | None,
    covariance_cluster_size: int | None,
    input_artifact_sha256: Sequence[str],
    metadata: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    str,
    str,
    str,
    str,
    str,
    tuple[int, int] | None,
    int | None,
    int | None,
    int | None,
    tuple[str, ...],
    Mapping[str, Any],
]:
    case_ids = _validated_case_ids(calibration_case_ids)
    repository = str(source_repository)
    identifier = str(model_identifier)
    method = str(covariance_method)
    if not repository or not identifier or not method:
        raise ValueError(
            "source_repository, model_identifier, and covariance_method must be nonempty"
        )
    resolution = _validated_resolution(image_resolution)
    normalized_window_size = None if window_size is None else int(window_size)
    normalized_overlap = None if window_overlap is None else int(window_overlap)
    normalized_cluster_size = (
        None if covariance_cluster_size is None else int(covariance_cluster_size)
    )
    if normalized_window_size is not None and normalized_window_size <= 0:
        raise ValueError("window_size must be positive when supplied")
    if normalized_overlap is not None and normalized_overlap < 0:
        raise ValueError("window_overlap must be non-negative when supplied")
    if (
        normalized_window_size is not None
        and normalized_overlap is not None
        and normalized_overlap >= normalized_window_size
    ):
        raise ValueError("window_overlap must be smaller than window_size")
    if normalized_cluster_size is not None and normalized_cluster_size <= 0:
        raise ValueError("covariance_cluster_size must be positive when supplied")
    digests = tuple(
        _validate_sha256(str(value), name="input_artifact_sha256")
        for value in input_artifact_sha256
    )
    if not digests:
        raise ValueError("input_artifact_sha256 must contain at least one source digest")
    if len(set(digests)) != len(digests):
        raise ValueError("input_artifact_sha256 values must be unique")
    return (
        case_ids,
        repository,
        _validate_revision(source_revision, name="source_revision"),
        _validate_revision(motioncrafter_revision, name="motioncrafter_revision"),
        identifier,
        method,
        resolution,
        normalized_window_size,
        normalized_overlap,
        normalized_cluster_size,
        digests,
        _validated_metadata(metadata),
    )


def _common_descriptor(artifact: Any) -> dict[str, Any]:
    return {
        "calibration_case_ids": list(artifact.calibration_case_ids),
        "source_repository": artifact.source_repository,
        "source_revision": artifact.source_revision,
        "motioncrafter_revision": artifact.motioncrafter_revision,
        "model_identifier": artifact.model_identifier,
        "covariance_method": artifact.covariance_method,
        "image_resolution": (
            None if artifact.image_resolution is None else list(artifact.image_resolution)
        ),
        "window_size": artifact.window_size,
        "window_overlap": artifact.window_overlap,
        "covariance_cluster_size": artifact.covariance_cluster_size,
        "input_artifact_sha256": list(artifact.input_artifact_sha256),
        "metadata": _thaw_json(artifact.metadata),
    }
