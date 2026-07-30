"""Strict compatibility checks for claim-bearing covariance calibration artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .alignment import (
    DEFAULT_COVARIANCE_CLUSTER_SIZE,
    DENSE_ALIGNMENT_COVARIANCE_METHOD,
)
from .motioncrafter import (
    MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    validate_motioncrafter_seed_schedule,
)
from .observation_contract import file_sha256

PROB4D_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V1 = "prob4d.motioncrafter-model.v1"
MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA = "prob4d.motioncrafter-model.v2"
POINT_UNCERTAINTY_COVARIANCE_METHOD = "depth_disagreement_anisotropic_v1"

_MODEL_CONFIG_KEYS_V1 = (
    "model_type",
    "unet_path",
    "vae_path",
    "num_inference_steps",
    "guidance_scale",
    "decode_chunk_size",
    "low_memory_usage",
    "seed",
    "frame_stride",
)
_MODEL_CONFIG_KEYS_V2 = (*_MODEL_CONFIG_KEYS_V1, "seed_policy")


class CalibrationCompatibilityError(ValueError):
    """Raised when a calibration artifact does not match a prediction run."""


class CalibrationArtifactV1(Protocol):
    """Structural provenance shared by the two version-1 calibration artifacts."""

    @property
    def artifact_id(self) -> str: ...

    calibration_case_ids: tuple[str, ...]
    source_repository: str
    source_revision: str
    motioncrafter_revision: str
    model_identifier: str
    covariance_method: str
    image_resolution: tuple[int, int] | None
    window_size: int | None
    window_overlap: int | None
    covariance_cluster_size: int | None


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validated_revision(value: object, *, name: str) -> str:
    revision = str(value)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase 40- or 64-character revision")
    return revision


def _required_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def motioncrafter_model_identifier(manifest: Mapping[str, Any]) -> str:
    """Return a canonical identifier for prediction-affecting model settings.

    Historical manifests without ``seed_policy`` and new manifests that explicitly
    request ``legacy-common`` retain the version-1 identifier because their inference
    behavior is identical. ``derived-per-call`` uses a version-2 descriptor that binds
    the seed policy into covariance-calibration compatibility.
    """

    if manifest.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    config = _required_mapping(manifest.get("config"), name="prediction manifest config")
    seed_policy = str(
        config.get("seed_policy", MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON)
    )
    if seed_policy == MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON:
        schema = MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V1
        config_keys = _MODEL_CONFIG_KEYS_V1
    elif seed_policy == MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL:
        schema = MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA
        config_keys = _MODEL_CONFIG_KEYS_V2
    else:
        raise ValueError(f"unsupported MotionCrafter seed policy {seed_policy!r}")
    missing = [key for key in config_keys if key not in config]
    if missing:
        raise ValueError(
            "prediction manifest is missing model-identifier settings: "
            + ", ".join(missing)
        )
    descriptor = {
        "schema": schema,
        "config": {key: config[key] for key in config_keys},
    }
    digest = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    return f"{schema}:{digest}"


@dataclass(frozen=True)
class PredictionCalibrationTargetV1:
    """Runtime settings that a claim-bearing calibration must match exactly."""

    manifest_sha256: str
    source_repository: str
    motioncrafter_revision: str
    model_identifier: str
    image_resolution: tuple[int, int]
    window_size: int
    window_overlap: int
    covariance_cluster_size: int
    gauge_covariance_method: str
    point_covariance_method: str

    def __post_init__(self) -> None:
        digest = str(self.manifest_sha256)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        repository = str(self.source_repository)
        model_identifier = str(self.model_identifier)
        gauge_method = str(self.gauge_covariance_method)
        point_method = str(self.point_covariance_method)
        resolution = tuple(int(value) for value in self.image_resolution)
        window_size = int(self.window_size)
        overlap = int(self.window_overlap)
        cluster_size = int(self.covariance_cluster_size)
        if not repository or not model_identifier or not gauge_method or not point_method:
            raise ValueError("calibration target identifiers and methods must be nonempty")
        if len(resolution) != 2 or any(value <= 0 for value in resolution):
            raise ValueError("image_resolution must contain two positive integers")
        if window_size <= 0 or overlap < 0 or overlap >= window_size:
            raise ValueError("window_size and window_overlap are inconsistent")
        if cluster_size <= 0:
            raise ValueError("covariance_cluster_size must be positive")
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(self, "source_repository", repository)
        object.__setattr__(
            self,
            "motioncrafter_revision",
            _validated_revision(
                self.motioncrafter_revision,
                name="motioncrafter_revision",
            ),
        )
        object.__setattr__(self, "model_identifier", model_identifier)
        object.__setattr__(self, "image_resolution", (resolution[0], resolution[1]))
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "window_overlap", overlap)
        object.__setattr__(self, "covariance_cluster_size", cluster_size)
        object.__setattr__(self, "gauge_covariance_method", gauge_method)
        object.__setattr__(self, "point_covariance_method", point_method)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "prob4d.prediction-calibration-target",
            "version": 1,
            "manifest_sha256": self.manifest_sha256,
            "source_repository": self.source_repository,
            "motioncrafter_revision": self.motioncrafter_revision,
            "model_identifier": self.model_identifier,
            "image_resolution": list(self.image_resolution),
            "window_size": self.window_size,
            "window_overlap": self.window_overlap,
            "covariance_cluster_size": self.covariance_cluster_size,
            "gauge_covariance_method": self.gauge_covariance_method,
            "point_covariance_method": self.point_covariance_method,
        }


def load_prediction_calibration_target(
    manifest_path: str | Path,
    *,
    covariance_cluster_size: int = DEFAULT_COVARIANCE_CLUSTER_SIZE,
    gauge_covariance_method: str = DENSE_ALIGNMENT_COVARIANCE_METHOD,
    point_covariance_method: str = POINT_UNCERTAINTY_COVARIANCE_METHOD,
) -> PredictionCalibrationTargetV1:
    """Read only manifest metadata and construct the strict compatibility target."""

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = _required_mapping(payload, name="prediction manifest")
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    config = _required_mapping(manifest.get("config"), name="prediction manifest config")
    validate_motioncrafter_seed_schedule(manifest)
    try:
        resolution = (int(config["height"]), int(config["width"]))
        window_size = int(config["window_size"])
        overlap = int(config["overlap"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "prediction manifest lacks valid resolution or window settings"
        ) from error
    return PredictionCalibrationTargetV1(
        manifest_sha256=file_sha256(path),
        source_repository=PROB4D_SOURCE_REPOSITORY,
        motioncrafter_revision=_validated_revision(
            manifest.get("motioncrafter_commit"),
            name="motioncrafter_commit",
        ),
        model_identifier=motioncrafter_model_identifier(manifest),
        image_resolution=resolution,
        window_size=window_size,
        window_overlap=overlap,
        covariance_cluster_size=covariance_cluster_size,
        gauge_covariance_method=gauge_covariance_method,
        point_covariance_method=point_covariance_method,
    )


def calibration_compatibility_mismatches(
    artifact: CalibrationArtifactV1,
    target: PredictionCalibrationTargetV1,
    *,
    expected_covariance_method: str,
) -> tuple[str, ...]:
    """Return deterministic field-level incompatibilities for one artifact."""

    comparisons = (
        ("source_repository", artifact.source_repository, target.source_repository),
        (
            "motioncrafter_revision",
            artifact.motioncrafter_revision,
            target.motioncrafter_revision,
        ),
        ("model_identifier", artifact.model_identifier, target.model_identifier),
        ("image_resolution", artifact.image_resolution, target.image_resolution),
        ("window_size", artifact.window_size, target.window_size),
        ("window_overlap", artifact.window_overlap, target.window_overlap),
        (
            "covariance_cluster_size",
            artifact.covariance_cluster_size,
            target.covariance_cluster_size,
        ),
        ("covariance_method", artifact.covariance_method, expected_covariance_method),
    )
    return tuple(
        f"{name}: artifact={actual!r}, prediction={expected!r}"
        for name, actual, expected in comparisons
        if actual != expected
    )


def assert_calibration_compatible(
    artifact: CalibrationArtifactV1,
    target: PredictionCalibrationTargetV1,
    *,
    expected_covariance_method: str,
    artifact_name: str,
) -> None:
    """Fail closed when one calibration was fitted for different runtime semantics."""

    mismatches = calibration_compatibility_mismatches(
        artifact,
        target,
        expected_covariance_method=expected_covariance_method,
    )
    if mismatches:
        raise CalibrationCompatibilityError(
            f"{artifact_name} calibration {artifact.artifact_id} is incompatible with "
            f"prediction manifest {target.manifest_sha256}: "
            + "; ".join(mismatches)
        )


def assert_calibration_pair_compatible(
    gauge_calibration: CalibrationArtifactV1,
    point_calibration: CalibrationArtifactV1,
    target: PredictionCalibrationTargetV1,
) -> None:
    """Validate both calibrations before any prediction payload is opened."""

    assert_calibration_compatible(
        gauge_calibration,
        target,
        expected_covariance_method=target.gauge_covariance_method,
        artifact_name="gauge",
    )
    assert_calibration_compatible(
        point_calibration,
        target,
        expected_covariance_method=target.point_covariance_method,
        artifact_name="point",
    )


__all__ = [
    "MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA",
    "MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V1",
    "POINT_UNCERTAINTY_COVARIANCE_METHOD",
    "PROB4D_SOURCE_REPOSITORY",
    "CalibrationCompatibilityError",
    "PredictionCalibrationTargetV1",
    "assert_calibration_compatible",
    "assert_calibration_pair_compatible",
    "calibration_compatibility_mismatches",
    "load_prediction_calibration_target",
    "motioncrafter_model_identifier",
]
