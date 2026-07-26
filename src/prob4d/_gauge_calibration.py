"""Content-addressed calibration for dense-overlap Sim(3) covariance."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._calibration_common import (
    GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
    GAUGE_COVARIANCE_CALIBRATION_VERSION,
    _artifact_id,
    _common_descriptor,
    _validate_common_provenance,
)
from .gauge import GaugeCovarianceCalibration

@dataclass(frozen=True)
class GaugeCovarianceCalibrationV1:
    """Reusable blockwise Sim(3) covariance inflation with exact provenance."""

    scale: float
    rotation: float
    translation: float
    count: int
    trim_quantile: float
    calibration_case_ids: tuple[str, ...]
    source_repository: str
    source_revision: str
    motioncrafter_revision: str
    model_identifier: str
    covariance_method: str
    image_resolution: tuple[int, int] | None = None
    window_size: int | None = None
    window_overlap: int | None = None
    covariance_cluster_size: int | None = None
    input_artifact_sha256: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factors = np.asarray([self.scale, self.rotation, self.translation], dtype=np.float64)
        if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
            raise ValueError("gauge covariance inflation factors must be finite and positive")
        count = int(self.count)
        trim_quantile = float(self.trim_quantile)
        if count < 1:
            raise ValueError("calibration count must be positive")
        if not np.isfinite(trim_quantile) or not 0.0 < trim_quantile <= 1.0:
            raise ValueError("trim_quantile must lie in (0, 1]")
        common = _validate_common_provenance(
            calibration_case_ids=self.calibration_case_ids,
            source_repository=self.source_repository,
            source_revision=self.source_revision,
            motioncrafter_revision=self.motioncrafter_revision,
            model_identifier=self.model_identifier,
            covariance_method=self.covariance_method,
            image_resolution=self.image_resolution,
            window_size=self.window_size,
            window_overlap=self.window_overlap,
            covariance_cluster_size=self.covariance_cluster_size,
            input_artifact_sha256=self.input_artifact_sha256,
            metadata=self.metadata,
        )
        object.__setattr__(self, "scale", float(factors[0]))
        object.__setattr__(self, "rotation", float(factors[1]))
        object.__setattr__(self, "translation", float(factors[2]))
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "trim_quantile", trim_quantile)
        (
            case_ids,
            repository,
            source_revision,
            motioncrafter_revision,
            model_identifier,
            method,
            resolution,
            window_size,
            overlap,
            cluster_size,
            digests,
            metadata,
        ) = common
        object.__setattr__(self, "calibration_case_ids", case_ids)
        object.__setattr__(self, "source_repository", repository)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "motioncrafter_revision", motioncrafter_revision)
        object.__setattr__(self, "model_identifier", model_identifier)
        object.__setattr__(self, "covariance_method", method)
        object.__setattr__(self, "image_resolution", resolution)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "window_overlap", overlap)
        object.__setattr__(self, "covariance_cluster_size", cluster_size)
        object.__setattr__(self, "input_artifact_sha256", digests)
        object.__setattr__(self, "metadata", metadata)

    @property
    def runtime_calibration(self) -> GaugeCovarianceCalibration:
        return GaugeCovarianceCalibration(
            scale=self.scale,
            rotation=self.rotation,
            translation=self.translation,
            trim_quantile=self.trim_quantile,
            count=self.count,
        )

    def apply(self, covariance: np.ndarray) -> np.ndarray:
        """Inflate one seven-dimensional covariance using the fitted blocks."""

        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.shape != (7, 7) or not np.all(np.isfinite(matrix)):
            raise ValueError("gauge covariance must have finite shape (7, 7)")
        symmetric = 0.5 * (matrix + matrix.T)
        if not np.allclose(matrix, symmetric, atol=1e-12, rtol=1e-10):
            raise ValueError("gauge covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(symmetric)) < -1e-10:
            raise ValueError("gauge covariance must be positive semidefinite")
        inflated = self.runtime_calibration.apply(symmetric)
        inflated.setflags(write=False)
        return inflated

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
            "version": GAUGE_COVARIANCE_CALIBRATION_VERSION,
            "calibration": {
                "scale": self.scale,
                "rotation": self.rotation,
                "translation": self.translation,
                "count": self.count,
                "trim_quantile": self.trim_quantile,
            },
            "provenance": _common_descriptor(self),
        }

    @property
    def artifact_id(self) -> str:
        return _artifact_id(self.descriptor())

    def to_dict(self) -> dict[str, Any]:
        descriptor = self.descriptor()
        return {"artifact_id": self.artifact_id, **descriptor}

    @classmethod
    def fit(
        cls,
        errors: np.ndarray,
        covariances: np.ndarray,
        *,
        calibration_case_ids: Sequence[str],
        source_repository: str,
        source_revision: str,
        motioncrafter_revision: str,
        model_identifier: str,
        covariance_method: str,
        trim_quantile: float = 0.99,
        image_resolution: tuple[int, int] | None = None,
        window_size: int | None = None,
        window_overlap: int | None = None,
        covariance_cluster_size: int | None = None,
        input_artifact_sha256: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> GaugeCovarianceCalibrationV1:
        fitted = GaugeCovarianceCalibration.fit(
            errors,
            covariances,
            trim_quantile=trim_quantile,
        )
        return cls(
            scale=fitted.scale,
            rotation=fitted.rotation,
            translation=fitted.translation,
            count=fitted.count,
            trim_quantile=fitted.trim_quantile,
            calibration_case_ids=tuple(calibration_case_ids),
            source_repository=source_repository,
            source_revision=source_revision,
            motioncrafter_revision=motioncrafter_revision,
            model_identifier=model_identifier,
            covariance_method=covariance_method,
            image_resolution=image_resolution,
            window_size=window_size,
            window_overlap=window_overlap,
            covariance_cluster_size=covariance_cluster_size,
            input_artifact_sha256=tuple(input_artifact_sha256),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GaugeCovarianceCalibrationV1:
        if payload.get("schema") != GAUGE_COVARIANCE_CALIBRATION_SCHEMA:
            raise ValueError("unexpected gauge covariance calibration schema")
        if payload.get("version") != GAUGE_COVARIANCE_CALIBRATION_VERSION:
            raise ValueError("unsupported gauge covariance calibration version")
        calibration = payload.get("calibration")
        provenance = payload.get("provenance")
        if not isinstance(calibration, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("calibration artifact is missing calibration or provenance")
        resolution = provenance.get("image_resolution")
        try:
            artifact = cls(
                scale=float(calibration["scale"]),
                rotation=float(calibration["rotation"]),
                translation=float(calibration["translation"]),
                count=int(calibration["count"]),
                trim_quantile=float(calibration["trim_quantile"]),
                calibration_case_ids=tuple(provenance["calibration_case_ids"]),
                source_repository=str(provenance["source_repository"]),
                source_revision=str(provenance["source_revision"]),
                motioncrafter_revision=str(provenance["motioncrafter_revision"]),
                model_identifier=str(provenance["model_identifier"]),
                covariance_method=str(provenance["covariance_method"]),
                image_resolution=None if resolution is None else tuple(resolution),
                window_size=provenance.get("window_size"),
                window_overlap=provenance.get("window_overlap"),
                covariance_cluster_size=provenance.get("covariance_cluster_size"),
                input_artifact_sha256=tuple(
                    provenance.get("input_artifact_sha256", ())
                ),
                metadata=provenance.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid gauge covariance calibration payload") from error
        supplied_id = str(payload.get("artifact_id", ""))
        if supplied_id != artifact.artifact_id:
            raise ValueError("gauge covariance calibration artifact_id does not match content")
        return artifact


def save_gauge_covariance_calibration(
    artifact: GaugeCovarianceCalibrationV1,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(artifact.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_gauge_covariance_calibration(path: str | Path) -> GaugeCovarianceCalibrationV1:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("gauge covariance calibration must contain one JSON object")
    return GaugeCovarianceCalibrationV1.from_dict(payload)



__all__ = [
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "GaugeCovarianceCalibrationV1",
    "load_gauge_covariance_calibration",
    "save_gauge_covariance_calibration",
]
