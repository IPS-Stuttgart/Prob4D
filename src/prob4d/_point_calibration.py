"""Content-addressed calibration for dense point uncertainty."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._calibration_common import (
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION,
    _artifact_id,
    _common_descriptor,
    _validate_common_provenance,
)
from .uncertainty import CalibrationReport, DepthDisagreementModel, StructuredCovariance

@dataclass(frozen=True)
class PointUncertaintyCalibrationV1:
    """Content-addressed held-out calibration of the dense point variance model."""

    parallel_floor: float
    parallel_depth_coefficient: float
    lateral_floor: float
    lateral_depth_coefficient: float
    disagreement_gain: float
    parallel_scale: float
    lateral_scale: float
    count: int
    trim_quantile: float
    parallel_scale_update: float
    lateral_scale_update: float
    parallel_normalized_mse: float
    lateral_normalized_mse: float
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
        nonnegative = np.asarray(
            [
                self.parallel_floor,
                self.parallel_depth_coefficient,
                self.lateral_floor,
                self.lateral_depth_coefficient,
                self.disagreement_gain,
            ],
            dtype=np.float64,
        )
        positive = np.asarray(
            [
                self.parallel_scale,
                self.lateral_scale,
                self.parallel_scale_update,
                self.lateral_scale_update,
            ],
            dtype=np.float64,
        )
        diagnostics = np.asarray(
            [self.parallel_normalized_mse, self.lateral_normalized_mse],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(nonnegative)) or np.any(nonnegative < 0.0):
            raise ValueError("point uncertainty model coefficients must be finite and non-negative")
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("point uncertainty scales must be finite and positive")
        if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
            raise ValueError("point calibration diagnostics must be finite and non-negative")
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
        for name, value in (
            ("parallel_floor", nonnegative[0]),
            ("parallel_depth_coefficient", nonnegative[1]),
            ("lateral_floor", nonnegative[2]),
            ("lateral_depth_coefficient", nonnegative[3]),
            ("disagreement_gain", nonnegative[4]),
            ("parallel_scale", positive[0]),
            ("lateral_scale", positive[1]),
            ("parallel_scale_update", positive[2]),
            ("lateral_scale_update", positive[3]),
            ("parallel_normalized_mse", diagnostics[0]),
            ("lateral_normalized_mse", diagnostics[1]),
        ):
            object.__setattr__(self, name, float(value))
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
    def model(self) -> DepthDisagreementModel:
        return DepthDisagreementModel(
            parallel_floor=self.parallel_floor,
            parallel_depth_coefficient=self.parallel_depth_coefficient,
            lateral_floor=self.lateral_floor,
            lateral_depth_coefficient=self.lateral_depth_coefficient,
            disagreement_gain=self.disagreement_gain,
            parallel_scale=self.parallel_scale,
            lateral_scale=self.lateral_scale,
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
            "version": POINT_UNCERTAINTY_CALIBRATION_VERSION,
            "calibration": {
                "parallel_floor": self.parallel_floor,
                "parallel_depth_coefficient": self.parallel_depth_coefficient,
                "lateral_floor": self.lateral_floor,
                "lateral_depth_coefficient": self.lateral_depth_coefficient,
                "disagreement_gain": self.disagreement_gain,
                "parallel_scale": self.parallel_scale,
                "lateral_scale": self.lateral_scale,
                "count": self.count,
                "trim_quantile": self.trim_quantile,
                "parallel_scale_update": self.parallel_scale_update,
                "lateral_scale_update": self.lateral_scale_update,
                "parallel_normalized_mse": self.parallel_normalized_mse,
                "lateral_normalized_mse": self.lateral_normalized_mse,
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
    def from_model(
        cls,
        model: DepthDisagreementModel,
        report: CalibrationReport,
        *,
        calibration_case_ids: Sequence[str],
        source_repository: str,
        source_revision: str,
        motioncrafter_revision: str,
        model_identifier: str,
        covariance_method: str,
        trim_quantile: float,
        image_resolution: tuple[int, int] | None = None,
        window_size: int | None = None,
        window_overlap: int | None = None,
        covariance_cluster_size: int | None = None,
        input_artifact_sha256: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> PointUncertaintyCalibrationV1:
        return cls(
            parallel_floor=model.parallel_floor,
            parallel_depth_coefficient=model.parallel_depth_coefficient,
            lateral_floor=model.lateral_floor,
            lateral_depth_coefficient=model.lateral_depth_coefficient,
            disagreement_gain=model.disagreement_gain,
            parallel_scale=model.parallel_scale,
            lateral_scale=model.lateral_scale,
            count=report.count,
            trim_quantile=trim_quantile,
            parallel_scale_update=report.parallel_scale_update,
            lateral_scale_update=report.lateral_scale_update,
            parallel_normalized_mse=report.parallel_normalized_mse,
            lateral_normalized_mse=report.lateral_normalized_mse,
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
    def fit(
        cls,
        model: DepthDisagreementModel,
        errors: np.ndarray,
        covariance: StructuredCovariance,
        *,
        calibration_case_ids: Sequence[str],
        source_repository: str,
        source_revision: str,
        motioncrafter_revision: str,
        model_identifier: str,
        covariance_method: str,
        mask: np.ndarray | None = None,
        trim_quantile: float = 0.99,
        image_resolution: tuple[int, int] | None = None,
        window_size: int | None = None,
        window_overlap: int | None = None,
        covariance_cluster_size: int | None = None,
        input_artifact_sha256: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> PointUncertaintyCalibrationV1:
        calibrated, report = model.calibrate(
            errors,
            covariance,
            mask=mask,
            trim_quantile=trim_quantile,
        )
        return cls.from_model(
            calibrated,
            report,
            calibration_case_ids=calibration_case_ids,
            source_repository=source_repository,
            source_revision=source_revision,
            motioncrafter_revision=motioncrafter_revision,
            model_identifier=model_identifier,
            covariance_method=covariance_method,
            trim_quantile=trim_quantile,
            image_resolution=image_resolution,
            window_size=window_size,
            window_overlap=window_overlap,
            covariance_cluster_size=covariance_cluster_size,
            input_artifact_sha256=input_artifact_sha256,
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PointUncertaintyCalibrationV1:
        if payload.get("schema") != POINT_UNCERTAINTY_CALIBRATION_SCHEMA:
            raise ValueError("unexpected point uncertainty calibration schema")
        if payload.get("version") != POINT_UNCERTAINTY_CALIBRATION_VERSION:
            raise ValueError("unsupported point uncertainty calibration version")
        calibration = payload.get("calibration")
        provenance = payload.get("provenance")
        if not isinstance(calibration, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("calibration artifact is missing calibration or provenance")
        resolution = provenance.get("image_resolution")
        try:
            artifact = cls(
                parallel_floor=float(calibration["parallel_floor"]),
                parallel_depth_coefficient=float(
                    calibration["parallel_depth_coefficient"]
                ),
                lateral_floor=float(calibration["lateral_floor"]),
                lateral_depth_coefficient=float(
                    calibration["lateral_depth_coefficient"]
                ),
                disagreement_gain=float(calibration["disagreement_gain"]),
                parallel_scale=float(calibration["parallel_scale"]),
                lateral_scale=float(calibration["lateral_scale"]),
                count=int(calibration["count"]),
                trim_quantile=float(calibration["trim_quantile"]),
                parallel_scale_update=float(calibration["parallel_scale_update"]),
                lateral_scale_update=float(calibration["lateral_scale_update"]),
                parallel_normalized_mse=float(
                    calibration["parallel_normalized_mse"]
                ),
                lateral_normalized_mse=float(calibration["lateral_normalized_mse"]),
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
            raise ValueError("invalid point uncertainty calibration payload") from error
        supplied_id = str(payload.get("artifact_id", ""))
        if supplied_id != artifact.artifact_id:
            raise ValueError("point uncertainty calibration artifact_id does not match content")
        return artifact


def save_point_uncertainty_calibration(
    artifact: PointUncertaintyCalibrationV1,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(artifact.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_point_uncertainty_calibration(path: str | Path) -> PointUncertaintyCalibrationV1:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("point uncertainty calibration must contain one JSON object")
    return PointUncertaintyCalibrationV1.from_dict(payload)



__all__ = [
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "PointUncertaintyCalibrationV1",
    "load_point_uncertainty_calibration",
    "save_point_uncertainty_calibration",
]
