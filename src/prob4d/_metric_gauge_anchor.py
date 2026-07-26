"""Content-addressed metric priors for the first retained Prob4D gauge."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .observation_contract import canonical_json_sha256, file_sha256
from .sim3 import Sim3

METRIC_GAUGE_ANCHOR_SCHEMA = "prob4d.metric-gauge-anchor"
METRIC_GAUGE_ANCHOR_VERSION = 2
LEGACY_METRIC_GAUGE_ANCHOR_VERSION = 1
FIXED_EXTERNAL_CALIBRATION = "fixed_external_calibration"
PROPAGATED_JOINT_GAUGE_COVARIANCE = "propagated_joint_gauge_covariance"


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite_json_copy(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error


def _safe_manifest_path(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("prediction manifest entry has no path")
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("prediction manifest references a path outside its root") from error
    return target


def prediction_window_sha256(
    manifest_path: str | Path,
    reference_window_id: str,
) -> str:
    """Hash exactly one independently decoded reference-window payload."""

    if not reference_window_id:
        raise ValueError("reference_window_id must be nonempty")
    manifest = Path(manifest_path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    windows = payload.get("overlap_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("prediction manifest has no overlap windows")
    matches = [
        item
        for item in windows
        if isinstance(item, Mapping)
        and str(item.get("window_id", "")) == reference_window_id
    ]
    if len(matches) != 1:
        raise ValueError("reference window must identify exactly one manifest entry")
    target = _safe_manifest_path(manifest.parent, str(matches[0].get("path", "")))
    if not target.is_file():
        raise ValueError("reference-window prediction payload does not exist")
    return file_sha256(target)


@dataclass(frozen=True)
class MetricGaugeAnchor:
    """Metric prior for the first retained overlap-window gauge.

    ``calibration_artifact_sha256`` is optional only for compatibility with old
    in-process experiments. Portable observation export calls
    :meth:`contract_metadata`, which rejects anchors without calibration
    provenance.
    """

    window_id: str
    global_from_local: Sim3
    covariance: np.ndarray
    coordinate_frame: str
    source_kind: str
    source_artifact_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    calibration_artifact_sha256: str | None = None
    case_id: str | None = None

    def __post_init__(self) -> None:
        if not self.window_id or not self.coordinate_frame or not self.source_kind:
            raise ValueError("metric gauge-anchor identities must be nonempty")
        if self.case_id is not None and not str(self.case_id):
            raise ValueError("metric gauge-anchor case_id must be nonempty when supplied")
        _require_sha256(
            self.source_artifact_sha256,
            name="metric gauge-anchor source_artifact_sha256",
        )
        if self.calibration_artifact_sha256 is not None:
            _require_sha256(
                self.calibration_artifact_sha256,
                name="metric gauge-anchor calibration_artifact_sha256",
            )
        covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        if covariance.shape != (7, 7) or not np.all(np.isfinite(covariance)):
            raise ValueError(
                "metric gauge-anchor covariance must have finite shape (7, 7)"
            )
        symmetric = 0.5 * (covariance + covariance.T)
        if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
            raise ValueError("metric gauge-anchor covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(symmetric)) < -1e-12:
            raise ValueError(
                "metric gauge-anchor covariance must be positive semidefinite"
            )
        symmetric.setflags(write=False)
        object.__setattr__(self, "covariance", symmetric)
        object.__setattr__(
            self,
            "case_id",
            None if self.case_id is None else str(self.case_id),
        )
        object.__setattr__(
            self,
            "metadata",
            _finite_json_copy(self.metadata, name="metric gauge-anchor metadata"),
        )

    @property
    def world_frame_id(self) -> str:
        """Machine-readable alias used by downstream contract validators."""

        return self.coordinate_frame

    @property
    def covariance_treatment(self) -> str:
        """Describe how the anchor covariance enters the portable artifact."""

        if np.max(np.abs(self.covariance), initial=0.0) <= 1e-18:
            return FIXED_EXTERNAL_CALIBRATION
        return PROPAGATED_JOINT_GAUGE_COVARIANCE

    @property
    def is_portable(self) -> bool:
        """Return whether the anchor has complete cross-repository provenance."""

        return self.calibration_artifact_sha256 is not None

    def require_portable(self) -> None:
        """Reject legacy anchors that do not identify their calibration artifact."""

        if not self.is_portable:
            raise ValueError(
                "portable observation export requires a metric gauge anchor with "
                "calibration_artifact_sha256; recreate the anchor with "
                "prob4d observation create-anchor"
            )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
            "schema_version": METRIC_GAUGE_ANCHOR_VERSION,
            "case_id": self.case_id,
            "window_id": self.window_id,
            "global_from_local": self.global_from_local.as_vector().tolist(),
            "covariance": self.covariance.tolist(),
            "coordinate_frame": self.coordinate_frame,
            "world_frame_id": self.world_frame_id,
            "metric_units": "m",
            "source_kind": self.source_kind,
            "source_artifact_sha256": self.source_artifact_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "covariance_treatment": self.covariance_treatment,
            "metadata": self.metadata,
        }

    def contract_metadata(self) -> dict[str, Any]:
        """Return the complete validated anchor record embedded in observations."""

        self.require_portable()
        return {
            "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
            "schema_version": METRIC_GAUGE_ANCHOR_VERSION,
            "artifact_id": self.artifact_id,
            "case_id": self.case_id,
            "window_id": self.window_id,
            "world_frame_id": self.world_frame_id,
            "source_kind": self.source_kind,
            "source_artifact_sha256": self.source_artifact_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "covariance_treatment": self.covariance_treatment,
        }

    @property
    def artifact_id(self) -> str:
        return canonical_json_sha256(self.descriptor())


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(descriptor.name)
    try:
        with descriptor:
            descriptor.write(content)
            descriptor.flush()
            os.fsync(descriptor.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_metric_gauge_anchor(path: str | Path) -> MetricGaugeAnchor:
    """Load and content-validate a metric first-window gauge prior."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_name") != METRIC_GAUGE_ANCHOR_SCHEMA:
        raise ValueError("unsupported metric gauge-anchor schema")
    version = int(payload.get("schema_version", -1))
    if version not in {
        LEGACY_METRIC_GAUGE_ANCHOR_VERSION,
        METRIC_GAUGE_ANCHOR_VERSION,
    }:
        raise ValueError("unsupported metric gauge-anchor version")
    if payload.get("metric_units") != "m":
        raise ValueError("metric gauge anchor must declare metric_units='m'")
    expected_artifact_id = payload.pop("artifact_id", None)
    metadata = payload.get("metadata", {})
    calibration_sha256 = payload.get("calibration_artifact_sha256")
    if calibration_sha256 is None and isinstance(metadata, Mapping):
        calibration_sha256 = metadata.get("calibration_artifact_sha256")
    coordinate_frame = str(
        payload.get("world_frame_id", payload.get("coordinate_frame", ""))
    )
    anchor = MetricGaugeAnchor(
        window_id=str(payload["window_id"]),
        global_from_local=Sim3.from_vector(
            np.asarray(payload["global_from_local"], dtype=np.float64)
        ),
        covariance=np.asarray(payload["covariance"], dtype=np.float64),
        coordinate_frame=coordinate_frame,
        source_kind=str(payload["source_kind"]),
        source_artifact_sha256=str(payload["source_artifact_sha256"]),
        calibration_artifact_sha256=(
            None if calibration_sha256 is None else str(calibration_sha256)
        ),
        case_id=(None if payload.get("case_id") is None else str(payload["case_id"])),
        metadata=metadata,
    )
    if expected_artifact_id is not None and expected_artifact_id != anchor.artifact_id:
        if version == LEGACY_METRIC_GAUGE_ANCHOR_VERSION:
            legacy_descriptor = {
                "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
                "schema_version": LEGACY_METRIC_GAUGE_ANCHOR_VERSION,
                "window_id": anchor.window_id,
                "global_from_local": anchor.global_from_local.as_vector().tolist(),
                "covariance": anchor.covariance.tolist(),
                "coordinate_frame": anchor.coordinate_frame,
                "metric_units": "m",
                "source_kind": anchor.source_kind,
                "source_artifact_sha256": anchor.source_artifact_sha256,
                "metadata": anchor.metadata,
            }
            if expected_artifact_id != canonical_json_sha256(legacy_descriptor):
                raise ValueError(
                    "metric gauge-anchor artifact_id does not match its content"
                )
        else:
            raise ValueError("metric gauge-anchor artifact_id does not match its content")
    return anchor


def save_metric_gauge_anchor(path: str | Path, anchor: MetricGaugeAnchor) -> None:
    """Write a canonical metric gauge-anchor JSON document atomically."""

    payload = anchor.descriptor()
    payload["artifact_id"] = anchor.artifact_id
    _atomic_write_text(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


__all__ = [
    "FIXED_EXTERNAL_CALIBRATION",
    "LEGACY_METRIC_GAUGE_ANCHOR_VERSION",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "PROPAGATED_JOINT_GAUGE_COVARIANCE",
    "MetricGaugeAnchor",
    "load_metric_gauge_anchor",
    "prediction_window_sha256",
    "save_metric_gauge_anchor",
]
