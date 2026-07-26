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

from .observation_contract import canonical_json_sha256
from .sim3 import Sim3

METRIC_GAUGE_ANCHOR_SCHEMA = "prob4d.metric-gauge-anchor"
METRIC_GAUGE_ANCHOR_VERSION = 1
CALIBRATION_ARTIFACT_SHA256_KEY = "calibration_artifact_sha256"
FIXED_EXTERNAL_CALIBRATION = "fixed_external_calibration"
PROPAGATED_EXTERNAL_PRIOR = "propagated_external_prior"


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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class MetricGaugeAnchor:
    """Metric prior for the first retained overlap-window gauge.

    The anchor content-addresses its transform, covariance, reference prediction
    payload, and finite metadata. Strict causal-stream export additionally
    requires ``metadata["calibration_artifact_sha256"]`` so the metric prior is
    traceable to the exact external calibration artifact that produced it.
    """

    window_id: str
    global_from_local: Sim3
    covariance: np.ndarray
    coordinate_frame: str
    source_kind: str
    source_artifact_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.window_id or not self.coordinate_frame or not self.source_kind:
            raise ValueError("metric gauge-anchor identities must be nonempty")
        _require_sha256(
            self.source_artifact_sha256,
            name="metric gauge-anchor source_artifact_sha256",
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
        metadata = _finite_json_copy(
            self.metadata,
            name="metric gauge-anchor metadata",
        )
        calibration_digest = metadata.get(CALIBRATION_ARTIFACT_SHA256_KEY)
        if calibration_digest is not None:
            _require_sha256(
                str(calibration_digest),
                name="metric gauge-anchor calibration_artifact_sha256",
            )
            metadata[CALIBRATION_ARTIFACT_SHA256_KEY] = str(calibration_digest)
        object.__setattr__(self, "covariance", symmetric)
        object.__setattr__(self, "metadata", metadata)

    @property
    def calibration_artifact_sha256(self) -> str | None:
        """Return the exact external-calibration digest when declared."""

        value = self.metadata.get(CALIBRATION_ARTIFACT_SHA256_KEY)
        return None if value is None else str(value)

    @property
    def covariance_treatment(self) -> str:
        """Describe whether anchor uncertainty is fixed or propagated."""

        if np.max(np.abs(self.covariance), initial=0.0) <= 1e-18:
            return FIXED_EXTERNAL_CALIBRATION
        return PROPAGATED_EXTERNAL_PRIOR

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
            "schema_version": METRIC_GAUGE_ANCHOR_VERSION,
            "window_id": self.window_id,
            "global_from_local": self.global_from_local.as_vector().tolist(),
            "covariance": self.covariance.tolist(),
            "coordinate_frame": self.coordinate_frame,
            "metric_units": "m",
            "source_kind": self.source_kind,
            "source_artifact_sha256": self.source_artifact_sha256,
            "metadata": self.metadata,
        }

    def contract_metadata(self, *, case_id: str) -> dict[str, Any]:
        """Return the complete anchor record embedded in strict stream v2."""

        calibration_digest = self.calibration_artifact_sha256
        if calibration_digest is None:
            raise ValueError(
                "strict causal-stream export requires metric-anchor metadata "
                "with calibration_artifact_sha256"
            )
        if not case_id:
            raise ValueError("metric gauge-anchor case_id must be nonempty")
        return {
            "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
            "schema_version": METRIC_GAUGE_ANCHOR_VERSION,
            "artifact_id": self.artifact_id,
            "case_id": case_id,
            "window_id": self.window_id,
            "coordinate_frame": self.coordinate_frame,
            "world_frame_id": self.coordinate_frame,
            "metric_units": "m",
            "source_kind": self.source_kind,
            "source_artifact_sha256": self.source_artifact_sha256,
            "calibration_artifact_sha256": calibration_digest,
            "covariance_treatment": self.covariance_treatment,
            "metadata": dict(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return canonical_json_sha256(self.descriptor())


def load_metric_gauge_anchor(path: str | Path) -> MetricGaugeAnchor:
    """Load and content-validate a metric first-window gauge prior."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_name") != METRIC_GAUGE_ANCHOR_SCHEMA:
        raise ValueError("unsupported metric gauge-anchor schema")
    if int(payload.get("schema_version", -1)) != METRIC_GAUGE_ANCHOR_VERSION:
        raise ValueError("unsupported metric gauge-anchor version")
    if payload.get("metric_units") != "m":
        raise ValueError("metric gauge anchor must declare metric_units='m'")
    expected_artifact_id = payload.pop("artifact_id", None)
    anchor = MetricGaugeAnchor(
        window_id=str(payload["window_id"]),
        global_from_local=Sim3.from_vector(
            np.asarray(payload["global_from_local"], dtype=np.float64)
        ),
        covariance=np.asarray(payload["covariance"], dtype=np.float64),
        coordinate_frame=str(payload["coordinate_frame"]),
        source_kind=str(payload["source_kind"]),
        source_artifact_sha256=str(payload["source_artifact_sha256"]),
        metadata=payload.get("metadata", {}),
    )
    if expected_artifact_id is not None and expected_artifact_id != anchor.artifact_id:
        raise ValueError("metric gauge-anchor artifact_id does not match its content")
    return anchor


def save_metric_gauge_anchor(path: str | Path, anchor: MetricGaugeAnchor) -> None:
    """Atomically write a canonical metric gauge-anchor JSON document."""

    payload = anchor.descriptor()
    payload["artifact_id"] = anchor.artifact_id
    _atomic_write_text(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


__all__ = [
    "CALIBRATION_ARTIFACT_SHA256_KEY",
    "FIXED_EXTERNAL_CALIBRATION",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "PROPAGATED_EXTERNAL_PRIOR",
    "MetricGaugeAnchor",
    "load_metric_gauge_anchor",
    "save_metric_gauge_anchor",
]
