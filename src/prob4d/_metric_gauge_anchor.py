"""Content-addressed metric priors for the first retained Prob4D gauge."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_json_number,
    require_mapping,
    require_sha256,
)
from .observation_contract import canonical_json_sha256
from .sim3 import Sim3

METRIC_GAUGE_ANCHOR_SCHEMA = "prob4d.metric-gauge-anchor"
METRIC_GAUGE_ANCHOR_VERSION = 1
CALIBRATION_ARTIFACT_SHA256_KEY = "calibration_artifact_sha256"
FIXED_EXTERNAL_CALIBRATION = "fixed_external_calibration"
PROPAGATED_EXTERNAL_PRIOR = "propagated_external_prior"

_ANCHOR_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "window_id",
        "global_from_local",
        "covariance",
        "coordinate_frame",
        "metric_units",
        "source_kind",
        "source_artifact_sha256",
        "metadata",
    }
)
_ANCHOR_SERIALIZED_FIELDS = _ANCHOR_DESCRIPTOR_FIELDS | {"artifact_id"}


def _require_numeric_array(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must have shape {shape}")
    try:
        object_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must have shape {shape}") from error
    if object_array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    normalized = np.empty(shape, dtype=np.float64)
    for index in np.ndindex(shape):
        normalized[index] = require_json_number(
            object_array[index],
            name=f"{name}{index}",
        )
    return normalized


def _require_anchor_fields(payload: Mapping[str, Any]) -> None:
    actual = frozenset(payload)
    if actual in {_ANCHOR_DESCRIPTOR_FIELDS, _ANCHOR_SERIALIZED_FIELDS}:
        return
    require_exact_fields(
        payload,
        _ANCHOR_SERIALIZED_FIELDS,
        name="metric gauge-anchor artifact",
    )


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
        window_id = require_exact_string(
            self.window_id,
            name="metric gauge-anchor window_id",
        )
        coordinate_frame = require_exact_string(
            self.coordinate_frame,
            name="metric gauge-anchor coordinate_frame",
        )
        source_kind = require_exact_string(
            self.source_kind,
            name="metric gauge-anchor source_kind",
        )
        source_artifact_sha256 = require_sha256(
            self.source_artifact_sha256,
            name="metric gauge-anchor source_artifact_sha256",
        )
        if not isinstance(self.global_from_local, Sim3):
            raise TypeError("metric gauge-anchor global_from_local must be a Sim3")

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

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="metric gauge-anchor metadata",
        )
        calibration_digest = metadata.get(CALIBRATION_ARTIFACT_SHA256_KEY)
        if calibration_digest is not None:
            require_sha256(
                calibration_digest,
                name="metric gauge-anchor calibration_artifact_sha256",
            )

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(
            self,
            "source_artifact_sha256",
            source_artifact_sha256,
        )
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
            "metadata": plain_json(self.metadata),
        }

    def contract_metadata(self, *, case_id: str) -> dict[str, Any]:
        """Return the complete anchor record embedded in strict stream v2."""

        calibration_digest = self.calibration_artifact_sha256
        if calibration_digest is None:
            raise ValueError(
                "strict causal-stream export requires metric-anchor metadata "
                "with calibration_artifact_sha256"
            )
        validated_case_id = require_exact_string(
            case_id,
            name="metric gauge-anchor case_id",
        )
        return {
            "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
            "schema_version": METRIC_GAUGE_ANCHOR_VERSION,
            "artifact_id": self.artifact_id,
            "case_id": validated_case_id,
            "window_id": self.window_id,
            "coordinate_frame": self.coordinate_frame,
            "world_frame_id": self.coordinate_frame,
            "metric_units": "m",
            "source_kind": self.source_kind,
            "source_artifact_sha256": self.source_artifact_sha256,
            "calibration_artifact_sha256": calibration_digest,
            "covariance_treatment": self.covariance_treatment,
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return canonical_json_sha256(self.descriptor())


def load_metric_gauge_anchor(path: str | Path) -> MetricGaugeAnchor:
    """Load and content-validate a metric first-window gauge prior."""

    payload = load_json_object(path, name="metric gauge anchor")
    _require_anchor_fields(payload)

    schema_name = require_exact_string(
        payload["schema_name"],
        name="metric gauge-anchor schema_name",
    )
    if schema_name != METRIC_GAUGE_ANCHOR_SCHEMA:
        raise ValueError("unsupported metric gauge-anchor schema")
    schema_version = require_exact_integer(
        payload["schema_version"],
        name="metric gauge-anchor schema_version",
        minimum=1,
    )
    if schema_version != METRIC_GAUGE_ANCHOR_VERSION:
        raise ValueError("unsupported metric gauge-anchor version")
    metric_units = require_exact_string(
        payload["metric_units"],
        name="metric gauge-anchor metric_units",
    )
    if metric_units != "m":
        raise ValueError("metric gauge anchor must declare metric_units='m'")

    expected_artifact_id = (
        None
        if "artifact_id" not in payload
        else require_sha256(
            payload["artifact_id"],
            name="metric gauge-anchor artifact_id",
        )
    )
    metadata = require_mapping(
        payload["metadata"],
        name="metric gauge-anchor metadata",
    )
    anchor = MetricGaugeAnchor(
        window_id=require_exact_string(
            payload["window_id"],
            name="metric gauge-anchor window_id",
        ),
        global_from_local=Sim3.from_vector(
            _require_numeric_array(
                payload["global_from_local"],
                shape=(7,),
                name="metric gauge-anchor global_from_local",
            )
        ),
        covariance=_require_numeric_array(
            payload["covariance"],
            shape=(7, 7),
            name="metric gauge-anchor covariance",
        ),
        coordinate_frame=require_exact_string(
            payload["coordinate_frame"],
            name="metric gauge-anchor coordinate_frame",
        ),
        source_kind=require_exact_string(
            payload["source_kind"],
            name="metric gauge-anchor source_kind",
        ),
        source_artifact_sha256=require_sha256(
            payload["source_artifact_sha256"],
            name="metric gauge-anchor source_artifact_sha256",
        ),
        metadata=metadata,
    )
    if expected_artifact_id is not None and expected_artifact_id != anchor.artifact_id:
        raise ValueError("metric gauge-anchor artifact_id does not match its content")
    return anchor


def save_metric_gauge_anchor(
    path: str | Path,
    anchor: MetricGaugeAnchor,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish and verify a canonical metric gauge-anchor artifact."""

    destination = Path(path)
    payload = anchor.descriptor()
    payload["artifact_id"] = anchor.artifact_id
    atomic_write_text(
        destination,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        overwrite=overwrite,
    )
    restored = load_metric_gauge_anchor(destination)
    if restored.descriptor() != anchor.descriptor():
        raise RuntimeError("published metric gauge-anchor differs from its source")


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
