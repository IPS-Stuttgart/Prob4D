"""Content-addressed metric-frame anchors for portable observation exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .observation_contract import file_sha256
from .sim3 import Sim3

METRIC_GAUGE_ANCHOR_SCHEMA = "prob4d.metric-gauge-anchor"
METRIC_GAUGE_ANCHOR_VERSION = 1


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validated_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metric-anchor metadata must be finite JSON data") from error


def prediction_bundle_sha256(manifest_path: str | Path) -> str:
    """Hash a prediction manifest and every archive it references.

    This is stronger than hashing ``predictions.json`` alone: replacing a referenced
    NPZ while leaving the manifest untouched changes the source artifact identity.
    """

    manifest = Path(manifest_path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    referenced: list[str] = []
    for key in ("disjoint_baseline", "latent_linear_baseline"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"prediction manifest is missing {key}")
        referenced.append(value)
    windows = payload.get("overlap_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("prediction manifest has no overlap windows")
    for item in windows:
        if not isinstance(item, Mapping):
            raise ValueError("overlap-window manifest entry must be a mapping")
        value = item.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError("overlap-window manifest entry has no path")
        referenced.append(value)

    root = manifest.parent.resolve()
    digest = hashlib.sha256()
    digest.update(_canonical_json(payload))
    for relative in sorted(set(referenced)):
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("prediction manifest references a path outside its root") from error
        if not target.is_file():
            raise ValueError(f"prediction bundle file does not exist: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(file_sha256(target).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class MetricGaugeAnchorV1:
    """A metric ``Sim(3)`` for the first retained Prob4D window.

    Version 1 of ``phys4d.observation_belief`` supports one coherent low-rank
    factor group per row. The causal exporter therefore accepts this anchor only
    when its covariance is zero and uses the low-rank factor for relative
    per-window gauge uncertainty. Uncertain metric anchors remain available
    through Prob4D's explicit-gauge ``ObservationFactorBundle`` path.
    """

    case_id: str
    world_frame_id: str
    reference_window_id: str
    source_artifact_sha256: str
    calibration_artifact_sha256: str
    global_from_reference: Sim3
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((7, 7)))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id or not self.world_frame_id or not self.reference_window_id:
            raise ValueError("metric-anchor identities must be nonempty")
        if not isinstance(self.global_from_reference, Sim3):
            raise TypeError("global_from_reference must be a Sim3")
        _validate_sha256(
            self.source_artifact_sha256,
            name="source_artifact_sha256",
        )
        _validate_sha256(
            self.calibration_artifact_sha256,
            name="calibration_artifact_sha256",
        )
        covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        if covariance.shape != (7, 7) or not np.all(np.isfinite(covariance)):
            raise ValueError("metric-anchor covariance must have finite shape (7, 7)")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("metric-anchor covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T))) < -1e-12:
            raise ValueError("metric-anchor covariance must be positive semidefinite")
        covariance.setflags(write=False)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
            "schema_version": METRIC_GAUGE_ANCHOR_VERSION,
            "case_id": self.case_id,
            "world_frame_id": self.world_frame_id,
            "reference_window_id": self.reference_window_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "global_from_reference_vector": self.global_from_reference.as_vector().tolist(),
            "covariance": self.covariance.tolist(),
            "metadata": self.metadata,
        }

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    @property
    def is_fixed(self) -> bool:
        return bool(np.max(np.abs(self.covariance), initial=0.0) <= 1e-18)

    def require_fixed(self) -> None:
        if not self.is_fixed:
            raise ValueError(
                "ObservationBeliefV1 cannot encode both uncertain global and "
                "per-window gauges; provide a fixed metric calibration or use "
                "ObservationFactorBundle"
            )


def save_metric_gauge_anchor(path: str | Path, anchor: MetricGaugeAnchorV1) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = anchor.descriptor()
    payload["artifact_id"] = anchor.artifact_id
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_metric_gauge_anchor(path: str | Path) -> MetricGaugeAnchorV1:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_name") != METRIC_GAUGE_ANCHOR_SCHEMA:
        raise ValueError("unsupported metric-gauge-anchor schema")
    if int(payload.get("schema_version", -1)) != METRIC_GAUGE_ANCHOR_VERSION:
        raise ValueError("unsupported metric-gauge-anchor version")
    anchor = MetricGaugeAnchorV1(
        case_id=str(payload["case_id"]),
        world_frame_id=str(payload["world_frame_id"]),
        reference_window_id=str(payload["reference_window_id"]),
        source_artifact_sha256=str(payload["source_artifact_sha256"]),
        calibration_artifact_sha256=str(payload["calibration_artifact_sha256"]),
        global_from_reference=Sim3.from_vector(
            np.asarray(payload["global_from_reference_vector"], dtype=np.float64)
        ),
        covariance=np.asarray(payload["covariance"], dtype=np.float64),
        metadata=payload.get("metadata", {}),
    )
    expected = str(payload.get("artifact_id", ""))
    _validate_sha256(expected, name="artifact_id")
    if anchor.artifact_id != expected:
        raise ValueError("metric-gauge-anchor digest does not match its payload")
    return anchor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--world-frame-id", required=True)
    parser.add_argument("--reference-window-id", required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument(
        "--sim3-vector",
        type=float,
        nargs=7,
        metavar=("LOG_SCALE", "RX", "RY", "RZ", "TX", "TY", "TZ"),
        required=True,
    )
    args = parser.parse_args(argv)

    anchor = MetricGaugeAnchorV1(
        case_id=args.case_id,
        world_frame_id=args.world_frame_id,
        reference_window_id=args.reference_window_id,
        source_artifact_sha256=prediction_bundle_sha256(args.predictions_manifest),
        calibration_artifact_sha256=file_sha256(args.calibration_artifact),
        global_from_reference=Sim3.from_vector(np.asarray(args.sim3_vector)),
        covariance=np.zeros((7, 7)),
        metadata={
            "calibration_artifact_name": args.calibration_artifact.name,
            "covariance_treatment": "fixed_external_calibration",
        },
    )
    save_metric_gauge_anchor(args.output_json, anchor)
    print(
        json.dumps(
            {
                "artifact_id": anchor.artifact_id,
                "source_artifact_sha256": anchor.source_artifact_sha256,
                "world_frame_id": anchor.world_frame_id,
                "reference_window_id": anchor.reference_window_id,
                "output": str(args.output_json.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "MetricGaugeAnchorV1",
    "load_metric_gauge_anchor",
    "prediction_bundle_sha256",
    "save_metric_gauge_anchor",
]
