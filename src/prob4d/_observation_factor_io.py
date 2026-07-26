"""Checksum-bound serialization for Prob4D observation-factor bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ._observation_factor_bundle import (
    GAUGE_PARAMETERIZATION,
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
)
from ._observation_factor_types import ObservationFactor
from .gauge import GaugeEstimate
from .sim3 import Sim3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_observation_factor_bundle(
    bundle: ObservationFactorBundle,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write one checksum-bound JSON manifest and non-pickled NPZ payload."""

    manifest = Path(manifest_path)
    payload = (
        Path(payload_path)
        if payload_path is not None
        else manifest.with_suffix(".npz")
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    gauges: list[dict[str, Any]] = []
    for index, gauge in enumerate(bundle.gauges):
        prefix = f"gauge_{index:04d}"
        arrays[f"{prefix}__mean"] = gauge.global_from_local.as_vector()
        arrays[f"{prefix}__covariance"] = np.asarray(gauge.covariance)
        gauges.append(
            {
                "gauge_id": gauge.window_id,
                "mean_key": f"{prefix}__mean",
                "covariance_key": f"{prefix}__covariance",
            }
        )
    factors: list[dict[str, Any]] = []
    for index, factor in enumerate(bundle.factors):
        prefix = f"factor_{index:04d}"
        array_names = {
            "point_ids": f"{prefix}__point_ids",
            "points_local_m": f"{prefix}__points_local_m",
            "valid_mask": f"{prefix}__valid_mask",
            "local_covariance_m2": f"{prefix}__local_covariance_m2",
            "association_probability": f"{prefix}__association_probability",
        }
        arrays[array_names["point_ids"]] = factor.point_ids
        arrays[array_names["points_local_m"]] = factor.points_local_m
        arrays[array_names["valid_mask"]] = factor.valid_mask
        arrays[array_names["local_covariance_m2"]] = factor.local_covariance_m2
        arrays[array_names["association_probability"]] = factor.association_probability
        ray_key = None
        if factor.ray_directions_local is not None:
            ray_key = f"{prefix}__ray_directions_local"
            arrays[ray_key] = factor.ray_directions_local
        factors.append(
            {
                "factor_id": factor.factor_id,
                "frame_index": factor.frame_index,
                "view_id": factor.view_id,
                "window_id": factor.window_id,
                "gauge_id": factor.gauge_id,
                "correlation_group_id": factor.correlation_group_id,
                "causal_frame_limit": factor.causal_frame_limit,
                "arrays": array_names,
                "ray_directions_local_key": ray_key,
            }
        )
    np.savez_compressed(payload, **arrays)
    record = {
        "schema": OBSERVATION_FACTOR_SCHEMA,
        "schema_version": bundle.schema_version,
        "gauge_parameterization": GAUGE_PARAMETERIZATION,
        "sequence_id": bundle.sequence_id,
        "source_revision": bundle.source_revision,
        "causal_frame_limit": bundle.causal_frame_limit,
        "metadata": dict(bundle.metadata),
        "payload": {
            "path": os.path.relpath(payload, manifest.parent),
            "sha256": _sha256(payload),
            "allow_pickle": False,
        },
        "gauges": gauges,
        "factors": factors,
    }
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest, payload


def load_observation_factor_bundle(
    manifest_path: str | Path,
) -> ObservationFactorBundle:
    """Load and validate a checksum-bound observation-factor bundle."""

    manifest = Path(manifest_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") != OBSERVATION_FACTOR_SCHEMA:
        raise ValueError("manifest is not a Prob4D observation-factor bundle")
    if record.get("schema_version") != OBSERVATION_FACTOR_SCHEMA_VERSION:
        raise ValueError("unsupported observation-factor schema version")
    if record.get("gauge_parameterization") != GAUGE_PARAMETERIZATION:
        raise ValueError("unsupported gauge parameterization")
    payload = manifest.parent / record["payload"]["path"]
    if _sha256(payload) != record["payload"]["sha256"]:
        raise ValueError("observation-factor payload checksum mismatch")
    gauges: list[GaugeEstimate] = []
    factors: list[ObservationFactor] = []
    with np.load(payload, allow_pickle=False) as arrays:
        for gauge_record in record["gauges"]:
            gauges.append(
                GaugeEstimate(
                    window_id=str(gauge_record["gauge_id"]),
                    global_from_local=Sim3.from_vector(
                        arrays[gauge_record["mean_key"]]
                    ),
                    covariance=arrays[gauge_record["covariance_key"]],
                )
            )
        for factor_record in record["factors"]:
            keys = factor_record["arrays"]
            ray_key = factor_record.get("ray_directions_local_key")
            factors.append(
                ObservationFactor(
                    factor_id=str(factor_record["factor_id"]),
                    frame_index=int(factor_record["frame_index"]),
                    view_id=str(factor_record["view_id"]),
                    window_id=str(factor_record["window_id"]),
                    gauge_id=str(factor_record["gauge_id"]),
                    point_ids=arrays[keys["point_ids"]],
                    points_local_m=arrays[keys["points_local_m"]],
                    valid_mask=arrays[keys["valid_mask"]],
                    local_covariance_m2=arrays[keys["local_covariance_m2"]],
                    association_probability=arrays[
                        keys["association_probability"]
                    ],
                    correlation_group_id=str(
                        factor_record["correlation_group_id"]
                    ),
                    causal_frame_limit=int(factor_record["causal_frame_limit"]),
                    ray_directions_local=(
                        arrays[ray_key] if ray_key is not None else None
                    ),
                )
            )
    return ObservationFactorBundle(
        sequence_id=str(record["sequence_id"]),
        factors=tuple(factors),
        gauges=tuple(gauges),
        source_revision=str(record["source_revision"]),
        causal_frame_limit=int(record["causal_frame_limit"]),
        metadata=record.get("metadata", {}),
        schema_version=int(record["schema_version"]),
    )
