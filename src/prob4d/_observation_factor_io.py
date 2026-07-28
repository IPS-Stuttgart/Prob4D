"""Checksum-bound serialization for Prob4D observation-factor bundles."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ._observation_factor_bundle import (
    GAUGE_PARAMETERIZATION,
    LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION,
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
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


def _write_observation_factor_bundle(
    bundle: ObservationFactorBundle,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
    schema_version: int,
) -> tuple[Path, Path]:
    if schema_version not in {
        PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
        OBSERVATION_FACTOR_SCHEMA_VERSION,
    }:
        raise ValueError("writer supports only observation-factor schema v3 or v4")
    if (
        schema_version == PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION
        and bundle.gauge_covariance_semantics != "marginal-blocks-only"
    ):
        raise ValueError(
            "schema-v3 compatibility output cannot represent joint cross-window "
            "gauge covariance"
        )

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
    joint_covariance_key = "joint_gauge_covariance"
    if schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION:
        arrays[joint_covariance_key] = bundle.joint_gauge_covariance
    factors: list[dict[str, Any]] = []
    for index, factor in enumerate(bundle.factors):
        prefix = f"factor_{index:04d}"
        array_names = {
            "point_ids": f"{prefix}__point_ids",
            "points_local_m": f"{prefix}__points_local_m",
            "valid_mask": f"{prefix}__valid_mask",
            "local_covariance_m2": f"{prefix}__local_covariance_m2",
            "association_probability": f"{prefix}__association_probability",
            "prior_reliability": f"{prefix}__prior_reliability",
        }
        arrays[array_names["point_ids"]] = factor.point_ids
        arrays[array_names["points_local_m"]] = factor.points_local_m
        arrays[array_names["valid_mask"]] = factor.valid_mask
        arrays[array_names["local_covariance_m2"]] = factor.local_covariance_m2
        arrays[array_names["association_probability"]] = factor.association_probability
        arrays[array_names["prior_reliability"]] = factor.prior_reliability
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
                "causal_frame_stop": factor.causal_frame_stop,
                "prior_nominal_probability": factor.prior_nominal_probability,
                "composite_weight": factor.composite_weight,
                "arrays": array_names,
                "ray_directions_local_key": ray_key,
            }
        )
    np.savez_compressed(payload, **arrays)
    record = {
        "schema": OBSERVATION_FACTOR_SCHEMA,
        "schema_version": schema_version,
        "gauge_parameterization": GAUGE_PARAMETERIZATION,
        "sequence_id": bundle.sequence_id,
        "case_id": bundle.case_id,
        "stream_id": bundle.stream_id,
        "source_repository": bundle.source_repository,
        "source_revision": bundle.source_revision,
        "causal_frame_stop": bundle.causal_frame_stop,
        "causal_frame_stop_convention": "exclusive",
        "metadata": dict(bundle.metadata),
        "payload": {
            "path": os.path.relpath(payload, manifest.parent),
            "sha256": _sha256(payload),
            "allow_pickle": False,
        },
        "gauges": gauges,
        "factors": factors,
    }
    if schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION:
        record["gauge_covariance"] = {
            "semantics": bundle.gauge_covariance_semantics,
            "joint_covariance_key": joint_covariance_key,
            "ordered_gauge_ids": [gauge.window_id for gauge in bundle.gauges],
            "cross_window_covariance_preserved": (
                bundle.cross_window_gauge_covariance_preserved
            ),
            "diagonal_blocks_match_gauge_marginals": True,
        }
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest, payload


def write_observation_factor_bundle(
    bundle: ObservationFactorBundle,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a schema-v4 manifest with an ordered joint gauge covariance."""

    return _write_observation_factor_bundle(
        bundle,
        manifest_path,
        payload_path=payload_path,
        schema_version=OBSERVATION_FACTOR_SCHEMA_VERSION,
    )


def write_observation_factor_bundle_v3(
    bundle: ObservationFactorBundle,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the frozen schema-v3 marginal-block compatibility representation."""

    return _write_observation_factor_bundle(
        bundle,
        manifest_path,
        payload_path=payload_path,
        schema_version=PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
    )


def _causal_frame_stop(record: dict[str, Any], *, schema_version: int) -> int:
    if schema_version in {
        PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
        OBSERVATION_FACTOR_SCHEMA_VERSION,
    }:
        convention = record.get("causal_frame_stop_convention")
        if convention is not None and convention != "exclusive":
            raise ValueError("nonlegacy causal frame stop must be exclusive")
        return int(record["causal_frame_stop"])
    if schema_version == LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION:
        return int(record["causal_frame_limit"]) + 1
    raise ValueError("unsupported observation-factor schema version")


def load_observation_factor_bundle(
    manifest_path: str | Path,
) -> ObservationFactorBundle:
    """Load schema v4 or upgrade schema-v2/v3 bundles without ambiguity."""

    manifest = Path(manifest_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") != OBSERVATION_FACTOR_SCHEMA:
        raise ValueError("manifest is not a Prob4D observation-factor bundle")
    schema_version = int(record.get("schema_version", -1))
    if schema_version not in {
        LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION,
        PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
        OBSERVATION_FACTOR_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported observation-factor schema version")
    if record.get("gauge_parameterization") != GAUGE_PARAMETERIZATION:
        raise ValueError("unsupported gauge parameterization")
    bundle_causal_frame_stop = _causal_frame_stop(
        record,
        schema_version=schema_version,
    )
    payload = manifest.parent / record["payload"]["path"]
    if _sha256(payload) != record["payload"]["sha256"]:
        raise ValueError("observation-factor payload checksum mismatch")
    gauges: list[GaugeEstimate] = []
    factors: list[ObservationFactor] = []
    joint_gauge_covariance: np.ndarray | None = None
    gauge_covariance_semantics = "marginal-blocks-only"
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
        if schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION:
            covariance_record = record.get("gauge_covariance")
            if not isinstance(covariance_record, dict):
                raise ValueError("schema-v4 manifest is missing gauge_covariance")
            gauge_covariance_semantics = str(covariance_record.get("semantics", ""))
            preserved = covariance_record.get("cross_window_covariance_preserved")
            if not isinstance(preserved, bool):
                raise ValueError(
                    "schema-v4 cross_window_covariance_preserved must be Boolean"
                )
            if preserved != (gauge_covariance_semantics == "joint-cross-window"):
                raise ValueError(
                    "schema-v4 gauge covariance semantics contradict the preservation flag"
                )
            if covariance_record.get("diagonal_blocks_match_gauge_marginals") is not True:
                raise ValueError(
                    "schema-v4 gauge covariance must declare matching marginal blocks"
                )
            joint_key = covariance_record.get("joint_covariance_key")
            if not isinstance(joint_key, str) or not joint_key:
                raise ValueError(
                    "schema-v4 gauge covariance is missing joint_covariance_key"
                )
            ordered_gauge_ids = covariance_record.get("ordered_gauge_ids")
            if ordered_gauge_ids != [gauge.window_id for gauge in gauges]:
                raise ValueError(
                    "schema-v4 joint gauge covariance order differs from gauge records"
                )
            joint_gauge_covariance = np.asarray(arrays[joint_key])
        for factor_record in record["factors"]:
            keys = factor_record["arrays"]
            ray_key = factor_record.get("ray_directions_local_key")
            association = arrays[keys["association_probability"]]
            factor_causal_frame_stop = _causal_frame_stop(
                factor_record,
                schema_version=schema_version,
            )
            reliability_key = keys.get("prior_reliability")
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
                    association_probability=association,
                    prior_reliability=(
                        np.ones(len(association), dtype=np.float64)
                        if reliability_key is None
                        else arrays[reliability_key]
                    ),
                    prior_nominal_probability=float(
                        factor_record.get("prior_nominal_probability", 1.0)
                    ),
                    composite_weight=float(
                        factor_record.get("composite_weight", 1.0)
                    ),
                    correlation_group_id=str(
                        factor_record["correlation_group_id"]
                    ),
                    causal_frame_stop=factor_causal_frame_stop,
                    ray_directions_local=(
                        arrays[ray_key] if ray_key is not None else None
                    ),
                )
            )
    metadata = dict(record.get("metadata", {}))
    if schema_version != OBSERVATION_FACTOR_SCHEMA_VERSION:
        metadata = {
            **metadata,
            "loaded_from_schema_version": schema_version,
            "source_gauge_covariance_semantics": "marginal-blocks-only",
            "source_cross_window_gauge_covariance_preserved": False,
        }
        if schema_version == LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION:
            metadata.update(
                {
                    "legacy_causal_frame_limit_upgraded_to_exclusive_stop": True,
                    "legacy_missing_reliability_defaults": "ones",
                }
            )
    return ObservationFactorBundle(
        sequence_id=str(record["sequence_id"]),
        factors=tuple(factors),
        gauges=tuple(gauges),
        source_revision=str(record["source_revision"]),
        causal_frame_stop=bundle_causal_frame_stop,
        case_id=str(record.get("case_id", record["sequence_id"])),
        stream_id=str(record.get("stream_id", record["sequence_id"])),
        source_repository=str(
            record.get("source_repository", "FlorianPfaff/Prob4D")
        ),
        metadata=metadata,
        schema_version=OBSERVATION_FACTOR_SCHEMA_VERSION,
        joint_gauge_covariance=joint_gauge_covariance,
        gauge_covariance_semantics=gauge_covariance_semantics,
    )


def load_observation_factor_bundle_v3(
    manifest_path: str | Path,
) -> ObservationFactorBundle:
    """Load through the frozen provider-v1 marginal-block schema boundary."""

    bundle = load_observation_factor_bundle(manifest_path)
    if bundle.gauge_covariance_semantics != "marginal-blocks-only":
        raise ValueError(
            "provider-v1 cannot load joint cross-window gauge covariance; "
            "use prob4d.provider_v2"
        )
    return replace(
        bundle,
        schema_version=PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
        joint_gauge_covariance=None,
        gauge_covariance_semantics="marginal-blocks-only",
    )
