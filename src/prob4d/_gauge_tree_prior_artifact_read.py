"""Strict loader for portable sparse gauge-tree prior artifacts."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._gauge_tree_common import canonical_array_descriptor
from ._gauge_tree_prior_artifact_json import (
    exact_fields,
    exact_integer,
    load_json,
    mapping,
    resolve_payload_path,
    sha256,
    string,
)
from ._gauge_tree_prior_artifact_schema import (
    ARRAY_NAMES,
    GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY,
    GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
    GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
    gauge_tree_prior_artifact_id,
    sha256_file,
)
from ._gauge_tree_prior_artifact_validation import (
    ARRAY_RECORD_FIELDS,
    PAYLOAD_FIELDS,
    ROOT_FIELDS,
    validate_array_descriptor,
    validate_prior_record,
)
from .gauge_tree_prior import GaugeTreeSquareRootPriorV1


def _load_payload_arrays(
    payload: Path,
    *,
    keys: Mapping[str, str],
) -> dict[str, np.ndarray]:
    try:
        with np.load(payload, allow_pickle=False) as arrays:
            if set(arrays.files) != set(keys.values()):
                raise ValueError(
                    "gauge-tree prior payload contains unexpected array keys"
                )
            return {name: np.asarray(arrays[keys[name]]) for name in ARRAY_NAMES}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "gauge-tree prior payload"
        ):
            raise
        raise ValueError(
            "gauge-tree prior payload is unreadable or invalid"
        ) from error


def _validate_payload_records(
    record: Mapping[str, Any],
    manifest: Path,
) -> tuple[Path, dict[str, str], dict[str, Mapping[str, Any]]]:
    payload_record = mapping(record["payload"], name="payload")
    exact_fields(payload_record, PAYLOAD_FIELDS, name="payload")
    if payload_record["format"] != "numpy-npz-v1":
        raise ValueError("unsupported gauge-tree prior payload format")
    if payload_record["allow_pickle"] is not False:
        raise ValueError("payload.allow_pickle must be the literal Boolean false")
    payload = resolve_payload_path(manifest, payload_record["path"])
    expected_sha256 = sha256(payload_record["sha256"], name="payload.sha256")
    try:
        observed_sha256 = sha256_file(payload)
    except OSError as error:
        raise ValueError("gauge-tree prior payload is unreadable") from error
    if observed_sha256 != expected_sha256:
        raise ValueError("gauge-tree prior payload checksum mismatch")

    array_record = mapping(record["arrays"], name="arrays")
    exact_fields(array_record, frozenset(ARRAY_NAMES), name="arrays")
    keys: dict[str, str] = {}
    descriptors: dict[str, Mapping[str, Any]] = {}
    for name in ARRAY_NAMES:
        item = mapping(array_record[name], name=f"arrays.{name}")
        exact_fields(item, ARRAY_RECORD_FIELDS, name=f"arrays.{name}")
        keys[name] = string(item["key"], name=f"arrays.{name}.key")
        descriptors[name] = validate_array_descriptor(
            item["descriptor"],
            name=f"arrays.{name}.descriptor",
        )
    if len(set(keys.values())) != len(keys):
        raise ValueError("array payload keys must be unique")
    return payload, keys, descriptors


def load_gauge_tree_prior(
    manifest_path: str | os.PathLike[str],
) -> GaugeTreeSquareRootPriorV1:
    """Load, checksum, reconstruct, and content-verify one sparse prior artifact."""

    manifest = Path(manifest_path)
    record = load_json(manifest)
    exact_fields(record, ROOT_FIELDS, name="gauge-tree prior manifest")
    if record["schema"] != GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA:
        raise ValueError("unsupported gauge-tree prior artifact schema")
    if exact_integer(
        record["schema_version"], name="schema_version"
    ) != GAUGE_TREE_PRIOR_ARTIFACT_VERSION:
        raise ValueError("unsupported gauge-tree prior artifact version")
    if record["claim_boundary"] != GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY:
        raise ValueError("gauge-tree prior artifact claim_boundary mismatch")
    observed_artifact_id = sha256(record["artifact_id"], name="artifact_id")
    prior_record = validate_prior_record(record["prior"])

    payload, keys, descriptors = _validate_payload_records(record, manifest)
    loaded = _load_payload_arrays(payload, keys=keys)
    for name, array in loaded.items():
        observed_descriptor = canonical_array_descriptor(array)
        if observed_descriptor != dict(descriptors[name]):
            raise ValueError(f"gauge-tree prior {name} descriptor mismatch")

    prior = GaugeTreeSquareRootPriorV1(
        gauge_ids=tuple(str(item) for item in prior_record["gauge_ids"]),
        parent_indices=loaded["parent_indices"],
        transition_matrices=loaded["transition_matrices"],
        innovation_scale_tril=loaded["innovation_scale_tril"],
        source_joint_covariance_sha256=(
            None
            if prior_record["source_joint_covariance_sha256"] is None
            else str(prior_record["source_joint_covariance_sha256"])
        ),
        representation_semantics=str(prior_record["representation_semantics"]),
    )
    if prior.to_dict() != dict(prior_record):
        raise ValueError(
            "gauge-tree prior manifest does not match reconstructed factors"
        )
    expected_artifact_id = gauge_tree_prior_artifact_id(prior)
    if observed_artifact_id != expected_artifact_id:
        raise ValueError("gauge-tree prior artifact_id mismatch")
    return prior
