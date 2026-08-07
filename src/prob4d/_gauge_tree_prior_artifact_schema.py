"""Canonical identity and manifest records for portable sparse gauge-tree priors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import numpy as np

from ._gauge_tree_common import canonical_array_descriptor, canonical_json_sha256
from .gauge_tree_prior import GaugeTreeSquareRootPriorV1

GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA: Final = (
    "prob4d.gauge-tree-square-root-prior-artifact"
)
GAUGE_TREE_PRIOR_ARTIFACT_VERSION: Final = 1
GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY: Final = (
    "This artifact preserves one exact zero-mean linearized causal gauge-tree "
    "Gaussian prior and its content identity without serializing a dense joint "
    "covariance. It establishes representation integrity and exact sparse "
    "algebra only; it does not establish provider competence, covariance "
    "calibration, physical-query benefit, or deployment safety."
)

ARRAY_NAMES: Final = (
    "parent_indices",
    "transition_matrices",
    "innovation_scale_tril",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_arrays(prior: GaugeTreeSquareRootPriorV1) -> dict[str, np.ndarray]:
    """Return the three arrays that define the sparse square-root prior."""

    return {
        "parent_indices": prior.parent_indices,
        "transition_matrices": prior.transition_matrices,
        "innovation_scale_tril": prior.innovation_scale_tril,
    }


def artifact_descriptor(prior: GaugeTreeSquareRootPriorV1) -> dict[str, object]:
    """Return the path-independent descriptor bound by ``artifact_id``."""

    arrays = factor_arrays(prior)
    return {
        "schema": GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
        "schema_version": GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
        "prior": prior.to_dict(),
        "arrays": {
            name: {
                "key": name,
                "descriptor": canonical_array_descriptor(arrays[name]),
            }
            for name in ARRAY_NAMES
        },
        "claim_boundary": GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY,
    }


def gauge_tree_prior_artifact_id(prior: GaugeTreeSquareRootPriorV1) -> str:
    """Return the path- and container-independent artifact identity."""

    if not isinstance(prior, GaugeTreeSquareRootPriorV1):
        raise TypeError("prior must be a GaugeTreeSquareRootPriorV1")
    return canonical_json_sha256(artifact_descriptor(prior))


def manifest_record(
    prior: GaugeTreeSquareRootPriorV1,
    *,
    payload_relative_path: str,
    payload_sha256: str,
) -> dict[str, object]:
    """Return the strict manifest record for one checksum-bound NPZ payload."""

    descriptor = artifact_descriptor(prior)
    return {
        "schema": descriptor["schema"],
        "schema_version": descriptor["schema_version"],
        "artifact_id": canonical_json_sha256(descriptor),
        "prior": descriptor["prior"],
        "payload": {
            "format": "numpy-npz-v1",
            "path": payload_relative_path,
            "sha256": payload_sha256,
            "allow_pickle": False,
        },
        "arrays": descriptor["arrays"],
        "claim_boundary": descriptor["claim_boundary"],
    }


def artifact_summary(prior: GaugeTreeSquareRootPriorV1) -> dict[str, object]:
    """Return a JSON-compatible verification summary without densifying the prior."""

    return {
        "artifact_id": gauge_tree_prior_artifact_id(prior),
        "prior_id": prior.prior_id,
        "gauge_count": prior.gauge_count,
        "factor_storage_nbytes": prior.factor_storage_nbytes,
        "dense_covariance_nbytes": prior.dense_covariance_nbytes,
        "storage_ratio_to_dense": prior.storage_ratio_to_dense,
        "source_joint_covariance_sha256": prior.source_joint_covariance_sha256,
    }
