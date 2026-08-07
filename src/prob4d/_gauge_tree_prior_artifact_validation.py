"""Manifest-record validation for portable sparse gauge-tree priors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ._gauge_tree_common import (
    GAUGE_DIMENSION,
    GAUGE_TREE_PRIOR_SCHEMA,
    GAUGE_TREE_PRIOR_SEMANTICS,
    GAUGE_TREE_PRIOR_VERSION,
)
from ._gauge_tree_prior_artifact_json import (
    exact_fields,
    exact_integer,
    json_list,
    mapping,
    sha256,
    string,
)

ROOT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "prior",
        "payload",
        "arrays",
        "claim_boundary",
    }
)
PRIOR_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "representation_semantics",
        "gauge_dimension",
        "gauge_ids",
        "parent_indices",
        "transition_matrices",
        "innovation_scale_tril",
        "source_joint_covariance_sha256",
        "prior_id",
        "gauge_count",
        "factor_storage_nbytes",
        "dense_covariance_nbytes",
    }
)
PAYLOAD_FIELDS: Final = frozenset({"format", "path", "sha256", "allow_pickle"})
ARRAY_FIELDS: Final = frozenset({"dtype", "shape", "sha256"})
ARRAY_RECORD_FIELDS: Final = frozenset({"key", "descriptor"})


def validate_array_descriptor(value: Any, *, name: str) -> Mapping[str, Any]:
    descriptor = mapping(value, name=name)
    exact_fields(descriptor, ARRAY_FIELDS, name=name)
    string(descriptor["dtype"], name=f"{name}.dtype")
    shape = json_list(descriptor["shape"], name=f"{name}.shape")
    for index, item in enumerate(shape):
        exact_integer(item, name=f"{name}.shape[{index}]", minimum=0)
    sha256(descriptor["sha256"], name=f"{name}.sha256")
    return descriptor


def validate_prior_record(value: Any) -> Mapping[str, Any]:
    """Validate the dense-free prior descriptor stored inside the manifest."""

    prior = mapping(value, name="prior")
    exact_fields(prior, PRIOR_FIELDS, name="prior")
    if prior["schema"] != GAUGE_TREE_PRIOR_SCHEMA:
        raise ValueError("prior.schema mismatch")
    if (
        exact_integer(prior["version"], name="prior.version")
        != GAUGE_TREE_PRIOR_VERSION
    ):
        raise ValueError("prior.version mismatch")
    if prior["representation_semantics"] != GAUGE_TREE_PRIOR_SEMANTICS:
        raise ValueError("prior representation semantics mismatch")
    if (
        exact_integer(prior["gauge_dimension"], name="prior.gauge_dimension")
        != GAUGE_DIMENSION
    ):
        raise ValueError("prior gauge dimension mismatch")
    gauge_ids = json_list(prior["gauge_ids"], name="prior.gauge_ids")
    if not gauge_ids or any(
        not isinstance(item, str) or not item for item in gauge_ids
    ):
        raise ValueError("prior.gauge_ids must contain nonempty strings")
    if len(set(gauge_ids)) != len(gauge_ids):
        raise ValueError("prior.gauge_ids must be unique")
    if exact_integer(
        prior["gauge_count"], name="prior.gauge_count", minimum=1
    ) != len(gauge_ids):
        raise ValueError("prior.gauge_count does not match prior.gauge_ids")
    exact_integer(
        prior["factor_storage_nbytes"],
        name="prior.factor_storage_nbytes",
        minimum=1,
    )
    exact_integer(
        prior["dense_covariance_nbytes"],
        name="prior.dense_covariance_nbytes",
        minimum=1,
    )
    parent_indices = json_list(
        prior["parent_indices"], name="prior.parent_indices"
    )
    if len(parent_indices) != len(gauge_ids):
        raise ValueError("prior.parent_indices must contain one entry per gauge")
    for index, item in enumerate(parent_indices):
        exact_integer(item, name=f"prior.parent_indices[{index}]")
    for name in ("transition_matrices", "innovation_scale_tril"):
        validate_array_descriptor(prior[name], name=f"prior.{name}")
    source_digest = prior["source_joint_covariance_sha256"]
    if source_digest is not None:
        sha256(source_digest, name="prior.source_joint_covariance_sha256")
    sha256(prior["prior_id"], name="prior.prior_id")
    return prior
