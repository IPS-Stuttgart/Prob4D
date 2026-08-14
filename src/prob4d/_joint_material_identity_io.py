"""Strict persistence and replay for joint material-identity posteriors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._atomic_file import atomic_write_bytes
from ._joint_material_identity_common import (
    CLAIM_BOUNDARY,
    JOINT_ASSIGNMENT_SEMANTICS,
    JOINT_MATERIAL_IDENTITY_POSTERIOR_SCHEMA,
    JOINT_MATERIAL_IDENTITY_POSTERIOR_VERSION,
    _canonical_json,
)
from ._joint_material_identity_compute import build_joint_material_identity_posterior
from ._joint_material_identity_json import (
    _POSTERIOR_FIELDS,
    _fields,
    _list,
    _load,
    _mapping,
)
from ._joint_material_identity_mixture_io import _mixture
from ._joint_material_identity_model import JointMaterialIdentityPosteriorV1


def _close(actual: object, expected: float, *, name: str) -> None:
    try:
        value = float(cast(Any, actual))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real number") from error
    if not np.isfinite(value) or not np.isclose(
        value,
        expected,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(f"{name} differs from exact joint-posterior replay")


def _replay(value: Mapping[str, Any]) -> JointMaterialIdentityPosteriorV1:
    _fields(value, _POSTERIOR_FIELDS, name="posterior")
    if value["schema"] != JOINT_MATERIAL_IDENTITY_POSTERIOR_SCHEMA:
        raise ValueError("unsupported joint material-identity posterior schema")
    if value["schema_version"] != JOINT_MATERIAL_IDENTITY_POSTERIOR_VERSION:
        raise ValueError("unsupported joint material-identity posterior version")
    if value["constraint_semantics"] != JOINT_ASSIGNMENT_SEMANTICS:
        raise ValueError("unsupported joint assignment semantics")
    if value["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("joint material-identity claim boundary changed")
    mixtures = tuple(
        _mixture(raw, index=index)
        for index, raw in enumerate(_list(value["mixtures"], name="mixtures"))
    )
    posterior = build_joint_material_identity_posterior(
        mixtures,
        window_order=tuple(_list(value["window_order"], name="window_order")),
        maximum_joint_assignments=value["maximum_joint_assignments"],
        metadata=_mapping(value["metadata"], name="metadata"),
    )
    if value["posterior_id"] != posterior.posterior_id:
        raise ValueError("posterior_id differs from exact joint-posterior replay")
    for name in (
        "unconstrained_assignment_count",
        "feasible_assignment_count",
        "rejected_assignment_count",
    ):
        if value[name] != getattr(posterior, name):
            raise ValueError(f"{name} differs from exact joint-posterior replay")
    _close(value["log_normalizer"], posterior.log_normalizer, name="log_normalizer")
    _close(
        value["joint_entropy_nats"],
        posterior.joint_entropy_nats,
        name="joint_entropy_nats",
    )
    _close(
        value["effective_assignment_count"],
        posterior.effective_assignment_count,
        name="effective_assignment_count",
    )
    assignments = _list(value["assignments"], name="assignments")
    if len(assignments) != len(posterior.assignments):
        raise ValueError("assignments differ from exact joint-posterior replay")
    for index, (raw, expected_assignment) in enumerate(
        zip(assignments, posterior.assignments, strict=True)
    ):
        record = _mapping(raw, name=f"assignments[{index}]")
        if (
            record.get("assignment_id") != expected_assignment.assignment_id
            or record.get("mixture_ids") != list(expected_assignment.mixture_ids)
            or record.get("candidate_ids") != list(expected_assignment.candidate_ids)
        ):
            raise ValueError("assignments differ from exact joint-posterior replay")
        _close(
            record.get("log_weight"),
            expected_assignment.log_weight,
            name="assignment.log_weight",
        )
        _close(
            record.get("probability"),
            expected_assignment.probability,
            name="assignment.probability",
        )
    marginals = _list(value["marginals"], name="marginals")
    if len(marginals) != len(posterior.marginals):
        raise ValueError("marginals differ from exact joint-posterior replay")
    for index, (raw, expected_marginal) in enumerate(
        zip(marginals, posterior.marginals, strict=True)
    ):
        record = _mapping(raw, name=f"marginals[{index}]")
        if (
            record.get("mixture_id") != expected_marginal.mixture_id
            or record.get("candidate_ids") != list(expected_marginal.candidate_ids)
            or record.get("target_endpoint")
            != expected_marginal.target_endpoint.to_dict()
        ):
            raise ValueError("marginals differ from exact joint-posterior replay")
        probabilities = np.asarray(record.get("probabilities"), dtype=np.float64)
        if not np.allclose(
            probabilities,
            expected_marginal.probabilities,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("marginals differ from exact joint-posterior replay")
    return posterior


def write_joint_material_identity_posterior(
    path: str | Path,
    posterior: JointMaterialIdentityPosteriorV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write one replay-complete joint posterior."""

    if not isinstance(posterior, JointMaterialIdentityPosteriorV1):
        raise ValueError("posterior must be JointMaterialIdentityPosteriorV1")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    atomic_write_bytes(
        path,
        _canonical_json(posterior.to_record()) + b"\n",
        overwrite=overwrite,
    )


def load_joint_material_identity_posterior(
    path: str | Path,
) -> JointMaterialIdentityPosteriorV1:
    """Load, independently reconstruct, and replay one joint posterior."""

    return _replay(_load(Path(path)))


__all__ = [
    "load_joint_material_identity_posterior",
    "write_joint_material_identity_posterior",
]
