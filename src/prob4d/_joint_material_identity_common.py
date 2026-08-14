"""Shared validation and graph helpers for joint local material identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import plain_json
from .material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityMixtureV1,
)

FloatArray: TypeAlias = NDArray[np.floating[Any]]

JOINT_MATERIAL_IDENTITY_POSTERIOR_SCHEMA = (
    "prob4d.joint-material-identity-posterior"
)
JOINT_MATERIAL_IDENTITY_POSTERIOR_VERSION = 1
JOINT_ASSIGNMENT_SEMANTICS: Literal[
    "conditioned-local-mixtures-window-unique-forest-v1"
] = "conditioned-local-mixtures-window-unique-forest-v1"
JOINT_LIKELIHOOD_SEMANTICS: Literal[
    "logsumexp-discrete-joint-identity-v1"
] = "logsumexp-discrete-joint-identity-v1"
CLAIM_BOUNDARY = (
    "Source-calibrated material-identity mixtures conditioned on the declared "
    "window-unique forest constraint. Endpoints remain window-local, every "
    "local null candidate remains available, and no provider competence, "
    "physical-state update, BayesianPhysTwin benefit, or Causal4D benefit is "
    "established by this artifact."
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: object, *, name: str, minimum: float | None = None) -> float:
    if type(value) not in {int, float} and not isinstance(
        value, (np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    result = float(cast(Any, value))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _sha256(value: object, *, name: str) -> str:
    digest = _string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _readonly(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _logsumexp(values: FloatArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("logsumexp values must be a non-empty vector")
    if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
        raise ValueError("logsumexp values may not contain NaN or positive infinity")
    maximum = float(np.max(array))
    if np.isneginf(maximum):
        return float("-inf")
    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


def _endpoint_key(
    endpoint: LocalTrackEndpoint,
    *,
    window_indices: Mapping[str, int],
) -> tuple[int, int, str]:
    return window_indices[endpoint.window_id], endpoint.track_id, endpoint.window_id


def _canonical_mixtures(
    mixtures: Sequence[MaterialIdentityMixtureV1],
    *,
    window_order: tuple[str, ...],
) -> tuple[MaterialIdentityMixtureV1, ...]:
    if type(window_order) is not tuple or not window_order:
        raise ValueError("window_order must be a non-empty tuple")
    order = tuple(
        _string(window_id, name=f"window_order[{index}]")
        for index, window_id in enumerate(window_order)
    )
    if len(set(order)) != len(order):
        raise ValueError("window_order must contain unique window IDs")
    values = tuple(mixtures)
    if not values:
        raise ValueError("at least one material-identity mixture is required")
    if any(not isinstance(value, MaterialIdentityMixtureV1) for value in values):
        raise ValueError("mixtures must contain MaterialIdentityMixtureV1 values")

    indices = {window_id: index for index, window_id in enumerate(order)}
    for mixture in values:
        target_window = mixture.target_endpoint.window_id
        if target_window not in indices:
            raise ValueError("mixture target window is absent from window_order")
        expected_prefix = order[: indices[target_window] + 1]
        if mixture.window_order != expected_prefix:
            raise ValueError(
                "every mixture window_order must equal the global prefix ending "
                "at its target window"
            )

    canonical = tuple(
        sorted(
            values,
            key=lambda mixture: (
                indices[mixture.target_endpoint.window_id],
                mixture.target_endpoint.track_id,
                cast(str, mixture.mixture_id),
            ),
        )
    )
    if len({mixture.target_endpoint for mixture in canonical}) != len(canonical):
        raise ValueError("joint mixtures must have unique target endpoints")
    if len({mixture.mixture_id for mixture in canonical}) != len(canonical):
        raise ValueError("joint mixture IDs must be unique")
    for field_name in (
        "association_rule_id",
        "calibration_id",
        "tracklet_producer_revision",
        "association_revision",
        "weight_semantics",
        "null_hypothesis_semantics",
    ):
        if len({getattr(mixture, field_name) for mixture in canonical}) != 1:
            raise ValueError(f"joint mixtures disagree on {field_name}")
    return canonical


class _WindowUniqueForest:
    """Union-find that rejects a second endpoint from an occupied window."""

    def __init__(self) -> None:
        self.parent: dict[LocalTrackEndpoint, LocalTrackEndpoint] = {}
        self.rank: dict[LocalTrackEndpoint, int] = {}
        self.windows: dict[LocalTrackEndpoint, frozenset[str]] = {}

    def add(self, endpoint: LocalTrackEndpoint) -> None:
        if endpoint in self.parent:
            return
        self.parent[endpoint] = endpoint
        self.rank[endpoint] = 0
        self.windows[endpoint] = frozenset({endpoint.window_id})

    def find(self, endpoint: LocalTrackEndpoint) -> LocalTrackEndpoint:
        parent = self.parent[endpoint]
        if parent != endpoint:
            self.parent[endpoint] = self.find(parent)
        return self.parent[endpoint]

    def union(self, first: LocalTrackEndpoint, second: LocalTrackEndpoint) -> bool:
        self.add(first)
        self.add(second)
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.windows[first_root] & self.windows[second_root]:
            return False
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.windows[first_root] |= self.windows[second_root]
        del self.windows[second_root]
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1
        return True

    def components(self) -> tuple[tuple[LocalTrackEndpoint, ...], ...]:
        grouped: dict[LocalTrackEndpoint, list[LocalTrackEndpoint]] = {}
        for endpoint in self.parent:
            grouped.setdefault(self.find(endpoint), []).append(endpoint)
        return tuple(tuple(values) for values in grouped.values())


def _selection_is_feasible(
    mixtures: tuple[MaterialIdentityMixtureV1, ...],
    candidate_indices: tuple[int, ...],
) -> bool:
    forest = _WindowUniqueForest()
    for mixture, candidate_index in zip(mixtures, candidate_indices, strict=True):
        target = mixture.target_endpoint
        forest.add(target)
        source = mixture.candidates[candidate_index].source_endpoint
        if source is not None and not forest.union(source, target):
            return False
    return True
