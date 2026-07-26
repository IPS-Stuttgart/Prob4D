"""Writer for the portable Phys4D observation-belief contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

OBSERVATION_BELIEF_SCHEMA = "phys4d.observation_belief"
OBSERVATION_BELIEF_VERSION = 1


def array_sha256(values: np.ndarray) -> str:
    """Hash an array including its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash finite JSON data using the observation-contract canonicalization."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _artifact_id(
    descriptor: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(descriptor))
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ObservationBeliefExportV1:
    """Validated data used to emit an ``ObservationBeliefV1`` archive."""

    case_id: str
    stream_id: str
    causal_frame_stop: int
    view_names: tuple[str, ...]
    window_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    source_repository: str
    source_revision: str
    source_artifact_sha256: str

    declared_frame_ids: np.ndarray
    mean_xyz_m: np.ndarray
    frame_ids: np.ndarray
    entity_ids: np.ndarray
    view_indices: np.ndarray
    window_indices: np.ndarray
    correlation_group_ids: np.ndarray
    factor_group_ids: np.ndarray
    prior_reliability: np.ndarray
    association_probability: np.ndarray
    local_covariance_m2: np.ndarray
    low_rank_factor_m: np.ndarray
    group_ids: np.ndarray
    group_prior_nominal_probability: np.ndarray
    group_composite_weight: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id or not self.stream_id:
            raise ValueError("case_id and stream_id must be nonempty")
        if self.causal_frame_stop < 1:
            raise ValueError("causal_frame_stop must be positive")
        if (
            not self.view_names
            or any(not name for name in self.view_names)
            or not self.window_names
            or any(not name for name in self.window_names)
        ):
            raise ValueError("view and window names must be nonempty")
        if any(not name for name in self.factor_names):
            raise ValueError("factor names must be nonempty when present")
        if not self.source_repository or not self.source_revision:
            raise ValueError("source provenance must be nonempty")
        _validate_sha256(
            self.source_artifact_sha256,
            name="source_artifact_sha256",
        )

        normalized = {
            "declared_frame_ids": np.asarray(
                self.declared_frame_ids, dtype=np.int64
            ),
            "mean_xyz_m": np.asarray(self.mean_xyz_m, dtype=np.float64),
            "frame_ids": np.asarray(self.frame_ids, dtype=np.int64),
            "entity_ids": np.asarray(self.entity_ids, dtype=np.int64),
            "view_indices": np.asarray(self.view_indices, dtype=np.int64),
            "window_indices": np.asarray(
                self.window_indices, dtype=np.int64
            ),
            "correlation_group_ids": np.asarray(
                self.correlation_group_ids, dtype=np.int64
            ),
            "factor_group_ids": np.asarray(
                self.factor_group_ids, dtype=np.int64
            ),
            "prior_reliability": np.asarray(
                self.prior_reliability, dtype=np.float64
            ),
            "association_probability": np.asarray(
                self.association_probability, dtype=np.float64
            ),
            "local_covariance_m2": np.asarray(
                self.local_covariance_m2, dtype=np.float64
            ),
            "low_rank_factor_m": np.asarray(
                self.low_rank_factor_m, dtype=np.float64
            ),
            "group_ids": np.asarray(self.group_ids, dtype=np.int64),
            "group_prior_nominal_probability": np.asarray(
                self.group_prior_nominal_probability, dtype=np.float64
            ),
            "group_composite_weight": np.asarray(
                self.group_composite_weight, dtype=np.float64
            ),
        }
        count = len(normalized["mean_xyz_m"])
        if normalized["mean_xyz_m"].shape != (count, 3) or count == 0:
            raise ValueError("mean_xyz_m must have nonempty shape (N, 3)")
        for name in (
            "frame_ids",
            "entity_ids",
            "view_indices",
            "window_indices",
            "correlation_group_ids",
            "factor_group_ids",
            "prior_reliability",
            "association_probability",
        ):
            if normalized[name].shape != (count,):
                raise ValueError(f"{name} must have shape ({count},)")
        if normalized["local_covariance_m2"].shape != (count, 3, 3):
            raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
        if normalized["low_rank_factor_m"].shape != (
            count,
            3,
            len(self.factor_names),
        ):
            raise ValueError("low_rank_factor_m rank does not match factor_names")
        frames = normalized["declared_frame_ids"]
        if (
            frames.ndim != 1
            or len(frames) == 0
            or np.any(np.diff(frames) <= 0)
            or np.any(frames < 0)
            or np.any(frames >= self.causal_frame_stop)
        ):
            raise ValueError("declared frames violate the causal boundary")
        if not np.all(np.isin(normalized["frame_ids"], frames)):
            raise ValueError("observation frames are not declared")
        if np.any(normalized["entity_ids"] < 0):
            raise ValueError("entity_ids must be nonnegative")
        if np.any(normalized["view_indices"] < 0) or np.any(
            normalized["view_indices"] >= len(self.view_names)
        ):
            raise ValueError("view_indices reference unavailable views")
        if np.any(normalized["window_indices"] < 0) or np.any(
            normalized["window_indices"] >= len(self.window_names)
        ):
            raise ValueError("window_indices reference unavailable windows")
        if np.any(normalized["correlation_group_ids"] < 0) or np.any(
            normalized["factor_group_ids"] < 0
        ):
            raise ValueError("group identifiers must be nonnegative")
        for name in ("prior_reliability", "association_probability"):
            values = normalized[name]
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        if not np.array_equal(
            normalized["group_ids"],
            np.unique(normalized["correlation_group_ids"]),
        ):
            raise ValueError("group_ids do not match correlation groups")
        group_count = len(normalized["group_ids"])
        if normalized["group_prior_nominal_probability"].shape != (
            group_count,
        ) or normalized["group_composite_weight"].shape != (group_count,):
            raise ValueError("group metadata shape changed")
        group_prior = normalized["group_prior_nominal_probability"]
        group_weight = normalized["group_composite_weight"]
        if not np.all(np.isfinite(group_prior)) or np.any(
            (group_prior < 0.0) | (group_prior > 1.0)
        ):
            raise ValueError("group prior probabilities must lie in [0, 1]")
        if not np.all(np.isfinite(group_weight)) or np.any(
            (group_weight <= 0.0) | (group_weight > 1.0)
        ):
            raise ValueError("group composite weights must lie in (0, 1]")
        if not np.all(np.isfinite(normalized["mean_xyz_m"])):
            raise ValueError("observation means must be finite")
        if not np.all(np.isfinite(normalized["local_covariance_m2"])):
            raise ValueError("local covariance must be finite")
        if not np.all(np.isfinite(normalized["low_rank_factor_m"])):
            raise ValueError("low-rank factors must be finite")
        symmetric = 0.5 * (
            normalized["local_covariance_m2"]
            + np.swapaxes(normalized["local_covariance_m2"], 1, 2)
        )
        if not np.allclose(
            symmetric,
            normalized["local_covariance_m2"],
            atol=1e-12,
            rtol=1e-10,
        ) or np.any(np.min(np.linalg.eigvalsh(symmetric), axis=1) <= 0.0):
            raise ValueError("local covariance must be symmetric positive definite")
        order = np.lexsort(
            (
                normalized["window_indices"],
                normalized["view_indices"],
                normalized["entity_ids"],
                normalized["frame_ids"],
            )
        )
        sorted_keys = np.column_stack(
            (
                normalized["frame_ids"][order],
                normalized["entity_ids"][order],
                normalized["view_indices"][order],
                normalized["window_indices"][order],
            )
        )
        if len(sorted_keys) > 1 and np.any(
            np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
        ):
            raise ValueError(
                "observation identity (frame, entity, view, window) must be unique"
            )
        try:
            metadata = json.loads(
                json.dumps(dict(self.metadata), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be finite JSON data") from error
        for name, values in normalized.items():
            values = values.copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "metadata", metadata)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": OBSERVATION_BELIEF_SCHEMA,
            "schema_version": OBSERVATION_BELIEF_VERSION,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "causal_frame_stop": self.causal_frame_stop,
            "view_names": list(self.view_names),
            "window_names": list(self.window_names),
            "factor_names": list(self.factor_names),
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_artifact_sha256": self.source_artifact_sha256,
            "metadata": self.metadata,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "declared_frame_ids": self.declared_frame_ids,
            "mean_xyz_m": self.mean_xyz_m,
            "frame_ids": self.frame_ids,
            "entity_ids": self.entity_ids,
            "view_indices": self.view_indices,
            "window_indices": self.window_indices,
            "correlation_group_ids": self.correlation_group_ids,
            "factor_group_ids": self.factor_group_ids,
            "prior_reliability": self.prior_reliability,
            "association_probability": self.association_probability,
            "local_covariance_m2": self.local_covariance_m2,
            "low_rank_factor_m": self.low_rank_factor_m,
            "group_ids": self.group_ids,
            "group_prior_nominal_probability": (
                self.group_prior_nominal_probability
            ),
            "group_composite_weight": self.group_composite_weight,
        }

    @property
    def artifact_id(self) -> str:
        return _artifact_id(self.descriptor(), self.arrays())


def save_observation_belief_export(
    path: str | Path, artifact: ObservationBeliefExportV1
) -> None:
    """Write the exact archive consumed by Bayesian-PhysTwin and Causal4D."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = artifact.descriptor()
    descriptor["artifact_id"] = artifact.artifact_id
    np.savez_compressed(
        target,
        descriptor_json=np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        **artifact.arrays(),
    )


__all__ = [
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "ObservationBeliefExportV1",
    "array_sha256",
    "canonical_json_sha256",
    "file_sha256",
    "save_observation_belief_export",
]
