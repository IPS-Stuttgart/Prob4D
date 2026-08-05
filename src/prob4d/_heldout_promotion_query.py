"""Sealed guarded-query result contract for held-out promotion."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._heldout_promotion_common import (
    HELDOUT_QUERY_RESULTS_SCHEMA,
    HELDOUT_QUERY_RESULTS_VERSION,
    QUERY_RESULTS_CLAIM_BOUNDARY,
    _SHA256,
    _atomic_write_json,
    _exact_keys,
    _load_json,
    _nonnegative_real,
    _optional_real,
    _optional_string,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._heldout_promotion_lock import HeldoutProviderPromotionLockV1
from ._immutable_json import frozen_finite_json_mapping, plain_json


@dataclass(frozen=True, slots=True)
class PromotionQueryRowV1:
    """One complete guarded physical-query outcome for one group and arm."""

    group_id: str
    arm_id: str
    query_rmse_mm: float
    deployed_artifact_id: str
    fallback_artifact_id: str
    accepted: bool | None
    exact_fallback_reproduced: bool | None
    accepted_coverage: float | None
    accepted_width_mm: float | None
    technical_failure: bool = False
    technical_failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_id",
            _strict_string(self.group_id, name="group_id"),
        )
        object.__setattr__(
            self,
            "arm_id",
            _strict_string(self.arm_id, name="arm_id"),
        )
        object.__setattr__(
            self,
            "query_rmse_mm",
            _nonnegative_real(self.query_rmse_mm, name="query_rmse_mm"),
        )
        for field_name in ("deployed_artifact_id", "fallback_artifact_id"):
            object.__setattr__(
                self,
                field_name,
                _strict_digest(
                    getattr(self, field_name),
                    name=field_name,
                    pattern=_SHA256,
                ),
            )
        if self.accepted is not None:
            object.__setattr__(
                self,
                "accepted",
                _strict_bool(self.accepted, name="accepted"),
            )
        if self.exact_fallback_reproduced is not None:
            object.__setattr__(
                self,
                "exact_fallback_reproduced",
                _strict_bool(
                    self.exact_fallback_reproduced,
                    name="exact_fallback_reproduced",
                ),
            )
        coverage = _optional_real(
            self.accepted_coverage,
            name="accepted_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        width = _optional_real(
            self.accepted_width_mm,
            name="accepted_width_mm",
            minimum=0.0,
        )
        technical_failure = _strict_bool(
            self.technical_failure,
            name="technical_failure",
        )
        reason = _optional_string(
            self.technical_failure_reason,
            name="technical_failure_reason",
        )
        if technical_failure and reason is None:
            raise ValueError("technical failures require a reason")
        if not technical_failure and reason is not None:
            raise ValueError("technical_failure_reason requires technical_failure=true")
        if technical_failure and self.accepted is not False:
            raise ValueError("technical failures must deploy fallback, not accept an update")
        if self.accepted is True and self.exact_fallback_reproduced is not None:
            raise ValueError("accepted updates must use null exact_fallback_reproduced")
        if self.accepted is None:
            if self.exact_fallback_reproduced is not None:
                raise ValueError("reference rows must not carry fallback decisions")
            if coverage is not None or width is not None:
                raise ValueError("reference rows must not carry accepted-update intervals")
        elif self.accepted is False and (coverage is not None or width is not None):
            raise ValueError("rejected updates must not carry accepted-update intervals")
        object.__setattr__(self, "accepted_coverage", coverage)
        object.__setattr__(self, "accepted_width_mm", width)
        object.__setattr__(self, "technical_failure", technical_failure)
        object.__setattr__(self, "technical_failure_reason", reason)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="query row metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "arm_id": self.arm_id,
            "query_rmse_mm": self.query_rmse_mm,
            "deployed_artifact_id": self.deployed_artifact_id,
            "fallback_artifact_id": self.fallback_artifact_id,
            "accepted": self.accepted,
            "exact_fallback_reproduced": self.exact_fallback_reproduced,
            "accepted_coverage": self.accepted_coverage,
            "accepted_width_mm": self.accepted_width_mm,
            "technical_failure": self.technical_failure,
            "technical_failure_reason": self.technical_failure_reason,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PromotionQueryRowV1:
        mapping = _strict_mapping(value, name="promotion query row")
        _exact_keys(
            mapping,
            {
                "group_id",
                "arm_id",
                "query_rmse_mm",
                "deployed_artifact_id",
                "fallback_artifact_id",
                "accepted",
                "exact_fallback_reproduced",
                "accepted_coverage",
                "accepted_width_mm",
                "technical_failure",
                "technical_failure_reason",
                "metadata",
            },
            name="promotion query row",
        )
        return cls(
            group_id=mapping["group_id"],
            arm_id=mapping["arm_id"],
            query_rmse_mm=mapping["query_rmse_mm"],
            deployed_artifact_id=mapping["deployed_artifact_id"],
            fallback_artifact_id=mapping["fallback_artifact_id"],
            accepted=mapping["accepted"],
            exact_fallback_reproduced=mapping["exact_fallback_reproduced"],
            accepted_coverage=mapping["accepted_coverage"],
            accepted_width_mm=mapping["accepted_width_mm"],
            technical_failure=mapping["technical_failure"],
            technical_failure_reason=mapping["technical_failure_reason"],
            metadata=_strict_mapping(mapping["metadata"], name="query row metadata"),
        )


@dataclass(frozen=True, slots=True)
class HeldoutPromotionQueryResultsV1:
    """Complete target result matrix sealed under one promotion lock."""

    promotion_lock_id: str
    rows: tuple[PromotionQueryRowV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "promotion_lock_id",
            _strict_digest(
                self.promotion_lock_id,
                name="promotion_lock_id",
                pattern=_SHA256,
            ),
        )
        if (
            type(self.rows) is not tuple
            or not self.rows
            or not all(isinstance(row, PromotionQueryRowV1) for row in self.rows)
        ):
            raise ValueError("rows must be a nonempty tuple of PromotionQueryRowV1")
        rows = tuple(self.rows)
        keys = tuple((row.group_id, row.arm_id) for row in rows)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("query rows must be sorted by unique group/arm key")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="query results metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": HELDOUT_QUERY_RESULTS_SCHEMA,
            "schema_version": HELDOUT_QUERY_RESULTS_VERSION,
            "promotion_lock_id": self.promotion_lock_id,
            "rows": [row.to_dict() for row in self.rows],
            "metadata": plain_json(self.metadata),
            "claim_boundary": QUERY_RESULTS_CLAIM_BOUNDARY,
        }

    @property
    def query_results_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "query_results_id": self.query_results_id}


_QUERY_RESULTS_FIELDS = {
    "schema_name",
    "schema_version",
    "promotion_lock_id",
    "rows",
    "metadata",
    "claim_boundary",
    "query_results_id",
}
_RAW_QUERY_RESULTS_FIELDS = {"promotion_lock_id", "rows", "metadata"}


def build_query_results(
    lock: HeldoutProviderPromotionLockV1,
    *,
    rows: Sequence[PromotionQueryRowV1],
    metadata: Mapping[str, Any] | None = None,
) -> HeldoutPromotionQueryResultsV1:
    if not isinstance(lock, HeldoutProviderPromotionLockV1):
        raise ValueError("lock must be HeldoutProviderPromotionLockV1")
    return HeldoutPromotionQueryResultsV1(
        promotion_lock_id=lock.promotion_lock_id,
        rows=tuple(sorted(rows, key=lambda row: (row.group_id, row.arm_id))),
        metadata={} if metadata is None else metadata,
    )


def query_results_from_raw(
    value: Any,
    *,
    lock: HeldoutProviderPromotionLockV1,
) -> HeldoutPromotionQueryResultsV1:
    mapping = _strict_mapping(value, name="raw promotion query results")
    _exact_keys(mapping, _RAW_QUERY_RESULTS_FIELDS, name="raw promotion query results")
    supplied_lock_id = _strict_digest(
        mapping["promotion_lock_id"],
        name="promotion_lock_id",
        pattern=_SHA256,
    )
    if supplied_lock_id != lock.promotion_lock_id:
        raise ValueError("raw query results reference a different promotion lock")
    raw_rows = _strict_list(mapping["rows"], name="rows")
    return build_query_results(
        lock,
        rows=tuple(PromotionQueryRowV1.from_dict(item) for item in raw_rows),
        metadata=_strict_mapping(mapping["metadata"], name="query results metadata"),
    )


def query_results_from_dict(value: Any) -> HeldoutPromotionQueryResultsV1:
    mapping = _strict_mapping(value, name="promotion query results")
    _exact_keys(mapping, _QUERY_RESULTS_FIELDS, name="promotion query results")
    if mapping["schema_name"] != HELDOUT_QUERY_RESULTS_SCHEMA:
        raise ValueError("unsupported promotion query-results schema")
    if mapping["schema_version"] != HELDOUT_QUERY_RESULTS_VERSION:
        raise ValueError("unsupported promotion query-results version")
    if mapping["claim_boundary"] != QUERY_RESULTS_CLAIM_BOUNDARY:
        raise ValueError("promotion query-results claim boundary changed")
    raw_rows = _strict_list(mapping["rows"], name="rows")
    results = HeldoutPromotionQueryResultsV1(
        promotion_lock_id=mapping["promotion_lock_id"],
        rows=tuple(PromotionQueryRowV1.from_dict(item) for item in raw_rows),
        metadata=_strict_mapping(mapping["metadata"], name="query results metadata"),
    )
    supplied = _strict_digest(
        mapping["query_results_id"],
        name="query_results_id",
        pattern=_SHA256,
    )
    if supplied != results.query_results_id:
        raise ValueError("query_results_id mismatch")
    return results


def write_query_results(
    results: HeldoutPromotionQueryResultsV1,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(results, HeldoutPromotionQueryResultsV1):
        raise ValueError("results must be HeldoutPromotionQueryResultsV1")
    _atomic_write_json(Path(path), results.to_dict(), overwrite=overwrite)


def load_query_results(
    path: str | os.PathLike[str],
) -> HeldoutPromotionQueryResultsV1:
    mapping, _ = _load_json(Path(path), name="promotion query results")
    return query_results_from_dict(mapping)

