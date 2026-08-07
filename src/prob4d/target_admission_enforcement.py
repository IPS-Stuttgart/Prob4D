"""Require one sealed target admission on cohort-bound promotion execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._heldout_promotion_common import (
    _SHA256,
    _strict_digest,
    _strict_mapping,
)
from ._heldout_promotion_lock import HeldoutProviderPromotionLockV1
from .target_provider_admission import (
    HeldoutTargetProviderAdmissionV1,
    load_target_provider_admission,
    validate_target_provider_admission_against_lock,
)

TARGET_PROVIDER_ADMISSION_METADATA_KEY = "target_provider_admission_id"


def _metadata_admission_id(value: object, *, name: str) -> str:
    metadata = _strict_mapping(value, name=name)
    if TARGET_PROVIDER_ADMISSION_METADATA_KEY not in metadata:
        raise ValueError(
            f"{name} must bind {TARGET_PROVIDER_ADMISSION_METADATA_KEY!r}"
        )
    return _strict_digest(
        metadata[TARGET_PROVIDER_ADMISSION_METADATA_KEY],
        name=f"{name}.{TARGET_PROVIDER_ADMISSION_METADATA_KEY}",
        pattern=_SHA256,
    )


def validate_target_admission_execution_binding(
    lock: HeldoutProviderPromotionLockV1,
    admission: HeldoutTargetProviderAdmissionV1,
    *,
    provider_report: Mapping[str, object],
    query_metadata: Mapping[str, object],
) -> None:
    """Bind provider and query streams to one exact target admission."""

    cohort_binding_id = lock.frozen_artifact_ids.get("cohort_binding")
    if cohort_binding_id is None:
        raise ValueError("target admission may only be used with a cohort-bound promotion lock")
    validate_target_provider_admission_against_lock(admission, lock)
    expected = admission.target_provider_admission_id
    provider_id = _metadata_admission_id(
        provider_report.get("manifest_metadata"),
        name="provider report manifest_metadata",
    )
    query_id = _metadata_admission_id(
        query_metadata,
        name="promotion query metadata",
    )
    if provider_id != expected:
        raise ValueError("provider report uses another target provider admission")
    if query_id != expected:
        raise ValueError("promotion query results use another target provider admission")


def load_target_admission_for_execution(
    lock: HeldoutProviderPromotionLockV1,
    admission_path: str | Path | None,
    *,
    provider_report: Mapping[str, object],
    query_metadata: Mapping[str, object],
) -> HeldoutTargetProviderAdmissionV1 | None:
    """Load the mandatory admission for a real cohort, preserving legacy controls."""

    cohort_binding_id = lock.frozen_artifact_ids.get("cohort_binding")
    if cohort_binding_id is None:
        if admission_path is not None:
            raise ValueError(
                "target provider admission was supplied for a promotion lock without a "
                "frozen cohort binding"
            )
        return None
    if admission_path is None:
        raise ValueError(
            "cohort-bound promotion requires --target-provider-admission before "
            "provider or query outcomes are evaluated"
        )
    path = Path(admission_path)
    if path.is_symlink():
        raise ValueError("target provider admission must not be a symbolic link")
    admission = load_target_provider_admission(path)
    validate_target_admission_execution_binding(
        lock,
        admission,
        provider_report=provider_report,
        query_metadata=query_metadata,
    )
    return admission


__all__ = [
    "TARGET_PROVIDER_ADMISSION_METADATA_KEY",
    "load_target_admission_for_execution",
    "validate_target_admission_execution_binding",
]
