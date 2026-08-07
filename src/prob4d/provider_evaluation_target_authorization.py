"""Authorize target-bound provider evaluation before target artifact I/O."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ._heldout_promotion_common import (
    _SHA256,
    _exact_keys,
    _strict_bool,
    _strict_digest,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._heldout_promotion_lock import HeldoutProviderPromotionLockV1
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._provider_evaluation_manifest import (
    PROVIDER_EVALUATION_DECISION_VERSION,
    PROVIDER_EVALUATION_SCHEMA,
    EvaluationModeName,
    ProviderEvaluationDecisionPolicy,
    _decision_policy,
    validate_finite_json,
)
from ._selection_evidence_common import _sha256_json, _strict_integer
from .target_provider_admission import (
    HeldoutTargetProviderAdmissionV1,
    validate_target_provider_admission_against_lock,
)

TARGET_PROVIDER_ADMISSION_METADATA_KEY = "target_provider_admission_id"
PROVIDER_EVALUATION_TARGET_AUTHORIZATION_FIELD = "target_admission_authorization"
PROVIDER_EVALUATION_TARGET_AUTHORIZATION_SCHEMA = (
    "prob4d.provider-evaluation-target-authorization"
)
PROVIDER_EVALUATION_TARGET_AUTHORIZATION_VERSION = 1
PROVIDER_EVALUATION_TARGET_AUTHORIZED_REPORT_VERSION = 4
PROVIDER_EVALUATION_TARGET_AUTHORIZATION_CLAIM_BOUNDARY = (
    "This receipt proves that one exact frozen provider-evaluation manifest, "
    "promotion lock, and target-provider admission were mutually consistent before "
    "truth or prediction artifacts were opened. It does not establish provider "
    "competence, uncertainty calibration, BayesianPhysTwin benefit, Causal4D "
    "benefit, deployment safety, or state of the art."
)

_ROOT_FIELDS = {
    "schema_name",
    "schema_version",
    "primary_mode",
    "reference_method",
    "cases",
    "metadata",
    "decision_policy",
}
_CASE_FIELDS = {
    "case_id",
    "group_id",
    "truth",
    "predictions",
    "boundary_frames",
    "prefix_frame_stop_exclusive",
}
_AUTHORIZATION_FIELDS = {
    "schema_name",
    "schema_version",
    "promotion_lock_id",
    "target_provider_admission_id",
    "cohort_binding_id",
    "source_manifest_sha256",
    "target_group_ids",
    "registered_method_ids",
    "reference_method",
    "case_count",
    "decision_minimum_group_count",
    "bootstrap_resamples",
    "bootstrap_seed",
    "legacy_artifacts_allowed",
    "target_outcomes_opened_during_authorization",
    "claim_boundary",
    "provider_evaluation_target_authorization_id",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate provider-evaluation manifest key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite provider-evaluation manifest constant {value!r}")


def _canonical_strings(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be a list of strings")
    result = tuple(
        _strict_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(cast(Sequence[object], value))
    )
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _absolute_target_path(root: Path, value: object, *, name: str) -> str:
    raw = Path(_strict_string(value, name=name))
    return str((raw if raw.is_absolute() else root / raw).resolve())


@dataclass(frozen=True, slots=True)
class ProviderEvaluationManifestSnapshotV1:
    """Exact manifest bytes plus target-free structural inspection."""

    source_path: Path
    source_payload: bytes
    source_manifest_sha256: str
    manifest: Mapping[str, Any]
    primary_mode: EvaluationModeName
    reference_method: str
    case_ids: tuple[str, ...]
    case_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    decision_policy: ProviderEvaluationDecisionPolicy
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        input_path = Path(self.source_path)
        if input_path.is_symlink():
            raise ValueError("provider-evaluation manifest must not be a symbolic link")
        source = input_path.resolve()
        if not source.is_file():
            raise ValueError("provider-evaluation manifest must be a regular file")
        if type(self.source_payload) is not bytes:
            raise ValueError("source_payload must contain exact manifest bytes")
        digest = _strict_digest(
            self.source_manifest_sha256,
            name="source_manifest_sha256",
            pattern=_SHA256,
        )
        if hashlib.sha256(self.source_payload).hexdigest() != digest:
            raise ValueError("source_manifest_sha256 does not match source_payload")
        if not isinstance(self.decision_policy, ProviderEvaluationDecisionPolicy):
            raise ValueError("decision_policy must be ProviderEvaluationDecisionPolicy")
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "source_manifest_sha256", digest)
        object.__setattr__(
            self,
            "manifest",
            frozen_finite_json_mapping(self.manifest, name="provider-evaluation manifest"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="provider-evaluation metadata"),
        )

    def assert_source_unchanged(self) -> None:
        try:
            current = self.source_path.read_bytes()
        except OSError as error:
            raise ValueError("provider-evaluation manifest became unreadable") from error
        if current != self.source_payload:
            raise ValueError("provider-evaluation manifest changed during evaluation")

    def execution_manifest(self) -> dict[str, object]:
        """Rewrite only target paths for a private exact-semantics snapshot."""

        result = copy.deepcopy(plain_json(self.manifest))
        if not isinstance(result, dict):
            raise AssertionError("validated provider-evaluation manifest is not a mapping")
        raw_cases = result["cases"]
        if not isinstance(raw_cases, list):
            raise AssertionError("validated provider-evaluation cases are not a list")
        root = self.source_path.parent
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                raise AssertionError("validated provider-evaluation case is not a mapping")
            raw_case["truth"] = _absolute_target_path(
                root,
                raw_case["truth"],
                name=f"cases[{index}].truth",
            )
            predictions = raw_case["predictions"]
            if not isinstance(predictions, dict):
                raise AssertionError("validated provider predictions are not a mapping")
            raw_case["predictions"] = {
                method: _absolute_target_path(
                    root,
                    path,
                    name=f"cases[{index}].predictions[{method!r}]",
                )
                for method, path in predictions.items()
            }
        return cast(dict[str, object], result)

    @contextmanager
    def materialize_execution_manifest(self) -> Iterator[Path]:
        """Materialize a private manifest while preserving original path semantics."""

        with tempfile.TemporaryDirectory(prefix="prob4d-provider-evaluation-") as temporary:
            destination = Path(temporary) / "provider-evaluation.json"
            destination.write_text(
                json.dumps(
                    self.execution_manifest(),
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            yield destination


def load_provider_evaluation_manifest_snapshot(
    manifest_path: str | Path,
) -> ProviderEvaluationManifestSnapshotV1:
    """Inspect exact manifest bytes without resolving or opening target paths."""

    input_path = Path(manifest_path)
    if input_path.is_symlink():
        raise ValueError("provider-evaluation manifest must not be a symbolic link")
    source = input_path.resolve()
    try:
        payload = source.read_bytes()
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read provider-evaluation manifest: {source}") from error
    manifest = _strict_mapping(decoded, name="provider-evaluation manifest")
    validate_finite_json(manifest, name="provider-evaluation manifest")
    _exact_keys(manifest, _ROOT_FIELDS, name="provider-evaluation manifest")
    if manifest["schema_name"] != PROVIDER_EVALUATION_SCHEMA:
        raise ValueError("unsupported provider-evaluation manifest schema")
    version = _strict_integer(manifest["schema_version"], name="schema_version", minimum=1)
    if version != PROVIDER_EVALUATION_DECISION_VERSION:
        raise ValueError(
            "target-authorized provider evaluation requires a decision-bearing "
            "schema-v2 manifest"
        )
    primary_mode_value = _strict_string(manifest["primary_mode"], name="primary_mode")
    if primary_mode_value == "oracle_aligned":
        raise ValueError("target-authorized provider evaluation cannot use oracle alignment")
    if primary_mode_value not in {"metric", "prefix_aligned"}:
        raise ValueError("target-authorized primary_mode must be metric or prefix_aligned")
    primary_mode = cast(EvaluationModeName, primary_mode_value)
    reference_method = _strict_string(manifest["reference_method"], name="reference_method")
    metadata = _strict_mapping(manifest["metadata"], name="provider-evaluation metadata")
    if TARGET_PROVIDER_ADMISSION_METADATA_KEY in metadata:
        raise ValueError(
            "provider-evaluation manifest metadata must not contain "
            "target_provider_admission_id because that creates a circular content identity; "
            "supply the admission separately at execution"
        )

    raw_cases = _strict_list(manifest["cases"], name="provider-evaluation cases")
    if not raw_cases:
        raise ValueError("provider-evaluation cases must not be empty")
    case_ids: list[str] = []
    case_group_ids: list[str] = []
    expected_methods: tuple[str, ...] | None = None
    for index, raw_case in enumerate(raw_cases):
        case = _strict_mapping(raw_case, name=f"cases[{index}]")
        _exact_keys(case, _CASE_FIELDS, name=f"cases[{index}]")
        case_id = _strict_string(case["case_id"], name=f"cases[{index}].case_id")
        group_id = _strict_string(case["group_id"], name=f"cases[{index}].group_id")
        _strict_string(case["truth"], name=f"cases[{index}].truth")
        predictions = _strict_mapping(
            case["predictions"],
            name=f"cases[{index}].predictions",
        )
        if not predictions:
            raise ValueError(f"cases[{index}].predictions must not be empty")
        methods = tuple(
            sorted(
                _strict_string(method, name=f"cases[{index}].prediction method")
                for method in predictions
            )
        )
        for method, path in predictions.items():
            _strict_string(path, name=f"cases[{index}].predictions[{method!r}]")
        if expected_methods is None:
            expected_methods = methods
        elif methods != expected_methods:
            raise ValueError("provider-evaluation method set changes across cases")
        boundary = _strict_list(case["boundary_frames"], name=f"cases[{index}].boundary_frames")
        boundary_values = tuple(
            _strict_integer(
                value,
                name=f"cases[{index}].boundary_frames",
                minimum=0,
            )
            for value in boundary
        )
        if boundary_values != tuple(sorted(set(boundary_values))):
            raise ValueError(f"cases[{index}].boundary_frames must be sorted and unique")
        prefix_stop = case["prefix_frame_stop_exclusive"]
        if prefix_stop is not None:
            _strict_integer(
                prefix_stop,
                name=f"cases[{index}].prefix_frame_stop_exclusive",
                minimum=1,
            )
        if primary_mode == "prefix_aligned" and prefix_stop is None:
            raise ValueError("prefix_aligned evaluation requires a prefix stop for every case")
        case_ids.append(case_id)
        case_group_ids.append(group_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("provider-evaluation case IDs must be unique")
    if expected_methods is None or reference_method not in expected_methods:
        raise ValueError("reference_method must identify one registered prediction method")

    decision_policy = _decision_policy(
        manifest["decision_policy"],
        methods=expected_methods,
        reference_method=reference_method,
    )
    return ProviderEvaluationManifestSnapshotV1(
        source_path=source,
        source_payload=payload,
        source_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        manifest=manifest,
        primary_mode=primary_mode,
        reference_method=reference_method,
        case_ids=tuple(case_ids),
        case_group_ids=tuple(case_group_ids),
        target_group_ids=tuple(sorted(set(case_group_ids))),
        method_ids=expected_methods,
        decision_policy=decision_policy,
        metadata=metadata,
    )


def build_provider_evaluation_target_authorization(
    snapshot: ProviderEvaluationManifestSnapshotV1,
    lock: HeldoutProviderPromotionLockV1,
    admission: HeldoutTargetProviderAdmissionV1,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    legacy_artifacts_allowed: bool,
) -> dict[str, object]:
    """Authorize one exact target evaluation before truth or predictions are opened."""

    if not isinstance(snapshot, ProviderEvaluationManifestSnapshotV1):
        raise ValueError("snapshot must be ProviderEvaluationManifestSnapshotV1")
    if not isinstance(lock, HeldoutProviderPromotionLockV1):
        raise ValueError("lock must be HeldoutProviderPromotionLockV1")
    if not isinstance(admission, HeldoutTargetProviderAdmissionV1):
        raise ValueError("admission must be HeldoutTargetProviderAdmissionV1")
    validate_target_provider_admission_against_lock(admission, lock)
    if lock.frozen_artifact_ids.get("cohort_binding") is None:
        raise ValueError("target authorization requires a cohort-bound promotion lock")
    if snapshot.source_manifest_sha256 != lock.provider_evaluation_manifest_sha256:
        raise ValueError("provider-evaluation manifest bytes differ from the promotion lock")
    if snapshot.target_group_ids != lock.target_group_ids:
        raise ValueError("provider-evaluation target groups differ from the promotion lock")
    if snapshot.method_ids != lock.provider_method_ids:
        raise ValueError("provider-evaluation methods differ from the promotion lock")
    if snapshot.reference_method != lock.provider_reference_method_id:
        raise ValueError("provider-evaluation reference method differs from the promotion lock")
    if snapshot.decision_policy.minimum_group_count != lock.minimum_target_group_count:
        raise ValueError("provider decision minimum group count differs from the promotion lock")
    resamples = _strict_integer(
        bootstrap_resamples,
        name="bootstrap_resamples",
        minimum=1,
    )
    seed = _strict_integer(bootstrap_seed, name="bootstrap_seed", minimum=0)
    if resamples != lock.bootstrap_resamples:
        raise ValueError("provider bootstrap resamples differ from the promotion lock")
    if seed != lock.bootstrap_seed:
        raise ValueError("provider bootstrap seed differs from the promotion lock")
    legacy = _strict_bool(legacy_artifacts_allowed, name="legacy_artifacts_allowed")
    if legacy:
        raise ValueError("target-authorized provider evaluation cannot admit legacy artifacts")

    descriptor: dict[str, object] = {
        "schema_name": PROVIDER_EVALUATION_TARGET_AUTHORIZATION_SCHEMA,
        "schema_version": PROVIDER_EVALUATION_TARGET_AUTHORIZATION_VERSION,
        "promotion_lock_id": lock.promotion_lock_id,
        "target_provider_admission_id": admission.target_provider_admission_id,
        "cohort_binding_id": admission.cohort_binding_id,
        "source_manifest_sha256": snapshot.source_manifest_sha256,
        "target_group_ids": list(snapshot.target_group_ids),
        "registered_method_ids": list(snapshot.method_ids),
        "reference_method": snapshot.reference_method,
        "case_count": len(snapshot.case_ids),
        "decision_minimum_group_count": snapshot.decision_policy.minimum_group_count,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "legacy_artifacts_allowed": legacy,
        "target_outcomes_opened_during_authorization": False,
        "claim_boundary": PROVIDER_EVALUATION_TARGET_AUTHORIZATION_CLAIM_BOUNDARY,
    }
    return {
        **descriptor,
        "provider_evaluation_target_authorization_id": _sha256_json(descriptor),
    }


def validate_provider_evaluation_target_authorization(
    provider_report: object,
    lock: HeldoutProviderPromotionLockV1,
    admission: HeldoutTargetProviderAdmissionV1,
) -> Mapping[str, Any]:
    """Replay the provider report's pre-I/O target authorization receipt."""

    validate_target_provider_admission_against_lock(admission, lock)
    report = _strict_mapping(provider_report, name="provider evaluation report")
    if report.get("schema_name") != "prob4d.provider-evaluation-report":
        raise ValueError("provider report has the wrong schema")
    report_version = _strict_integer(
        report.get("schema_version"),
        name="provider report schema_version",
        minimum=1,
    )
    if report_version != PROVIDER_EVALUATION_TARGET_AUTHORIZED_REPORT_VERSION:
        raise ValueError("cohort-bound provider report must retain target authorization")
    authorization = _strict_mapping(
        report.get(PROVIDER_EVALUATION_TARGET_AUTHORIZATION_FIELD),
        name="provider target authorization",
    )
    _exact_keys(
        authorization,
        _AUTHORIZATION_FIELDS,
        name="provider target authorization",
    )
    if authorization["schema_name"] != PROVIDER_EVALUATION_TARGET_AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported provider target authorization schema")
    version = _strict_integer(
        authorization["schema_version"],
        name="provider target authorization schema_version",
        minimum=1,
    )
    if version != PROVIDER_EVALUATION_TARGET_AUTHORIZATION_VERSION:
        raise ValueError("unsupported provider target authorization version")
    if (
        authorization["claim_boundary"]
        != PROVIDER_EVALUATION_TARGET_AUTHORIZATION_CLAIM_BOUNDARY
    ):
        raise ValueError("provider target authorization claim boundary changed")
    supplied_id = _strict_digest(
        authorization["provider_evaluation_target_authorization_id"],
        name="provider_evaluation_target_authorization_id",
        pattern=_SHA256,
    )
    descriptor = dict(authorization)
    descriptor.pop("provider_evaluation_target_authorization_id")
    if supplied_id != _sha256_json(descriptor):
        raise ValueError("provider target authorization ID mismatch")

    expected_pairs = {
        "promotion_lock_id": lock.promotion_lock_id,
        "target_provider_admission_id": admission.target_provider_admission_id,
        "cohort_binding_id": admission.cohort_binding_id,
        "source_manifest_sha256": lock.provider_evaluation_manifest_sha256,
        "reference_method": lock.provider_reference_method_id,
        "decision_minimum_group_count": lock.minimum_target_group_count,
        "bootstrap_resamples": lock.bootstrap_resamples,
        "bootstrap_seed": lock.bootstrap_seed,
        "legacy_artifacts_allowed": False,
        "target_outcomes_opened_during_authorization": False,
    }
    for field_name, expected in expected_pairs.items():
        if authorization[field_name] != expected:
            raise ValueError(f"provider target authorization changed {field_name}")
    if _canonical_strings(
        authorization["target_group_ids"],
        name="provider target authorization target_group_ids",
    ) != lock.target_group_ids:
        raise ValueError("provider target authorization changed target groups")
    if _canonical_strings(
        authorization["registered_method_ids"],
        name="provider target authorization registered_method_ids",
    ) != lock.provider_method_ids:
        raise ValueError("provider target authorization changed registered methods")

    report_manifest_sha = _strict_digest(
        report.get("source_manifest_sha256"),
        name="provider report source_manifest_sha256",
        pattern=_SHA256,
    )
    if report_manifest_sha != authorization["source_manifest_sha256"]:
        raise ValueError("provider report and target authorization use different manifests")
    if report.get("reference_method") != authorization["reference_method"]:
        raise ValueError("provider report and target authorization use different references")
    if report.get("bootstrap_resamples") != authorization["bootstrap_resamples"]:
        raise ValueError("provider report and target authorization use different resamples")
    if report.get("bootstrap_seed") != authorization["bootstrap_seed"]:
        raise ValueError("provider report and target authorization use different seeds")
    if report.get("legacy_artifacts_allowed") is not False:
        raise ValueError("target-authorized provider report cannot admit legacy artifacts")

    raw_records = _strict_list(report.get("cases"), name="provider report cases")
    case_ids: set[str] = set()
    observed_groups: set[str] = set()
    observed_methods: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        record = _strict_mapping(raw_record, name=f"provider report cases[{index}]")
        case_ids.add(_strict_string(record.get("case_id"), name="provider case_id"))
        observed_groups.add(_strict_string(record.get("group_id"), name="provider group_id"))
        observed_methods.add(_strict_string(record.get("method_id"), name="provider method_id"))
    case_count = _strict_integer(
        authorization["case_count"],
        name="provider target authorization case_count",
        minimum=1,
    )
    if len(case_ids) != case_count:
        raise ValueError("provider report case count differs from target authorization")
    if tuple(sorted(observed_groups)) != lock.target_group_ids:
        raise ValueError("provider report target groups differ from target authorization")
    if tuple(sorted(observed_methods)) != lock.provider_method_ids:
        raise ValueError("provider report methods differ from target authorization")

    report_metadata = _strict_mapping(
        report.get("manifest_metadata"),
        name="provider report manifest_metadata",
    )
    if TARGET_PROVIDER_ADMISSION_METADATA_KEY in report_metadata:
        raise ValueError(
            "provider report manifest metadata contains the circular target admission field"
        )
    return authorization


__all__ = [
    "PROVIDER_EVALUATION_TARGET_AUTHORIZATION_CLAIM_BOUNDARY",
    "PROVIDER_EVALUATION_TARGET_AUTHORIZATION_FIELD",
    "PROVIDER_EVALUATION_TARGET_AUTHORIZATION_SCHEMA",
    "PROVIDER_EVALUATION_TARGET_AUTHORIZATION_VERSION",
    "PROVIDER_EVALUATION_TARGET_AUTHORIZED_REPORT_VERSION",
    "TARGET_PROVIDER_ADMISSION_METADATA_KEY",
    "ProviderEvaluationManifestSnapshotV1",
    "build_provider_evaluation_target_authorization",
    "load_provider_evaluation_manifest_snapshot",
    "validate_provider_evaluation_target_authorization",
]
