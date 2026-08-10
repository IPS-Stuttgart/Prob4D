"""Outcome-blind selection among frozen provider-support designs.

A design request contains a finite candidate set of complete
:class:`~prob4d.provider_support_feasibility.ProviderSupportFeasibilityRequestV1`
objects. Selection uses only causal-prefix support metadata. Prediction payloads,
provider residuals, and target outcomes remain closed.

The selected request and result retain their existing version-1 schemas, so
current promotion authorization consumes them without a compatibility change.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from ._heldout_promotion_common import _atomic_write_json, _load_json
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from .provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportFeasibilityV1,
    ProviderSupportStreamV1,
    evaluate_provider_support_feasibility,
    write_provider_support_feasibility,
    write_provider_support_feasibility_request,
)

PROVIDER_SUPPORT_DESIGN_REQUEST_SCHEMA = "prob4d.provider-support-design-request"
PROVIDER_SUPPORT_DESIGN_REQUEST_VERSION = 1
PROVIDER_SUPPORT_DESIGN_SCHEMA = "prob4d.provider-support-design"
PROVIDER_SUPPORT_DESIGN_VERSION = 1
PROVIDER_SUPPORT_DESIGN_SELECTION_RULE = (
    "support-feasible-then-maximin-group-support-then-supported-frame-cells-"
    "then-shortest-causal-span-v1"
)
PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY = (
    "This artifact selects one exact provider-support configuration from a finite "
    "candidate set frozen before prediction payloads, provider residuals, or target "
    "outcomes are opened. Selection uses only causal-prefix metadata, geometry, "
    "camera-calibration, metric-anchor, and predeclared technical-support evidence. "
    "A selected support-feasible design does not establish provider competence, "
    "calibrated uncertainty, BayesianPhysTwin benefit, Causal4D benefit, deployment "
    "safety, or state of the art."
)

_CANDIDATE_FIELDS = {"candidate_id", "feasibility_request", "metadata"}
_REQUEST_FIELDS = {
    "schema_name",
    "schema_version",
    "protocol_id",
    "selection_rule",
    "prediction_payloads_opened",
    "residuals_used",
    "target_outcomes_used",
    "candidates",
    "metadata",
    "claim_boundary",
    "request_id",
}
_RESULT_FIELDS = {
    "schema_name",
    "schema_version",
    "request",
    "candidate_evaluations",
    "feasible_candidate_count",
    "selected_candidate_id",
    "selected_support_request_id",
    "selected_support_feasibility_id",
    "support_design_feasible",
    "decision_reason",
    "claim_boundary",
    "provider_support_design_id",
}
_COMMON_REQUEST_FIELDS = (
    "source_repository",
    "source_revision",
    "provider_family",
    "provider_repository",
    "provider_revision",
    "model_set_id",
    "loader_id",
    "cohort_binding_id",
    "promotion_lock_id",
    "coordinate_semantics",
    "admission_rule",
    "minimum_supported_fraction",
    "permitted_technical_exclusion_codes",
    "maximum_technical_exclusions",
)


def _strict_false(value: object, *, name: str) -> bool:
    result = _strict_bool(value, name=name)
    if result:
        raise ValueError(f"{name} must be false for provider support design")
    return result


def _request_signature(
    request: ProviderSupportFeasibilityRequestV1,
) -> tuple[object, ...]:
    return tuple(getattr(request, name) for name in _COMMON_REQUEST_FIELDS)


def _stream_requirement_signature(stream: ProviderSupportStreamV1) -> tuple[object, ...]:
    return (
        stream.key,
        stream.minimum_geometry_support_fraction,
        stream.intrinsics_required,
        stream.extrinsics_required,
        stream.metric_anchor_required,
    )


@dataclass(frozen=True, slots=True)
class ProviderSupportDesignCandidateV1:
    """One prospectively frozen provider-support configuration."""

    candidate_id: str
    feasibility_request: ProviderSupportFeasibilityRequestV1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _strict_string(self.candidate_id, name="candidate_id"),
        )
        if not isinstance(
            self.feasibility_request,
            ProviderSupportFeasibilityRequestV1,
        ):
            raise TypeError(
                "feasibility_request must be ProviderSupportFeasibilityRequestV1"
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="provider support design candidate metadata",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "feasibility_request": self.feasibility_request.to_dict(),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportDesignCandidateV1:
        mapping = _strict_mapping(value, name="provider support design candidate")
        _exact_keys(
            mapping,
            _CANDIDATE_FIELDS,
            name="provider support design candidate",
        )
        return cls(
            candidate_id=mapping["candidate_id"],
            feasibility_request=ProviderSupportFeasibilityRequestV1.from_dict(
                mapping["feasibility_request"]
            ),
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class ProviderSupportDesignRequestV1:
    """Finite, outcome-blind candidate set and deterministic selection rule."""

    protocol_id: str
    candidates: tuple[ProviderSupportDesignCandidateV1, ...]
    prediction_payloads_opened: bool = False
    residuals_used: bool = False
    target_outcomes_used: bool = False
    selection_rule: str = PROVIDER_SUPPORT_DESIGN_SELECTION_RULE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            _strict_string(self.protocol_id, name="protocol_id"),
        )
        rule = _strict_string(self.selection_rule, name="selection_rule")
        if rule != PROVIDER_SUPPORT_DESIGN_SELECTION_RULE:
            raise ValueError("unsupported provider support design selection rule")
        object.__setattr__(self, "selection_rule", rule)
        for name in (
            "prediction_payloads_opened",
            "residuals_used",
            "target_outcomes_used",
        ):
            object.__setattr__(self, name, _strict_false(getattr(self, name), name=name))
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError(
                "candidates must be a nonempty tuple of "
                "ProviderSupportDesignCandidateV1"
            )
        if not all(
            isinstance(candidate, ProviderSupportDesignCandidateV1)
            for candidate in self.candidates
        ):
            raise ValueError(
                "candidates must contain only ProviderSupportDesignCandidateV1"
            )
        candidates = tuple(
            sorted(self.candidates, key=lambda candidate: candidate.candidate_id)
        )
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique")
        request_ids = tuple(
            candidate.feasibility_request.request_id for candidate in candidates
        )
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("candidate feasibility requests must be unique")
        first = candidates[0].feasibility_request
        signature = _request_signature(first)
        roster = tuple(stream.key for stream in first.streams)
        requirements = tuple(
            _stream_requirement_signature(stream) for stream in first.streams
        )
        for candidate in candidates[1:]:
            request = candidate.feasibility_request
            if _request_signature(request) != signature:
                raise ValueError(
                    "candidate feasibility requests must share provider, cohort, "
                    "promotion-lock, admission, and exclusion identities"
                )
            if tuple(stream.key for stream in request.streams) != roster:
                raise ValueError(
                    "candidate feasibility requests must share the exact stream roster"
                )
            if tuple(
                _stream_requirement_signature(stream) for stream in request.streams
            ) != requirements:
                raise ValueError(
                    "candidate feasibility requests must not change support thresholds "
                    "or required calibration classes"
                )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="provider support design request metadata",
            ),
        )

    def _identity_dict(self) -> dict[str, object]:
        return {
            "schema_name": PROVIDER_SUPPORT_DESIGN_REQUEST_SCHEMA,
            "schema_version": PROVIDER_SUPPORT_DESIGN_REQUEST_VERSION,
            "protocol_id": self.protocol_id,
            "selection_rule": self.selection_rule,
            "prediction_payloads_opened": self.prediction_payloads_opened,
            "residuals_used": self.residuals_used,
            "target_outcomes_used": self.target_outcomes_used,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY,
        }

    @property
    def request_id(self) -> str:
        return _sha256_json(self._identity_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_dict(), "request_id": self.request_id}

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportDesignRequestV1:
        mapping = _strict_mapping(value, name="provider support design request")
        _exact_keys(mapping, _REQUEST_FIELDS, name="provider support design request")
        if mapping["schema_name"] != PROVIDER_SUPPORT_DESIGN_REQUEST_SCHEMA:
            raise ValueError("unsupported provider support design request schema")
        if mapping["schema_version"] != PROVIDER_SUPPORT_DESIGN_REQUEST_VERSION:
            raise ValueError("unsupported provider support design request version")
        if mapping["claim_boundary"] != PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY:
            raise ValueError("provider support design claim boundary changed")
        request = cls(
            protocol_id=mapping["protocol_id"],
            candidates=tuple(
                ProviderSupportDesignCandidateV1.from_dict(candidate)
                for candidate in _strict_list(mapping["candidates"], name="candidates")
            ),
            prediction_payloads_opened=mapping["prediction_payloads_opened"],
            residuals_used=mapping["residuals_used"],
            target_outcomes_used=mapping["target_outcomes_used"],
            selection_rule=mapping["selection_rule"],
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )
        identifier = _strict_digest(
            mapping["request_id"],
            name="request_id",
            pattern=_SHA256,
        )
        if identifier != request.request_id:
            raise ValueError("provider support design request identity mismatch")
        return request


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    candidate_id: str
    support_feasibility: ProviderSupportFeasibilityV1
    group_count: int
    supported_group_count: int
    minimum_group_support_fraction: float
    supported_required_frame_count: int
    geometry_supported_required_frame_count: int
    required_frame_count: int
    maximum_causal_span_frames: int
    total_causal_span_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "support_feasibility": self.support_feasibility.to_dict(),
            "group_count": self.group_count,
            "supported_group_count": self.supported_group_count,
            "minimum_group_support_fraction": self.minimum_group_support_fraction,
            "supported_required_frame_count": self.supported_required_frame_count,
            "geometry_supported_required_frame_count": (
                self.geometry_supported_required_frame_count
            ),
            "required_frame_count": self.required_frame_count,
            "maximum_causal_span_frames": self.maximum_causal_span_frames,
            "total_causal_span_frames": self.total_causal_span_frames,
        }


def _group_counts(
    result: ProviderSupportFeasibilityV1,
) -> tuple[tuple[str, int, int], ...]:
    counts: dict[str, list[int]] = {}
    for stream in result.stream_results:
        values = counts.setdefault(stream.group_id, [0, 0])
        if not stream.excluded_from_admission:
            values[1] += 1
            if stream.supported:
                values[0] += 1
    return tuple(
        (group_id, values[0], values[1])
        for group_id, values in sorted(counts.items())
    )


def _evaluate_candidate(candidate: ProviderSupportDesignCandidateV1) -> _CandidateEvaluation:
    result = evaluate_provider_support_feasibility(candidate.feasibility_request)
    groups = _group_counts(result)
    fractions = tuple(
        Fraction(supported, evaluable) if evaluable else Fraction(0, 1)
        for _, supported, evaluable in groups
    )
    spans = tuple(
        stream.causal_frame_stop_exclusive - stream.causal_frame_start
        for stream in result.request.streams
    )
    return _CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        support_feasibility=result,
        group_count=len(groups),
        supported_group_count=sum(supported > 0 for _, supported, _ in groups),
        minimum_group_support_fraction=float(min(fractions)),
        supported_required_frame_count=sum(
            item.required_frame_count for item in result.stream_results if item.supported
        ),
        geometry_supported_required_frame_count=sum(
            item.geometry_supported_required_frame_count
            for item in result.stream_results
            if not item.excluded_from_admission
        ),
        required_frame_count=sum(
            item.required_frame_count
            for item in result.stream_results
            if not item.excluded_from_admission
        ),
        maximum_causal_span_frames=max(spans),
        total_causal_span_frames=sum(spans),
    )


def _selection_key(
    evaluation: _CandidateEvaluation,
) -> tuple[bool, float, int, int, int, int, int, int, str]:
    return (
        not evaluation.support_feasibility.support_feasible,
        -evaluation.minimum_group_support_fraction,
        -evaluation.supported_group_count,
        -evaluation.supported_required_frame_count,
        -evaluation.geometry_supported_required_frame_count,
        -evaluation.support_feasibility.supported_stream_count,
        evaluation.maximum_causal_span_frames,
        evaluation.total_causal_span_frames,
        evaluation.candidate_id,
    )


@dataclass(frozen=True, slots=True)
class ProviderSupportDesignV1:
    """Replayable selection from one exact frozen support-design request."""

    request: ProviderSupportDesignRequestV1

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProviderSupportDesignRequestV1):
            raise TypeError("request must be ProviderSupportDesignRequestV1")

    @property
    def candidate_evaluations(self) -> tuple[_CandidateEvaluation, ...]:
        return tuple(_evaluate_candidate(candidate) for candidate in self.request.candidates)

    @property
    def selected_evaluation(self) -> _CandidateEvaluation:
        return min(self.candidate_evaluations, key=_selection_key)

    @property
    def feasible_candidate_count(self) -> int:
        return sum(
            item.support_feasibility.support_feasible
            for item in self.candidate_evaluations
        )

    @property
    def selected_candidate_id(self) -> str:
        return self.selected_evaluation.candidate_id

    @property
    def selected_support_request(self) -> ProviderSupportFeasibilityRequestV1:
        return self.selected_evaluation.support_feasibility.request

    @property
    def selected_support_request_id(self) -> str:
        return self.selected_support_request.request_id

    @property
    def selected_support_feasibility(self) -> ProviderSupportFeasibilityV1:
        return self.selected_evaluation.support_feasibility

    @property
    def selected_support_feasibility_id(self) -> str:
        return self.selected_support_feasibility.provider_support_feasibility_id

    @property
    def support_design_feasible(self) -> bool:
        return self.selected_support_feasibility.support_feasible

    @property
    def decision_reason(self) -> str:
        return (
            "selected-support-feasible"
            if self.support_design_feasible
            else "no-support-feasible-candidate"
        )

    def _identity_dict(self) -> dict[str, object]:
        return {
            "schema_name": PROVIDER_SUPPORT_DESIGN_SCHEMA,
            "schema_version": PROVIDER_SUPPORT_DESIGN_VERSION,
            "request": self.request.to_dict(),
            "candidate_evaluations": [
                item.to_dict() for item in self.candidate_evaluations
            ],
            "feasible_candidate_count": self.feasible_candidate_count,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_support_request_id": self.selected_support_request_id,
            "selected_support_feasibility_id": self.selected_support_feasibility_id,
            "support_design_feasible": self.support_design_feasible,
            "decision_reason": self.decision_reason,
            "claim_boundary": PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY,
        }

    @property
    def provider_support_design_id(self) -> str:
        return _sha256_json(self._identity_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_dict(),
            "provider_support_design_id": self.provider_support_design_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportDesignV1:
        mapping = _strict_mapping(value, name="provider support design")
        _exact_keys(mapping, _RESULT_FIELDS, name="provider support design")
        if mapping["schema_name"] != PROVIDER_SUPPORT_DESIGN_SCHEMA:
            raise ValueError("unsupported provider support design schema")
        if mapping["schema_version"] != PROVIDER_SUPPORT_DESIGN_VERSION:
            raise ValueError("unsupported provider support design version")
        if mapping["claim_boundary"] != PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY:
            raise ValueError("provider support design claim boundary changed")
        expected = cls(ProviderSupportDesignRequestV1.from_dict(mapping["request"]))
        identifier = _strict_digest(
            mapping["provider_support_design_id"],
            name="provider_support_design_id",
            pattern=_SHA256,
        )
        if identifier != expected.provider_support_design_id:
            raise ValueError("provider support design identity mismatch")
        if plain_json(mapping) != expected.to_dict():
            raise ValueError("provider support design does not replay from its request")
        return expected


def evaluate_provider_support_design(
    request: ProviderSupportDesignRequestV1,
) -> ProviderSupportDesignV1:
    """Select one exact candidate with the frozen outcome-blind ranking rule."""

    if not isinstance(request, ProviderSupportDesignRequestV1):
        raise TypeError("request must be ProviderSupportDesignRequestV1")
    return ProviderSupportDesignV1(request)


def write_provider_support_design_request(
    path: Path,
    request: ProviderSupportDesignRequestV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(request, ProviderSupportDesignRequestV1):
        raise TypeError("request must be ProviderSupportDesignRequestV1")
    _atomic_write_json(path, request.to_dict(), overwrite=overwrite)


def load_provider_support_design_request(path: Path) -> ProviderSupportDesignRequestV1:
    value, _ = _load_json(path, name="provider support design request")
    return ProviderSupportDesignRequestV1.from_dict(value)


def write_provider_support_design(
    path: Path,
    result: ProviderSupportDesignV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(result, ProviderSupportDesignV1):
        raise TypeError("result must be ProviderSupportDesignV1")
    _atomic_write_json(path, result.to_dict(), overwrite=overwrite)


def load_provider_support_design(path: Path) -> ProviderSupportDesignV1:
    value, _ = _load_json(path, name="provider support design")
    return ProviderSupportDesignV1.from_dict(value)


def write_selected_provider_support_request(
    path: Path,
    result: ProviderSupportDesignV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(result, ProviderSupportDesignV1):
        raise TypeError("result must be ProviderSupportDesignV1")
    write_provider_support_feasibility_request(
        path,
        result.selected_support_request,
        overwrite=overwrite,
    )


def write_selected_provider_support_feasibility(
    path: Path,
    result: ProviderSupportDesignV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(result, ProviderSupportDesignV1):
        raise TypeError("result must be ProviderSupportDesignV1")
    write_provider_support_feasibility(
        path,
        result.selected_support_feasibility,
        overwrite=overwrite,
    )


def _summary(result: ProviderSupportDesignV1) -> dict[str, object]:
    selected = result.selected_evaluation
    return {
        "provider_support_design_id": result.provider_support_design_id,
        "request_id": result.request.request_id,
        "support_design_feasible": result.support_design_feasible,
        "decision_reason": result.decision_reason,
        "candidate_count": len(result.candidate_evaluations),
        "feasible_candidate_count": result.feasible_candidate_count,
        "selected_candidate_id": result.selected_candidate_id,
        "selected_support_request_id": result.selected_support_request_id,
        "selected_support_feasibility_id": result.selected_support_feasibility_id,
        "minimum_group_support_fraction": (
            selected.minimum_group_support_fraction
        ),
        "supported_required_frame_count": (
            selected.supported_required_frame_count
        ),
        "maximum_causal_span_frames": selected.maximum_causal_span_frames,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser(
        "select",
        help="select and persist one frozen support design",
    )
    select.add_argument("--request", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--selected-request-output", type=Path)
    select.add_argument("--selected-feasibility-output", type=Path)
    select.add_argument("--overwrite", action="store_true")
    select.add_argument("--compact", action="store_true")
    verify = subparsers.add_parser(
        "verify",
        help="verify and replay a provider support design",
    )
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        request = load_provider_support_design_request(args.request)
        result = evaluate_provider_support_design(request)
        paths = tuple(
            path
            for path in (
                args.output,
                args.selected_request_output,
                args.selected_feasibility_output,
            )
            if path is not None
        )
        resolved = tuple(path.resolve(strict=False) for path in paths)
        if len(resolved) != len(set(resolved)):
            raise ValueError("provider support design output paths must be distinct")
        if not args.overwrite:
            existing = [path for path in paths if path.exists()]
            if existing:
                raise FileExistsError(existing[0])
        if args.selected_request_output is not None:
            write_selected_provider_support_request(
                args.selected_request_output,
                result,
                overwrite=args.overwrite,
            )
        if args.selected_feasibility_output is not None:
            write_selected_provider_support_feasibility(
                args.selected_feasibility_output,
                result,
                overwrite=args.overwrite,
            )
        write_provider_support_design(args.output, result, overwrite=args.overwrite)
    else:
        result = load_provider_support_design(args.artifact)
    print(
        json.dumps(
            _summary(result),
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
        )
    )
    return 0 if result.support_design_feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROVIDER_SUPPORT_DESIGN_CLAIM_BOUNDARY",
    "PROVIDER_SUPPORT_DESIGN_REQUEST_SCHEMA",
    "PROVIDER_SUPPORT_DESIGN_REQUEST_VERSION",
    "PROVIDER_SUPPORT_DESIGN_SCHEMA",
    "PROVIDER_SUPPORT_DESIGN_SELECTION_RULE",
    "PROVIDER_SUPPORT_DESIGN_VERSION",
    "ProviderSupportDesignCandidateV1",
    "ProviderSupportDesignRequestV1",
    "ProviderSupportDesignV1",
    "evaluate_provider_support_design",
    "load_provider_support_design",
    "load_provider_support_design_request",
    "main",
    "write_provider_support_design",
    "write_provider_support_design_request",
    "write_selected_provider_support_feasibility",
    "write_selected_provider_support_request",
]
