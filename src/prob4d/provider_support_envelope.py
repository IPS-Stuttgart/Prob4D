"""Outcome-blind contiguous support envelopes for frozen provider streams.

The existing provider-support feasibility contract answers whether one exact set
of required frames is supported.  This module derives a reusable per-stream
frame envelope from the same pre-residual metadata.  The envelope can be used to
freeze a *new* support request or support-design candidate before prediction
payloads or residuals are opened; it never mutates the request it was derived
from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)
from .provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    evaluate_provider_support_feasibility,
    load_provider_support_feasibility_request,
)

PROVIDER_SUPPORT_ENVELOPE_SCHEMA = "prob4d.provider-support-envelope"
PROVIDER_SUPPORT_ENVELOPE_VERSION = 1
PROVIDER_SUPPORT_ENVELOPE_CLAIM_BOUNDARY = (
    "This artifact derives contiguous frame-support envelopes only from an exact "
    "outcome-blind ProviderSupportFeasibilityRequestV1. It does not change that "
    "request, select a causal prefix after seeing provider residuals, establish "
    "provider competence or calibration, authorize a BayesianPhysTwin update, "
    "establish Causal4D benefit, deployment safety, or state of the art."
)

_STREAM_FIELDS = frozenset(
    {
        "group_id",
        "stream_id",
        "causal_frame_start",
        "causal_frame_stop_exclusive",
        "admissible_intervals",
        "admissible_frame_count",
        "earliest_admissible_frame",
        "latest_admissible_frame_exclusive",
        "maximum_contiguous_frame_count",
        "required_frame_count",
        "required_admissible_frame_count",
        "required_support_fraction",
        "static_support_complete",
        "technical_failure_code",
        "feasibility_supported",
        "excluded_from_admission",
        "feasibility_reason_codes",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "request",
        "streams",
        "stream_count",
        "stream_with_nonempty_envelope_count",
        "total_admissible_frame_count",
        "metadata",
        "claim_boundary",
        "provider_support_envelope_id",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _optional_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return require_exact_integer(value, name=name, minimum=0)


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _string_tuple_from_json(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return _string_tuple(tuple(value), name=name)


def _intervals(frames: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ordered = tuple(sorted(set(int(frame) for frame in frames)))
    if not ordered:
        return ()
    result: list[tuple[int, int]] = []
    start = ordered[0]
    previous = start
    for frame in ordered[1:]:
        if frame != previous + 1:
            result.append((start, previous + 1))
            start = frame
        previous = frame
    result.append((start, previous + 1))
    return tuple(result)


def _interval_tuple(value: object, *, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result: list[tuple[int, int]] = []
    previous_stop: int | None = None
    for index, interval in enumerate(value):
        if type(interval) is not tuple or len(interval) != 2:
            raise ValueError(f"{name}[{index}] must be a (start, stop) tuple")
        start = require_exact_integer(interval[0], name=f"{name}[{index}][0]", minimum=0)
        stop = require_exact_integer(interval[1], name=f"{name}[{index}][1]", minimum=1)
        if stop <= start:
            raise ValueError(f"{name}[{index}] stop must exceed start")
        if previous_stop is not None and start <= previous_stop:
            raise ValueError(f"{name} intervals must be strictly separated")
        result.append((start, stop))
        previous_stop = stop
    return tuple(result)


def _interval_tuple_from_json(value: object, *, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    tuples: list[tuple[int, int]] = []
    for index, interval in enumerate(value):
        if type(interval) is not list or len(interval) != 2:
            raise ValueError(f"{name}[{index}] must be a two-element JSON array")
        tuples.append((interval[0], interval[1]))  # type: ignore[arg-type]
    return _interval_tuple(tuple(tuples), name=name)


@dataclass(frozen=True, slots=True)
class ProviderSupportStreamEnvelopeV1:
    """Derived contiguous support summary for one frozen stream."""

    group_id: str
    stream_id: str
    causal_frame_start: int
    causal_frame_stop_exclusive: int
    admissible_intervals: tuple[tuple[int, int], ...]
    admissible_frame_count: int
    earliest_admissible_frame: int | None
    latest_admissible_frame_exclusive: int | None
    maximum_contiguous_frame_count: int
    required_frame_count: int
    required_admissible_frame_count: int
    required_support_fraction: float
    static_support_complete: bool
    technical_failure_code: str | None
    feasibility_supported: bool
    excluded_from_admission: bool
    feasibility_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        group_id = require_exact_string(self.group_id, name="group_id")
        stream_id = require_exact_string(self.stream_id, name="stream_id")
        start = require_exact_integer(
            self.causal_frame_start,
            name="causal_frame_start",
            minimum=0,
        )
        stop = require_exact_integer(
            self.causal_frame_stop_exclusive,
            name="causal_frame_stop_exclusive",
            minimum=1,
        )
        if stop <= start:
            raise ValueError("causal_frame_stop_exclusive must exceed causal_frame_start")
        intervals = _interval_tuple(self.admissible_intervals, name="admissible_intervals")
        if any(
            interval_start < start or interval_stop > stop
            for interval_start, interval_stop in intervals
        ):
            raise ValueError("admissible_intervals cross the frozen causal span")
        frame_count = require_exact_integer(
            self.admissible_frame_count,
            name="admissible_frame_count",
            minimum=0,
        )
        expected_count = sum(
            interval_stop - interval_start
            for interval_start, interval_stop in intervals
        )
        if frame_count != expected_count:
            raise ValueError("admissible_frame_count is inconsistent")
        earliest = _optional_integer(
            self.earliest_admissible_frame,
            name="earliest_admissible_frame",
        )
        latest = _optional_integer(
            self.latest_admissible_frame_exclusive,
            name="latest_admissible_frame_exclusive",
        )
        if intervals:
            if earliest != intervals[0][0] or latest != intervals[-1][1]:
                raise ValueError("admissible frame bounds are inconsistent")
        elif earliest is not None or latest is not None:
            raise ValueError("empty envelopes require null admissible frame bounds")
        maximum = require_exact_integer(
            self.maximum_contiguous_frame_count,
            name="maximum_contiguous_frame_count",
            minimum=0,
        )
        expected_maximum = max(
            (interval_stop - interval_start for interval_start, interval_stop in intervals),
            default=0,
        )
        if maximum != expected_maximum:
            raise ValueError("maximum_contiguous_frame_count is inconsistent")
        required = require_exact_integer(
            self.required_frame_count,
            name="required_frame_count",
            minimum=1,
        )
        required_admissible = require_exact_integer(
            self.required_admissible_frame_count,
            name="required_admissible_frame_count",
            minimum=0,
        )
        if required_admissible > required:
            raise ValueError("required_admissible_frame_count exceeds required_frame_count")
        fraction = float(self.required_support_fraction)
        if not 0.0 <= fraction <= 1.0 or abs(fraction - required_admissible / required) > 1.0e-15:
            raise ValueError("required_support_fraction is inconsistent")
        static_complete = _strict_bool(self.static_support_complete, name="static_support_complete")
        technical_failure = self.technical_failure_code
        if technical_failure is not None:
            technical_failure = require_exact_string(
                technical_failure,
                name="technical_failure_code",
            )
        feasibility_supported = _strict_bool(
            self.feasibility_supported,
            name="feasibility_supported",
        )
        excluded = _strict_bool(self.excluded_from_admission, name="excluded_from_admission")
        reasons = _string_tuple(self.feasibility_reason_codes, name="feasibility_reason_codes")

        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "causal_frame_start", start)
        object.__setattr__(self, "causal_frame_stop_exclusive", stop)
        object.__setattr__(self, "admissible_intervals", intervals)
        object.__setattr__(self, "admissible_frame_count", frame_count)
        object.__setattr__(self, "earliest_admissible_frame", earliest)
        object.__setattr__(self, "latest_admissible_frame_exclusive", latest)
        object.__setattr__(self, "maximum_contiguous_frame_count", maximum)
        object.__setattr__(self, "required_frame_count", required)
        object.__setattr__(self, "required_admissible_frame_count", required_admissible)
        object.__setattr__(self, "required_support_fraction", fraction)
        object.__setattr__(self, "static_support_complete", static_complete)
        object.__setattr__(self, "technical_failure_code", technical_failure)
        object.__setattr__(self, "feasibility_supported", feasibility_supported)
        object.__setattr__(self, "excluded_from_admission", excluded)
        object.__setattr__(self, "feasibility_reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "stream_id": self.stream_id,
            "causal_frame_start": self.causal_frame_start,
            "causal_frame_stop_exclusive": self.causal_frame_stop_exclusive,
            "admissible_intervals": [list(interval) for interval in self.admissible_intervals],
            "admissible_frame_count": self.admissible_frame_count,
            "earliest_admissible_frame": self.earliest_admissible_frame,
            "latest_admissible_frame_exclusive": self.latest_admissible_frame_exclusive,
            "maximum_contiguous_frame_count": self.maximum_contiguous_frame_count,
            "required_frame_count": self.required_frame_count,
            "required_admissible_frame_count": self.required_admissible_frame_count,
            "required_support_fraction": self.required_support_fraction,
            "static_support_complete": self.static_support_complete,
            "technical_failure_code": self.technical_failure_code,
            "feasibility_supported": self.feasibility_supported,
            "excluded_from_admission": self.excluded_from_admission,
            "feasibility_reason_codes": list(self.feasibility_reason_codes),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportStreamEnvelopeV1:
        mapping = require_mapping(value, name="provider support stream envelope")
        require_exact_fields(mapping, _STREAM_FIELDS, name="provider support stream envelope")
        return cls(
            group_id=mapping["group_id"],
            stream_id=mapping["stream_id"],
            causal_frame_start=mapping["causal_frame_start"],
            causal_frame_stop_exclusive=mapping["causal_frame_stop_exclusive"],
            admissible_intervals=_interval_tuple_from_json(
                mapping["admissible_intervals"],
                name="admissible_intervals",
            ),
            admissible_frame_count=mapping["admissible_frame_count"],
            earliest_admissible_frame=mapping["earliest_admissible_frame"],
            latest_admissible_frame_exclusive=mapping[
                "latest_admissible_frame_exclusive"
            ],
            maximum_contiguous_frame_count=mapping["maximum_contiguous_frame_count"],
            required_frame_count=mapping["required_frame_count"],
            required_admissible_frame_count=mapping["required_admissible_frame_count"],
            required_support_fraction=mapping["required_support_fraction"],
            static_support_complete=mapping["static_support_complete"],
            technical_failure_code=mapping["technical_failure_code"],
            feasibility_supported=mapping["feasibility_supported"],
            excluded_from_admission=mapping["excluded_from_admission"],
            feasibility_reason_codes=_string_tuple_from_json(
                mapping["feasibility_reason_codes"],
                name="feasibility_reason_codes",
            ),
        )


def _derive_streams(
    request: ProviderSupportFeasibilityRequestV1,
) -> tuple[ProviderSupportStreamEnvelopeV1, ...]:
    feasibility = evaluate_provider_support_feasibility(request)
    evaluation_by_key = {
        (item.group_id, item.stream_id): item for item in feasibility.stream_results
    }
    result: list[ProviderSupportStreamEnvelopeV1] = []
    for stream in request.streams:
        evaluation = evaluation_by_key[stream.key]
        static_complete = (
            (not stream.intrinsics_required or stream.intrinsics_id is not None)
            and (not stream.extrinsics_required or stream.extrinsics_id is not None)
            and (not stream.metric_anchor_required or stream.metric_anchor_id is not None)
        )
        admissible = set(stream.available_frame_ids).intersection(
            stream.geometry_supported_frame_ids
        )
        if not static_complete or stream.technical_failure_code is not None:
            admissible.clear()
        intervals = _intervals(tuple(admissible))
        required_admissible = len(set(stream.required_frame_ids).intersection(admissible))
        result.append(
            ProviderSupportStreamEnvelopeV1(
                group_id=stream.group_id,
                stream_id=stream.stream_id,
                causal_frame_start=stream.causal_frame_start,
                causal_frame_stop_exclusive=stream.causal_frame_stop_exclusive,
                admissible_intervals=intervals,
                admissible_frame_count=len(admissible),
                earliest_admissible_frame=None if not intervals else intervals[0][0],
                latest_admissible_frame_exclusive=None if not intervals else intervals[-1][1],
                maximum_contiguous_frame_count=max(
                    (interval_stop - interval_start for interval_start, interval_stop in intervals),
                    default=0,
                ),
                required_frame_count=len(stream.required_frame_ids),
                required_admissible_frame_count=required_admissible,
                required_support_fraction=required_admissible / len(stream.required_frame_ids),
                static_support_complete=static_complete,
                technical_failure_code=stream.technical_failure_code,
                feasibility_supported=evaluation.supported,
                excluded_from_admission=evaluation.excluded_from_admission,
                feasibility_reason_codes=evaluation.reason_codes,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ProviderSupportEnvelopeV1:
    """Replayable support envelope derived from one exact feasibility request."""

    request: ProviderSupportFeasibilityRequestV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    streams: tuple[ProviderSupportStreamEnvelopeV1, ...] = field(init=False)
    stream_count: int = field(init=False)
    stream_with_nonempty_envelope_count: int = field(init=False)
    total_admissible_frame_count: int = field(init=False)
    provider_support_envelope_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProviderSupportFeasibilityRequestV1):
            raise TypeError("request must be ProviderSupportFeasibilityRequestV1")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )
        streams = _derive_streams(self.request)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "stream_count", len(streams))
        object.__setattr__(
            self,
            "stream_with_nonempty_envelope_count",
            sum(item.admissible_frame_count > 0 for item in streams),
        )
        object.__setattr__(
            self,
            "total_admissible_frame_count",
            sum(item.admissible_frame_count for item in streams),
        )
        object.__setattr__(
            self,
            "provider_support_envelope_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_SUPPORT_ENVELOPE_SCHEMA,
            "schema_version": PROVIDER_SUPPORT_ENVELOPE_VERSION,
            "request": self.request.to_dict(),
            "streams": [item.to_dict() for item in self.streams],
            "stream_count": self.stream_count,
            "stream_with_nonempty_envelope_count": self.stream_with_nonempty_envelope_count,
            "total_admissible_frame_count": self.total_admissible_frame_count,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_SUPPORT_ENVELOPE_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "provider_support_envelope_id": self.provider_support_envelope_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportEnvelopeV1:
        mapping = require_mapping(value, name="provider support envelope")
        require_exact_fields(mapping, _ARTIFACT_FIELDS, name="provider support envelope")
        if mapping["schema"] != PROVIDER_SUPPORT_ENVELOPE_SCHEMA:
            raise ValueError("provider support envelope schema changed")
        if mapping["schema_version"] != PROVIDER_SUPPORT_ENVELOPE_VERSION:
            raise ValueError("provider support envelope version changed")
        if mapping["claim_boundary"] != PROVIDER_SUPPORT_ENVELOPE_CLAIM_BOUNDARY:
            raise ValueError("provider support envelope claim boundary changed")
        result = cls(
            request=ProviderSupportFeasibilityRequestV1.from_dict(mapping["request"]),
            metadata=require_finite_json_mapping(mapping["metadata"], name="metadata"),
        )
        supplied_id = require_sha256(
            mapping["provider_support_envelope_id"],
            name="provider_support_envelope_id",
        )
        if supplied_id != result.provider_support_envelope_id:
            raise ValueError("provider support envelope identity mismatch")
        if plain_json(mapping) != result.to_dict():
            raise ValueError("provider support envelope derived fields changed")
        return result


def derive_provider_support_envelope(
    request: ProviderSupportFeasibilityRequestV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ProviderSupportEnvelopeV1:
    return ProviderSupportEnvelopeV1(
        request=request,
        metadata={} if metadata is None else metadata,
    )


def write_provider_support_envelope(
    path: str | Path,
    envelope: ProviderSupportEnvelopeV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(envelope, ProviderSupportEnvelopeV1):
        raise TypeError("envelope must be ProviderSupportEnvelopeV1")
    payload = json.dumps(envelope.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_provider_support_envelope(path: str | Path) -> ProviderSupportEnvelopeV1:
    return ProviderSupportEnvelopeV1.from_dict(
        load_json_object(path, name="provider support envelope")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    derive = subparsers.add_parser("derive")
    derive.add_argument("--request", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    derive.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "derive":
        request = load_provider_support_feasibility_request(arguments.request)
        envelope = derive_provider_support_envelope(request)
        write_provider_support_envelope(
            arguments.output,
            envelope,
            overwrite=arguments.overwrite,
        )
    else:
        envelope = load_provider_support_envelope(arguments.artifact)
    print(
        json.dumps(
            {
                "provider_support_envelope_id": envelope.provider_support_envelope_id,
                "stream_count": envelope.stream_count,
                "stream_with_nonempty_envelope_count": (
                    envelope.stream_with_nonempty_envelope_count
                ),
                "total_admissible_frame_count": envelope.total_admissible_frame_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PROVIDER_SUPPORT_ENVELOPE_CLAIM_BOUNDARY",
    "PROVIDER_SUPPORT_ENVELOPE_SCHEMA",
    "PROVIDER_SUPPORT_ENVELOPE_VERSION",
    "ProviderSupportEnvelopeV1",
    "ProviderSupportStreamEnvelopeV1",
    "derive_provider_support_envelope",
    "load_provider_support_envelope",
    "write_provider_support_envelope",
]
