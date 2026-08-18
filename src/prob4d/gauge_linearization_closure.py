"""Source-only closure diagnostic for first-order joint ``Sim(3)`` propagation.

The diagnostic compares analytic first-order propagation against a deterministic
nonlinear sigma-point reference and localizes gauge nonlinearity before a residual
failure is attributed to conditional point covariance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._atomic_file import atomic_write_text
from ._gauge_linearization_contract import (
    GAUGE_LINEARIZATION_CLOSURE_CLAIM_BOUNDARY,
    GAUGE_LINEARIZATION_CLOSURE_SCHEMA,
    GAUGE_LINEARIZATION_CLOSURE_VERSION,
    GAUGE_LINEARIZATION_EVIDENCE_PARTITION,
    GAUGE_LINEARIZATION_SIGMA_POINT_RULE,
    GaugeLinearizationCaseV1,
    GaugeLinearizationPolicyV1,
    JsonReport,
    _load_json,
    _probability,
)
from ._gauge_linearization_numerics import (
    _case_reports,
    _group_reports,
    _reports_as_plain,
)
from ._gauge_linearization_numerics import (
    evaluate_gauge_linearization_case as evaluate_gauge_linearization_case,
)
from ._gauge_linearization_numerics import (
    linearize_sim3_chain as linearize_sim3_chain,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_string,
)


@dataclass(frozen=True, slots=True)
class GaugeLinearizationClosureV1:
    """Content-addressed replayable source-only closure decision."""

    representation_name: str
    policy: GaugeLinearizationPolicyV1
    cases: tuple[GaugeLinearizationCaseV1, ...]
    reports: tuple[JsonReport, ...]
    group_reports: tuple[JsonReport, ...]
    evidence_partition: str = GAUGE_LINEARIZATION_EVIDENCE_PARTITION
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, GaugeLinearizationPolicyV1):
            raise ValueError("policy must be GaugeLinearizationPolicyV1")
        cases = tuple(self.cases)
        if not cases or not all(isinstance(case, GaugeLinearizationCaseV1) for case in cases):
            raise ValueError("cases must be a nonempty tuple of GaugeLinearizationCaseV1")
        case_ids = tuple(case.case_id for case in cases)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("cases must be sorted by unique case_id")
        replayed_reports = _case_reports(cases, self.policy)
        replayed_groups = _group_reports(replayed_reports)
        if _reports_as_plain(self.reports) != _reports_as_plain(replayed_reports):
            raise ValueError("case reports do not match deterministic replay")
        if _reports_as_plain(self.group_reports) != _reports_as_plain(replayed_groups):
            raise ValueError("group reports do not match deterministic replay")
        partition = _strict_string(self.evidence_partition, name="evidence_partition")
        if partition != GAUGE_LINEARIZATION_EVIDENCE_PARTITION:
            raise ValueError("gauge linearization evidence must be source-diagnostic")
        target_used = _strict_bool(self.target_outcomes_used, name="target_outcomes_used")
        if target_used:
            raise ValueError("gauge linearization closure forbids target outcomes")
        object.__setattr__(
            self,
            "representation_name",
            _strict_string(self.representation_name, name="representation_name"),
        )
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "reports", replayed_reports)
        object.__setattr__(self, "group_reports", replayed_groups)
        object.__setattr__(self, "evidence_partition", partition)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="gauge linearization closure metadata",
            ),
        )

    @property
    def group_count(self) -> int:
        return len(self.group_reports)

    @property
    def passing_group_count(self) -> int:
        return sum(bool(report["closure_passed"]) for report in self.group_reports)

    @property
    def group_pass_fraction(self) -> float:
        return self.passing_group_count / self.group_count

    @property
    def decision(self) -> str:
        if self.group_count < self.policy.minimum_group_count:
            return "insufficient-independent-groups"
        if self.group_pass_fraction >= self.policy.minimum_group_pass_fraction:
            return "linearization-closure-adequate"
        return "linearization-closure-negative"

    @property
    def passed(self) -> bool:
        return self.decision == "linearization-closure-adequate"

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": GAUGE_LINEARIZATION_CLOSURE_SCHEMA,
            "schema_version": GAUGE_LINEARIZATION_CLOSURE_VERSION,
            "sigma_point_rule": GAUGE_LINEARIZATION_SIGMA_POINT_RULE,
            "representation_name": self.representation_name,
            "policy": self.policy.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "reports": _reports_as_plain(self.reports),
            "group_reports": _reports_as_plain(self.group_reports),
            "group_count": self.group_count,
            "passing_group_count": self.passing_group_count,
            "group_pass_fraction": self.group_pass_fraction,
            "decision": self.decision,
            "passed": self.passed,
            "point_covariance_development_authorized": False,
            "evidence_partition": self.evidence_partition,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": GAUGE_LINEARIZATION_CLOSURE_CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_dict(cls, value: Any) -> GaugeLinearizationClosureV1:
        mapping = _strict_mapping(value, name="gauge linearization closure")
        expected = {
            "schema_name",
            "schema_version",
            "sigma_point_rule",
            "representation_name",
            "policy",
            "cases",
            "reports",
            "group_reports",
            "group_count",
            "passing_group_count",
            "group_pass_fraction",
            "decision",
            "passed",
            "point_covariance_development_authorized",
            "evidence_partition",
            "target_outcomes_used",
            "metadata",
            "claim_boundary",
            "artifact_id",
        }
        _exact_keys(mapping, expected, name="gauge linearization closure")
        if mapping["schema_name"] != GAUGE_LINEARIZATION_CLOSURE_SCHEMA:
            raise ValueError("unsupported gauge linearization closure schema")
        if mapping["schema_version"] != GAUGE_LINEARIZATION_CLOSURE_VERSION:
            raise ValueError("unsupported gauge linearization closure version")
        if mapping["sigma_point_rule"] != GAUGE_LINEARIZATION_SIGMA_POINT_RULE:
            raise ValueError("gauge linearization sigma-point rule changed")
        if mapping["claim_boundary"] != GAUGE_LINEARIZATION_CLOSURE_CLAIM_BOUNDARY:
            raise ValueError("gauge linearization claim boundary changed")
        if _strict_bool(
            mapping["point_covariance_development_authorized"],
            name="point_covariance_development_authorized",
        ):
            raise ValueError("linearization closure cannot authorize point covariance")
        reports = tuple(
            frozen_finite_json_mapping(
                _strict_mapping(report, name="case report"),
                name="gauge linearization case report",
            )
            for report in _strict_list(mapping["reports"], name="reports")
        )
        group_reports = tuple(
            frozen_finite_json_mapping(
                _strict_mapping(report, name="group report"),
                name="gauge linearization group report",
            )
            for report in _strict_list(mapping["group_reports"], name="group_reports")
        )
        artifact = cls(
            representation_name=mapping["representation_name"],
            policy=GaugeLinearizationPolicyV1.from_dict(mapping["policy"]),
            cases=tuple(
                GaugeLinearizationCaseV1.from_dict(case)
                for case in _strict_list(mapping["cases"], name="cases")
            ),
            reports=reports,
            group_reports=group_reports,
            evidence_partition=mapping["evidence_partition"],
            target_outcomes_used=mapping["target_outcomes_used"],
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )
        if _strict_integer(mapping["group_count"], name="group_count") != artifact.group_count:
            raise ValueError("group_count does not match deterministic replay")
        if _strict_integer(
            mapping["passing_group_count"],
            name="passing_group_count",
        ) != artifact.passing_group_count:
            raise ValueError("passing_group_count does not match deterministic replay")
        if _strict_string(mapping["decision"], name="decision") != artifact.decision:
            raise ValueError("decision does not match deterministic replay")
        if _strict_bool(mapping["passed"], name="passed") is not artifact.passed:
            raise ValueError("passed does not match deterministic replay")
        if not math.isclose(
            _probability(mapping["group_pass_fraction"], name="group_pass_fraction"),
            artifact.group_pass_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("group_pass_fraction does not match deterministic replay")
        supplied_id = _strict_digest(
            mapping["artifact_id"],
            name="artifact_id",
            pattern=_SHA256,
        )
        if supplied_id != artifact.artifact_id:
            raise ValueError("gauge linearization closure artifact_id mismatch")
        return artifact


def build_gauge_linearization_closure(
    *,
    representation_name: str,
    policy: GaugeLinearizationPolicyV1,
    cases: Sequence[GaugeLinearizationCaseV1],
    evidence_partition: str = GAUGE_LINEARIZATION_EVIDENCE_PARTITION,
    target_outcomes_used: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> GaugeLinearizationClosureV1:
    supplied = tuple(cases)
    if not supplied or not all(isinstance(case, GaugeLinearizationCaseV1) for case in supplied):
        raise ValueError("cases must be a nonempty sequence of GaugeLinearizationCaseV1")
    ordered = tuple(sorted(supplied, key=lambda case: case.case_id))
    reports = _case_reports(ordered, policy)
    return GaugeLinearizationClosureV1(
        representation_name=representation_name,
        policy=policy,
        cases=ordered,
        reports=reports,
        group_reports=_group_reports(reports),
        evidence_partition=evidence_partition,
        target_outcomes_used=target_outcomes_used,
        metadata={} if metadata is None else metadata,
    )


def gauge_linearization_closure_from_raw(value: Any) -> GaugeLinearizationClosureV1:
    mapping = _strict_mapping(value, name="raw gauge linearization closure")
    _exact_keys(
        mapping,
        {
            "representation_name",
            "policy",
            "cases",
            "evidence_partition",
            "target_outcomes_used",
            "metadata",
        },
        name="raw gauge linearization closure",
    )
    return build_gauge_linearization_closure(
        representation_name=mapping["representation_name"],
        policy=GaugeLinearizationPolicyV1.from_dict(mapping["policy"]),
        cases=tuple(
            GaugeLinearizationCaseV1.from_dict(case)
            for case in _strict_list(mapping["cases"], name="cases")
        ),
        evidence_partition=mapping["evidence_partition"],
        target_outcomes_used=mapping["target_outcomes_used"],
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )


def write_gauge_linearization_closure(
    artifact: GaugeLinearizationClosureV1,
    path: str | Path,
) -> None:
    if not isinstance(artifact, GaugeLinearizationClosureV1):
        raise ValueError("artifact must be GaugeLinearizationClosureV1")
    destination = Path(path)
    payload = json.dumps(
        artifact.to_dict(),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    try:
        atomic_write_text(destination, payload, overwrite=False)
    except FileExistsError:
        if load_gauge_linearization_closure(destination).to_dict() == artifact.to_dict():
            return
        raise FileExistsError(
            f"refusing to replace a different gauge linearization closure: {destination}"
        ) from None


def load_gauge_linearization_closure(path: str | Path) -> GaugeLinearizationClosureV1:
    return GaugeLinearizationClosureV1.from_dict(
        _load_json(path, name="gauge linearization closure")
    )


def _build_cli(arguments: argparse.Namespace) -> int:
    artifact = gauge_linearization_closure_from_raw(
        _load_json(arguments.input, name="raw gauge linearization closure")
    )
    write_gauge_linearization_closure(artifact, arguments.output)
    print(artifact.artifact_id)
    return 0 if artifact.passed or not arguments.require_pass else 3


def _verify_cli(arguments: argparse.Namespace) -> int:
    artifact = load_gauge_linearization_closure(arguments.artifact)
    print(artifact.artifact_id)
    return 0 if artifact.passed or not arguments.require_pass else 3


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify the source-only joint Sim(3) closure diagnostic."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the closure diagnostic")
    build.add_argument("input", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--require-pass", action="store_true")
    build.set_defaults(handler=_build_cli)
    verify = subparsers.add_parser("verify", help="verify and replay a closure artifact")
    verify.add_argument("artifact", type=Path)
    verify.add_argument("--require-pass", action="store_true")
    verify.set_defaults(handler=_verify_cli)
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


__all__ = [
    "GAUGE_LINEARIZATION_CLOSURE_CLAIM_BOUNDARY",
    "GAUGE_LINEARIZATION_CLOSURE_SCHEMA",
    "GAUGE_LINEARIZATION_CLOSURE_VERSION",
    "GAUGE_LINEARIZATION_EVIDENCE_PARTITION",
    "GAUGE_LINEARIZATION_SIGMA_POINT_RULE",
    "GaugeLinearizationCaseV1",
    "GaugeLinearizationClosureV1",
    "GaugeLinearizationPolicyV1",
    "build_gauge_linearization_closure",
    "evaluate_gauge_linearization_case",
    "gauge_linearization_closure_from_raw",
    "linearize_sim3_chain",
    "load_gauge_linearization_closure",
    "main",
    "write_gauge_linearization_closure",
]


if __name__ == "__main__":
    raise SystemExit(main())
