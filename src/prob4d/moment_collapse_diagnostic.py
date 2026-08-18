"""Controlled diagnostic for Gaussian moment collapse of multimodal 3-D evidence.

The current observation-factor contracts preserve means, covariance, gauges, and
source dependence.  A genuinely bimodal point likelihood can nevertheless be
made qualitatively different when it is replaced by its moment-matched Gaussian.
This module quantifies that mechanism for symmetric Gaussian mixtures without
claiming that any real provider exhibits it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_text
from ._immutable_array import immutable_array
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
    _strict_real,
    _strict_string,
)

FloatArray: TypeAlias = NDArray[np.float64]

MOMENT_COLLAPSE_DIAGNOSTIC_SCHEMA = "prob4d.gaussian-moment-collapse-diagnostic"
MOMENT_COLLAPSE_DIAGNOSTIC_VERSION = 1
MOMENT_COLLAPSE_DIAGNOSTIC_CLAIM_BOUNDARY = (
    "This artifact is controlled mechanism evidence for a symmetric Gaussian-mixture "
    "replacement by its moment-matched Gaussian. It does not establish that a real "
    "provider is multimodal, that the current Prob4D contract loses material evidence, "
    "or that BayesianPhysTwin or Causal4D performance improves."
)


def _positive_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _nonnegative_real(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: str | Path, *, name: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable or invalid JSON: {source}") from error
    mapping = _strict_mapping(value, name=name)
    if any(type(key) is not str for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    return mapping


@dataclass(frozen=True, slots=True)
class SymmetricGaussianMixtureCaseV1:
    """One 50/50 mixture with component means ``-offset`` and ``+offset``."""

    case_id: str
    offset_xyz: FloatArray
    component_covariance: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = _strict_string(self.case_id, name="case_id")
        offset = np.asarray(self.offset_xyz, dtype=np.float64)
        covariance = np.asarray(self.component_covariance, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError("offset_xyz must have shape (3,)")
        if covariance.shape != (3, 3):
            raise ValueError("component_covariance must have shape (3, 3)")
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(covariance)):
            raise ValueError("mixture case arrays must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
            raise ValueError("component_covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("component_covariance must be positive definite") from error
        if not np.any(offset != 0.0):
            raise ValueError("offset_xyz must define two distinct component means")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "offset_xyz", immutable_array(offset, dtype=np.float64))
        object.__setattr__(
            self,
            "component_covariance",
            immutable_array(covariance, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="moment collapse case metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "offset_xyz": self.offset_xyz.tolist(),
            "component_covariance": self.component_covariance.tolist(),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> SymmetricGaussianMixtureCaseV1:
        mapping = _strict_mapping(value, name="moment collapse case")
        _exact_keys(
            mapping,
            {"case_id", "offset_xyz", "component_covariance", "metadata"},
            name="moment collapse case",
        )
        offset_items = _strict_list(mapping["offset_xyz"], name="offset_xyz")
        offset = [
            _strict_real(item, name=f"offset_xyz[{index}]")
            for index, item in enumerate(offset_items)
        ]
        covariance_rows = _strict_list(
            mapping["component_covariance"],
            name="component_covariance",
        )
        covariance = [
            [
                _strict_real(
                    item,
                    name=f"component_covariance[{row_index}][{column_index}]",
                )
                for column_index, item in enumerate(
                    _strict_list(
                        row,
                        name=f"component_covariance[{row_index}]",
                    )
                )
            ]
            for row_index, row in enumerate(covariance_rows)
        ]
        return cls(
            case_id=mapping["case_id"],
            offset_xyz=np.asarray(offset, dtype=np.float64),
            component_covariance=np.asarray(covariance, dtype=np.float64),
            metadata=_strict_mapping(mapping["metadata"], name="case metadata"),
        )


@dataclass(frozen=True, slots=True)
class MomentCollapseThresholdsV1:
    """Frozen criteria for a material moment-collapse mechanism."""

    midpoint_half_width_component_sigma: float
    minimum_component_mean_separation_sigma: float
    maximum_mixture_midpoint_to_component_mean_density_ratio: float
    minimum_moment_gaussian_central_mass_inflation: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "midpoint_half_width_component_sigma",
            _positive_real(
                self.midpoint_half_width_component_sigma,
                name="midpoint_half_width_component_sigma",
            ),
        )
        object.__setattr__(
            self,
            "minimum_component_mean_separation_sigma",
            _positive_real(
                self.minimum_component_mean_separation_sigma,
                name="minimum_component_mean_separation_sigma",
            ),
        )
        object.__setattr__(
            self,
            "maximum_mixture_midpoint_to_component_mean_density_ratio",
            _nonnegative_real(
                self.maximum_mixture_midpoint_to_component_mean_density_ratio,
                name="maximum_mixture_midpoint_to_component_mean_density_ratio",
            ),
        )
        object.__setattr__(
            self,
            "minimum_moment_gaussian_central_mass_inflation",
            _probability(
                self.minimum_moment_gaussian_central_mass_inflation,
                name="minimum_moment_gaussian_central_mass_inflation",
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "midpoint_half_width_component_sigma": (
                self.midpoint_half_width_component_sigma
            ),
            "minimum_component_mean_separation_sigma": (
                self.minimum_component_mean_separation_sigma
            ),
            "maximum_mixture_midpoint_to_component_mean_density_ratio": (
                self.maximum_mixture_midpoint_to_component_mean_density_ratio
            ),
            "minimum_moment_gaussian_central_mass_inflation": (
                self.minimum_moment_gaussian_central_mass_inflation
            ),
        }

    @classmethod
    def from_dict(cls, value: Any) -> MomentCollapseThresholdsV1:
        mapping = _strict_mapping(value, name="moment collapse thresholds")
        _exact_keys(mapping, set(cls.__dataclass_fields__), name="moment collapse thresholds")
        return cls(**mapping)


@dataclass(frozen=True, slots=True)
class MomentCollapseCaseReportV1:
    """Analytic one-dimensional discriminant diagnostics for one case."""

    case_id: str
    mahalanobis_offset_squared: float
    component_mean_separation_sigma: float
    mixture_midpoint_to_component_mean_density_ratio: float
    mixture_central_mass: float
    moment_gaussian_central_mass: float
    moment_gaussian_central_mass_inflation: float
    moment_matched_excess_kurtosis: float
    material_moment_collapse: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _strict_string(self.case_id, name="case_id"))
        for field_name in (
            "mahalanobis_offset_squared",
            "component_mean_separation_sigma",
        ):
            value = _strict_real(getattr(self, field_name), name=field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        density_ratio = _strict_real(
            self.mixture_midpoint_to_component_mean_density_ratio,
            name="mixture_midpoint_to_component_mean_density_ratio",
        )
        if density_ratio < 0.0:
            raise ValueError(
                "mixture_midpoint_to_component_mean_density_ratio must be non-negative"
            )
        object.__setattr__(
            self,
            "mixture_midpoint_to_component_mean_density_ratio",
            density_ratio,
        )
        for field_name in (
            "mixture_central_mass",
            "moment_gaussian_central_mass",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(getattr(self, field_name), name=field_name),
            )
        inflation = _strict_real(
            self.moment_gaussian_central_mass_inflation,
            name="moment_gaussian_central_mass_inflation",
        )
        if inflation < -1.0 or inflation > 1.0:
            raise ValueError("central-mass inflation must lie in [-1, 1]")
        object.__setattr__(
            self,
            "moment_gaussian_central_mass_inflation",
            inflation,
        )
        kurtosis = _strict_real(
            self.moment_matched_excess_kurtosis,
            name="moment_matched_excess_kurtosis",
        )
        if kurtosis < -2.0 - 1e-12 or kurtosis > 0.0 + 1e-12:
            raise ValueError("symmetric-mixture excess kurtosis must lie in [-2, 0]")
        object.__setattr__(self, "moment_matched_excess_kurtosis", kurtosis)
        object.__setattr__(
            self,
            "material_moment_collapse",
            _strict_bool(
                self.material_moment_collapse,
                name="material_moment_collapse",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Any) -> MomentCollapseCaseReportV1:
        mapping = _strict_mapping(value, name="moment collapse case report")
        _exact_keys(mapping, set(cls.__dataclass_fields__), name="moment collapse case report")
        return cls(**mapping)


def moment_matched_gaussian(
    case: SymmetricGaussianMixtureCaseV1,
) -> tuple[FloatArray, FloatArray]:
    """Return the zero mean and covariance matching the mixture's first two moments."""

    if not isinstance(case, SymmetricGaussianMixtureCaseV1):
        raise ValueError("case must be SymmetricGaussianMixtureCaseV1")
    mean = immutable_array(np.zeros(3, dtype=np.float64), dtype=np.float64)
    covariance = case.component_covariance + np.outer(case.offset_xyz, case.offset_xyz)
    return mean, immutable_array(covariance, dtype=np.float64)


def evaluate_moment_collapse_case(
    case: SymmetricGaussianMixtureCaseV1,
    thresholds: MomentCollapseThresholdsV1,
) -> MomentCollapseCaseReportV1:
    """Compute analytic Fisher-axis diagnostics for one symmetric mixture."""

    if not isinstance(case, SymmetricGaussianMixtureCaseV1):
        raise ValueError("case must be SymmetricGaussianMixtureCaseV1")
    if not isinstance(thresholds, MomentCollapseThresholdsV1):
        raise ValueError("thresholds must be MomentCollapseThresholdsV1")
    solved = np.linalg.solve(case.component_covariance, case.offset_xyz)
    mahalanobis_squared = float(case.offset_xyz @ solved)
    offset_sigma = math.sqrt(max(0.0, mahalanobis_squared))
    separation = 2.0 * offset_sigma
    midpoint_ratio = (
        2.0 * math.exp(-0.5 * mahalanobis_squared)
        / (1.0 + math.exp(-2.0 * mahalanobis_squared))
    )
    half_width = thresholds.midpoint_half_width_component_sigma
    mixture_mass = _standard_normal_cdf(half_width - offset_sigma) - _standard_normal_cdf(
        -half_width - offset_sigma
    )
    matched_standard_deviation = math.sqrt(1.0 + mahalanobis_squared)
    gaussian_mass = 2.0 * _standard_normal_cdf(
        half_width / matched_standard_deviation
    ) - 1.0
    inflation = gaussian_mass - mixture_mass
    excess_kurtosis = (
        -2.0
        * mahalanobis_squared**2
        / (1.0 + mahalanobis_squared) ** 2
    )
    material = (
        separation >= thresholds.minimum_component_mean_separation_sigma
        and midpoint_ratio
        <= thresholds.maximum_mixture_midpoint_to_component_mean_density_ratio
        and inflation
        >= thresholds.minimum_moment_gaussian_central_mass_inflation
    )
    return MomentCollapseCaseReportV1(
        case_id=case.case_id,
        mahalanobis_offset_squared=mahalanobis_squared,
        component_mean_separation_sigma=separation,
        mixture_midpoint_to_component_mean_density_ratio=midpoint_ratio,
        mixture_central_mass=mixture_mass,
        moment_gaussian_central_mass=gaussian_mass,
        moment_gaussian_central_mass_inflation=inflation,
        moment_matched_excess_kurtosis=excess_kurtosis,
        material_moment_collapse=material,
    )


@dataclass(frozen=True, slots=True)
class MomentCollapseDiagnosticV1:
    """Content-addressed controlled mechanism study over canonical cases."""

    representation_name: str
    thresholds: MomentCollapseThresholdsV1
    cases: tuple[SymmetricGaussianMixtureCaseV1, ...]
    reports: tuple[MomentCollapseCaseReportV1, ...]
    material_case_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        representation = _strict_string(
            self.representation_name,
            name="representation_name",
        )
        if not isinstance(self.thresholds, MomentCollapseThresholdsV1):
            raise ValueError("thresholds must be MomentCollapseThresholdsV1")
        if type(self.cases) is not tuple or not self.cases or not all(
            isinstance(case, SymmetricGaussianMixtureCaseV1) for case in self.cases
        ):
            raise ValueError("cases must be a nonempty tuple of mixture cases")
        case_ids = tuple(case.case_id for case in self.cases)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("cases must be sorted by unique case_id")
        if type(self.reports) is not tuple or not all(
            isinstance(report, MomentCollapseCaseReportV1) for report in self.reports
        ):
            raise ValueError("reports must be a tuple of case reports")
        replayed = tuple(
            evaluate_moment_collapse_case(case, self.thresholds)
            for case in self.cases
        )
        if [report.to_dict() for report in self.reports] != [
            report.to_dict() for report in replayed
        ]:
            raise ValueError("moment collapse reports do not match deterministic replay")
        material_count = _strict_integer(
            self.material_case_count,
            name="material_case_count",
            minimum=0,
        )
        expected_count = sum(report.material_moment_collapse for report in replayed)
        if material_count != expected_count:
            raise ValueError("material_case_count does not match replayed reports")
        object.__setattr__(self, "representation_name", representation)
        object.__setattr__(self, "reports", replayed)
        object.__setattr__(self, "material_case_count", material_count)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="moment collapse metadata"),
        )

    @property
    def any_material_moment_collapse(self) -> bool:
        return self.material_case_count > 0

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": MOMENT_COLLAPSE_DIAGNOSTIC_SCHEMA,
            "schema_version": MOMENT_COLLAPSE_DIAGNOSTIC_VERSION,
            "representation_name": self.representation_name,
            "thresholds": self.thresholds.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "reports": [report.to_dict() for report in self.reports],
            "material_case_count": self.material_case_count,
            "any_material_moment_collapse": self.any_material_moment_collapse,
            "metadata": plain_json(self.metadata),
            "claim_boundary": MOMENT_COLLAPSE_DIAGNOSTIC_CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_dict(cls, value: Any) -> MomentCollapseDiagnosticV1:
        mapping = _strict_mapping(value, name="moment collapse diagnostic")
        _exact_keys(
            mapping,
            {
                "schema_name",
                "schema_version",
                "representation_name",
                "thresholds",
                "cases",
                "reports",
                "material_case_count",
                "any_material_moment_collapse",
                "metadata",
                "claim_boundary",
                "artifact_id",
            },
            name="moment collapse diagnostic",
        )
        if mapping["schema_name"] != MOMENT_COLLAPSE_DIAGNOSTIC_SCHEMA:
            raise ValueError("unsupported moment collapse diagnostic schema")
        if mapping["schema_version"] != MOMENT_COLLAPSE_DIAGNOSTIC_VERSION:
            raise ValueError("unsupported moment collapse diagnostic version")
        if mapping["claim_boundary"] != MOMENT_COLLAPSE_DIAGNOSTIC_CLAIM_BOUNDARY:
            raise ValueError("moment collapse claim boundary changed")
        cases = tuple(
            SymmetricGaussianMixtureCaseV1.from_dict(case)
            for case in _strict_list(mapping["cases"], name="cases")
        )
        reports = tuple(
            MomentCollapseCaseReportV1.from_dict(report)
            for report in _strict_list(mapping["reports"], name="reports")
        )
        artifact = cls(
            representation_name=mapping["representation_name"],
            thresholds=MomentCollapseThresholdsV1.from_dict(mapping["thresholds"]),
            cases=cases,
            reports=reports,
            material_case_count=mapping["material_case_count"],
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )
        supplied_any = _strict_bool(
            mapping["any_material_moment_collapse"],
            name="any_material_moment_collapse",
        )
        if supplied_any != artifact.any_material_moment_collapse:
            raise ValueError("any_material_moment_collapse does not match reports")
        supplied_id = _strict_digest(
            mapping["artifact_id"],
            name="artifact_id",
            pattern=_SHA256,
        )
        if supplied_id != artifact.artifact_id:
            raise ValueError("moment collapse diagnostic artifact_id mismatch")
        return artifact


def build_moment_collapse_diagnostic(
    *,
    representation_name: str,
    cases: Sequence[SymmetricGaussianMixtureCaseV1],
    thresholds: MomentCollapseThresholdsV1,
    metadata: Mapping[str, Any] | None = None,
) -> MomentCollapseDiagnosticV1:
    """Canonicalize cases and build a replayable controlled diagnostic."""

    supplied_cases = tuple(cases)
    if not supplied_cases or not all(
        isinstance(case, SymmetricGaussianMixtureCaseV1) for case in supplied_cases
    ):
        raise ValueError(
            "cases must be a nonempty sequence of SymmetricGaussianMixtureCaseV1"
        )
    ordered_cases = tuple(sorted(supplied_cases, key=lambda case: case.case_id))
    reports = tuple(
        evaluate_moment_collapse_case(case, thresholds)
        for case in ordered_cases
    )
    return MomentCollapseDiagnosticV1(
        representation_name=representation_name,
        thresholds=thresholds,
        cases=ordered_cases,
        reports=reports,
        material_case_count=sum(report.material_moment_collapse for report in reports),
        metadata={} if metadata is None else metadata,
    )


def moment_collapse_diagnostic_from_raw(value: Any) -> MomentCollapseDiagnosticV1:
    mapping = _strict_mapping(value, name="raw moment collapse diagnostic")
    _exact_keys(
        mapping,
        {"representation_name", "thresholds", "cases", "metadata"},
        name="raw moment collapse diagnostic",
    )
    return build_moment_collapse_diagnostic(
        representation_name=mapping["representation_name"],
        thresholds=MomentCollapseThresholdsV1.from_dict(mapping["thresholds"]),
        cases=tuple(
            SymmetricGaussianMixtureCaseV1.from_dict(case)
            for case in _strict_list(mapping["cases"], name="cases")
        ),
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )


def write_moment_collapse_diagnostic(
    artifact: MomentCollapseDiagnosticV1,
    path: str | Path,
) -> None:
    if not isinstance(artifact, MomentCollapseDiagnosticV1):
        raise ValueError("artifact must be MomentCollapseDiagnosticV1")
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
        existing = load_moment_collapse_diagnostic(destination)
        if existing.to_dict() == artifact.to_dict():
            return
        raise FileExistsError(
            f"refusing to replace a different moment collapse diagnostic: {destination}"
        ) from None


def load_moment_collapse_diagnostic(path: str | Path) -> MomentCollapseDiagnosticV1:
    return MomentCollapseDiagnosticV1.from_dict(
        _load_json(path, name="moment collapse diagnostic")
    )


def _build_cli(arguments: argparse.Namespace) -> int:
    artifact = moment_collapse_diagnostic_from_raw(
        _load_json(arguments.input, name="raw moment collapse diagnostic")
    )
    write_moment_collapse_diagnostic(artifact, arguments.output)
    print(artifact.artifact_id)
    return 0


def _verify_cli(arguments: argparse.Namespace) -> int:
    artifact = load_moment_collapse_diagnostic(arguments.artifact)
    print(artifact.artifact_id)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify the controlled Gaussian moment-collapse diagnostic."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the controlled diagnostic")
    build.add_argument("input", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(handler=_build_cli)
    verify = subparsers.add_parser("verify", help="verify and replay a diagnostic")
    verify.add_argument("artifact", type=Path)
    verify.set_defaults(handler=_verify_cli)
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


__all__ = [
    "MOMENT_COLLAPSE_DIAGNOSTIC_CLAIM_BOUNDARY",
    "MOMENT_COLLAPSE_DIAGNOSTIC_SCHEMA",
    "MOMENT_COLLAPSE_DIAGNOSTIC_VERSION",
    "MomentCollapseCaseReportV1",
    "MomentCollapseDiagnosticV1",
    "MomentCollapseThresholdsV1",
    "SymmetricGaussianMixtureCaseV1",
    "build_moment_collapse_diagnostic",
    "evaluate_moment_collapse_case",
    "load_moment_collapse_diagnostic",
    "main",
    "moment_collapse_diagnostic_from_raw",
    "moment_matched_gaussian",
    "write_moment_collapse_diagnostic",
]


if __name__ == "__main__":
    raise SystemExit(main())
