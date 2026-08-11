"""Source-fitted contaminated-Gaussian likelihood for correlation groups.

One latent contamination state is shared by every row in one coherent correlation
group. Independent source objects or sessions remain the outer statistical units.
Association probability and prior reliability remain separate from the posterior
contamination responsibility computed here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .joint_covariance_metrics import evaluate_joint_gaussian_group

FloatArray: TypeAlias = NDArray[np.floating[Any]]

CORRELATION_GROUP_ROBUST_SCHEMA = "prob4d.correlation-group-robust-likelihood"
CORRELATION_GROUP_ROBUST_VERSION = 1
CORRELATION_GROUP_ROBUST_CLAIM_BOUNDARY = (
    "This source-side diagnostic fits a finite contaminated-Gaussian candidate grid "
    "over correlation groups nested inside independent source objects or sessions. "
    "Posterior contamination responsibility is distinct from association probability "
    "and prior reliability. The result does not authorize a BayesianPhysTwin "
    "update, replace exact fallback, establish "
    "target calibration, or establish Causal4D intervention benefit."
)


def _strict_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a genuine real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    return int(value)


def _strict_group_id(value: object, *, name: str = "group_id") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _readonly(value: object, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _array_digest(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


@dataclass(frozen=True, slots=True)
class CorrelationGroupContaminationSpecV1:
    """One frozen group-level contaminated-Gaussian candidate."""

    contamination_probability: float
    inflation_factor: float
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        probability = _strict_real(
            self.contamination_probability,
            name="contamination_probability",
        )
        inflation = _strict_real(self.inflation_factor, name="inflation_factor")
        if not 0.0 <= probability < 1.0:
            raise ValueError("contamination_probability must lie in [0, 1)")
        if probability == 0.0:
            if inflation != 1.0:
                raise ValueError("the Gaussian fallback must use inflation_factor=1")
        elif inflation <= 1.0:
            raise ValueError("a contaminated candidate must use inflation_factor > 1")
        identity = {
            "schema": CORRELATION_GROUP_ROBUST_SCHEMA,
            "version": CORRELATION_GROUP_ROBUST_VERSION,
            "contamination_probability_hex": probability.hex(),
            "inflation_factor_hex": inflation.hex(),
        }
        object.__setattr__(self, "contamination_probability", probability)
        object.__setattr__(self, "inflation_factor", inflation)
        object.__setattr__(
            self,
            "spec_id",
            hashlib.sha256(_canonical_json(identity)).hexdigest(),
        )

    @property
    def is_gaussian_fallback(self) -> bool:
        return self.contamination_probability == 0.0

    def summary(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "contamination_probability": self.contamination_probability,
            "inflation_factor": self.inflation_factor,
            "is_gaussian_fallback": self.is_gaussian_fallback,
        }


GAUSSIAN_GROUP_LIKELIHOOD_V1 = CorrelationGroupContaminationSpecV1(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class CorrelationGroupResidualV1:
    """Immutable matched residual bundle for one coherent correlation group."""

    group_id: str
    residual_xyz_m: FloatArray
    local_covariance_m2: FloatArray
    low_rank_factor_m: FloatArray
    source_id: str = field(init=False)
    sample_count: int = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        group_id = _strict_group_id(self.group_id)
        residual = np.asarray(self.residual_xyz_m, dtype=np.float64)
        local = np.asarray(self.local_covariance_m2, dtype=np.float64)
        factor = np.asarray(self.low_rank_factor_m, dtype=np.float64)
        if residual.ndim != 2 or residual.shape[1:] != (3,) or residual.shape[0] < 1:
            raise ValueError("residual_xyz_m must have nonempty shape (N, 3)")
        if local.shape != (residual.shape[0], 3, 3):
            raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
        if factor.ndim != 3 or factor.shape[:2] != residual.shape:
            raise ValueError("low_rank_factor_m must have shape (N, 3, R)")
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual_xyz_m must be finite")
        if not np.all(np.isfinite(local)):
            raise ValueError("local_covariance_m2 must be finite")
        if not np.all(np.isfinite(factor)):
            raise ValueError("low_rank_factor_m must be finite")
        evaluate_joint_gaussian_group(residual, local, factor)
        symmetric = 0.5 * (local + local.swapaxes(1, 2))

        residual_owned = _readonly(residual)
        local_owned = _readonly(symmetric)
        factor_owned = _readonly(factor)
        digest = hashlib.sha256()
        digest.update(
            _canonical_json(
                {
                    "schema": CORRELATION_GROUP_ROBUST_SCHEMA,
                    "version": CORRELATION_GROUP_ROBUST_VERSION,
                    "record": "correlation-group-residual",
                    "group_id": group_id,
                }
            )
        )
        _array_digest(digest, "residual_xyz_m", residual_owned)
        _array_digest(digest, "local_covariance_m2", local_owned)
        _array_digest(digest, "low_rank_factor_m", factor_owned)

        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "residual_xyz_m", residual_owned)
        object.__setattr__(self, "local_covariance_m2", local_owned)
        object.__setattr__(self, "low_rank_factor_m", factor_owned)
        object.__setattr__(self, "source_id", digest.hexdigest())
        object.__setattr__(self, "sample_count", int(residual.shape[0]))
        object.__setattr__(self, "dimension", int(3 * residual.shape[0]))


@dataclass(frozen=True, slots=True)
class SourceCorrelationGroupUnitV1:
    """One independent source object/session containing correlation groups."""

    source_unit_id: str
    correlation_groups: tuple[CorrelationGroupResidualV1, ...]
    source_id: str = field(init=False)
    correlation_group_count: int = field(init=False)
    sample_count: int = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        source_unit_id = _strict_group_id(
            self.source_unit_id,
            name="source_unit_id",
        )
        groups = tuple(self.correlation_groups)
        if not groups:
            raise ValueError("correlation_groups must not be empty")
        if not all(isinstance(item, CorrelationGroupResidualV1) for item in groups):
            raise TypeError(
                "correlation_groups must contain CorrelationGroupResidualV1 values"
            )
        groups = tuple(sorted(groups, key=lambda item: item.group_id))
        if len({item.group_id for item in groups}) != len(groups):
            raise ValueError("correlation-group IDs must be unique within a source unit")

        identity = {
            "schema": CORRELATION_GROUP_ROBUST_SCHEMA,
            "version": CORRELATION_GROUP_ROBUST_VERSION,
            "record": "source-correlation-group-unit",
            "source_unit_id": source_unit_id,
            "correlation_groups": [
                {"group_id": group.group_id, "source_id": group.source_id}
                for group in groups
            ],
        }
        source_id = hashlib.sha256(_canonical_json(identity)).hexdigest()

        object.__setattr__(self, "source_unit_id", source_unit_id)
        object.__setattr__(self, "correlation_groups", groups)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "correlation_group_count", len(groups))
        object.__setattr__(
            self,
            "sample_count",
            sum(item.sample_count for item in groups),
        )
        object.__setattr__(
            self,
            "dimension",
            sum(item.dimension for item in groups),
        )


@dataclass(frozen=True, slots=True)
class CorrelationGroupMixtureEvaluationV1:
    """Likelihood and one shared posterior contamination responsibility."""

    group_id: str
    source_id: str
    spec: CorrelationGroupContaminationSpecV1
    sample_count: int
    dimension: int
    mahalanobis_squared: float
    joint_log_determinant: float
    gaussian_nll: float
    inflated_nll: float
    mixture_nll: float
    posterior_contamination_probability: float
    posterior_expected_precision_multiplier: float

    @property
    def mixture_nll_per_dimension(self) -> float:
        return self.mixture_nll / self.dimension

    @property
    def gaussian_nll_per_dimension(self) -> float:
        return self.gaussian_nll / self.dimension

    @property
    def nll_advantage_over_gaussian_per_dimension(self) -> float:
        return (self.gaussian_nll - self.mixture_nll) / self.dimension

    def summary(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "source_id": self.source_id,
            "spec_id": self.spec.spec_id,
            "sample_count": self.sample_count,
            "dimension": self.dimension,
            "mahalanobis_squared": self.mahalanobis_squared,
            "joint_log_determinant": self.joint_log_determinant,
            "gaussian_nll": self.gaussian_nll,
            "inflated_nll": self.inflated_nll,
            "mixture_nll": self.mixture_nll,
            "mixture_nll_per_dimension": self.mixture_nll_per_dimension,
            "nll_advantage_over_gaussian_per_dimension": (
                self.nll_advantage_over_gaussian_per_dimension
            ),
            "posterior_contamination_probability": (
                self.posterior_contamination_probability
            ),
            "posterior_expected_precision_multiplier": (
                self.posterior_expected_precision_multiplier
            ),
        }


def evaluate_correlation_group_mixture(
    group: CorrelationGroupResidualV1,
    spec: CorrelationGroupContaminationSpecV1,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> CorrelationGroupMixtureEvaluationV1:
    """Evaluate a group-shared two-scale Gaussian mixture without densification."""

    if not isinstance(group, CorrelationGroupResidualV1):
        raise TypeError("group must be CorrelationGroupResidualV1")
    if not isinstance(spec, CorrelationGroupContaminationSpecV1):
        raise TypeError("spec must be CorrelationGroupContaminationSpecV1")
    metrics = evaluate_joint_gaussian_group(
        group.residual_xyz_m,
        group.local_covariance_m2,
        group.low_rank_factor_m,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    dimension = int(metrics["dimension"])
    mahalanobis = float(metrics["mahalanobis_squared"])
    log_determinant = float(metrics["joint_log_determinant"])
    gaussian_nll = float(metrics["gaussian_nll"])

    if spec.is_gaussian_fallback:
        inflated_nll = gaussian_nll
        mixture_nll = gaussian_nll
        posterior_contamination_probability = 0.0
        expected_precision = 1.0
    else:
        inflation = spec.inflation_factor
        inflated_nll = 0.5 * (
            dimension * math.log(2.0 * math.pi)
            + log_determinant
            + dimension * math.log(inflation)
            + mahalanobis / inflation
        )
        nominal_log_component = math.log1p(-spec.contamination_probability) - gaussian_nll
        contaminated_log_component = (
            math.log(spec.contamination_probability) - inflated_nll
        )
        mixture_log_likelihood = _logaddexp(
            nominal_log_component,
            contaminated_log_component,
        )
        mixture_nll = -mixture_log_likelihood
        posterior_contamination_probability = math.exp(
            contaminated_log_component - mixture_log_likelihood
        )
        expected_precision = (
            1.0 - posterior_contamination_probability
            + posterior_contamination_probability / inflation
        )

    return CorrelationGroupMixtureEvaluationV1(
        group_id=group.group_id,
        source_id=group.source_id,
        spec=spec,
        sample_count=group.sample_count,
        dimension=dimension,
        mahalanobis_squared=mahalanobis,
        joint_log_determinant=log_determinant,
        gaussian_nll=gaussian_nll,
        inflated_nll=inflated_nll,
        mixture_nll=mixture_nll,
        posterior_contamination_probability=posterior_contamination_probability,
        posterior_expected_precision_multiplier=expected_precision,
    )


def _validated_probability(value: object, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _candidate_complexity_key(
    spec: CorrelationGroupContaminationSpecV1,
) -> tuple[int, float, float, str]:
    return (
        0 if spec.is_gaussian_fallback else 1,
        spec.contamination_probability,
        spec.inflation_factor,
        spec.spec_id,
    )


def _choose_candidate(
    scores: np.ndarray,
    candidates: tuple[CorrelationGroupContaminationSpecV1, ...],
    *,
    tie_tolerance: float,
) -> int:
    best = float(np.min(scores))
    eligible = [
        index
        for index, score in enumerate(scores)
        if float(score) <= best + tie_tolerance
    ]
    return min(
        eligible,
        key=lambda index: _candidate_complexity_key(candidates[index]),
    )


@dataclass(frozen=True, slots=True)
class SourceUnitSelectionFoldV1:
    heldout_source_unit_id: str
    selected_spec_id: str
    selected_is_robust: bool
    heldout_selected_nll_per_dimension: float
    heldout_gaussian_nll_per_dimension: float
    heldout_nll_advantage_per_dimension: float
    harmful_relative_to_gaussian: bool

    def summary(self) -> dict[str, object]:
        return {
            "heldout_source_unit_id": self.heldout_source_unit_id,
            "selected_spec_id": self.selected_spec_id,
            "selected_is_robust": self.selected_is_robust,
            "heldout_selected_nll_per_dimension": (
                self.heldout_selected_nll_per_dimension
            ),
            "heldout_gaussian_nll_per_dimension": (
                self.heldout_gaussian_nll_per_dimension
            ),
            "heldout_nll_advantage_per_dimension": (
                self.heldout_nll_advantage_per_dimension
            ),
            "harmful_relative_to_gaussian": self.harmful_relative_to_gaussian,
        }


@dataclass(frozen=True, slots=True)
class SourceCorrelationGroupMixtureSelectionV1:
    """Replayable source-only selection over a finite frozen candidate grid."""

    source_unit_ids: tuple[str, ...]
    source_unit_source_ids: tuple[str, ...]
    candidates: tuple[CorrelationGroupContaminationSpecV1, ...]
    nll_per_dimension: FloatArray
    minimum_source_unit_count: int = 4
    minimum_mean_heldout_advantage_per_dimension: float = 0.0
    maximum_heldout_nll_harm_per_dimension: float = 0.1
    maximum_harmful_source_unit_fraction: float = 0.0
    minimum_final_candidate_fold_fraction: float = 0.5
    tie_tolerance: float = 1e-12
    relative_rank_tolerance: float = 1e-10
    unconstrained_spec_id: str = field(init=False)
    selected_spec_id: str = field(init=False)
    robust_supported: bool = field(init=False)
    decision_reasons: tuple[str, ...] = field(init=False)
    candidate_equal_source_unit_mean_nll_per_dimension: tuple[float, ...] = field(
        init=False
    )
    folds: tuple[SourceUnitSelectionFoldV1, ...] = field(init=False)
    mean_heldout_advantage_per_dimension: float = field(init=False)
    harmful_source_unit_fraction: float = field(init=False)
    final_candidate_fold_fraction: float = field(init=False)
    selection_id: str = field(init=False)

    def __post_init__(self) -> None:
        source_unit_ids = tuple(
            _strict_group_id(value, name=f"source_unit_ids[{index}]")
            for index, value in enumerate(self.source_unit_ids)
        )
        if len(source_unit_ids) < 2:
            raise ValueError("at least two independent source units are required")
        if tuple(sorted(source_unit_ids)) != source_unit_ids:
            raise ValueError("source_unit_ids must be sorted")
        if len(set(source_unit_ids)) != len(source_unit_ids):
            raise ValueError("source_unit_ids must be unique")
        source_ids = tuple(self.source_unit_source_ids)
        if len(source_ids) != len(source_unit_ids):
            raise ValueError("source_unit_source_ids must match source_unit_ids")
        for index, value in enumerate(source_ids):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
            ):
                raise ValueError(f"source_unit_source_ids[{index}] must be SHA-256 text")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(
                    f"source_unit_source_ids[{index}] must be SHA-256 text"
                ) from error

        raw_candidates = tuple(self.candidates)
        if not raw_candidates:
            raise ValueError("candidates must not be empty")
        if not all(
            isinstance(value, CorrelationGroupContaminationSpecV1)
            for value in raw_candidates
        ):
            raise TypeError("candidates must contain contamination specifications")
        candidate_order = tuple(
            sorted(
                range(len(raw_candidates)),
                key=lambda index: raw_candidates[index].spec_id,
            )
        )
        candidates = tuple(raw_candidates[index] for index in candidate_order)
        if len({item.spec_id for item in candidates}) != len(candidates):
            raise ValueError("candidate specifications must be unique")
        gaussian_indices = [
            index for index, item in enumerate(candidates) if item.is_gaussian_fallback
        ]
        if len(gaussian_indices) != 1:
            raise ValueError("the candidate grid must contain one Gaussian fallback")
        gaussian_index = gaussian_indices[0]

        raw_scores = np.asarray(self.nll_per_dimension, dtype=np.float64)
        if raw_scores.shape != (len(candidates), len(source_unit_ids)):
            raise ValueError(
                "nll_per_dimension must have shape "
                "(candidate_count, source_unit_count)"
            )
        if not np.all(np.isfinite(raw_scores)):
            raise ValueError("nll_per_dimension must be finite")
        scores = raw_scores[np.asarray(candidate_order, dtype=np.int64)]

        minimum_source_unit_count = _strict_integer(
            self.minimum_source_unit_count,
            name="minimum_source_unit_count",
        )
        if minimum_source_unit_count < 2:
            raise ValueError("minimum_source_unit_count must be at least two")
        minimum_advantage = _strict_real(
            self.minimum_mean_heldout_advantage_per_dimension,
            name="minimum_mean_heldout_advantage_per_dimension",
        )
        if minimum_advantage < 0.0:
            raise ValueError(
                "minimum_mean_heldout_advantage_per_dimension must be nonnegative"
            )
        maximum_harm_amount = _strict_real(
            self.maximum_heldout_nll_harm_per_dimension,
            name="maximum_heldout_nll_harm_per_dimension",
        )
        if maximum_harm_amount < 0.0:
            raise ValueError(
                "maximum_heldout_nll_harm_per_dimension must be nonnegative"
            )
        maximum_harm = _validated_probability(
            self.maximum_harmful_source_unit_fraction,
            name="maximum_harmful_source_unit_fraction",
        )
        minimum_fold_fraction = _validated_probability(
            self.minimum_final_candidate_fold_fraction,
            name="minimum_final_candidate_fold_fraction",
        )
        tie_tolerance = _strict_real(self.tie_tolerance, name="tie_tolerance")
        if tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be nonnegative")
        relative_rank_tolerance = _strict_real(
            self.relative_rank_tolerance,
            name="relative_rank_tolerance",
        )
        if not 0.0 <= relative_rank_tolerance < 1.0:
            raise ValueError("relative_rank_tolerance must lie in [0, 1)")

        candidate_means = tuple(float(np.mean(row)) for row in scores)
        unconstrained_index = _choose_candidate(
            np.asarray(candidate_means),
            candidates,
            tie_tolerance=tie_tolerance,
        )
        folds: list[SourceUnitSelectionFoldV1] = []
        selected_indices: list[int] = []
        advantages: list[float] = []
        harmful_count = 0
        for heldout_index, source_unit_id in enumerate(source_unit_ids):
            retained = np.ones(len(source_unit_ids), dtype=bool)
            retained[heldout_index] = False
            training_scores = np.mean(scores[:, retained], axis=1)
            selected_index = _choose_candidate(
                training_scores,
                candidates,
                tie_tolerance=tie_tolerance,
            )
            selected_indices.append(selected_index)
            selected_score = float(scores[selected_index, heldout_index])
            gaussian_score = float(scores[gaussian_index, heldout_index])
            advantage = gaussian_score - selected_score
            harmful = advantage < -(maximum_harm_amount + tie_tolerance)
            harmful_count += int(harmful)
            advantages.append(advantage)
            folds.append(
                SourceUnitSelectionFoldV1(
                    heldout_source_unit_id=source_unit_id,
                    selected_spec_id=candidates[selected_index].spec_id,
                    selected_is_robust=(
                        not candidates[selected_index].is_gaussian_fallback
                    ),
                    heldout_selected_nll_per_dimension=selected_score,
                    heldout_gaussian_nll_per_dimension=gaussian_score,
                    heldout_nll_advantage_per_dimension=advantage,
                    harmful_relative_to_gaussian=harmful,
                )
            )

        mean_advantage = float(np.mean(advantages))
        harmful_fraction = harmful_count / len(source_unit_ids)
        final_fold_fraction = (
            selected_indices.count(unconstrained_index) / len(source_unit_ids)
        )
        reasons: list[str] = []
        if len(source_unit_ids) < minimum_source_unit_count:
            reasons.append("insufficient-independent-source-units")
        if candidates[unconstrained_index].is_gaussian_fallback:
            reasons.append("full-source-selection-is-gaussian")
        if mean_advantage + tie_tolerance < minimum_advantage:
            reasons.append("heldout-mean-nll-advantage-below-minimum")
        if harmful_fraction > maximum_harm + tie_tolerance:
            reasons.append("harmful-heldout-source-unit-fraction-exceeds-maximum")
        if final_fold_fraction + tie_tolerance < minimum_fold_fraction:
            reasons.append("final-candidate-fold-fraction-below-minimum")
        robust_supported = not reasons
        selected_index = unconstrained_index if robust_supported else gaussian_index

        identity = {
            "schema": CORRELATION_GROUP_ROBUST_SCHEMA,
            "version": CORRELATION_GROUP_ROBUST_VERSION,
            "source_unit_ids": list(source_unit_ids),
            "source_unit_source_ids": list(source_ids),
            "candidate_spec_ids": [item.spec_id for item in candidates],
            "nll_per_dimension": [
                [float(value).hex() for value in row] for row in scores
            ],
            "minimum_source_unit_count": minimum_source_unit_count,
            "minimum_mean_heldout_advantage_per_dimension_hex": minimum_advantage.hex(),
            "maximum_heldout_nll_harm_per_dimension_hex": (
                maximum_harm_amount.hex()
            ),
            "maximum_harmful_source_unit_fraction_hex": maximum_harm.hex(),
            "minimum_final_candidate_fold_fraction_hex": minimum_fold_fraction.hex(),
            "tie_tolerance_hex": tie_tolerance.hex(),
            "relative_rank_tolerance_hex": relative_rank_tolerance.hex(),
            "claim_boundary": CORRELATION_GROUP_ROBUST_CLAIM_BOUNDARY,
        }

        object.__setattr__(self, "source_unit_ids", source_unit_ids)
        object.__setattr__(self, "source_unit_source_ids", source_ids)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "nll_per_dimension", _readonly(scores))
        object.__setattr__(
            self,
            "minimum_source_unit_count",
            minimum_source_unit_count,
        )
        object.__setattr__(
            self,
            "minimum_mean_heldout_advantage_per_dimension",
            minimum_advantage,
        )
        object.__setattr__(
            self,
            "maximum_heldout_nll_harm_per_dimension",
            maximum_harm_amount,
        )
        object.__setattr__(
            self,
            "maximum_harmful_source_unit_fraction",
            maximum_harm,
        )
        object.__setattr__(
            self,
            "minimum_final_candidate_fold_fraction",
            minimum_fold_fraction,
        )
        object.__setattr__(self, "tie_tolerance", tie_tolerance)
        object.__setattr__(self, "relative_rank_tolerance", relative_rank_tolerance)
        object.__setattr__(
            self,
            "unconstrained_spec_id",
            candidates[unconstrained_index].spec_id,
        )
        object.__setattr__(
            self,
            "selected_spec_id",
            candidates[selected_index].spec_id,
        )
        object.__setattr__(self, "robust_supported", robust_supported)
        object.__setattr__(self, "decision_reasons", tuple(reasons))
        object.__setattr__(
            self,
            "candidate_equal_source_unit_mean_nll_per_dimension",
            candidate_means,
        )
        object.__setattr__(self, "folds", tuple(folds))
        object.__setattr__(
            self,
            "mean_heldout_advantage_per_dimension",
            mean_advantage,
        )
        object.__setattr__(
            self,
            "harmful_source_unit_fraction",
            harmful_fraction,
        )
        object.__setattr__(
            self,
            "final_candidate_fold_fraction",
            final_fold_fraction,
        )
        object.__setattr__(
            self,
            "selection_id",
            hashlib.sha256(_canonical_json(identity)).hexdigest(),
        )

    @property
    def selected_spec(self) -> CorrelationGroupContaminationSpecV1:
        return next(
            item for item in self.candidates if item.spec_id == self.selected_spec_id
        )

    @property
    def unconstrained_spec(self) -> CorrelationGroupContaminationSpecV1:
        return next(
            item for item in self.candidates if item.spec_id == self.unconstrained_spec_id
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": CORRELATION_GROUP_ROBUST_SCHEMA,
            "version": CORRELATION_GROUP_ROBUST_VERSION,
            "selection_id": self.selection_id,
            "source_unit_ids": list(self.source_unit_ids),
            "source_unit_source_ids": list(self.source_unit_source_ids),
            "candidate_specs": [item.summary() for item in self.candidates],
            "candidate_equal_source_unit_mean_nll_per_dimension": list(
                self.candidate_equal_source_unit_mean_nll_per_dimension
            ),
            "candidate_by_source_unit_nll_per_dimension": (
                self.nll_per_dimension.tolist()
            ),
            "unconstrained_spec_id": self.unconstrained_spec_id,
            "selected_spec_id": self.selected_spec_id,
            "robust_supported": self.robust_supported,
            "decision_reasons": list(self.decision_reasons),
            "mean_heldout_advantage_per_dimension": (
                self.mean_heldout_advantage_per_dimension
            ),
            "harmful_source_unit_fraction": self.harmful_source_unit_fraction,
            "final_candidate_fold_fraction": self.final_candidate_fold_fraction,
            "minimum_source_unit_count": self.minimum_source_unit_count,
            "minimum_mean_heldout_advantage_per_dimension": (
                self.minimum_mean_heldout_advantage_per_dimension
            ),
            "maximum_heldout_nll_harm_per_dimension": (
                self.maximum_heldout_nll_harm_per_dimension
            ),
            "maximum_harmful_source_unit_fraction": (
                self.maximum_harmful_source_unit_fraction
            ),
            "minimum_final_candidate_fold_fraction": (
                self.minimum_final_candidate_fold_fraction
            ),
            "tie_tolerance": self.tie_tolerance,
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "folds": [fold.summary() for fold in self.folds],
            "claim_boundary": CORRELATION_GROUP_ROBUST_CLAIM_BOUNDARY,
        }


def select_source_correlation_group_mixture(
    source_units: Sequence[SourceCorrelationGroupUnitV1],
    candidates: Sequence[CorrelationGroupContaminationSpecV1],
    *,
    minimum_source_unit_count: int = 4,
    minimum_mean_heldout_advantage_per_dimension: float = 0.0,
    maximum_heldout_nll_harm_per_dimension: float = 0.1,
    maximum_harmful_source_unit_fraction: float = 0.0,
    minimum_final_candidate_fold_fraction: float = 0.5,
    tie_tolerance: float = 1e-12,
    relative_rank_tolerance: float = 1e-10,
) -> SourceCorrelationGroupMixtureSelectionV1:
    """Select or reject a robust likelihood using independent source units."""

    if isinstance(source_units, (str, bytes)) or not isinstance(
        source_units,
        Sequence,
    ):
        raise TypeError("source_units must be a sequence")
    raw_source_units = tuple(source_units)
    if not all(
        isinstance(item, SourceCorrelationGroupUnitV1) for item in raw_source_units
    ):
        raise TypeError(
            "source_units must contain SourceCorrelationGroupUnitV1 values"
        )
    ordered_source_units = tuple(
        sorted(raw_source_units, key=lambda item: item.source_unit_id)
    )
    if len(ordered_source_units) < 2:
        raise ValueError("at least two independent source units are required")
    if len({item.source_unit_id for item in ordered_source_units}) != len(
        ordered_source_units
    ):
        raise ValueError("source-unit IDs must be unique")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")
    raw_candidates = tuple(candidates)
    if not all(
        isinstance(item, CorrelationGroupContaminationSpecV1)
        for item in raw_candidates
    ):
        raise TypeError("candidates must contain contamination specifications")
    ordered_candidates = tuple(
        sorted(raw_candidates, key=lambda item: item.spec_id)
    )

    scores = np.empty(
        (len(ordered_candidates), len(ordered_source_units)),
        dtype=np.float64,
    )
    for candidate_index, candidate in enumerate(ordered_candidates):
        for source_index, source_unit in enumerate(ordered_source_units):
            total_nll = 0.0
            total_dimension = 0
            for group in source_unit.correlation_groups:
                evaluation = evaluate_correlation_group_mixture(
                    group,
                    candidate,
                    relative_rank_tolerance=relative_rank_tolerance,
                )
                total_nll += evaluation.mixture_nll
                total_dimension += evaluation.dimension
            scores[candidate_index, source_index] = total_nll / total_dimension

    return SourceCorrelationGroupMixtureSelectionV1(
        source_unit_ids=tuple(item.source_unit_id for item in ordered_source_units),
        source_unit_source_ids=tuple(
            item.source_id for item in ordered_source_units
        ),
        candidates=ordered_candidates,
        nll_per_dimension=scores,
        minimum_source_unit_count=minimum_source_unit_count,
        minimum_mean_heldout_advantage_per_dimension=(
            minimum_mean_heldout_advantage_per_dimension
        ),
        maximum_heldout_nll_harm_per_dimension=(
            maximum_heldout_nll_harm_per_dimension
        ),
        maximum_harmful_source_unit_fraction=maximum_harmful_source_unit_fraction,
        minimum_final_candidate_fold_fraction=minimum_final_candidate_fold_fraction,
        tie_tolerance=tie_tolerance,
        relative_rank_tolerance=relative_rank_tolerance,
    )


__all__ = [
    "CORRELATION_GROUP_ROBUST_CLAIM_BOUNDARY",
    "CORRELATION_GROUP_ROBUST_SCHEMA",
    "CORRELATION_GROUP_ROBUST_VERSION",
    "GAUSSIAN_GROUP_LIKELIHOOD_V1",
    "CorrelationGroupContaminationSpecV1",
    "CorrelationGroupMixtureEvaluationV1",
    "CorrelationGroupResidualV1",
    "SourceUnitSelectionFoldV1",
    "SourceCorrelationGroupMixtureSelectionV1",
    "SourceCorrelationGroupUnitV1",
    "evaluate_correlation_group_mixture",
    "select_source_correlation_group_mixture",
]
