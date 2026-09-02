"""Group-calibrated tubes around estimated residual transformation orbits.

The finite-orbit certificate in :mod:`prob4d.axial_query_certificate` is exact
when the compatible state is known to lie on the declared orbit.  Real
providers usually supply only an estimated orbit.  This module adds a small,
auditable bridge: calibrate a radius from *complete independent groups* and
penalize query diameter and candidate advantage by Lipschitz bounds.

The statistical guarantee is marginal and conditional on exchangeability of
the calibration groups and the future group.  It is not a guarantee that the
orbit estimator, metric, or Lipschitz constants are physically correct.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import math


_CALIBRATION_SCHEMA = "prob4d.split-conformal-orbit-tube.v1"


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def group_max_nonconformity(groups: Iterable[Iterable[float]]) -> tuple[float, ...]:
    """Collapse nested observations to one maximum score per independent group.

    A trajectory-level statement requires trajectories (or another declared
    complete group) as the exchangeable units.  Pooling frames would calibrate
    frame coverage and does not generally control simultaneous coverage of a
    future trajectory.
    """

    maxima: list[float] = []
    for group_index, group in enumerate(groups):
        values = tuple(
            _finite_nonnegative(value, name=f"groups[{group_index}] score")
            for value in group
        )
        if not values:
            raise ValueError(f"groups[{group_index}] must not be empty")
        maxima.append(max(values))
    if not maxima:
        raise ValueError("at least one calibration group is required")
    return tuple(maxima)


@dataclass(frozen=True, slots=True)
class SplitConformalOrbitTube:
    """A split-conformal radius calibrated from complete-group scores."""

    miscoverage: float
    radius: float
    calibration_group_count: int
    quantile_rank: int
    coverage_lower_bound: float
    sorted_group_scores: tuple[float, ...]
    calibration_id: str

    @property
    def finite_radius(self) -> bool:
        """Whether the requested confidence is supportable with finite radius."""

        return math.isfinite(self.radius)

    @property
    def minimum_miscoverage_for_finite_radius(self) -> float:
        """Smallest nominal miscoverage supportable by this sample size."""

        return 1.0 / (self.calibration_group_count + 1.0)

    def covers(self, group_score: float) -> bool:
        """Return whether one complete-group score lies inside the tube."""

        score = _finite_nonnegative(group_score, name="group_score")
        return score <= self.radius


def fit_split_conformal_orbit_tube(
    group_scores: Iterable[float],
    *,
    miscoverage: float,
) -> SplitConformalOrbitTube:
    """Fit the standard finite-sample split-conformal ``higher`` quantile.

    Let ``n`` be the number of complete calibration groups and
    ``k = ceil((n + 1) * (1 - miscoverage))``.  The radius is the ``k``-th
    calibration order statistic.  When ``k == n + 1``, finite calibration is
    impossible from ``n`` groups, so the radius is infinity and every downstream
    certificate fails closed.
    """

    alpha = float(miscoverage)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")

    scores = tuple(
        sorted(
            _finite_nonnegative(score, name="group_score")
            for score in group_scores
        )
    )
    if not scores:
        raise ValueError("at least one calibration group score is required")

    count = len(scores)
    rank = math.ceil((count + 1) * (1.0 - alpha))
    radius = scores[rank - 1] if rank <= count else math.inf
    coverage_lower_bound = min(rank / (count + 1.0), 1.0)

    payload = {
        "schema": _CALIBRATION_SCHEMA,
        "miscoverage": alpha,
        "quantile_rank": rank,
        "radius": radius if math.isfinite(radius) else "infinity",
        "sorted_group_scores": scores,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    calibration_id = hashlib.sha256(encoded).hexdigest()

    return SplitConformalOrbitTube(
        miscoverage=alpha,
        radius=radius,
        calibration_group_count=count,
        quantile_rank=rank,
        coverage_lower_bound=coverage_lower_bound,
        sorted_group_scores=scores,
        calibration_id=calibration_id,
    )


def fit_groupwise_orbit_tube(
    groups: Iterable[Iterable[float]],
    *,
    miscoverage: float,
) -> SplitConformalOrbitTube:
    """Collapse complete groups and fit a split-conformal orbit tube."""

    return fit_split_conformal_orbit_tube(
        group_max_nonconformity(groups),
        miscoverage=miscoverage,
    )


@dataclass(frozen=True, slots=True)
class OrbitTubeDecision:
    """Auditable result of a query-and-advantage tube certificate."""

    accepted: bool
    reasons: tuple[str, ...]
    calibration_id: str
    tube_radius: float
    exact_orbit_query_diameter: float
    query_tube_penalty: float
    query_diameter_upper_bound: float
    query_tolerance: float
    exact_orbit_advantage_lower_bound: float
    advantage_tube_penalty: float
    omitted_effect_bound: float
    numerical_slack: float
    robust_advantage_lower_bound: float
    required_advantage_margin: float


def query_diameter_tube_bound(
    *,
    exact_orbit_diameter: float,
    query_lipschitz: float,
    radius: float,
) -> float:
    """Bound query diameter on a radius-``rho`` tube around an orbit.

    If ``q`` is ``L_q``-Lipschitz and every compatible state lies within
    ``rho`` of the estimated orbit, then

    ``diam(q(C)) <= diam(q(O_hat)) + 2 * L_q * rho``.
    """

    diameter = _finite_nonnegative(
        exact_orbit_diameter,
        name="exact_orbit_diameter",
    )
    lipschitz = _finite_nonnegative(query_lipschitz, name="query_lipschitz")
    rho = float(radius)
    if math.isnan(rho) or rho < 0.0:
        raise ValueError("radius must be nonnegative and not NaN")
    if not math.isfinite(rho):
        return diameter if lipschitz == 0.0 else math.inf
    return diameter + 2.0 * lipschitz * rho


def robust_advantage_tube_bound(
    *,
    exact_orbit_advantage_lower_bound: float,
    advantage_lipschitz: float,
    radius: float,
    omitted_effect_bound: float = 0.0,
    numerical_slack: float = 0.0,
) -> float:
    """Lower-bound candidate advantage on an approximate-orbit tube.

    For fallback-minus-candidate advantage ``D`` that is ``L_D``-Lipschitz,

    ``inf_C D >= inf_O_hat D - L_D * rho``.

    Declared omitted-effect and numerical envelopes are subtracted as additional
    fail-closed penalties.
    """

    advantage = _finite(
        exact_orbit_advantage_lower_bound,
        name="exact_orbit_advantage_lower_bound",
    )
    lipschitz = _finite_nonnegative(
        advantage_lipschitz,
        name="advantage_lipschitz",
    )
    omitted = _finite_nonnegative(
        omitted_effect_bound,
        name="omitted_effect_bound",
    )
    slack = _finite_nonnegative(numerical_slack, name="numerical_slack")
    rho = float(radius)
    if math.isnan(rho) or rho < 0.0:
        raise ValueError("radius must be nonnegative and not NaN")
    if not math.isfinite(rho):
        return advantage - omitted - slack if lipschitz == 0.0 else -math.inf
    return advantage - lipschitz * rho - omitted - slack


def certify_orbit_tube(
    calibration: SplitConformalOrbitTube,
    *,
    exact_orbit_query_diameter: float,
    query_lipschitz: float,
    query_tolerance: float,
    exact_orbit_advantage_lower_bound: float,
    advantage_lipschitz: float,
    required_advantage_margin: float = 0.0,
    omitted_effect_bound: float = 0.0,
    numerical_slack: float = 0.0,
) -> OrbitTubeDecision:
    """Certify a complete candidate under group-calibrated approximate symmetry.

    Admission requires a finite calibrated radius, a query-diameter upper bound
    no larger than the registered tolerance, and a strict robust-advantage lower
    bound above the requested margin.  Otherwise callers must use the complete
    fallback; this function never constructs a partially updated state.
    """

    tolerance = float(query_tolerance)
    if math.isnan(tolerance) or tolerance < 0.0:
        raise ValueError("query_tolerance must be nonnegative and not NaN")
    margin = _finite_nonnegative(
        required_advantage_margin,
        name="required_advantage_margin",
    )

    exact_diameter = _finite_nonnegative(
        exact_orbit_query_diameter,
        name="exact_orbit_query_diameter",
    )
    q_lipschitz = _finite_nonnegative(query_lipschitz, name="query_lipschitz")
    exact_advantage = _finite(
        exact_orbit_advantage_lower_bound,
        name="exact_orbit_advantage_lower_bound",
    )
    d_lipschitz = _finite_nonnegative(
        advantage_lipschitz,
        name="advantage_lipschitz",
    )
    omitted = _finite_nonnegative(
        omitted_effect_bound,
        name="omitted_effect_bound",
    )
    slack = _finite_nonnegative(numerical_slack, name="numerical_slack")

    query_upper = query_diameter_tube_bound(
        exact_orbit_diameter=exact_diameter,
        query_lipschitz=q_lipschitz,
        radius=calibration.radius,
    )
    advantage_lower = robust_advantage_tube_bound(
        exact_orbit_advantage_lower_bound=exact_advantage,
        advantage_lipschitz=d_lipschitz,
        radius=calibration.radius,
        omitted_effect_bound=omitted,
        numerical_slack=slack,
    )

    if math.isfinite(calibration.radius):
        query_penalty = 2.0 * q_lipschitz * calibration.radius
        advantage_tube_penalty = d_lipschitz * calibration.radius
    else:
        query_penalty = 0.0 if q_lipschitz == 0.0 else math.inf
        advantage_tube_penalty = 0.0 if d_lipschitz == 0.0 else math.inf

    reasons: list[str] = []
    if not calibration.finite_radius:
        reasons.append("finite-sample-confidence-not-supportable")
    if query_upper > tolerance:
        reasons.append("query-diameter-exceeds-tolerance")
    if not advantage_lower > margin:
        reasons.append("robust-advantage-not-above-margin")
    if not reasons:
        reasons.append("certified")

    return OrbitTubeDecision(
        accepted=reasons == ["certified"],
        reasons=tuple(reasons),
        calibration_id=calibration.calibration_id,
        tube_radius=calibration.radius,
        exact_orbit_query_diameter=exact_diameter,
        query_tube_penalty=query_penalty,
        query_diameter_upper_bound=query_upper,
        query_tolerance=tolerance,
        exact_orbit_advantage_lower_bound=exact_advantage,
        advantage_tube_penalty=advantage_tube_penalty,
        omitted_effect_bound=omitted,
        numerical_slack=slack,
        robust_advantage_lower_bound=advantage_lower,
        required_advantage_margin=margin,
    )
