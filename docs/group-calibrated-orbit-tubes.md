# Group-calibrated approximate-orbit certificates

## Purpose

Prob4D's finite-orbit query certificate is exact when a compatible state is
known to lie on a declared residual orbit. A learned or geometric provider will
usually estimate that orbit imperfectly. This module adds an explicit tube around
the estimated orbit without pretending that approximate symmetry is exact.

The implementation is in `prob4d.orbit_tube`.

## Independent calibration unit

Let recording, object, or trajectory `i` contain nested residual scores
`r[i, t]`, measuring distance of the compatible complete state from the estimated
orbit at frame or query instance `t`. The calibration score is

```text
R_i = max_t r[i, t].
```

The maximum is deliberate. Treating frames as exchangeable would calibrate a
random frame, not simultaneous coverage of the complete future trajectory.

For `n` complete calibration groups and miscoverage `alpha`, define

```text
k = ceil((n + 1) * (1 - alpha)).
```

When `k <= n`, the tube radius is the `k`-th sorted group score. Under
exchangeability of the complete calibration groups and one future group,

```text
P(R_new <= rho_alpha) >= 1 - alpha.
```

When `k = n + 1`, the requested finite-sample confidence cannot be supported by
`n` groups. Prob4D records `rho_alpha = infinity` and the downstream certificate
rejects. It does not silently use the largest calibration score as if it supplied
the requested coverage.

## Query and decision bounds

Let `O_hat` be the estimated residual orbit, and suppose every compatible state
lies within metric distance `rho` of `O_hat`.

For an `L_q`-Lipschitz query,

```text
diam q(C) <= diam q(O_hat) + 2 L_q rho.
```

For fallback-minus-candidate advantage `D` that is `L_D`-Lipschitz,

```text
inf_C D >= inf_O_hat D - L_D rho.
```

With omitted-effect envelope `epsilon`, numerical slack `delta`, and required
margin `m`, the complete candidate is admitted only if

```text
inf_O_hat D - L_D rho - epsilon - delta > m
```

and the inflated query diameter is no larger than its registered tolerance.
Every other outcome is a rejection. The module returns a decision record only;
the caller remains responsible for returning the complete fallback state,
covariance, identity, and provenance atomically.

## Example

```python
from prob4d.orbit_tube import (
    certify_orbit_tube,
    fit_groupwise_orbit_tube,
)

calibration = fit_groupwise_orbit_tube(
    complete_recording_scores,
    miscoverage=0.1,
)

decision = certify_orbit_tube(
    calibration,
    exact_orbit_query_diameter=0.01,
    query_lipschitz=1.8,
    query_tolerance=0.05,
    exact_orbit_advantage_lower_bound=0.20,
    advantage_lipschitz=2.5,
    omitted_effect_bound=0.01,
    numerical_slack=1e-6,
    required_advantage_margin=0.02,
)

if decision.accepted:
    use_complete_candidate()
else:
    use_exact_complete_fallback()
```

## Guarantee boundary

The conformal statement is marginal over an exchangeable future group. It does
not establish that:

- the orbit estimator has the correct physical symmetry class;
- calibration and deployment groups are exchangeable;
- the distance metric is task-complete;
- the supplied Lipschitz constants or omitted-effect envelope are valid;
- conditional subgroup coverage holds; or
- an accepted update is generally safe outside the registered query and loss.

These assumptions must be source-qualified and retained in the evidence record.
Calibration groups must not be frames from the same trajectory, overlapping
windows, vertices, particles, or angular quadrature atoms.
