# Analytic gauge covariance propagation

`prob4d.gauge_analytic` is an additive experimental covariance-propagation path
for recursive `Sim(3)` gauges. The historical classes in `prob4d.gauge` retain
their finite-difference behavior so frozen provider-v1 and existing development
artifacts are not reinterpreted.

## Coordinate convention

Transforms use the seven-vector

```text
[log_scale, rotation_vector(3), translation(3)].
```

Composition uses the analytic Jacobian callable selected by the immutable
observation-export numerical policy. Compatibility context selection remains
available in `prob4d.composition_jacobian`, but it no longer relies on replacing
private exporter functions. Inversion uses the exact derivative of

```text
scale_inverse       = 1 / scale
rotation_inverse    = rotation.T
translation_inverse = -(1 / scale) * rotation.T @ translation.
```

The principal `SO(3)` logarithm is not differentiable at angle pi. Analytic
composition or inversion therefore fails closed within the declared branch-cut
tolerance rather than emitting platform-dependent covariance.

## Public experimental surface

```python
from prob4d.gauge_analytic import (
    AnalyticSequentialGaugeEstimatorV2,
    compose_sim3_with_covariance_analytic,
    invert_sim3_with_covariance_analytic,
)
```

`AnalyticSequentialGaugeEstimatorV2` preserves the historical transform
initialization and covariance-intersection policy. Only covariance derivatives
for composition and inversion change. The class records
`jacobian_method = "analytic_sim3_compose_inverse_v1"` for experiment manifests.
It is deliberately not exported from the stable `prob4d.api.v2` facade.

## Numerical validation

The focused validation covers:

- analytic inverse derivatives against central finite differences over varied
  scales, translations, and rotations;
- analytic composition and inversion covariance against a numerical oracle;
- exact transform parity with the historical sequential estimator;
- close covariance parity away from branch cuts;
- positive-semidefinite output validation; and
- branch-cut, invalid-covariance, and coercive-parameter rejection.

Dense alignment covariance now obtains IID inverses and cluster-robust sandwich
terms through Cholesky solves, with a validated eigendecomposition fallback for
numerically difficult but full-rank information matrices. It no longer uses a
Moore-Penrose pseudoinverse after declaring the alignment information full rank.
This changes numerical implementation, not the covariance method labels or
estimator formula.

## Claim boundary

Analytic first-order derivatives and factorized solves reduce numerical
approximation and make failure modes explicit. They do not establish target-data
calibration, provider competence, physical-query benefit, or Causal4D benefit.
Any promotion requires the same source/calibration separation and held-out
physical gate as other Prob4D covariance treatments.
