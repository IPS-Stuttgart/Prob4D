# Gauge covariance support diagnostics

A positive-semidefinite gauge covariance may be singular because a gauge is
partly deterministic, anchored, or unobservable. A Moore--Penrose pseudoinverse
alone is insufficient for evaluating its error: any error component in the
covariance nullspace receives zero quadratic cost and can therefore make an
invalid estimate look well calibrated.

Prob4D now evaluates each gauge error with
`covariance_support_diagnostic`:

```python
from prob4d.diagnostics.covariance_support import (
    covariance_support_diagnostic,
)

result = covariance_support_diagnostic(error, covariance)
```

The diagnostic validates the covariance as symmetric positive semidefinite
without adding jitter, defines its observable range with a scale-aware
eigenvalue threshold, and decomposes the error into observable and nullspace
components. It reports:

- covariance dimension and numerical rank;
- the observable Moore--Penrose quadratic;
- that quadratic divided by the effective rank;
- the covariance-nullspace error norm;
- the declared support tolerance; and
- whether the complete error lies in covariance support.

The observable quadratic is a valid NEES-style statistic only when
`support_consistent` is true. A nonzero nullspace error is a model or reporting
failure, not a low-uncertainty success.

## Ablation output

The historical `gauge_mean_normalized_squared_error` field remains available as
the mean observable quadratic. Ablation reports now add:

- `gauge_mean_rank_normalized_squared_error`;
- `gauge_mean_covariance_rank`;
- `gauge_minimum_covariance_rank`;
- `gauge_support_violation_count`;
- `gauge_maximum_nullspace_error_norm`; and
- `gauge_all_errors_in_covariance_support`.

Do not interpret the historical mean quadratic as a calibration statistic when
`gauge_support_violation_count` is nonzero. Rank-normalized values exclude
rank-zero deterministic gauges; the support fields still test those gauges and
will expose any incompatible error.

## Scope

This change hardens non-claim-bearing gauge diagnostics and synthetic or
manifest ablation reporting. It does not alter provider selection, gauge
estimation, covariance export, the frozen CUT3R protocol, BayesianPhysTwin
admission, exact fallback, or any scientific result.
