# Gauge Covariance Contracts

Prob4D treats gauge-domain covariance as part of the estimator and artifact
contract, not merely as an input to a numerical inverse. The gauge dataclasses
therefore defensively copy and validate covariance before storing it.

## Validation policy

A covariance must be finite, symmetric, and positive semidefinite. Materially
negative eigenvalues fail closed. Floating-point-scale negative eigenvalues are
projected to zero only after passing the same absolute and relative tolerance
used by the shared covariance utilities.

The contract does not add an eigenvalue floor to the stored covariance. This is
important for fixed external calibration: an exact zero covariance remains an
exact zero covariance. A numerical floor is introduced only by operations that
explicitly require an inverse or log determinant.

Gauge estimates, relative-gauge constraints, gauge anchors, and point anchors
store read-only defensive copies. Caller mutation after construction therefore
cannot change an estimator state, calibration input, or content-bound workflow.

## Metadata and anchors

The validation boundary also requires:

- nonempty and distinct relative-gauge window identities;
- finite nonnegative residual diagnostics and correspondence counts;
- finite positive covariance-inflation factors;
- finite positive scale-anchor values; and
- finite point-anchor coordinates with validated covariance.

Gauge covariance calibration rejects invalid covariance before normalized error
statistics are formed. Whitening and Mahalanobis computations use the shared
fail-closed eigendecomposition rather than silently repairing materially invalid
input.

## Claim boundary

These checks prevent malformed or caller-mutated uncertainty from entering gauge
estimation. They do not establish empirical covariance calibration, conservative
coverage on target data, or improved Bayesian physical-twin prediction.
