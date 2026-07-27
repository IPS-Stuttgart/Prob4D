# Joint Multi-Window Dense Fusion

Prob4D can receive more than two independently decoded predictions for the same
absolute frame and pixel. Sorting the windows makes a repeated pairwise update
deterministic, but it does not make covariance intersection associative. For
three estimates `a`, `b`, and `c`, in general,

```text
CI(CI(a, b), c) != CI(a, CI(b, c)).
```

The production dense-fusion path therefore groups every frame by its exact valid
contributor pattern and fuses all active estimates in one batch.

## Multi-Input Semantics

For `K` active means `x_k` and covariances `P_k`:

- **uniform fusion** exports the exact equally weighted Gaussian-mixture mean and
  second moment;
- **independent precision fusion** sums all `K` information matrices and
  information vectors in one operation; and
- **covariance intersection** solves one generalized simplex problem,

```text
P(w)^-1 = sum_k w_k P_k^-1,

x(w) = P(w) sum_k w_k P_k^-1 x_k,

w_k >= 0,  sum_k w_k = 1.
```

The covariance-intersection objective is the mean log determinant of `P(w)` on
a deterministic representative sample. A projected-gradient solver with
backtracking optimizes the simplex. Contributor order is canonicalized from the
input contents for the public numerical primitive; `fuse_windows` additionally
uses the repository's canonical `(start_frame, stop_frame, window_id)` ordering.

One contributor is returned unchanged. Two contributors retain the existing
grid-search implementation exactly, including its tie behavior. The generalized
solver is used only for three or more contributors.

## Mask and Flow Handling

Pixels need not share the same valid windows. For each absolute frame, Prob4D
partitions pixels by their Boolean contributor mask. Every nonempty mask pattern
is fused independently, so invalid rows cannot influence the mean, covariance,
or CI weights of valid rows.

Scene flow uses the same batching and covariance-intersection semantics as point
geometry. Gauge translation uncertainty is propagated to points but not to
vectors, preserving the existing `Sim(3)` transport contract.

## Computational Boundary

The solver chooses one weight vector per frame and contributor-mask pattern,
not one weight per pixel. Covariances and information vectors are then evaluated
in chunks. This preserves coherent weights for shared-backbone predictions and
avoids the memory cost of pointwise simplex optimization.

The implementation is deterministic, fail-closed on invalid covariance, and
covered by tests for:

- exact two-input parity;
- equal-covariance symmetry;
- contributor and window permutation invariance;
- agreement with a dense three-input simplex grid;
- heterogeneous masks and contributor counts;
- scene-flow parity; and
- a 4,096-row numerical smoke case.

## Provider Boundary

This changes the decoded reconstruction estimator only. The causally sealed
`ObservationBeliefV1` export retains unfused window rows and its existing joint
gauge-factor semantics, so the provider wire contract and artifact IDs are not
silently reinterpreted by this change.

## Claim Boundary

Joint fusion removes an implementation-level ordering artifact. It does not by
itself establish prospective covariance calibration, improved MotionCrafter
accuracy, or improved Bayesian physical-twin prediction. Those remain held-out
empirical gates under the project evidence policy.
