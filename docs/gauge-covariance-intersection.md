# Deterministic multi-estimate gauge initialization

`SequentialGaugeEstimator` may obtain several candidate global gauges for a new
MotionCrafter window. Those candidates share upstream windows and are therefore
not safely fused as independent Gaussian estimates.

## Previous behavior

The estimator previously applied two-estimate covariance intersection repeatedly
in the order in which overlap constraints were supplied. Pairwise covariance
intersection is not associative, so permuting equivalent constraint records could
change both the initialized gauge and its covariance. It also averaged global
axis-angle coordinates directly, which is poorly behaved when equivalent
rotations lie on opposite sides of the shortest-logarithm branch.

## Current initializer

The initializer now performs one deterministic n-way covariance-intersection
operation:

1. validate every candidate covariance and reject non-finite, asymmetric, or
   materially indefinite matrices;
2. canonically order the candidates independently of the input list;
3. choose the candidate with the smallest regularized covariance determinant as
   a deterministic local `Sim(3)` chart;
4. transport each candidate mean and covariance into that chart with a centered
   finite-difference Jacobian;
5. minimize the fused log-determinant over the weight simplex by deterministic
   pairwise coordinate minimization and one-dimensional derivative bisection;
6. transport the fused result back to the global seven-vector coordinates.

A small positive component-weight floor prevents a single overlap from being
silently discarded. The floor is reduced automatically when many candidate
parents are available so that the feasible simplex remains nonempty.

## Scope and claim boundary

This changes the recursive initializer used by reconstruction ablations and by
the approximate fixed-lag path. It does **not** replace the strict portable
causal export's selected spanning-tree covariance, whose cross-window covariance
semantics remain unchanged.

The change establishes deterministic numerical behavior and avoids a known
rotation-coordinate failure mode. It is not evidence that Prob4D improves
held-out reconstruction or Bayesian-PhysTwin prediction; those claims still
require the registered prospective experiments.
