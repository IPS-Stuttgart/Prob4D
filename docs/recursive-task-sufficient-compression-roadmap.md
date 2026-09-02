# Roadmap to a larger contribution

## Gate 1 — controlled mechanism

- CI-clean exact task-state closure implementation.
- Current-query-only compression exact at one step but demonstrably non-recursive.
- Closure-aware compression recursively exact in the registered linear-Gaussian model.
- Closure violation expands the audited state instead of being silently ignored.

## Gate 2 — theorem audit

- Formal necessary/sufficient statement for the chosen recursive model class.
- Independent proof review.
- Explicit novelty comparison against functional observers, state aggregation, reduced-order
  Kalman filtering, lossless sensor transformations, and goal-oriented Bayesian reduction.

## Gate 3 — time-varying / task-family extension

Generalize from one fixed LTI closure to a finite-horizon or time-varying sequence of task maps and
linearizations. The desired construction is a backward relevance recursion whose task-state rank
can change with the registered horizon and query family.

## Gate 4 — public real-data mechanism

Use an already public deformable-object sequence with correlated repeated pseudo-measurements.
Freeze the task family and closure construction before held-out scoring. Demonstrate full-versus-
compressed posterior parity and rank scaling over complete trajectories.

## Gate 5 — closed-loop decision evidence

Use a task family chosen after the correlated factor is produced and show that the recursively
compressed interface returns the same controller decisions as the full estimator. This is the
experiment that most directly defeats the fixed-query cache objection.

Only after Gates 1–2 should the current ICRA manuscript absorb the recursive theorem. Gates 4–5
would support the substantially larger robotics framing.
