# Next experiment: changing-task recursive exactness

The controlled LTI mechanism is not sufficient for a large robotics claim. The next experiment
should be preregistered before target outcomes are opened.

## Preferred design

Use a deformable-object sequence with a state representation that supports repeated correlated
pseudo-measurements over time. A rope or cloth manipulation episode is preferable to a static
centroid benchmark because the downstream task can change during one sequence.

At each update:

1. build the registered recursive task closure from the source-side model only;
2. form the full correlated innovation factor;
3. compress that factor for the recursive task state;
4. run the full and compressed filters on the same observations;
5. choose the downstream query/action from a frozen task family after the factor has been
   produced; and
6. record posterior and action parity, retained rank, factor bytes, and update time.

## Primary endpoints

- maximum task-posterior mean difference over the complete trajectory;
- maximum task-posterior covariance difference;
- fraction of controller decisions identical to the full estimator;
- full shared-factor rank versus recursive task-state rank versus retained factor rank;
- fraction of updates that require exact full-factor fallback; and
- representation bytes and update latency, with a direct fixed-query cache retained as a control.

## Required controls

- current-query-only compression, which should be allowed to fail recursively;
- equal-rank spectral/PCA truncation;
- direct `(K, P)` cache for a single immutable query;
- deliberately closure-violating task/action family, for which the registered closure must expand
  or fall back rather than silently claim exactness.

## Decisive positive result

A strong result would show that observation/history dimension and full correlated-noise rank grow
substantially over a sequence while the registered recursive task closure stays small, and that the
closure-aware factor produces numerically identical task posteriors and controller decisions to the
full estimator throughout the sequence.

The main scientific value is not a small runtime gain. It is demonstrating that the exact
correlated-uncertainty dimension is governed by the recursively relevant task state rather than by
the complete measurement history.
