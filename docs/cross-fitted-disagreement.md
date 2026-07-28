# Cluster-cross-fitted overlap disagreement

Prob4D normally estimates a relative `Sim(3)` gauge from overlapping
MotionCrafter windows and then evaluates their residual disagreement. That
in-sample residual is useful as an online consistency cue, but it can be
optimistic because the same correspondences helped fit the gauge.

`accumulate_cross_fitted_disagreement` provides an additive source-side
diagnostic. It partitions overlap rows into whole frame-by-spatial-tile
clusters, refits the relative gauge without one fold, and evaluates residual
energy only on the held-out clusters.

```python
from prob4d import accumulate_cross_fitted_disagreement

evidence, report = accumulate_cross_fitted_disagreement(
    windows,
    alignments,
    folds=4,
    cluster_size=32,
    maximum_training_correspondences=100_000,
    seed=0,
)

print(report.to_dict())
```

## Information boundary

The fit for a held-out fold cannot read any point in that fold. Fold assignment
is deterministic for a fixed seed, derives its random stream from the alignment
identity rather than list position, and uses the same frame-by-spatial-tile
clustering concept as the dense gauge-covariance estimator.

When a fold has fewer than four usable training correspondences, rank-deficient
geometry, or a failed numerical fit, Prob4D does **not** substitute the original
in-sample alignment. The affected rows retain zero evidence count. The returned
report records fitted and skipped folds, skipped alignments, overlap rows, and
the strictly out-of-fold evaluated fraction.

## Relationship to the existing uncertainty model

The returned values use the same `DisagreementEvidence` representation consumed
by `DepthDisagreementModel`. They can therefore be compared directly with the
ordinary in-sample result:

```python
from prob4d.uncertainty import accumulate_disagreement

in_sample = accumulate_disagreement(windows, alignments)
cross_fitted, report = accumulate_cross_fitted_disagreement(
    windows,
    alignments,
)
```

Cross-fitted evidence changes the statistical meaning and scale of the
disagreement feature. Existing gauge and point-uncertainty calibration artifacts
were fitted for the current production semantics and must not be reused
silently. Before cross-fitted evidence enters a claim-bearing provider export:

1. regenerate point-uncertainty calibration on disjoint sequence families;
2. bind the fold count, cluster size, seed rule, and sampling cap into the
   calibration compatibility record;
3. compare in-sample and cross-fitted coverage, likelihood, and selective risk;
4. preserve provider-v1 and frozen provider-v2 artifact reproduction.

Until those gates pass, this API is a diagnostic and development surface rather
than a new production default.
