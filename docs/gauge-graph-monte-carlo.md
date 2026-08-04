# Causal gauge-graph Monte Carlo study

`prob4d diagnostic gauge-graph-monte-carlo` is a controlled, target-separated
study of the experimental multi-edge gauge graph and its source-only cycle guard.
It addresses three mechanistic questions before scarce physical-object target data
are opened:

1. Does using every prefix-valid overlap edge reduce gauge drift when edge errors
   are predominantly independent?
2. Does full-joint covariance intersection avoid the severe overconfidence that a
   naive independent-edge graph would produce under shared-frame and shared-model
   correlation?
3. Does a cycle threshold frozen on clean source data reject inconsistent skip
   edges and return the exact production spanning tree without introducing a
   target-informed edge-selection rule?

The study uses Prob4D's existing synthetic long-sequence generator. That generator
creates metric truth, drifting local `Sim(3)` gauges, overlapping windows,
anisotropic depth-dependent point noise, and a configurable shared-noise fraction.
The normal robust overlap alignment, cluster/fallback covariance, production tree,
sequential marginal covariance intersection, full-joint graph, and guarded graph
are then executed unchanged.

## Calibration and target separation

The guard threshold is not selected from target gauge error. Clean calibration
trials are generated with disjoint seeds and balanced across all registered clean
correlation levels. For each calibration trial, the study records only the maximum
source-side direct-versus-two-edge cycle displacement. The threshold is the frozen
higher empirical quantile of those trial maxima.

Target trials use a separate seed range. The registered default scenarios are:

| Scenario | Shared-noise correlation | Skip-edge outliers |
| --- | ---: | ---: |
| `independent_clean` | 0.00 | none |
| `correlated_clean` | 0.75 | none |
| `highly_correlated_clean` | 0.95 | none |
| `correlated_mild_outliers` | 0.75 | 25% at 0.10 translation units |
| `correlated_strong_outliers` | 0.75 | 25% at 0.30 translation units |
| `highly_correlated_strong_outliers` | 0.95 | 25% at 0.30 translation units |

An injected outlier changes one direct skip edge that has an available directed
two-edge path. Its reported covariance and overlap residual are deliberately left
unchanged. This represents an overconfident source inconsistency and tests the
cycle guard rather than a trivial residual threshold.

## Compared estimators

Every target trial compares:

- `tree`: the analytic-Jacobian claim-bearing single-parent spanning tree;
- `marginal_ci`: sequential multi-parent marginal gauge covariance intersection;
- `full_joint_graph`: all prefix-valid edges fused as complete augmented joint
  gauge distributions by covariance intersection; and
- `guarded_graph`: the same graph after the frozen source-only cycle gate, with
  exact whole-case production-tree fallback.

The first metric anchor uses the known synthetic first-window gauge with a small,
explicit covariance. No method receives later true gauges.

## Metrics

The primary mechanistic metric is endpoint root-mean-square displacement over the
origin and positive/negative representative axes. The report also includes:

- mean and 90th-percentile window displacement;
- displacement drift slope over window index;
- approximate seven-dimensional 95% ellipsoid coverage;
- normalized NEES;
- mixed-unit covariance trace after representative-radius normalization;
- paired endpoint difference from the production tree;
- endpoint win and harm rates relative to the tree;
- guarded-graph fallback rate;
- injected-outlier detection rate; and
- clean false-fallback rate.

All confidence intervals are deterministic trial bootstraps. Pairwise differences
are calculated within the same synthetic target trial before aggregation.

## Command

```bash
prob4d diagnostic gauge-graph-monte-carlo \
  --output-dir outputs/gauge-graph-monte-carlo \
  --calibration-trials 48 \
  --target-trials-per-scenario 128 \
  --threshold-quantile 0.95 \
  --representative-radius 1.0 \
  --bootstrap-resamples 2000 \
  --source-revision "$(git rev-parse HEAD)"
```

The command writes:

- `gauge_graph_monte_carlo.json`: complete configuration, calibration records,
  trial records, aggregate intervals, provenance, and report ID;
- `gauge_graph_monte_carlo.csv`: one aggregate row per scenario and method;
- `gauge_graph_monte_carlo_trials.csv`: one raw row per trial and method;
- `gauge_graph_monte_carlo.md`: compact human-readable results; and
- `SHA256SUMS`: checksums for the evidence files.

The pull-request workflow runs a smaller deterministic pilot. A larger replication
can use the same command on `workstation2`; changing the trial count is recorded in
the report and does not change estimator semantics.

## Interpretation and promotion boundary

A useful positive result would show a clean-scenario endpoint improvement for the
full-joint graph without a material coverage or covariance-width regression, while
the guarded method remains close to the graph on clean trials and falls back on
inconsistent target trials. A negative result is equally informative: it supports
retaining the simpler production tree and quantifies why extra overlap edges do not
help under the tested dependence model.

This study cannot establish held-out physical-object provider competence,
BayesianPhysTwin acceptance, physical-prediction improvement, harmful accepted
update control, or Causal4D intervention benefit. Those remain separate sealed
experiments with real calibration and target units.
