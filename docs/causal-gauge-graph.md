# Correlation-aware causal multi-edge gauge graph

The claim-bearing provider-v2 estimator retains one causal parent edge for every
new MotionCrafter window. That spanning tree is transparent and preserves a full
joint cross-window covariance, but it discards other prefix-valid overlap edges.
Prob4D now provides a separately labelled experimental graph that admits all such
edges without treating their shared frames or shared visual backbone as
independent evidence.

## Full-joint covariance intersection

Assume the first `k` gauges have one joint Gaussian approximation

```text
x_0:k-1 ~ N(m, P).
```

Each prefix-valid edge from an earlier parent `p` to the new child `k` produces
an augmented candidate

```text
[x_0:k-1, x_k] ~ N(m_p, P_p).
```

`P_p` carries the complete previous covariance `P`, the analytic Sim(3)
composition cross-covariance between the parent and every previous gauge, and
the edge covariance propagated through the relative transform. Candidates share
prior, frames, weights, and often most correspondences, so the graph fuses the
**complete augmented distributions** with covariance intersection:

```text
P_CI^-1 = sum_j w_j P_j^-1,
sum_j w_j = 1,
w_j >= 0.
```

The weights minimize joint log determinant through deterministic simplex
coordinate searches. Information matrices are explicitly symmetrized and every
candidate and posterior covariance is validated positive semidefinite. Sim(3)
axis-angle coordinates fail closed near the SO(3) logarithm branch cut.

This is materially different from treating edge residuals as independent graph
factors. Repeated shared prior information is not summed with unit weight, and no
independence assumption is made between edges.

## API

```python
from prob4d import estimate_causal_multi_edge_gauge_graph

posterior, report = estimate_causal_multi_edge_gauge_graph(
    windows,
    alignments,
    initial_transform=metric_anchor.global_from_local,
    initial_covariance=metric_anchor.covariance,
)
```

The posterior contains all gauge means and the complete joint covariance. The
report records every admitted edge, its parent, alignment index, CI weight,
analytic-Jacobian mode, and dependence semantics.

The graph mode is `causal_full_joint_ci_graph_v1`. It is not accepted by the
claim-bearing provider-v2 loader and must not be relabelled as the production
`sequential_joint_spanning_tree_v1` model.

## Source-only cycle gate and exact fallback

Full-joint covariance intersection protects covariance against unknown cross-edge
correlation, but its weights are optimized from covariance rather than disagreement
between the candidate means. An overconfident inconsistent overlap can therefore
still move the graph mean. The optional guarded API uses the existing directed
three-window cycle audit before the graph is admitted:

```python
from prob4d import estimate_guarded_causal_multi_edge_gauge_graph

posterior, guarded_report = estimate_guarded_causal_multi_edge_gauge_graph(
    windows,
    alignments,
    initial_transform=metric_anchor.global_from_local,
    initial_covariance=metric_anchor.covariance,
    maximum_cycle_displacement=0.025,
    representative_radius=0.5,
    minimum_cycles_per_multi_edge_child=1,
)
```

The cycle score compares each direct `Sim(3)` edge with every available two-edge
path on the origin and the positive and negative coordinate axes at the declared
representative radius. It is an unnormalized source-side displacement, not a
chi-square statistic: overlap edges have unavailable cross-covariance because they
share frames, correspondences, and model weights. The displacement and
representative radius are expressed in the reference gauge units; calibration
and target cases therefore require the same metric anchor or frozen scale
convention.

The displacement threshold, representative radius, and minimum cycle count for
every multi-edge child must be frozen from source or calibration groups before
target outcomes are opened. The guard admits the unchanged full-joint graph only
when every audited cycle passes and the declared minimum number of cycles is
available for every child with multiple parents. Otherwise it returns the exact
analytic-Jacobian production spanning tree for the complete case. It does not drop
the case, remove an inconvenient edge, partially retain graph updates, or inspect a
Bayesian-PhysTwin innovation. The report records the ordered window IDs, complete
cycle audit, per-child support counts, unsupported multi-edge children, fallback
reason, returned posterior mode, and optional admitted graph report.

The guarded mode is `guarded_causal_full_joint_ci_graph_v1`. Like the unguarded
graph, it is diagnostic-only and is not admitted by claim-bearing provider-v2
export or consumer loaders.

## Paired diagnostic

Run the registered four-method comparison with:

```bash
prob4d diagnostic gauge-graph \
  --predictions outputs/test/predictions.json \
  --truth data/test_truth.npz \
  --calibration-predictions outputs/calibration/predictions.json \
  --calibration-truth data/calibration_truth.npz \
  --output-dir outputs/gauge-graph-ablation
```

By default it compares:

1. the provider-v2 causal spanning tree;
2. marginal multi-parent gauge CI;
3. the full-joint causal multi-edge graph; and
4. fixed-lag reconstruction control.

Add the source-only guarded candidate with a preregistered threshold:

```bash
prob4d diagnostic gauge-graph \
  --predictions outputs/test/predictions.json \
  --truth data/test_truth.npz \
  --calibration-predictions outputs/calibration/predictions.json \
  --calibration-truth data/calibration_truth.npz \
  --maximum-cycle-displacement 0.025 \
  --cycle-representative-radius 0.5 \
  --minimum-cycles-per-multi-edge-child 1 \
  --output-dir outputs/gauge-graph-ablation
```

This adds a fifth paired row for the guarded graph or its exact tree fallback. All
methods use the same independently calibrated dense uncertainty model and the same
covariance-intersection dense fusion. The output retains point, endpoint, seam,
drift, coverage, NLL, covariance-width, and gauge metrics from the normal ablation
contract, plus the complete graph and guard reports.

## Promotion rule

The production spanning tree remains the default. The graph may be considered
for a future provider version only after a frozen held-out, group-balanced study
shows lower seam, drift, endpoint, or point error without material regression in:

- one-sided coverage shortfall;
- covariance width;
- worst-group calibration;
- BayesianPhysTwin harmful accepted updates; or
- exact-fallback behavior.

A negative result is complete and retains the simpler tree. Passing unit tests or
producing a smaller covariance is infrastructure or diagnostic evidence, not a
scientific benefit claim.
