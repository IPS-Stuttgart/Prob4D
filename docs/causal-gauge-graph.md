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
    initial_transform=metric_anchor.mean,
    initial_covariance=metric_anchor.covariance,
)
```

The posterior contains all gauge means and the complete joint covariance. The
report records every admitted edge, its parent, alignment index, CI weight,
analytic-Jacobian mode, and dependence semantics.

The graph mode is `causal_full_joint_ci_graph_v1`. It is not accepted by the
claim-bearing provider-v2 loader and must not be relabelled as the production
`sequential_joint_spanning_tree_v1` model.

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

It compares:

1. the provider-v2 causal spanning tree;
2. marginal multi-parent gauge CI;
3. the full-joint causal multi-edge graph; and
4. fixed-lag reconstruction control.

All four use the same independently calibrated dense uncertainty model and the
same covariance-intersection dense fusion. The output retains point, endpoint,
seam, drift, coverage, NLL, covariance-width, and gauge metrics from the normal
ablation contract, plus the complete graph report.

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
