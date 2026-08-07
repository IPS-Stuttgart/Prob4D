# Sparse explicit-gauge observation-factor stacks

`ObservationFactorBundle` schema v4 preserves one local `Sim(3)` gauge per
prediction window and one ordered joint gauge covariance over all windows. The
historical in-memory `bundle.stack()` adapter expands every selected observation
row into a dense gauge Jacobian with shape

```text
M x 3 x 7K
```

where `M` is the selected observation count and `K` is the number of gauges.
Exactly one seven-column block is nonzero in each row. This is convenient for
small experiments, but its Jacobian storage grows as `O(MK)`.

`prob4d.sparse_observation_factors` provides an additive representation with:

```text
local_gauge_jacobian: M x 3 x 7
gauge_indices:        M
gauge_prior:           7K x 7K
```

The representation is mathematically equivalent to the dense stack and reduces
the row-design storage to `O(M)`. It does not change the neutral serialized
factor-bundle schema or its content identity.

## Usage

```python
from prob4d.provider_v2 import load_observation_factor_bundle
from prob4d.sparse_observation_factors import (
    stack_sparse_observation_factors,
)

bundle = load_observation_factor_bundle("outputs/case-a/factors.json")
stacked = stack_sparse_observation_factors(bundle)

# Apply one explicit gauge perturbation without constructing zero blocks.
predicted_gauge_displacement = stacked.apply_gauge_delta(gauge_delta)

# Materialize the historical dense design only for a compatibility boundary.
dense_design = stacked.dense_gauge_jacobian()
```

The sparse adapter preserves:

- conditional and marginal world covariance separately;
- the complete joint cross-window gauge prior;
- association probability;
- source-side prior reliability;
- nominal-component probability;
- composite information weight;
- row, factor, view, correlation-group, frame, and gauge identities; and
- the exclusive causal frame stop.

## Covariance rule

An estimator that keeps gauge errors explicit must use:

```text
conditional_world_covariance_m2
+ local_gauge_jacobian
+ gauge_indices
+ gauge_prior_covariance
```

It must not additionally use `marginal_world_covariance_m2`, because that would
count `J Sigma_gg J^T` twice. The method
`gauge_marginal_covariance_m2()` reconstructs that per-row contribution for
parity tests and diagnostics.

## Sparse gauge prior

For the production causal spanning tree, the dense prior can now be represented
by `prob4d.gauge_tree_prior.GaugeTreeSquareRootPriorV1`. It stores one parent,
`7 x 7` transition, and `7 x 7` innovation Cholesky factor per gauge. The
representation applies `Sigma_gg`, its inverse, selected marginals, and
`H Sigma_gg H^T` without expanding either the row design or the complete gauge
prior.

An existing dense bundle can be converted only when its declared parent order is
available:

```python
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1

prior = GaugeTreeSquareRootPriorV1.from_dense_covariance(
    gauge_ids=stacked.gauge_ids,
    parent_indices=parent_indices,
    joint_covariance=stacked.gauge_prior_covariance,
)

complete_covariance_action = prior.marginal_observation_covariance_action(
    stacked.local_gauge_jacobian,
    stacked.gauge_indices,
    stacked.conditional_world_covariance_m2,
    residual_direction,
)
```

The converter reconstructs and verifies the entire dense covariance. It fails
closed for a wrong parent tree, singular factors, or off-tree conditional
dependence. See [sparse square-root gauge-tree prior](sparse-gauge-tree-prior.md)
for the construction and operation contracts.

The current serialized schema-v4 factor bundle still carries the dense prior.
The new backend therefore reduces retained execution storage after direct sparse
construction or verified conversion, but does not yet reduce artifact size or
initial dense-loading peak. Eliminating the dense serialized prior requires a
future provider or artifact version.

## Compatibility and claim boundary

The existing `StackedObservationFactors`, factor-bundle schema v4, provider-v1
surface, provider-v2 artifacts, and content addresses remain unchanged. Both
sparse adapters are additive in-memory execution representations.

Lower memory use and exact dense parity are engineering properties. They do not
establish point-covariance calibration, physical-state identifiability, safe
BayesianPhysTwin updates, or improved Causal4D predictions.
