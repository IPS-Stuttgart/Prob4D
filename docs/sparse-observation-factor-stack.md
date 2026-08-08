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
+ one gauge prior
```

It must not additionally use `marginal_world_covariance_m2`, because that would
count `J Sigma_gg J^T` twice. The dense sparse stack method
`gauge_marginal_covariance_m2()` reconstructs that per-row contribution for
parity tests and diagnostics.

## Tree-backed sparse stack

For the production causal spanning tree, the dense prior can be replaced after
exact verification by `GaugeTreeSquareRootPriorV1`. The tree-backed adapter keeps
only:

```text
local_gauge_jacobian: M x 3 x 7
gauge_indices:        M
parent_indices:        K
transition_matrices:   K x 7 x 7
innovation_scale_tril: K x 7 x 7
```

Load a portable prior artifact and bind it to a schema-v4 bundle through the
stable provider-v2 factor facade:

```python
from prob4d.gauge_tree_prior_io import load_gauge_tree_prior
from prob4d.provider_v2_factors import (
    stack_tree_sparse_observation_factors,
)

prior = load_gauge_tree_prior("outputs/case-a/gauge-tree-prior.json")
tree_stacked = stack_tree_sparse_observation_factors(bundle, prior)

# These operations use the tree factors and do not retain a dense prior.
gauge_response = tree_stacked.gauge_covariance_action(gauge_rhs)
observation_response = tree_stacked.marginal_observation_covariance_action(
    residual_direction
)
```

Before releasing the dense prior, `stack_tree_sparse_observation_factors`
requires:

- exact gauge-order equality;
- complete numerical equality between the tree covariance and the schema-v4
  joint covariance;
- source-covariance digest equality when the tree prior carries that digest; and
- row-marginal covariance equality under the tree's diagonal gauge blocks.

A mismatch fails closed. The returned `TreeSparseStackedObservationFactors`
reuses the already immutable row arrays but contains no
`gauge_prior_covariance` field. It exposes matrix-free covariance and information
actions, observation-space gauge covariance, complete marginal observation
covariance, and explicitly guarded dense materialization for compatibility.

Binding an already-created `SparseStackedObservationFactors` is also available:

```python
from prob4d.provider_v2_factors import bind_gauge_tree_prior

tree_stacked = bind_gauge_tree_prior(stacked, prior)
```

The caller may retain the old dense stack separately; the returned tree-backed
object does not reference its dense covariance.

## Constructing the sparse gauge prior

An existing dense bundle can be converted only when its declared parent order is
available:

```python
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1

prior = GaugeTreeSquareRootPriorV1.from_dense_covariance(
    gauge_ids=stacked.gauge_ids,
    parent_indices=parent_indices,
    joint_covariance=stacked.gauge_prior_covariance,
)
```

The converter reconstructs and verifies the entire dense covariance. It fails
closed for a wrong parent tree, singular factors, or off-tree conditional
dependence. Direct construction from transition and innovation factors plus the
portable prior artifact is the intended low-memory producer path. See
[sparse square-root gauge-tree prior](sparse-gauge-tree-prior.md) and
[portable gauge-tree prior artifacts](gauge-tree-prior-artifact.md).

## Remaining serialization boundary

The current serialized schema-v4 factor bundle still carries the dense prior.
Consequently, loading schema v4 and converting or binding it incurs the original
dense file and transient loading cost once. The tree-backed stack removes the
dense prior from the retained execution object; it does not retroactively change
the bundle bytes or their content identity.

Eliminating the dense prior before initial factor-bundle loading requires a
separately versioned provider or observation-factor contract. Such a future
contract should bind the portable prior artifact identity rather than silently
reinterpret schema v4.

## Compatibility and claim boundary

The existing `StackedObservationFactors`, `SparseStackedObservationFactors`,
factor-bundle schema v4, provider-v1 surface, provider-v2 artifacts, and content
addresses remain unchanged. The tree-backed stack is an additive execution
representation.

Exact dense parity, matrix-free operations, and lower retained memory are
engineering properties. They do not establish point-covariance calibration,
physical-state identifiability, safe BayesianPhysTwin updates, improved Causal4D
predictions, deployment safety, or state of the art.
