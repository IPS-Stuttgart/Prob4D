# Sparse square-root gauge-tree prior

`GaugeTreeSquareRootPriorV1` is an additive execution backend for the production
causal spanning-tree gauge posterior. It replaces the dense in-memory
`7K x 7K` covariance with `O(K)` parent, transition, and innovation factors while
preserving the exact joint Gaussian statistics.

The separately versioned portable artifact in `prob4d.gauge_tree_artifact` stores
the same factors without materializing the dense prior. It does **not** change
provider-v2, `ObservationFactorBundle` schema v4, or any frozen artifact identity.
Current schema-v4 factor bundles still carry the dense compatibility covariance;
the sparse artifact is an additive hand-off for prospective workflows.

## Model

For ordered seven-dimensional linearized `Sim(3)` gauge errors,

```text
g_0 = e_0
g_i = F_i g_parent(i) + e_i
```

where all innovations are mutually independent and

```text
e_i ~ Normal(0, L_i L_i^T).
```

The backend stores:

```text
parent_indices:         K
transition_matrices:    K x 7 x 7
innovation_scale_tril:  K x 7 x 7
```

`innovation_scale_tril[0]` is the metric-root prior Cholesky factor. The root
transition is exactly zero. Every other parent must precede its child, so the
stored order is a causal topological order.

For the existing production propagation, the factors are available before a
dense covariance is formed:

```text
F_child = parent composition Jacobian
Q_child = relative Jacobian
          @ relative gauge covariance
          @ relative Jacobian.T
L_child = cholesky(Q_child)
```

Construct that route directly with:

```python
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1

prior = GaugeTreeSquareRootPriorV1.from_transition_covariances(
    gauge_ids=gauge_ids,
    parent_indices=parent_indices,
    transition_matrices=transition_matrices,
    innovation_covariances=innovation_covariances,
)
```

No jitter, eigenvalue clipping, pseudoinverse, or hidden regularization is used.
A non-positive-definite root or innovation covariance fails admission.

## Safe migration from the current dense contract

An existing schema-v4 bundle can be converted after loading when its declared
lineage is the production causal tree:

```python
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1

prior = GaugeTreeSquareRootPriorV1.from_dense_covariance(
    gauge_ids=stacked.gauge_ids,
    parent_indices=parent_indices,
    joint_covariance=stacked.gauge_prior_covariance,
)
```

The converter derives each exact Gaussian child-on-parent conditional, rebuilds
the complete joint covariance through the tree, and compares every block with
the supplied dense matrix. It rejects:

- an incorrect parent order;
- a covariance with non-tree conditional dependence;
- a singular parent marginal;
- a non-positive-definite root or innovation covariance; and
- any dense reconstruction mismatch beyond the explicit parity tolerance.

The resulting prior binds the canonical dense source digest. Call
`verify_dense_covariance(..., require_source_digest=True)` to require the same
canonical dense matrix and exact tree parity again.

The experimental full-joint covariance-intersection graph is not generally a
tree-structured Gaussian. It must not be forced into this representation; the
strict converter will reject it when its off-tree conditional structure is
nonzero.

## Portable sparse artifact

Publish the normalized tree factors as a JSON manifest plus compressed NPZ
payload:

```python
from prob4d.gauge_tree_artifact import (
    load_gauge_tree_prior_artifact,
    save_gauge_tree_prior_artifact,
)

manifest_path, payload_path = save_gauge_tree_prior_artifact(
    "outputs/gauge-prior.json",
    prior,
)

recovered = load_gauge_tree_prior_artifact(
    manifest_path,
    expected_prior_id=prior.prior_id,
)
```

The writer is no-clobber and idempotent. Repeating a write for the same semantic
prior reuses the validated files; attempting to replace either path with a
different prior fails. If publication stops after the payload is complete but
before the manifest is written, a repeated call verifies and reuses that orphan
payload before publishing the manifest.

The manifest binds:

- schema, version, representation semantics, and ordered gauge identities;
- the complete parent list and canonical descriptors of both factor tensors;
- the optional dense-source covariance digest;
- factor and dense-equivalent storage accounting;
- the semantic `prior_id` and artifact identity; and
- the relative payload path, exact payload SHA-256, fixed array inventory, and
  `allow_pickle=false` declaration.

The artifact identity is computed from the normalized semantic prior and fixed
storage contract, not from ZIP timestamps or the local path. Two valid transports
of the same prior therefore have the same identity. The NPZ SHA-256 remains a
separate byte-level transport-integrity check.

Loading rejects duplicate JSON keys, non-finite constants, unknown manifest or
payload fields, path traversal, symbolic links, checksum drift, missing or extra
arrays, changed array descriptors, mismatched storage accounting, and an
unexpected `prior_id`. Loaded factors pass through the normal immutable prior
constructor before they are returned.

## Matrix-free operations

The generative tree gives `Sigma = T D T^T`, where `D` is block diagonal. A
reverse pass applies `T^T`, independent square-root blocks apply `D`, and a
forward pass applies `T`. Consequently:

```python
sigma_rhs = prior.covariance_action(rhs)
solved_information = prior.solve_information(rhs)
lambda_rhs = prior.information_action(rhs)
solved_covariance = prior.solve_covariance(rhs)
```

All four paths operate on one vector or several right-hand sides without forming
the complete covariance or information matrix. The prior also provides:

- exact log determinant and information quadratic;
- deterministic-seed sampling;
- all marginal `7 x 7` covariance blocks in one forward pass;
- selected and cross-covariances for named gauges;
- a guarded dense compatibility conversion; and
- an immutable content identity over order, parents, factors, semantics, and
  optional source binding.

## Sparse observation-space covariance

`SparseStackedObservationFactors` already stores one `3 x 7` Jacobian and one
gauge index per row. The new prior applies the shared gauge contribution without
expanding either the row design or the gauge covariance:

```python
gauge_action = prior.observation_covariance_action(
    stacked.local_gauge_jacobian,
    stacked.gauge_indices,
    residual_direction,
)

complete_action = prior.marginal_observation_covariance_action(
    stacked.local_gauge_jacobian,
    stacked.gauge_indices,
    stacked.conditional_world_covariance_m2,
    residual_direction,
)
```

The second method applies

```text
R + H Sigma_gg H^T
```

where `R` is the row-block-diagonal conditional local covariance. It
deliberately does not consume `marginal_world_covariance_m2`, because that would
count the same gauge uncertainty twice.

## Storage behavior

For float64 factors, retained numerical storage is approximately

```text
K * (8 + 2 * 7 * 7 * 8) bytes,
```

rather than

```text
(7K)^2 * 8 bytes.
```

At 64 gauges this is about 50 KiB instead of 1.53 MiB, before Python-container
overhead. The advantage grows linearly with `K`. Dense materialization remains
available only behind an explicit maximum-gauge guard.

Direct sparse construction plus the portable artifact avoids dense prior storage
both in memory and in the standalone file. A verified conversion from an existing
dense bundle still incurs the original bundle load once. Serialized schema-v4
observation-factor bundles continue to carry the dense covariance for frozen
compatibility, so their file size and initial load peak are unchanged.

## Claim boundary

Exact dense parity, lower memory use, portable factor persistence, and
matrix-free algebra are engineering properties. They do not establish provider
competence, covariance calibration, physical-query identifiability, safe
BayesianPhysTwin acceptance, Causal4D intervention benefit, deployment safety,
or state of the art. Promotion requires the separately frozen independent-object
provider and guarded-query experiment.
