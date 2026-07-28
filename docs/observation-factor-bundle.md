# Unfused observation-factor bundles

Prob4D normally fuses overlapping windows into one reconstruction. A Bayesian
physical-twin update also needs an unfused interface: individual window and view
factors must remain separate so uncertain `Sim(3)` gauges, shared-backbone
dependence, reliability, and causal timing are not erased before inference.

`prob4d.observation_factors` exposes two deliberately distinct contracts:

- `ObservationFactorBundle` is the frozen schema-v3 compatibility surface. It
  stores one marginal covariance per gauge and stacks those marginals into a
  block-diagonal prior.
- `JointObservationFactorBundle` is schema v4 for new explicit-gauge work. It
  additionally stores one ordered full covariance over all gauge parameters,
  including cross-window blocks.

Provider v1 continues to advertise schema v3 for exact reproduction. Provider v2
advertises schema v4 and the
`joint_observation_factor_gauge_covariance` capability. The loader accepts schema
versions 2, 3, and 4 without reinterpreting an older artifact as having covariance
that it never carried.

## Contents

Both bundle types contain:

- local 3-D points, full local covariance, material or association IDs, and
  optional viewing rays for every factor;
- one `GaugeEstimate` for every local window gauge;
- separate association probability and residual-independent prior reliability;
- correlation-group nominal probabilities and composite-likelihood weights;
- case, stream, sequence, repository, revision, view, window, frame, and
  correlation-group provenance;
- one explicit **exclusive** causal frame stop; and
- a checksum-bound JSON manifest with a non-pickled NPZ payload.

Schema v4 additionally contains `joint_gauge_covariance` in the exact order of
`gauges`. Its manifest binds that order and the semantics identifier
`ordered-full-cross-window-covariance-v1`.

## Joint gauge covariance

For ordered gauge perturbations

```text
delta_g = [delta_g_0, ..., delta_g_(K-1)],
```

schema v4 carries

```text
P_g = Cov(delta_g) in R^(7K x 7K).
```

Every diagonal `7 x 7` block of `P_g` must equal the covariance in the
corresponding `GaugeEstimate`. Construction and loading fail when the matrix is
non-finite, asymmetric, non-positive-semidefinite, has the wrong dimension, uses
a different gauge order, or disagrees with a marginal block.

The redundancy is intentional. A factor can still be linearized against its local
`GaugeEstimate`, while stacking returns the complete nuisance prior without
silently replacing shared anchor and upstream-window uncertainty by independent
marginals.

For stacked conditional covariance `R` and gauge Jacobian `J`, a consumer that
marginalizes gauges obtains

```text
Cov(y) = R + J P_g J^T.
```

The off-diagonal blocks of this observation covariance are generally nonzero.
They are lost by the schema-v3 block-diagonal approximation.

## Conditional and marginal covariance

For a local point `p` and gauge vector `g`, Prob4D exports the linearization

```text
y = Sim3(g) p
J_g = dy / dg.
```

The stack exposes two row-level covariance products:

- `conditional_world_covariance_m2` transforms only local point covariance;
- `marginal_world_covariance_m2` additionally contains the row's marginal
  `J_g Sigma_g J_g^T` contribution.

A downstream estimator that keeps gauge errors as explicit nuisance variables
must use the **conditional** covariance together with `gauge_jacobian` and
`gauge_prior_covariance`. Adding the marginal covariance again would double-count
gauge uncertainty. A consumer that eliminates the gauges should use the complete
stacked expression above rather than treating row marginals as independent.

## Reliability boundary

The contract keeps four quantities separate:

- `association_probability` describes support for the named entity or material
  point;
- `prior_reliability` is source-side evidence that a row is nominal and must not
  depend on a downstream physical innovation;
- `prior_nominal_probability` is the fixed nominal-component prior for a
  correlation group; and
- `composite_weight` limits that group's contribution when rows, windows, or
  pixels are dependent.

Factors sharing one `correlation_group_id` must use the same nominal probability
and composite weight. Stacking repeats those values row-for-row without
multiplying them into association probability. A zero-reliability row is not
selected merely because its association probability is high.

## Schema-v4 example

```python
from prob4d.provider_v2 import (
    JointObservationFactorBundle,
    write_observation_factor_bundle,
)

bundle = JointObservationFactorBundle(
    sequence_id="double_stretch_sloth-camera0-prefix",
    case_id="double_stretch_sloth",
    stream_id="prob4d:motioncrafter-points:camera0",
    factors=tuple(factors),
    gauges=tuple(ordered_gauges),
    joint_gauge_covariance=joint_gauge_covariance,
    source_repository="FlorianPfaff/Prob4D",
    source_revision=prob4d_commit,
    causal_frame_stop=134,
    metadata={
        "upstream_repository": "TencentARC/MotionCrafter",
        "upstream_revision": motioncrafter_commit,
        "metric_anchor_used": True,
    },
)

manifest, payload = write_observation_factor_bundle(
    bundle,
    "outputs/double_stretch_sloth/observation_factors.json",
)
stacked = bundle.stack()
```

`stacked.gauge_jacobian` has one seven-dimensional block per gauge.
`stacked.gauge_prior_covariance` is the exact joint matrix supplied above. The
association, reliability, nominal-probability, and composite-weight arrays remain
separate inputs for the Bayesian estimator.

## Identity and causal boundary

`case_id` names the physical case consumed downstream, while `stream_id`
identifies the observation stream within that case. `sequence_id` remains the
producer-side bundle identity. `source_repository` and `source_revision` identify
the exact producer implementation.

Every factor must satisfy

```text
frame_index < causal_frame_stop
```

and every factor in a bundle must use the same exclusive stop. The contract does
not authorize a predictor to read later RGB, point maps, target tracks, or outcome
metrics. Reconstruction controls that use all frames must be written to a
separately labelled bundle.

## Legacy migration

The loader accepts schema v2, whose `causal_frame_limit` was inclusive, and
converts it deterministically to
`causal_frame_stop = causal_frame_limit + 1`. Missing reliability and group
weights are conservatively restored as one and the migration is recorded.

Schema v3 remains a frozen compatibility representation. Loading it does **not**
claim that cross-window gauge covariance was preserved; stacking reproduces its
historical block-diagonal prior. New multi-window explicit-gauge experiments
should construct schema v4 from the producer's joint gauge posterior rather than
upgrading v3 marginals after the fact.
