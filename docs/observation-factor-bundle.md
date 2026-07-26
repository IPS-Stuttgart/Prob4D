# Unfused observation-factor bundle

Prob4D normally produces fused trajectories for reconstruction benchmarks. A
Bayesian physical-twin update needs a second interface: individual window and
view factors must remain separate so that uncertain `Sim(3)` gauges, shared
backbone dependence, reliability, and causal timing are not erased before
inference.

`prob4d.observation_factors` implements this interface as schema version 3. The
bundle contains:

- local 3-D points, full local covariance, material/association IDs, and optional
  viewing rays for every factor;
- one `GaugeEstimate` for every local window gauge;
- separate association probability and residual-independent prior reliability;
- correlation-group nominal probabilities and composite-likelihood weights;
- view, window, frame, and correlation-group provenance;
- one explicit **exclusive** causal frame stop;
- a checksum-bound JSON manifest and non-pickled NPZ payload.

## Reliability boundary

The contract deliberately keeps four quantities separate:

- `association_probability` describes support for the named entity or material
  point;
- `prior_reliability` is source-side evidence that a row is nominal and must not
  depend on a downstream physical innovation;
- `prior_nominal_probability` is the fixed nominal-component prior for a
  correlation group;
- `composite_weight` limits that group's contribution when rows, windows, or
  pixels are dependent.

Factors sharing one `correlation_group_id` must use the same nominal probability
and composite weight. Stacking repeats those group values row-for-row without
multiplying them into association probability. A zero-reliability row is not
selected merely because its association probability is high.

## Conditional and marginal covariance

For a local point `p` and gauge vector `g`, Prob4D exports the linearization

```text
y = Sim3(g) p
J_g = dy / dg.
```

The stack deliberately exposes two covariance products:

- `conditional_world_covariance_m2` transforms only the local point covariance;
- `marginal_world_covariance_m2` additionally contains
  `J_g Sigma_g J_g^T`.

A downstream estimator that keeps gauge errors as explicit nuisance variables
must use the **conditional** covariance together with `gauge_jacobian` and
`gauge_prior_covariance`. Using the marginal covariance as well would count
uncertain gauge variation twice. The marginal covariance remains useful for a
consumer that does not estimate gauges explicitly.

## Example

```python
from prob4d.observation_factors import (
    ObservationFactor,
    ObservationFactorBundle,
    write_observation_factor_bundle,
)

factor = ObservationFactor(
    factor_id="camera0-window3-frame132",
    frame_index=132,
    view_id="camera0",
    window_id="window3",
    gauge_id="window3",
    point_ids=point_ids,
    points_local_m=points_local_m,
    valid_mask=valid_mask,
    local_covariance_m2=local_covariance_m2,
    association_probability=association_probability,
    prior_reliability=overlap_reliability,
    prior_nominal_probability=0.9,
    composite_weight=0.25,
    correlation_group_id="shared-backbone-frame132",
    causal_frame_stop=134,
)

bundle = ObservationFactorBundle(
    sequence_id="double_stretch_sloth-camera0",
    factors=(factor,),
    gauges=(gauge_window3_camera0,),
    source_revision=motioncrafter_commit,
    causal_frame_stop=134,
    metadata={"metric_anchor_used": False},
)

manifest, payload = write_observation_factor_bundle(
    bundle,
    "outputs/double_stretch_sloth/observation_factors.json",
)
stacked = bundle.stack()
```

`stacked.gauge_jacobian` has one seven-dimensional block per gauge. Rows from
another gauge are exactly zero in that block. The stacked association,
reliability, nominal-probability, and composite-weight arrays remain separate
inputs for the consuming Bayesian estimator.

## Information boundary

Every factor must satisfy

```text
frame_index < causal_frame_stop
```

and every factor in a bundle must use the same exclusive stop. This convention
matches `ObservationBeliefV1` and avoids adapter-specific `+1` conversions.
The contract does not authorize a predictor to read later RGB, point maps,
target tracks, or outcome metrics. Reconstruction controls that use all frames
must be written to a separately labelled bundle.

## Schema-v2 migration

The loader accepts legacy schema-v2 manifests, whose `causal_frame_limit` was
inclusive. It converts them deterministically to
`causal_frame_stop = causal_frame_limit + 1`. Because v2 had no distinct prior
reliability or group weighting fields, those values are conservatively restored
as one and the migration is recorded in bundle metadata. New writes always use
schema v3; the legacy inclusive aliases remain read-only compatibility
properties.
