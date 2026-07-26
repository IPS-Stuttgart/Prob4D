# Unfused observation-factor bundle

Prob4D normally produces fused trajectories for reconstruction benchmarks. A
Bayesian physical-twin update needs a second interface: individual window and
view factors must remain separate so that uncertain `Sim(3)` gauges, shared
backbone dependence, association reliability, and causal timing are not erased
before inference.

`prob4d.observation_factors` implements this interface as schema version 2.
The bundle contains:

- local 3-D points, full local covariance, material/association IDs, and optional
  viewing rays for every factor;
- one `GaugeEstimate` for every local window gauge;
- view, window, frame, and correlation-group provenance;
- an explicit causal frame limit;
- a checksum-bound JSON manifest and non-pickled NPZ payload.

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

bundle = ObservationFactorBundle(
    sequence_id="double_stretch_sloth-camera0",
    factors=(factor_camera0_window3, factor_camera1_window3),
    gauges=(gauge_window3_camera0, gauge_window3_camera1),
    source_revision=motioncrafter_commit,
    causal_frame_limit=133,
    metadata={"metric_anchor_used": False},
)

manifest, payload = write_observation_factor_bundle(
    bundle,
    "outputs/double_stretch_sloth/observation_factors.json",
)
stacked = bundle.stack()
```

`stacked.gauge_jacobian` has one seven-dimensional block per gauge. Rows from
another gauge are exactly zero in that block. `correlation_group_ids` are not
interpreted by Prob4D; they are retained for the consuming Bayesian estimator to
cap shared-backbone information.

## Information boundary

Every factor must satisfy `frame_index <= causal_frame_limit`, and all factors in
a bundle must use the same limit. The contract does not authorize a predictor to
read later RGB, point maps, target tracks, or outcome metrics. Reconstruction
controls that use all frames must be written to a separately labelled bundle.
