# Causal scene-flow tracklets

`prob4d.causal_tracklets` turns one decoded MotionCrafter window into sparse
persistent 3-D observation identities without reading prediction payloads at or
after an exclusive causal cutoff.

The ordinary dense observation exporter assigns an entity ID from image-grid
position. That is appropriate for same-frame duplicate-window fusion, but a
fixed pixel is not a persistent material point on a deforming object. Tracklets
supply the missing within-window temporal identity layer.

## Builder

```python
from prob4d import build_causal_scene_flow_tracklets

tracklets, report = build_causal_scene_flow_tracklets(
    prediction_window,
    causal_frame_stop=134,
    seed_stride=8,
    search_radius_pixels=4,
    maximum_step_error_local=0.05,
    association_sigma_local=0.02,
    minimum_link_probability=0.05,
    minimum_track_length=3,
)
```

The algorithm:

1. keeps only decoded frames whose absolute frame index is below
   `causal_frame_stop`;
2. seeds a deterministic sparse grid in the first retained frame;
3. predicts each active local 3-D point with its scene-flow vector;
4. searches a bounded pixel neighborhood in the next retained decoded frame;
5. chooses the nearest 3-D candidate and scores geometric fit plus nearest-versus-
   second-nearest separation;
6. resolves collisions deterministically by link probability, geometric error,
   and original track ID;
7. drops tracks shorter than the declared minimum length.

The distance threshold and scale are in the prediction window's **local gauge
coordinates**. They are not metric thresholds until an uncertain `Sim(3)` gauge
maps the window into the metric world frame.

`association_probability` is the cumulative product of the accepted link
probabilities. It expresses support that the current row still belongs to the
same seeded identity. It is not source reliability and must not be replaced by a
physical residual or downstream posterior responsibility.

## Unfused factors

```python
from prob4d import tracklets_to_observation_factors

factors = tracklets_to_observation_factors(
    tracklets,
    structured_local_covariance,
    view_id="camera0",
    prior_reliability=source_only_reliability,
    effective_samples_per_group=64.0,
)
```

The converter emits one `ObservationFactor` per absolute frame. The same
`point_id` is retained across frames of one track. The factor keeps separate:

- persistent association probability;
- residual-independent prior reliability;
- conditional local point covariance;
- frame correlation group and composite weight;
- source window and gauge identity;
- exclusive causal frame stop.

The resulting factors can be combined with the window's `GaugeEstimate` and, for
several windows, the schema-v4 joint gauge covariance in an
`ObservationFactorBundle`.

## Information boundary

The builder never links the last retained frame to a frame at or after the
cutoff. Mutating later point maps, masks, or scene flow therefore cannot alter
the tracklet result.

This is an additive development surface, not a claim-bearing provider-v2
default. Before it is promoted into a physical-twin experiment:

1. calibrate the local link thresholds on source sequences only;
2. calibrate association quality separately from point uncertainty and source
   reliability;
3. bind the configuration and exact producer revision into the experiment
   manifest;
4. compare persistent tracklets with framewise pixel identities under the same
   Bayesian-PhysTwin guard;
5. retain exact fallback when the persistent observations do not support a
   physical update.

The current method preserves identity only within one decoded window. It does
not claim cross-window or cross-view material correspondence, and bounded pixel
search can terminate under large image motion or occlusion. Those failures
remain explicit through track length and termination diagnostics.
