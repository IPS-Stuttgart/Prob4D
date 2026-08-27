# Persistent-point prediction windows

`prob4d.persistent_point_prediction` is an additive experimental artifact for
providers whose native output is a fixed sparse scene-point axis rather than a
dense image-grid point map.

It was introduced to resolve the source-side representation gate in the
PointWorld–Flat'n'Fold qualification protocol without rasterizing PointWorld
predictions onto a target-dependent RGB-D grid.

## Source-side PointWorld finding

At PointWorld revision `05484826dfef74cbe278a3974179a5a16705d35d`, the
released model:

- obtains `scene_coord0` from the first scene-point frame;
- predicts a `B x T x N_scene x 3` relative displacement tensor;
- forms absolute predicted positions as `scene_coord0 + displacement`; and
- emits a `B x T x N_scene x 1` log-variance tensor.

Consequently, the `N_scene` axis is a persistent provider seed axis within one
forecast window. It should be preserved, not reshaped into `H x W`.

The uncertainty head is trained against normalized initial-frame-relative
scene displacement. Its output is therefore **not** a calibrated metric point
covariance. The artifact retains the raw log variance together with the SHA-256
identity of the normalization statistics that give it meaning.

## Contract

```python
from prob4d.persistent_point_prediction import (
    persistent_point_window_from_pointworld,
)

window = persistent_point_window_from_pointworld(
    window_id="garment-07-demo-03-window-000",
    frame_indices=absolute_output_frames,
    scene_positions=pointworld_output["scene_flows"][0].cpu().numpy(),
    scene_valid_mask=scene_valid_mask[0].cpu().numpy(),
    reported_log_variance=pointworld_output["log_var"][0].cpu().numpy(),
    normalization_id=normalization_statistics_sha256,
)
window.to_npz("window-000.persistent-points.npz")
```

The version-1 archive stores:

- exact absolute output-frame indices;
- source point-axis indices;
- deterministic window-scoped integer point identities;
- `T x N x 3` point positions;
- `T x N` validity;
- explicit position and identity semantics; and
- optional raw provider log variance, its semantics, and its normalization ID.

Arrays are defensively owned and read-only after validation. The NPZ reader
rejects unknown fields, missing fields, dtype drift, malformed identities,
nonfinite active observations, incomplete uncertainty metadata, and schema drift.

## Identity boundary

PointWorld seed index `n` is persistent across frames **within one forecast
window**. The adapter hashes `(window_id, source_point_index)` into a positive
`int64` identifier so the same numeric source index in two different windows is
not silently interpreted as the same material point.

This does not establish cross-window correspondence. A future cross-window
association layer must declare and validate that relationship separately.

## Calibration boundary

`reported_log_variance` is retained exactly as provider output. It must not be
passed directly to `ObservationFactor.local_covariance_m2`.

Before downstream use, a source/calibration-only step must establish:

1. the exact PointWorld checkpoint and normalization-statistics identities;
2. a mapping from normalized displacement log variance to conditional metric
   point covariance, including anisotropy assumptions;
3. dependence groups for points, frames, stochastic members, and overlapping
   windows; and
4. held-out coverage, proper-score, and covariance-width behavior.

A valid negative calibration result retains the raw predictions but does not
authorize uncertainty-weighted fusion.

## Remaining PointWorld qualification work

This artifact resolves only the representation-shape question. Issue #333 still
requires source-only completion of:

- checkpoint and runtime binding;
- Flat'n'Fold dataset-byte and garment-roster inventory;
- three-camera timestamp and geometry verification;
- Baxter action conversion into PointWorld robot point flows;
- metric-frame verification;
- visual and action source-lineage semantics;
- covariance/reliability calibration design; and
- `ProviderSupportFeasibilityV1` evaluation before target outcomes.

No target garment outcome, BayesianPhysTwin innovation, or Causal4D result is
needed or permitted for these steps.

## Claim boundary

A valid persistent-point archive proves shape, identity, dtype, and semantic
interoperability for the exact bytes it contains. It does not establish provider
accuracy, calibrated uncertainty, cross-window material identity, Prob4D fusion
benefit, BayesianPhysTwin benefit, Causal4D benefit, or deployment safety.
