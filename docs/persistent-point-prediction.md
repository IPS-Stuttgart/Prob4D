# Persistent sparse point-prediction windows

`prob4d.persistent_point_prediction` defines a strict, versioned payload for
providers whose output is a set of point trajectories rather than a dense image
grid.

The first consumer is PointWorld. Its released data pipeline samples scene
points at time zero and applies the same selected indices to every later
timestep. Its model then predicts one absolute trajectory for each retained
source point. Preserving those identities is more faithful than rasterizing the
output into `PredictionWindow`'s dense `T x H x W` grid.

## Contract

`PersistentPointPredictionWindow` retains:

- strictly increasing absolute output-frame indices;
- strictly increasing non-negative point IDs;
- absolute point positions with shape `T x N x 3`;
- one `T x N` validity mask;
- the number of context frames;
- fixed point-identity and trajectory semantics; and
- optional provider uncertainty with shape `T x N x 1` or `T x N x 3`.

Every retained point must be valid in the first context frame. The same point ID
therefore denotes the same seeded material/sample point throughout one window.
The identity scope is deliberately **window-local**. Cross-window association is
a separate scientific problem and is not invented by this archive.

The archive schema is:

```text
prob4d.persistent-point-prediction-window-npz
```

version `1`. It rejects missing or additional members, dtype drift, duplicate or
reordered point IDs, non-finite values, ambiguous uncertainty semantics, and
non-Boolean masks. Loaded arrays use immutable byte-backed storage.

## PointWorld export

The source-only adapter accepts a strict runtime snapshot with schema:

```text
prob4d.pointworld-persistent-point-source-npz
```

version `1`, containing:

```text
schema_name
schema_version
frame_indices          int64, shape (T,)
scene_flows            float32/float64, shape (T,N,3) or (1,T,N,3)
scene_exists           bool, shape (T,N) or (1,T,N)
log_var                float32/float64, shape (T,N,1|3) or batched equivalent
context_frame_count    int64 scalar
point_ids              optional int64, shape (N,)
```

Run:

```bash
prob4d prediction import-pointworld-sparse \
  pointworld-source-window-0000.npz \
  persistent-window-0000.npz \
  --window-id pointworld-window-0000 \
  --storage-dtype float32
```

The adapter removes only source-frame padding, sorts explicit point IDs while
reordering all trajectory fields identically, writes a no-clobber canonical
archive, reloads it, and checks exact array equality.

## Uncertainty boundary

PointWorld's released `log_var` is trained in its normalized relative-displacement
space. The adapter retains it under the explicit semantic label:

```text
pointworld-normalized-relative-log-variance-v1
```

This value is **not** called metric covariance, predictive coverage, or a
calibrated likelihood. A later source/calibration-only study must bind the exact
normalization statistics and determine whether the raw quantity supports a
calibrated conditional covariance. Until then, it is an identity-bound provider
uncertainty feature only.

## What this resolves

For the PointWorld–Flat'n'Fold source qualification, this contract resolves the
representation choice without target outcomes:

- no target-truth nearest-neighbor rasterization;
- no target-tuned interpolation or masks;
- no loss of within-window point identity; and
- no reinterpretation of PointWorld's raw log variance.

It does not yet establish:

- executable PointWorld support on the Flat'n'Fold inventory;
- Baxter-action conversion;
- metric coordinate consistency;
- cross-window point association;
- recursive fusion benefit;
- provider calibration;
- BayesianPhysTwin benefit; or
- Causal4D benefit.

Those remain separate source-qualification and held-out gates.
