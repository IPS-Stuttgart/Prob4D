# Sparse persistent-point prediction windows

PointWorld does not emit a dense image-grid point map. Its released predictor
starts from a set of scene points and predicts one trajectory for every retained
scene-point index over a finite action-conditioned horizon. Forcing those outputs
into the dense `PredictionWindow` grid would introduce an interpolation and
identity assumption that the provider itself does not make.

Prob4D therefore provides an additive upstream artifact:

```python
from prob4d.sparse_prediction_window import SparsePredictionWindow
```

The contract is intentionally upstream of `ObservationFactorBundle`, calibrated
covariance, gauge fusion, and BayesianPhysTwin. It resolves representation and
provenance first; later source/calibration-only work must decide how the sparse
points become probabilistic observation factors.

## Contract

A `SparsePredictionWindow` retains:

- strictly increasing absolute forecast-frame IDs;
- unique nonnegative point IDs;
- one predicted 3-D position for every frame and point ID;
- a position-validity mask;
- explicit coordinate and identity semantics;
- optional provider-native uncertainty plus a separate uncertainty-validity mask;
- exact dense storage precision; and
- finite JSON metadata.

Every point ID must be valid in the seed frame. The ID denotes that same seeded
point throughout the window. This is only a **within-window** identity unless a
separate source-only association contract establishes more.

The versioned NPZ schema is
`prob4d.sparse-prediction-window-npz`, version 1. Loading rejects unknown or
missing fields, dtype drift, duplicate IDs, non-finite values, malformed masks,
uncertainty without semantics, and symbolic-link traversal. Writing is
no-clobber. A canonical content ID is computed from the semantics, arrays, and
canonical metadata rather than from incidental ZIP metadata.

## PointWorld source export

The dependency-light adapter is:

```python
from prob4d.pointworld_sparse_adapter import (
    write_pointworld_sparse_source_export,
    convert_pointworld_sparse_source_export,
)
```

A PointWorld-side runner writes the exact arrays it executed:

- `scene_coord0`: context scene points, shape `(N, 3)`;
- `predicted_displacement_from_context`: shape `(T, N, 3)`;
- `scene_exists`: context support, shape `(N,)`;
- `prediction_valid_mask`: shape `(T, N)`;
- `provider_log_variance`: shape `(T, N, 1)` or `(T, N, 3)`;
- absolute forecast-frame IDs;
- exact provider revision, checkpoint, loader, camera-geometry, and action IDs.

The first displacement must be exactly zero and the first validity row must equal
`scene_exists`. An inactive context point cannot become valid later. These checks
make the displacement-from-context and seeded-identity semantics explicit rather
than inferring them from tensor shape.

Convert without importing the PointWorld runtime:

```bash
python -m prob4d.pointworld_sparse_adapter convert \
  pointworld-source-window.npz \
  prob4d-sparse-window.npz

python -m prob4d.pointworld_sparse_adapter verify \
  prob4d-sparse-window.npz
```

The adapter filters unsupported context points while retaining their original
array indices as point IDs. Positions are reconstructed only as
`scene_coord0 + predicted_displacement_from_context`. No nearest-neighbor
projection, RGB-D rasterization, target-truth correspondence, or target-tuned
mask is used.

## Uncertainty boundary

PointWorld's native value is stored under the semantics
`pointworld-normalized-relative-flow-log-variance-uncalibrated-v1`. The context
frame is marked uncertainty-invalid because it is the conditioning state rather
than a forecast. The native value is not exponentiated into metric covariance and
is never relabelled as calibrated uncertainty.

A later Prob4D experiment must use source/calibration garments only to establish:

1. the normalization and coordinate transform from native flow error to metric
   position error;
2. anisotropy or correlation structure, if any;
3. cross-horizon and cross-window dependence;
4. calibration transport to the frozen target prefix; and
5. exact fallback when those requirements fail.

## Flat'n'Fold qualification order

For the proposed PointWorld--Flat'n'Fold study, the sparse contract resolves only
the representation branch of issue #333. Before any target garment is opened,
the study must still bind:

1. exact PointWorld checkpoint bytes and runtime;
2. exact Flat'n'Fold dataset bytes and complete garment roster;
3. three-camera timing and calibration;
4. Baxter-base action conversion into PointWorld robot point flow;
5. one source-only window and action schedule;
6. a support-feasibility artifact; and
7. source/calibration-only mappings for covariance, reliability, and downstream
   BayesianPhysTwin admission.

A representation-positive contract is not a provider-competence result. A valid
negative at any later gate remains a complete scientific outcome.
