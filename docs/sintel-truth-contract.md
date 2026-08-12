# Sintel truth geometry contract

`prob4d.sintel_uncertainty.load_sintel_truth` treats the processed Sintel HDF5
file as diagnostic ground truth. The loader validates the geometry before using
it in uncertainty evaluation.

## Required datasets

The file must contain:

- `point_map` with shape `(T, H, W, 3)`;
- `valid_mask` with shape `(T, H, W)`; and
- `camera_pose` with shape `(T, 4, 4)`.

At least one frame is required. Pose entries must be finite homogeneous rigid
transforms. Their last row must be `[0, 0, 0, 1]`; every rotation block must be
orthogonal and have determinant `+1`. Frame-count mismatches, reflections,
scaled or sheared rotations, non-finite poses, and malformed homogeneous rows
fail closed.

The positive finite `max_depth` limit and the resize support threshold are
validated separately. Invalid or out-of-range point coordinates are masked
before mask-normalized bilinear resizing, so missing coordinates cannot leak
into otherwise supported points.

## Coordinate convention

The stored poses are interpreted as world-from-camera transforms. Evaluation is
performed in the coordinate system of the first camera. Prob4D forms the first
pose inverse analytically using

```text
R_inverse = R_transpose
translation_inverse = -R_transpose @ translation
```

and left-composes it with every stored pose. A generic matrix inverse is not used
for this rigid-transform path.

## Claim boundary

This contract hardens diagnostic truth loading. It changes no Prob4D provider,
observation artifact, calibration fit, BayesianPhysTwin update, Causal4D
intervention, target-access rule, or frozen scientific result.
