# Group-balanced point-uncertainty calibration

Dense calibration sets often contain very different numbers of valid rows per
sequence, object, camera, or acquisition session. Pooling all rows gives every
pixel equal mass, which can let one long or densely valid group determine nearly
the entire variance scale.

`DepthDisagreementModel.calibrate_group_balanced` provides an explicit
alternative:

```python
calibrated, report = model.calibrate_group_balanced(
    errors,
    covariance,
    sequence_ids,
    mask=valid,
    trim_quantile=0.99,
)
```

For every declared group, Prob4D independently computes the along-ray and
per-lateral-axis normalized squared errors, trims those ratios at the requested
within-group quantile, and obtains one parallel and one lateral scale update.
The final update is the arithmetic mean of the per-group updates. Thus each
sequence contributes one calibration unit regardless of how many sampled rows
it contains.

The returned `GroupBalancedCalibrationReport` records:

- the total valid row count and number of groups;
- the common trim quantile;
- equal-group aggregate scale updates and normalized MSE;
- canonical sorted group IDs;
- every group's row count, scale update, and untrimmed normalized MSE.

`report.to_dict()` emits the explicit semantics identifier
`equal-group-mean-of-within-group-trimmed-ratios-v1`.

## Content-addressed provider artifact

Use the public helper when the fitted model will be carried into a provider-v2
calibration workflow:

```python
from prob4d.calibration import (
    fit_group_balanced_point_uncertainty_calibration,
)

artifact, report = fit_group_balanced_point_uncertainty_calibration(
    model,
    errors,
    covariance,
    sequence_ids,
    group_definition="scene sequence",
    calibration_case_ids=calibration_case_ids,
    source_repository="FlorianPfaff/Prob4D",
    source_revision=prob4d_commit,
    motioncrafter_revision=motioncrafter_commit,
    model_identifier=model_identifier,
    covariance_method="depth_disagreement_anisotropic_v1",
    trim_quantile=0.99,
    image_resolution=(320, 640),
    window_size=25,
    window_overlap=8,
    covariance_cluster_size=32,
    input_artifact_sha256=calibration_input_digests,
)
```

The helper keeps `PointUncertaintyCalibrationV1` schema v1 and its existing
aggregate fields. It additionally binds the complete equal-group report and the
human-readable grouping definition into content-addressed metadata under
`group_balanced_point_uncertainty_calibration`. A pooled artifact and an
equal-group artifact therefore receive different artifact IDs even when their
aggregate scales happen to be numerically equal.

Inspect the declaration without guessing from aggregate values:

```python
from prob4d.calibration import group_balanced_point_calibration_metadata

record = group_balanced_point_calibration_metadata(artifact)
assert record is not None
```

The accessor validates the recorded aggregation identifier, group count, and
presence of per-group diagnostics. It returns `None` for an ordinary pooled
artifact rather than silently labelling it group-balanced.

## Choosing the statistical unit

The grouping must match the intended transfer claim. Typical choices are:

- scene sequence for sequence-family-held-out reconstruction;
- physical object or acquisition session for deformable-object transfer;
- camera only when cameras are the independent deployment units;
- simulation run for controlled Monte Carlo evidence.

Neighboring pixels, overlapping windows, or repeated frames from one sequence
are not independent calibration units merely because they are separate rows.

## Compatibility and claim boundary

`DepthDisagreementModel.calibrate` remains unchanged and retains pooled-point
weighting for frozen reproduction. Group-balanced calibration is additive and
must be selected explicitly.

Changing the calibration aggregation changes the fitted covariance model.
Claim-bearing artifacts must therefore record the grouping definition and exact
calibration method, use disjoint calibration and target groups, and regenerate
content-addressed calibration artifacts rather than relabeling pooled results.

The provider compatibility field `covariance_method` continues to identify the
point covariance model itself. Equal-group versus pooled aggregation is bound
separately in artifact metadata. This avoids falsely advertising a new covariance
formula while still preventing the two fitting protocols from sharing an
artifact identity.
