# Covariance calibration contracts

Prob4D exposes two content-addressed calibration artifacts through
`prob4d.provider_v1`:

- `GaugeCovarianceCalibrationV1` inflates the scale, rotation, and translation
  blocks of dense-overlap Sim(3) covariance.
- `PointUncertaintyCalibrationV1` carries the calibrated
  `DepthDisagreementModel` used for conditional point covariance.

Both artifacts bind their fitted parameters to calibration case identifiers,
Prob4D and MotionCrafter revisions, model and covariance settings, source
artifact digests, and finite JSON metadata. The loaders recompute and verify the
artifact identifier, so edited calibration files fail validation.

## Claim-bearing export

```python
from prob4d.provider_v1 import (
    export_calibrated_observation_belief,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_point_uncertainty_calibration,
)

anchor = load_metric_gauge_anchor("metric-anchor.json")
gauge_calibration = load_gauge_covariance_calibration("gauge-calibration.json")
point_calibration = load_point_uncertainty_calibration("point-calibration.json")

artifact = export_calibrated_observation_belief(
    "predictions.json",
    case_id="held-out-case",
    causal_frame_stop=134,
    metric_anchor=anchor,
    gauge_covariance_calibration=gauge_calibration,
    point_uncertainty_calibration=point_calibration,
    source_revision="<exact Prob4D commit>",
)
```

The claim-bearing helper requires both artifacts. The more general
`export_observation_belief` remains usable for explicitly exploratory runs, but
records `uncalibrated_exploratory` or `partially_calibrated` in the observation
metadata.

## Dense-alignment fallback policy

The provider now fails closed when a requested frame-by-spatial-tile covariance
has fewer than eight active clusters. A pointwise-cluster fallback can be
enabled only with `allow_pointwise_covariance_fallback=True`. The observation
metadata records both whether the fallback was allowed and how many alignments
actually used each fallback mode.

Direct low-level calls to `align_windows` retain the historical pointwise
fallback for source compatibility. Provider exports use a task-local covariance
policy, so concurrent exports cannot leak calibration or fallback settings into
one another.

## Interpretation

A structurally calibrated artifact is not, by itself, evidence of prospective
coverage or downstream physical-twin improvement. Those claims remain gated on
held-out calibration and end-to-end Prob4D-to-Bayesian-PhysTwin evaluation.
