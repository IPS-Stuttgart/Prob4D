# Provider API version 2

`prob4d.provider_v2` is the safe-by-default Python surface for new claim-bearing
experiments. It does not change `ObservationBeliefV1`, the causal observation
stream contract, or the frozen `prob4d.provider_v1` behavior.

## Explicit export modes

Version 2 removes the ambiguous general export entry point. Callers must choose
one of two functions:

- `export_exploratory_observation_belief` permits uncalibrated or partially
  calibrated covariance and can explicitly enable the pointwise covariance
  fallback.
- `export_calibrated_observation_belief` requires both content-addressed
  calibration artifacts, an exact Prob4D source revision, sequential gauge
  covariance propagation, and the fail-closed spatial-cluster covariance path.

The calibrated function validates compatibility before any decoded prediction
payload is opened.

## Canonical MotionCrafter model identifier

Calibration artifacts used through version 2 must set `model_identifier` to the
value returned by:

```python
import json

from prob4d.provider_v2 import motioncrafter_model_identifier

manifest = json.loads(open("predictions.json", encoding="utf-8").read())
model_identifier = motioncrafter_model_identifier(manifest)
```

The identifier hashes the model type, UNet and VAE identifiers, inference-step
configuration, guidance scale, decode chunk size, and low-memory mode. The exact
MotionCrafter commit is checked separately.

## Compatibility fields

For a claim-bearing export, each calibration must match the prediction manifest
and runtime settings in all of the following fields:

- source repository;
- MotionCrafter revision;
- canonical model identifier;
- image resolution;
- window size and overlap;
- covariance cluster size; and
- the expected gauge or point covariance method.

The default methods are
`frame_spatial_cluster_robust_v1` for gauge covariance and
`depth_disagreement_anisotropic_v1` for conditional point covariance. A mismatch
raises `CalibrationCompatibilityError` with field-level diagnostics.

The calibration case identifiers and input digests are deliberately not compared
to the target sequence. They identify the independent calibration data and should
normally differ from the target artifact.

## Example

```python
from prob4d.provider_v2 import (
    export_calibrated_observation_belief,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_point_uncertainty_calibration,
)

artifact = export_calibrated_observation_belief(
    "predictions.json",
    case_id="held-out-case",
    causal_frame_stop=134,
    metric_anchor=load_metric_gauge_anchor("metric-anchor.json"),
    gauge_covariance_calibration=load_gauge_covariance_calibration(
        "gauge-calibration.json"
    ),
    point_uncertainty_calibration=load_point_uncertainty_calibration(
        "point-calibration.json"
    ),
    source_revision="<exact Prob4D commit>",
)
```

Version 1 remains available for frozen runs. New experiments should use version 2
unless they intentionally reproduce the earlier API semantics.
