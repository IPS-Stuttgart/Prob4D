# Provider API version 2

`prob4d.api.v2` is the stable ecosystem-facing surface for new claim-bearing
experiments. `prob4d.provider_v2` is its attested implementation module. The
provider preserves the portable `ObservationBeliefV1` schema and the Prob4D
causal-stream-v2 contract while removing provider-v1 execution from the current
package.

## Explicit export modes

Callers must choose one of two provider-v2 functions:

- `export_exploratory_observation_belief` permits uncalibrated or partially
  calibrated covariance and can explicitly enable pointwise covariance fallback;
- `export_calibrated_observation_belief` requires both content-addressed
  calibration artifacts, an exact Prob4D source revision, sequential gauge
  propagation, canonical covariance roots, analytic composition Jacobians, and
  the fail-closed spatial-cluster covariance path.

The calibrated function verifies the executing Prob4D revision and validates
prediction/calibration compatibility before any decoded prediction payload is
opened.

The grouped CLI makes the selection explicit:

```bash
prob4d observation export-calibrated \
  outputs/test/predictions.json \
  outputs/test/observation_belief.npz \
  --case-id held-out-case \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/test/metric-anchor.json \
  --gauge-covariance-calibration outputs/calibration/gauge.json \
  --point-uncertainty-calibration outputs/calibration/point.json \
  --source-revision "$(git rev-parse HEAD)" \
  --summary-json outputs/test/observation_belief_summary.json
```

Use `prob4d observation export-exploratory` for labelled reconstruction controls.
The bare `prob4d observation export` command is intentionally ambiguous: it
prints the two current choices and does not execute an exporter. Provider-v1
execution and standalone `prob4d-*` executables belong to the exact Prob4D 0.4.1
reproduction environment.

## Strict claim-bearing loading

Provider v2 exposes the corresponding admission boundary through the stable
façade:

```python
from prob4d.api.v2 import load_claim_bearing_observation_belief

validated = load_claim_bearing_observation_belief(
    "outputs/test/observation_belief.npz"
)
observation = validated.observation
```

This validates causal stream version 2, joint cross-window covariance, complete
alignment-level covariance calibration, zero fallback permission/use, canonical
numerical modes, calibration identities, and independently verified runtime
provenance. The neutral `load_observation_belief_export` remains available from
`prob4d.api.v2` for explicit artifact inspection.

## Provider and runtime attestation

Every provider-v2 artifact contains `metadata.prob4d_provider_attestation`. The
record binds:

- provider API version 2 and its content-addressed manifest;
- the artifact's exact Prob4D source revision;
- calibrated versus exploratory export mode;
- whether prediction/calibration compatibility was validated;
- gauge and point calibration artifact identifiers;
- covariance-root and composition-Jacobian modes; and
- the observed runtime revision, evidence source, checkout cleanliness, and
  independent-verification status.

Claim-bearing export fails closed when runtime provenance is unavailable,
mismatched, dirty, or not independently verified. It accepts a VCS-installed
package whose PEP 610 metadata identifies the commit or a clean source checkout
at the declared revision.

`PROB4D_RUNTIME_REVISION` may annotate an exploratory packaged deployment. An
unauthenticated environment variable cannot prove the executing code bytes and
never satisfies the claim-bearing entry point.

CI emits the current manifest with:

```bash
prob4d provider manifest --provider-revision "<commit>"
```

## Analytic `Sim(3)` composition Jacobians

Sequential gauge covariance propagation composes a parent gauge with an
uncertain relative gauge in coordinates

```text
[log scale, axis-angle rotation (3), translation (3)].
```

Provider v2 accounts for additive log scale, SO(3) right Jacobians, scale and
rotation transport of relative translation, and direct translation blocks. The
SO(3) logarithm is nondifferentiable at its pi branch cut; provider-v2 sequential
export fails closed there rather than exporting platform-dependent covariance.

Provider v2 passes an immutable `ExportNumericsPolicy` through the export
core. The policy binds the declared composition-Jacobian and covariance-root
callables for the complete export, so import order cannot change provider
semantics. Compatibility context managers remain task-local for existing
internal callers and tests, but importing the numerical modules never replaces
private functions in `observation_export`.

- provider-v2 sequential export uses `analytic`;
- exploratory fixed-lag reconstruction retains its declared finite-difference
  path; and
- explicit policies remain authoritative inside nested or concurrent contexts.

The historical provider-v1 defaults are reproduced only by Prob4D 0.4.1.

## Canonical covariance-root basis

Provider v2 derives a canonical basis from repeated-eigenspace projectors and
fails closed when an eigenvalue floor or rank boundary would split a numerically
repeated eigenspace. The claim-bearing entry point always uses
`canonical_eigenspaces`; exploratory callers can request legacy roots as a
labelled numerical control. Full provider-v1 reproduction remains pinned to
Prob4D 0.4.1.

## Calibration compatibility

For a claim-bearing export, each calibration must match the prediction manifest
and runtime in:

- source repository and MotionCrafter revision;
- canonical model identifier, including seed and temporal stride;
- image resolution;
- window size and overlap;
- covariance cluster size; and
- expected gauge or point covariance method.

The default methods are `frame_spatial_cluster_robust_v1` for gauge covariance
and `depth_disagreement_anisotropic_v1` for conditional point covariance. A
mismatch raises `CalibrationCompatibilityError` with field-level diagnostics.
Calibration case identifiers and input digests intentionally identify independent
calibration data and normally differ from the target artifact.

## Append-only factor streams

`ObservationFactorStreamV1` is an additive provider-v2 artifact for recursive
observation times. It references causally disjoint schema-v4
`ObservationFactorBundle` deltas and binds them through portable update IDs and a
previous-update hash chain. Paths are retrieval metadata; bundle and payload
hashes, frame intervals, observation identities, and gauge IDs determine content
identity. See [append-only observation-factor streams](observation-factor-stream.md).

## Python export example

```python
from prob4d.api.v2 import (
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

Prob4D 0.5 retains `prob4d.provider_v1` only as an artifact compatibility bridge.
It exposes no provider-v1 estimator or exporter. See the
[compatibility boundaries](compatibility.md) and pin Prob4D 0.4.1 for full
provider-v1 execution.
