# Observation timestamp lineage V1

Frame indices are not physical time. During fast deformation, even a small
camera-to-actuator or provider-to-simulator clock offset can create a coherent
spatial residual that resembles physical model discrepancy.

`ObservationTimestampLineageV1` is a content-addressed sidecar for an existing
observation-factor bundle. It preserves timing evidence without changing the
observation-factor schema-v4 wire format.

## Recorded contract

One sidecar binds, in exact factor order:

- sequence, case, stream, source revision, and exclusive causal frame stop;
- the raw timestamp-source artifact by SHA-256;
- clock domain, time scale, and timestamp-source identity;
- factor IDs and frame indices;
- integer observation timestamps in nanoseconds;
- per-factor conditional timestamp standard deviations; and
- an optional source/calibration-derived shared clock-offset prior artifact ID.

The fixed uncertainty semantics are:

```text
conditional-jitter-excludes-shared-clock-offset
```

The per-factor standard deviations may represent timestamp quantization or
local packet/capture jitter. They deliberately exclude a coherent clock offset.
When a downstream estimator retains the shared clock offset as an explicit
latent, adding that same uncertainty into every local observation covariance
would double count it.

## Python example

```python
import numpy as np

from prob4d.observation_timestamp_lineage import (
    ObservationTimestampLineageV1,
    validate_timestamp_lineage_for_bundle,
    write_observation_timestamp_lineage,
)

lineage = ObservationTimestampLineageV1(
    sequence_id=bundle.sequence_id,
    case_id=bundle.case_id,
    stream_id=bundle.stream_id,
    source_revision=bundle.source_revision,
    source_artifact_sha256=raw_timestamp_artifact_sha256,
    causal_frame_stop=bundle.causal_frame_stop,
    clock_domain="camera-hardware-clock",
    time_scale="device-monotonic",
    timestamp_source="camera-hardware-packet",
    factor_ids=tuple(factor.factor_id for factor in bundle.factors),
    frame_indices=np.asarray(
        [factor.frame_index for factor in bundle.factors],
        dtype=np.int64,
    ),
    timestamps_ns=np.asarray(factor_timestamps_ns, dtype=np.int64),
    conditional_timestamp_std_ns=np.asarray(timestamp_jitter_ns),
    shared_clock_offset_prior_artifact_id=sync_prior_artifact_id,
)
validate_timestamp_lineage_for_bundle(lineage, bundle)
write_observation_timestamp_lineage(lineage, "timestamps.json")
```

Loading revalidates the closed schema, finite values, exact integer fields,
factor identities, source revision, causal cutoff, content ID, and recursively
immutable metadata. Numeric arrays use byte-backed storage, so callers cannot
restore their NumPy write flag after validation. Publication is idempotent for
the same artifact and refuses to replace different content.

## Downstream timing nuisance

A downstream Bayesian physical update can convert nanoseconds to seconds and
linearize a timing offset as

```text
residual = state_jacobian * delta_state
         + bias_jacobian * spatial_bias
         + observation_time_derivative * delta_time
         + conditional_noise.
```

The sidecar does not estimate `delta_time`. A source-only synchronization
procedure, such as the Causal4D actuator/camera timing fit, may publish the
optional prior artifact. BayesianPhysTwin must still test whether the timing
column is distinguishable from state, gauge, spatial bias, and physical
relaxation modes. An unidentifiable update must retain the frozen fallback.

## Protocol boundary

Timestamp extraction, local jitter estimation, clock-domain definitions, and
any shared-offset prior must be frozen on source or calibration evidence before
a confirmation cohort is opened. This contract records timing provenance; it
does not establish timestamp accuracy, provider competence, physical-state
identifiability, calibrated coverage, downstream improvement, or deployment
safety.
