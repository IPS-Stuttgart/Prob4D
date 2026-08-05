# Append-only observation-factor streams

`ObservationFactorStreamV1` records a sequence of causal, unfused Prob4D
observation updates without rewriting or reopening previously admitted frame
intervals. It is intended for recursive Bayesian-PhysTwin experiments that need
several observation times rather than one persisted endpoint correction.

The stream is an orchestration artifact. Every update references a normal
schema-v4 `ObservationFactorBundle`; it does not introduce another point,
covariance, or gauge representation.

## Contract

Each update must reference a bundle that:

- uses observation-factor schema version 4;
- declares `joint-cross-window` gauge covariance semantics;
- contains only factors in the newly admitted frame interval;
- uses the same sequence, case, stream, source repository, and source revision as
  all previous updates; and
- lies inside the directory containing the stream manifest.

The first update declares `admitted_frame_start`. Every subsequent update starts
exactly at the preceding exclusive `causal_frame_stop`. Empty temporal regions
inside an admitted interval are allowed, but a later update may not reintroduce
a frame from an earlier interval.

## Portable identity

For every update, Prob4D records and validates:

- the bundle-manifest SHA-256 digest;
- the referenced NPZ payload SHA-256 digest;
- a digest of the ordered `(frame, view, window, point_id)` observations;
- the factor, observation, and persistent-identity counts;
- the ordered gauge IDs; and
- the preceding update ID.

The update ID excludes the local bundle path. Moving an unchanged stream tree
therefore preserves its content identity, while changing a bundle, payload,
identity, frame boundary, or chain predecessor changes the update and stream
IDs. Relative paths are still validated as retrieval metadata and may not escape
the stream directory.

## Strict loading and persistence

Stream and referenced bundle manifests are read as strict portable JSON:

duplicate object keys, non-finite numbers, coercion-dependent scalar values,
unknown fields, malformed digests, unsafe POSIX paths, and contradictory joint
gauge-covariance metadata fail closed. `validate_bundles=False` skips reopening
the bundle bytes, but it does not relax stream-manifest validation.

`write_observation_factor_stream` uses a unique temporary file, flushes and
synchronizes the complete bytes, and atomically replaces the manifest. When a
manifest already exists, the replacement must be either byte-semantically
idempotent or a strict extension of the existing update chain. Rewriting a
persisted stream with changed identifiers or metadata, truncating updates, or
forking any existing update is rejected.

These persistence checks do not change valid version-1 update or stream IDs.

## Python example

```python
from prob4d.provider_v2 import (
    append_observation_factor_bundle,
    load_observation_factor_stream,
    write_observation_factor_stream,
)

stream_path = "outputs/case-a/prob4d-factor-stream.json"

stream = append_observation_factor_bundle(
    None,
    "outputs/case-a/update-000/factors.json",
    stream_manifest_path=stream_path,
    admitted_frame_start=100,
    metadata={"protocol": "prob4d-bpt-recursive-prefix-v1"},
)
stream = append_observation_factor_bundle(
    stream,
    "outputs/case-a/update-001/factors.json",
    stream_manifest_path=stream_path,
)
write_observation_factor_stream(stream, stream_path)

validated = load_observation_factor_stream(stream_path)
assert validated.updates[1].previous_update_id == validated.updates[0].update_id
```

`load_observation_factor_stream` reopens and validates every referenced bundle by
default. Passing `validate_bundles=False` validates only the stream manifest and
its hash chain; it is useful for metadata inspection, not evidence admission.

## Identity semantics

Persistent identities are scoped by `(view_id, window_id, point_id)`. The
existing causal scene-flow tracklet builder can provide such point IDs inside a
window. A producer that changes identity semantics must use a distinct stream ID
or protocol identifier rather than silently reinterpreting an existing stream.

The stream does not assert that points from different windows or cameras denote
the same material point. Such associations remain explicit downstream model
inputs.

## Scientific boundary

A valid stream proves that the referenced factor updates are immutable,
causally ordered, non-overlapping, and checksum-bound. It does not prove that:

- the visual uncertainty is calibrated on a target cohort;
- the observations identify a physical state rather than a nuisance mode;
- an update should be accepted; or
- Prob4D improves future physical prediction.

Bayesian-PhysTwin remains responsible for nuisance-aware inference, the
baseline-relative regret guard, and exact fallback. Causal4D should consume only
a content-bound accepted twin belief rather than reinterpret the raw factor
stream.
