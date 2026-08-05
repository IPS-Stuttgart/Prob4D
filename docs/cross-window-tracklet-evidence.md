# Joint-gauge cross-window tracklet evidence

`prob4d.cross_window_tracklet_evidence` adds an opt-in evidence layer around the
source-only cross-window tracklet diagnostic. It addresses two gaps without
changing provider-v2 observation identities:

1. a `CausalTrackletSet` can now be bound to its complete array content,
   prediction-manifest identity, exact source revision, builder configuration,
   and nested metadata; and
2. every candidate can retain a normalized residual diagnostic from the complete
   joint cross-window `Sim(3)` covariance rather than treating the two window
   gauges as independent.

The module remains experimental. It scores and audits possible material
identities; it does not merge point IDs in a claim-bearing factor bundle.

## Content-addressed tracklet artifact

`CausalTrackletArtifactV1` binds:

- every track, frame, pixel, point, link-probability, and cumulative-association
  array through a canonical little-endian dtype, shape, and SHA-256 descriptor;
- the complete recursively immutable tracklet metadata;
- the prediction-manifest SHA-256;
- an exact 40-character Git source revision; and
- the finite JSON builder configuration.

The `tracklet_set_id` identifies only the complete `CausalTrackletSet` content.
The `artifact_id` additionally binds prediction and producer lineage. Therefore,
changing a point changes both IDs, while changing only the prediction manifest
preserves `tracklet_set_id` and changes `artifact_id`.

## Correlated residual covariance

For a candidate pair in a shared frame, define

\[
 r = p_l^g - p_r^g,
\]

where each global point is obtained from its window-local point through an
uncertain `Sim(3)` gauge. With gauge Jacobians `J_l` and `J_r` and joint gauge
blocks `P_ll`, `P_rr`, `P_lr`, and `P_rl`, the gauge contribution to the residual
covariance is

\[
 J_l P_{ll} J_l^T + J_r P_{rr} J_r^T
 - J_l P_{lr} J_r^T - J_r P_{rl} J_l^T.
\]

The implementation adds the two transformed conditional local point
covariances exactly once. Their cross-window conditional covariance is not
available from the current producer and is explicitly bound as
`assumed-zero-unavailable-v1`; the joint-gauge label must not be read as complete
residual dependence. Positive cross-window gauge correlation therefore reduces
the gauge-difference contribution, while negative correlation increases it. The
result fails closed when the supplied joint prior or any local covariance is
non-finite, asymmetric, or materially indefinite.

## Association evidence

`associate_cross_window_tracklets_joint_gauge` reuses the existing bounded
spatial candidate generation and causal shared-frame rules. The complete joint
covariance determines each candidate's `normalized_rms` diagnostic. It does not
alter mutual-best ranking or the compatibility margin: those retain the bounded
covariance-independent geometric score. This prevents a candidate from becoming
more likely merely because its predictive covariance is wider. A covariance-aware
admission rule requires a separately versioned, source-calibrated gate.

The returned `JointGaugeCrossWindowAssociationEvidenceV1` binds:

- both tracklet artifact IDs;
- the ordered gauge-ID list;
- both exact `Sim(3)` transform identities;
- the complete joint gauge-prior identity;
- both conditional covariance-stack identities;
- the exact tracklet-producer revision;
- the exact association-implementation revision;
- the geometric-ranking and unavailable-cross-covariance semantics; and
- the complete semantic association result.

Execution-only candidate tiling is excluded. Different
`candidate_chunk_size` values must produce identical evidence dictionaries and
the same `result_id`.

## Example

```python
import numpy as np

from prob4d.cross_window_tracklet_evidence import (
    CausalTrackletArtifactV1,
    associate_cross_window_tracklets_joint_gauge,
)
from prob4d.cross_window_tracklets import CrossWindowAssociationConfig

left_artifact = CausalTrackletArtifactV1(
    tracklets=left_tracklets,
    prediction_manifest_id=left_prediction_manifest_id,
    source_revision=prob4d_revision,
    builder_configuration=tracklet_builder_configuration,
)
right_artifact = CausalTrackletArtifactV1(
    tracklets=right_tracklets,
    prediction_manifest_id=right_prediction_manifest_id,
    source_revision=prob4d_revision,
    builder_configuration=tracklet_builder_configuration,
)

evidence = associate_cross_window_tracklets_joint_gauge(
    left_artifact,
    right_artifact,
    left_global_from_local=left_gauge,
    right_global_from_local=right_gauge,
    left_conditional_local_covariance_m2=left_local_covariance,
    right_conditional_local_covariance_m2=right_local_covariance,
    gauge_ids=ordered_gauge_ids,
    joint_gauge_covariance=joint_gauge_covariance,
    left_gauge_id=left_tracklets.window_id,
    right_gauge_id=right_tracklets.window_id,
    association_revision=prob4d_association_revision,
    configuration=CrossWindowAssociationConfig(
        minimum_shared_frames=3,
        minimum_effective_support=1.5,
        maximum_weighted_rms_m=0.025,
        maximum_shared_frame_distance_m=0.075,
        minimum_compatibility_score=0.20,
        minimum_score_margin=0.10,
    ),
)

print(evidence.result_id)
print(evidence.accepted_pairs)
```

## Promotion gate

The compatibility score remains a covariance-independent source-side ranking
statistic, and `normalized_rms` remains an uncalibrated covariance diagnostic;
neither is a posterior match probability. Before any accepted link rewrites a
provider-v2 observation identity, freeze all association settings on independent
development/calibration objects or sessions and report:

- association precision, retention, false merges, and identity switches where
  labels exist;
- deployed BayesianPhysTwin RMSE;
- harmful accepted updates;
- interval coverage and width;
- rejection and exact-fallback rates; and
- object/session-clustered uncertainty.

A negative prospective result should retain the existing within-window identity
semantics unchanged.
