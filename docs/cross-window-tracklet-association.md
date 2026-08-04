# Cross-window tracklet association diagnostic

`prob4d.cross_window_tracklets` evaluates whether two causally sealed
`CausalTrackletSet` objects contain geometrically compatible material tracks in
their shared absolute frames. It addresses a deliberately narrow gap: existing
track IDs are persistent inside one prediction window, but they are not evidence
that tracks from different windows represent the same material point.

The diagnostic does **not** rewrite observation-factor point IDs and is not part
of the claim-bearing provider-v2 export path. It produces auditable candidate
scores and admits only unambiguous mutual-best links. Downstream experiments can
therefore measure whether cross-window identity has useful oracle headroom before
introducing it into a Bayesian update.

## Information boundary

The association consumes only:

- two prefix-only `CausalTrackletSet` objects;
- one global-from-local `Sim3` transform for each window;
- optional per-observation global point covariance; and
- a frozen source-only association configuration.

It does not consume target truth, BayesianPhysTwin innovations, physical-model
residuals, intervention outcomes, or post-cutoff observations. Only absolute
frames present in both tracklets contribute to a candidate. A nonshared causal
suffix in either window therefore cannot change the score.

## Candidate score

Before scoring complete track pairs, the diagnostic applies a bounded spatial
gate on every shared absolute frame. The default gate admits a pair only when its
global points come within `maximum_shared_frame_distance_m` on at least one
shared frame. Distance matrices are evaluated in bounded chunks;
`candidate_chunk_size` changes temporary memory only, not the selected pairs. Set
the distance gate to `None` only for an explicitly exhaustive small diagnostic.

For each spatial candidate with enough shared frames, the diagnostic transforms
both point sequences into the common global frame. It reports:

- shared absolute frame IDs;
- association-probability-weighted RMS and maximum displacement;
- effective support, using the product of the two source-side tracklet
  association probabilities;
- an uncertainty-normalized RMS; and
- a bounded compatibility score.

Without covariance input, the normalized residual uses the frozen isotropic
`isotropic_distance_scale_m`. When covariance is supplied for both windows, each
residual uses the sum of the two global point covariances plus an explicit
positive floor. The covariance may already include local point uncertainty,
gauge uncertainty, or both; the diagnostic never silently adds or removes a
gauge term. Cross-window covariance is not inferred: callers with shared source
errors must supply an appropriately conservative marginal scale.

The compatibility score is a source-side ranking statistic, not a calibrated
posterior probability. It combines the Gaussian-shaped normalized residual score
with an effective-support factor. Promotion would require independent calibration
and a downstream guarded physical-prediction gate.

## Conservative link admission

A candidate becomes a link only when all of the following hold:

1. it is the deterministic best candidate for the left track;
2. it is also the deterministic best candidate for the right track;
3. both best-versus-second-best score margins pass the frozen ambiguity gate;
4. effective support passes its minimum;
5. weighted RMS passes its maximum; and
6. the compatibility score passes its minimum.

The output remains one-to-one on both sides and records spatially rejected,
non-mutual, ambiguous, threshold-rejected, low-support, zero-support, and
insufficient-overlap counts. Rejected tracks remain explicitly unmatched rather
than being forced into an identity.

## Example

```python
from prob4d.cross_window_tracklets import (
    CrossWindowAssociationConfig,
    associate_cross_window_tracklets,
)

result = associate_cross_window_tracklets(
    left_tracklets,
    right_tracklets,
    left_global_from_local=left_gauge,
    right_global_from_local=right_gauge,
    configuration=CrossWindowAssociationConfig(
        minimum_shared_frames=3,
        minimum_effective_support=1.5,
        maximum_weighted_rms_m=0.025,
        maximum_shared_frame_distance_m=0.075,
        minimum_compatibility_score=0.20,
        minimum_score_margin=0.10,
    ),
    left_global_covariance_m2=left_covariance,
    right_global_covariance_m2=right_covariance,
    candidate_chunk_size=256,
)

for link in result.links:
    print(link.left_track_id, link.right_track_id, link.compatibility_score)
```

## Required experiment before integration

Compare, by independent physical object or acquisition session:

1. existing within-window persistent identities;
2. cross-window links from this diagnostic;
3. a framewise identity control; and
4. exact physical fallback.

Freeze all association gates on development/calibration objects. Report source
association precision and retention where labels exist, but keep the decisive
endpoints downstream: accepted-update RMSE, harmful accepted updates, interval
coverage and width, rejection rate, and exact fallback. A negative result should
leave provider-v2 identity semantics unchanged.
