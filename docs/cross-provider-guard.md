# Cross-provider corroboration guard

Prob4D can compare two **distinct provider-neutral prediction contracts** before a
visual factor is offered to BayesianPhysTwin. The guard is source-only and
calibration-separated. It answers one narrow question:

> Do the registered providers corroborate one another on the matched rows under
> the frozen covariance and support semantics?

An admitted case is not a physical-state update. BayesianPhysTwin still owns the
baseline-relative regret guard and exact physical fallback.

## Why this exists

MotionCrafter and VGGT consume the same video and can share camera, dataset, and
model biases. Different implementations therefore do not imply independent
errors. The guard never defaults to

```text
Cov(e_first - e_second) = C_first + C_second.
```

When an explicit cross-provider covariance is unavailable, it uses the PSD upper
bound

\[
\operatorname{Cov}(e_1-e_2)
\preceq
(1+\beta)C_1 + (1+\beta^{-1})C_2,
\]

with \(\beta\) chosen per row to minimize the trace of the bound. If an explicit
cross covariance \(C_{12}\) is supplied, the registered difference covariance is

\[
C_1+C_2-C_{12}-C_{12}^{\mathsf T}.
\]

The row score is the square root of normalized Mahalanobis energy. One complete
object or acquisition session contributes one case score: the frozen higher
quantile of its valid row scores. A split-conformal upper order statistic is fit
across clean calibration cases.

## Provider and dependence requirements

Every panel case references two strict `PredictionProviderManifestV1` artifacts
and the exact payload IDs used to generate the matched rows. Prob4D verifies all
payload bytes before opening the matched comparison file.

The selected payload sets must:

- refer to the same sequence;
- have identical output-frame and view support;
- share exactly one `input-video:<sha256>` dependence group; and
- come from distinct provider contracts.

The last rule prevents VGGT `world_points` and `depth_unprojected`, two seeds of
one model, or two representations from one run from being presented as two
independent provider votes.

## Matched observation file

Each case points to a closed NPZ containing exactly:

```text
first_points_m          float64 [N, 3]
second_points_m         float64 [N, 3]
first_covariance_m2     float64 [N, 3, 3]
second_covariance_m2    float64 [N, 3, 3]
valid_mask              bool    [N]
```

It may additionally contain:

```text
cross_covariance_m2     float64 [N, 3, 3]
```

The case record binds the exact NPZ SHA-256, row-identity SHA-256, alignment
artifact ID, coordinate-frame ID, provider manifests, and selected payload IDs.
Truth, BayesianPhysTwin innovations, and target outcomes are not permitted in the
panel semantics.

## Panel format

Paths are relative to the panel file and are retrieval metadata. All referenced
content identities are recomputed.

```json
{
  "schema": "prob4d.cross-provider-panel",
  "schema_version": 1,
  "purpose": "calibration",
  "cases": [
    {
      "case_id": "object-01/session-02",
      "first_manifest": "motioncrafter/provider.json",
      "second_manifest": "vggt/provider.json",
      "first_payload_ids": ["<sha256>"],
      "second_payload_ids": ["<sha256>"],
      "matched_observations": "matched/object-01-session-02.npz",
      "matched_observations_sha256": "<sha256>",
      "alignment_artifact_id": "<sha256>",
      "row_identity_sha256": "<sha256>",
      "coordinate_frame_id": "registered-world-object-01-session-02"
    }
  ],
  "metadata": {
    "uses_truth": false,
    "uses_downstream_physical_innovation": false
  }
}
```

Calibration case IDs are the exchangeable statistical units. Do not place frames,
pixels, or correlated windows into the panel as separate calibration cases.

## Commands

Fit the clean calibration threshold before target access:

```bash
prob4d diagnostic cross-provider-guard calibrate \
  calibration-panel.json \
  cross-provider-calibration.json \
  --miscoverage 0.05 \
  --row-quantile 0.95 \
  --minimum-support-fraction 0.8
```

Evaluate a frozen target panel:

```bash
prob4d diagnostic cross-provider-guard evaluate \
  cross-provider-calibration.json \
  target-panel.json \
  cross-provider-decision.json
```

Replay artifact validation:

```bash
prob4d diagnostic cross-provider-guard verify-calibration \
  cross-provider-calibration.json

prob4d diagnostic cross-provider-guard verify-decision \
  cross-provider-decision.json \
  --calibration cross-provider-calibration.json
```

A target case is admitted only when its case score does not exceed the frozen
finite-sample threshold and its common support is not below the frozen minimum.
Rejection reasons are explicit:

- `cross-provider-disagreement`;
- `insufficient-common-support`.

## Controlled stress workflow

The `Cross-provider corroboration guard` workflow runs a fresh-seed stress study
with clean calibration cases, clean target cases, provider-specific corruption,
and a stronger bias shared by both providers. The registered expected behavior is:

- clean false rejection remains near the requested miscoverage;
- provider-specific corruption is detected; and
- shared common bias is **not** misrepresented as detected merely because two
  providers agree.

The shared-bias arm is an explicit limitation control. Agreement between two
monocular providers cannot establish absolute correctness. Independent metric
anchors, provider-specific bias evidence, or downstream physical regret evidence
remain necessary for that claim.
