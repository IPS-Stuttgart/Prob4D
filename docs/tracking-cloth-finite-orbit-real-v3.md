# Tracking Cloth: support-qualified finite-orbit query validation

Status: **passed recording-disjoint public real-geometry replication**.

This result validates one narrow distinction required by Prob4D: local
first-order sensitivity at one gauge representative is not sufficient to decide
whether a physical query is identifiable over a complete unresolved finite
orbit. It does not evaluate a learned visual provider or claim recovery of the
physical cloth state.

## Why a new replication was necessary

The original v1 execution (`33361712662`) terminated during source marker
discovery with `fewer than three common 3-D marker groups: []`. It did not emit
a source seal or scientific result and did not open target CSV contents.
OptiTrack Motive exports marker type, label, unique ID, quantity, and `X/Y/Z`
axes on separate header rows; the old reader interpreted only the first row as a
conventional flat header.

A source-only header audit then found a second support issue. Cotton, denim, and
wool A2 recordings share labelled marker identities `1` through `20`, whereas
polyester hand-held recordings use a distinct unlabeled-marker namespace. A
filename-only roster audit and a source-header audit were completed before any
target trajectory value was used.

The v2 support stage therefore froze:

- 24 A2 shake/twist source recordings from cotton, denim, and wool;
- the exact source-selected marker triplet `1`, `20`, `5`;
- a 42-recording A2 collision-family candidate target roster;
- unchanged v1 controlled-factor, inference, bootstrap, and decision criteria.

After source sealing, v2 inspected target marker headers only. All 15 Hitting
and Tablecloth recordings supported the selected labels. All 27 Self-collisions
recordings used an incompatible marker namespace. No target trajectory value was
parsed, and the unsupported recordings were reported rather than replaced.

The v3 protocol then froze exactly those 15 support-positive recordings. It
uses 128 deterministic samples per recording, selected before target trajectory
access solely to preserve the parent minimum of 1,000 target cases. No outcome,
threshold, marker, or method was tuned on the target trajectories.

## Immutable evidence

- protocol ID: `9c0fb1a4191743a5038a2f26e521db1640fd5abfc3cac389e851485b7836a472`;
- result ID: `1441a141a8eccc1ae3a503c701c72e8d702a0bcb17226238f3d32effa3e58111`;
- source seal ID: `b6819675387890dd786b22168638f1c1ee33558f88d8d6fa6359a0f6df403e6d`;
- target support ID: `f9867227ef9e5a780c49fbbe3ce646205795d9fda504c13b535523d654ef8812`;
- Actions run: `33534703640`;
- artifact ID: `9811148956`;
- artifact digest: `sha256:817c86588bc5673529fc4ae41196f88741a23dd36a7347caeeca5bd92d3a177d`;
- official archive MD5: `b4868b702f8a42b2ea1069d0f1a3b8f6`;
- official archive SHA-256:
  `14916efa89a26d991c024024cc9449397d3a6f654311e621bb91e9602e231e1a`;
- runtime: Python 3.12.14, NumPy 2.3.5, Linux X64.

The workflow verified the frozen scientific blobs, the exact v2 support
artifact, all 15 target group identities, the public archive checksum, source
sealing before target-header access, and removal of raw data before publication.
An independent verification step recomputed the content identities and accepted
the terminal result.

## Held-out result

The 15 recordings contributed 1,803 controlled geometry cases. All ten
registered criteria passed.

| Equal-recording endpoint | Estimate | 95% recording bootstrap interval |
|---|---:|---:|
| Finite-orbit invariant-query acceptance | 1.0000 | [1.0000, 1.0000] |
| Finite-orbit radial-query rejection | 1.0000 | [1.0000, 1.0000] |
| Local-gate harmful radial updates | 0.6506 | [0.6232, 0.6770] |
| Finite-orbit harmful accepted radial updates | 0.0000 | [0.0000, 0.0000] |
| Exact fallback fraction for radial queries | 1.0000 | [1.0000, 1.0000] |
| Invariant fallback RMSE | 15.0605 mm | [14.5504, 15.5934] |
| Invariant finite-orbit RMSE | 0.0000 mm | [0.0000, 0.0000] |
| Radial local-gate RMSE | 337.9311 mm | [273.3206, 393.3651] |
| Radial finite-orbit/fallback RMSE | 197.2634 mm | [159.6446, 229.6299] |
| Radial local Gaussian coverage | 0.0485 | [0.0380, 0.0607] |
| Radial finite-orbit/fallback coverage | 1.0000 | [1.0000, 1.0000] |
| Full orbit support coverage | 1.0000 | [1.0000, 1.0000] |

The local gate admitted every radial query although the query varied over the
hidden SO(2) orbit. Approximately 65.1% of those admitted updates were harmful
under the registered criterion. The finite-orbit gate rejected every radial
query and therefore made zero harmful accepted radial updates, returning the
registered exact fallback. In contrast, it admitted every invariant query and
represented that controlled query exactly.

The radial RMSE reduction relative to the locally admitted estimate is about
41.6%, but it must be interpreted as the benefit of **correct rejection plus
exact fallback**, not as a more accurate visual or geometric point predictor.
The finite-orbit radial RMSE and the fallback RMSE are intentionally identical.

The effect is not confined to one motion family. Descriptively, without a new
inferential claim, the mean harmful local-radial fraction was 0.6449 across the
three Hitting groups and 0.6520 across the twelve Tablecloth groups. Every
recording had zero harmful finite-orbit-accepted radial updates and complete
orbit-support coverage.

## Registered criteria

All passed:

1. every target group contributes;
2. at least 1,000 target cases;
3. invariant queries are admitted;
4. radial queries are rejected over the finite orbit;
5. the local gate admits the radial counterexample;
6. finite-orbit admission produces no harmful radial updates;
7. the local gate exposes the global-identifiability failure;
8. rejected queries use exact fallback;
9. the complete orbit support contains the controlled truth;
10. the invariant query improves over unnecessary fallback.

## Scientific interpretation

The result provides public real-trajectory validation of the following
mechanism:

> A query may have benign or zero local sensitivity at a chosen gauge
> representative while varying strongly across the complete transformation
> orbit left unresolved by the observation model.

The appropriate decision is query-specific. An invariant query can be accepted
without injecting arbitrary gauge uncertainty. A radial query that changes over
the hidden orbit must be rejected or integrated over that orbit; local
linearization alone is not a global identifiability certificate.

This complements the Deform DLO4/DLO5 result. DLO4/DLO5 showed useful selective
query admission under local rank deficiency, but its registered query bank did
not contain off-axis accepted cases. Tracking Cloth now supplies the missing
real-trajectory finite-orbit counterexample and demonstrates the consequence of
incorrect local admission across independent recordings and materials.

## Limits

This experiment applies a controlled hidden SO(2) gauge to real motion-capture
geometry. It does not infer the gauge from a learned provider likelihood. It
therefore establishes mechanism validity and gate semantics, not learned visual
provider competence.

The 27 Self-collisions recordings remain a separately retained support-negative
cohort because their marker identity namespace is incompatible with the
source-selected labelled triplet. They were not silently replaced, relabelled,
or used for physical-outcome optimization.

The result does not establish:

- recovery of the physical cloth state;
- calibration of CUT3R or another learned provider;
- BayesianPhysTwin downstream benefit;
- arbitrary-cloth or arbitrary-gauge generalization;
- deployment safety;
- state-of-the-art perception performance.

The remaining ICRA-critical experiment is a support-feasible learned-provider
and downstream physical-query study. This real-geometry result substantially
strengthens the method section and failure-mode evidence but cannot substitute
for that separate provider/utility result.
