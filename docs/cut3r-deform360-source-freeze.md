# CUT3R Deform360 source freeze

This source-only stage turns the already selected ten Deform360 development
object sessions into the exact retained input bundle required by the frozen
CUT3R native-versus-Prob4D comparison. It closes the last metadata and byte
identity gap before provider inference.

It does **not** run CUT3R, decode source RGB frames, inspect residuals or future
geometry, fit uncertainty, or open any of the twelve confirmation objects.

## Frozen source roles

The independent unit is the complete physical object session. Camera streams
are nested cases, not independent replicates.

The ten source objects are partitioned before provider output as follows:

| Role | Sheet | Volumetric |
| --- | --- | --- |
| Development | `031-cotton-cloth`, episode 0 | `193-frog`, episode 7 |
| Calibration | `026-sock-cloth`, episode 7; `167-glove-gray-cloth`, episode 0 | `153-cake`, episode 5; `186-monster`, episode 6 |
| Source evaluation | `036-napkin-cloth`, episode 9; `198-kneepad-cloth`, episode 2 | `058-roll-napkin`, episode 1; `152-slime`, episode 8 |

The assignment is the deterministic within-stratum SHA-256 ranking declared in
`protocols/cut3r_deform360_source_v1.json`. It gives two development groups, four
calibration groups, and four source-evaluation groups. The four evaluation
groups also match the minimum independent-group count in the frozen diagnostic
strata.

## Prefix and camera panel

Every selected source camera case uses frames `[0, 58)` and evaluates frames
`[24, 58)`. A 25-frame window with 8-frame overlap is therefore fully defined
before the first evaluated frame. The eventual restarted execution must append
one end-anchored final window when the ordinary stride would leave the last
prefix frames uncovered.

The freeze first identifies camera streams that, in every source object session,
have all of the following:

- official aligned `undistorted.mp4` bytes;
- at least 58 aligned timestamps;
- alignment and provenance sidecars; and
- calibrated intrinsics and camera-to-world extrinsics.

From their intersection it selects four cameras by deterministic
geometry-balanced farthest-point selection on calibrated camera-center
directions. This is a source-support design decision made before residuals. It
is not a claim about the omitted cameras. The selected camera names, directions,
maximum cross-episode calibration deviations, video hashes, byte counts, and
sidecar hashes are retained in the freeze artifact.

If fewer than four common streams are available, the workflow writes the valid
terminal decision `insufficient-common-camera-support`, uploads it, and creates
no comparison specification.

## Identity binding

A passing freeze binds:

- `CUT3R/CUT3R` revision `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`;
- the exact retained `cut3r_512_dpt_4_64.pth` SHA-256 and byte count;
- the exact installed Prob4D wheel SHA-256 and source revision;
- the committed Deform360 selection-lock bytes and semantic identities;
- all forty source cases: ten object sessions by four camera streams; and
- every source video and provenance-sidecar content identity.

The emitted `cut3r-comparison-spec.json` is passed unchanged to:

```bash
prob4d prediction cut3r-comparison build \
  cut3r-comparison-spec.json \
  --output cut3r-comparison-lock.json

prob4d prediction cut3r-comparison verify \
  cut3r-comparison-lock.json
```

## Portable retained-artifact verification

The protected builder has access to the source videos and calibration files, but
reviewers of the retained result should not need those protected paths merely to
check artifact integrity. The portable verifier rechecks the source-freeze
content address, exact field sets, source/target roster separation, reconstructed
common-camera support, camera-panel membership, calibration identities, the full
source-group-by-camera case product, every source-case content address, and the
schema-v1 no-access boundary.

For a support-positive freeze, supply all three external bindings to obtain a
complete verification:

```bash
python scripts/science/verify_cut3r_deform360_source_freeze.py \
  outputs/cut3r/source-freeze/cut3r-deform360-source-freeze.json \
  --comparison-spec \
    outputs/cut3r/source-freeze/cut3r-comparison-spec.json \
  --protocol protocols/cut3r_deform360_source_v1.json \
  --selection /path/to/deform360_official_hub_selection.json \
  --require-complete-bindings \
  --require-support-pass \
  --json
```

With the protocol present, the verifier independently reconstructs the complete
comparison specification from the retained source cases and frozen windowing
policy. A comparison file whose digest was merely rebound after changing an
evaluation interval therefore fails verification. With the selection lock
present, it also rechecks the calibration and forbidden-confirmation rosters,
not only the selection file hash.

A valid support-negative freeze needs no comparison specification. Verification
returns status `0` for either a valid pass or a valid negative, status `2` for an
invalid or noncanonical artifact, and status `3` for a valid support negative
when `--require-support-pass` is requested. The command reads no source video,
prediction, residual, or confirmation payload.

## Protected runner configuration

The `cut3r-deform360-source-freeze` profile of the repository's single
`Trusted exact-head validation` workflow uses the protected
`trusted-self-hosted-validation` environment and requires these repository
variables on the self-hosted runner:

| Variable | Meaning |
| --- | --- |
| `BPT_CHECKOUT` | Trusted BayesianPhysTwin checkout containing the exact committed Deform360 selection lock |
| `CUT3R_CHECKOUT` | Trusted CUT3R checkout at the frozen revision |
| `CUT3R_CHECKPOINT` | Retained final CUT3R checkpoint path |
| `DEFORM360_PROCESSED_ROOT` | Official processed Deform360 root containing only the already opened source objects needed by this stage |

The workflow records no absolute protected path in the public summary or
content-addressed artifacts.

## Publication order

1. Merge the protocol, builder, verifier, tests, documentation, hosted contract
   workflow, and the fixed protected-runner profile.
2. Create a dedicated same-repository execution/lock pull request from that
   merged `main` revision. Keep the scientific inputs unchanged.
3. From `main`, dispatch `Trusted exact-head validation` with that open pull
   request number, its exact reviewed head SHA, and profile
   `cut3r-deform360-source-freeze`; approve the protected environment
   independently.
4. Retain the uploaded artifact. Run the portable verifier against the exact
   retained source-freeze, protocol, and selection bytes. For a support-positive
   result, also verify and commit the exact generated source-freeze and
   comparison-lock JSON under `protocols/locks/` to the execution/lock pull
   request.
5. Only after those exact locks are merged, execute the three causal arms:
   `native-continuous`, `restarted-newest`, and
   `restarted-prob4d-fused` on the frozen source inputs.
6. Stop at the first negative ordered gate. Do not open a confirmation object or
   add a new covariance/provider method unless the source result explicitly
   localizes that missing capability.

## Claim boundary

A successful freeze proves source-input support, identity, and information-order
closure only. It is not provider accuracy, calibration, BayesianPhysTwin value,
Causal4D value, deployment evidence, benchmark parity, or state of the art. A
support-negative freeze is a complete result for this exact source design.
