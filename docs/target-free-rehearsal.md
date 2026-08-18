# Target-free observation rehearsal

Prob4D claim-bearing source and target executions can be intentionally one-shot.
A serializer, path, packaging, or loader failure discovered only after an
outcome boundary opens cannot safely be repaired by retrying the same registered
execution. The target-free rehearsal exercises the portable observation boundary
before any source suffix or target payload is used.

The hosted CI lane is read-only, receives no dataset secrets, and operates only
on synthetic contract vectors generated inside the job.

## What the rehearsal proves

The command publishes a content-addressed `TargetFreeRehearsalReceiptV1` after
all of the following pass:

1. Prob4D's official writer serializes the normative
   `phys4d.observation_belief` version-1 vector.
2. Prob4D's official loader reconstructs the artifact and verifies its content
   address and numerical invariants.
3. `prob4d_independent_verifier` validates the same NPZ through a separate
   implementation that never imports `prob4d`.
4. The unattested normative artifact is rejected by the strict claim-bearing
   provider-v2 loader.
5. Both validation implementations reject five registered controls:
   - a future observation frame at the exclusive causal cutoff;
   - a duplicate observation identity;
   - a non-positive-definite local covariance;
   - an integer array with the wrong dtype; and
   - a changed payload retaining the pre-mutation artifact ID.
6. The installed-wheel workflow reruns and verifies the sealed receipt and
   independently compares the report and observation content identities.

The receipt contains no wall-clock timestamp. It binds the exact source revision,
package version, project identity, public-API manifest, provider manifest,
normative contract-bundle digest, environment, positive artifact and report IDs,
negative-control file digests, and zero-access declarations.

## Run before privileged execution

From the exact reviewed source revision:

```bash
prob4d diagnostic target-free-rehearsal run \
  outputs/target-free-rehearsal \
  --source-revision "$(git rev-parse HEAD)"

prob4d diagnostic target-free-rehearsal verify \
  outputs/target-free-rehearsal/target_free_rehearsal_receipt.json
```

The output directory must be new or empty. Publication is no-clobber so a second
run cannot silently replace an earlier receipt.

The standalone verifier can also be used by consumers that do not import the
Prob4D package:

```bash
python -m prob4d_independent_verifier \
  observation_belief.npz \
  --report independent_verification.json
```

It first checks the closed ZIP member set, duplicate or unsafe member names,
compressed and uncompressed size bounds, and compression ratio. It then loads
with `allow_pickle=False`, parses finite JSON while rejecting duplicate object
keys, validates exact dtypes and shapes, checks the causal and covariance
invariants, and independently recomputes the artifact ID.

Resource limits are deliberately generous for ordinary observation products and
can be tightened with `--max-archive-mib`, `--max-uncompressed-mib`, and
`--max-compression-ratio`.

## Relation to the ecosystem capsule

This rehearsal is intentionally narrower than the three-repository release
capsule. The release capsule exercises accepted and rejected Prob4D-to-
BayesianPhysTwin paths and the Causal4D decision trace. The rehearsal is a
fast, target-free pre-execution check for the observation serialization and
admission boundary, including an implementation-diverse verifier and explicit
adversarial controls.

Run both before a release or one-shot physical evaluation. Neither substitutes
for source competence, calibration, a target decision, or scientific evidence.

## Claim boundary

A passing receipt establishes only that the retained test artifact obeys the
portable schema and content lock, that the installed command path is executable,
and that the registered controls fail closed. The positive vector is synthetic
contract data. The command opens zero source-suffix payloads, zero target
payloads, and zero target outcomes. It does not establish observation accuracy,
uncertainty calibration, BayesianPhysTwin benefit, Causal4D intervention
benefit, deployment safety, or state of the art.
