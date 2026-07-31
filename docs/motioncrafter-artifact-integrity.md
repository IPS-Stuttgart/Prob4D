# MotionCrafter artifact integrity and resume contract

`prob4d-motioncrafter` publishes prediction bundles through the crash-safe runner
in `prob4d.motioncrafter_safe`. The runner wraps the GPU adapter with an evidence
boundary around every expensive inference call.

## Publication rules

A new run refuses a nonempty output directory. Each baseline and independently
decoded overlap window is written to a temporary file, flushed, and atomically
renamed before its SHA-256 digest and byte count enter the progress journal. The
final `predictions.json` is published only after all referenced members validate.
A crash therefore cannot leave a newly published manifest pointing to a mixture
of old and new prediction files.

The hidden `.motioncrafter-progress.json` journal binds:

- the input-video SHA-256 digest and byte count;
- the MotionCrafter Git object ID, clean/dirty state, and status digest;
- all output-affecting inference settings and the stochastic seed policy;
- the exact executed Prob4D producer-module bytes;
- every completed member's safe relative path, product kind, byte count, and
  SHA-256 digest.

The cache directory and machine-local retrieval paths are excluded from the
portable run identity.

## Exact model snapshots

New runs can bind immutable Hugging Face snapshots for all three model products:

```bash
prob4d-motioncrafter input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/sequence_name \
  --unet-revision <exact-40-or-64-character-revision> \
  --vae-revision <exact-40-or-64-character-revision> \
  --base-model-revision <exact-40-or-64-character-revision>
```

The corresponding repository references are configurable through `--unet-path`,
`--vae-path`, and `--base-model-path`. Supplied revisions are passed directly to
all three `from_pretrained` calls and become part of the content-addressed run
specification.

Snapshot-aware manifests use the version-3 MotionCrafter model identifier.
Claim-bearing calibration and provider-v2 export fail closed unless all three
revisions are exact lowercase 40- or 64-character object IDs. Historical
manifests without snapshot fields retain their frozen version-1/version-2 model
identifiers.

## Resume

Resume only the identical recorded run:

```bash
prob4d-motioncrafter input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/sequence_name \
  --resume
```

Before loading MotionCrafter, resume recomputes the input-video hash, checks the
upstream checkout state, recomputes the run-spec digest, and verifies every
journaled member. Valid members are skipped. Missing, corrupted, path-escaping,
or configuration-incompatible evidence fails closed rather than being silently
reused. A completed bundle is verified and returned without loading the GPU
models.

`prob4d-benchmark --skip-existing` uses the same runner. It resumes only a
matching content-verified producer bundle and regenerates fused outputs; it no
longer accepts result files merely because their destination paths exist.

## Verification-only operation

Verify a portable bundle without the MotionCrafter environment:

```bash
prob4d-motioncrafter \
  --output-dir outputs/sequence_name \
  --verify-only
```

New manifests contain `artifact_integrity` with the complete run spec and an
exact descriptor for every referenced NPZ member. `load_prediction_bundle` also
performs this verification before opening arrays. Legacy format-version-1
manifests remain readable, but they are marked as integrity-unbound and still
receive strict relative-path and bundle-root containment checks.

Causal export follows a narrower rule. It validates the manifest, stochastic
seed schedule, run-spec digest, and all member descriptors without opening
payloads after the causal cutoff. It then verifies the size and SHA-256 digest of
only the admitted prefix windows. For non-unit frame strides, admission uses the
maximum source frame actually consumed by a window, not the next sampling-grid
boundary recorded as `stop_frame`.

## Dirty upstream checkouts

A dirty MotionCrafter checkout is rejected by default. Exploratory runs may opt
in explicitly:

```bash
prob4d-motioncrafter input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/exploratory \
  --allow-dirty-upstream
```

The dirty-state digest is then part of the run identity, so a later resume must
match the same checkout state.
