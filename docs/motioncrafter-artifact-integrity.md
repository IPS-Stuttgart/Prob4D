# MotionCrafter artifact integrity and resume contract

`prob4d-motioncrafter` publishes prediction bundles through the crash-safe runner
in `prob4d.motioncrafter_safe`. The low-level GPU adapter remains unchanged; the
runner adds an evidence boundary around each expensive inference call.

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
- every completed member's safe relative path, product kind, byte count, and
  SHA-256 digest.

The cache directory and retrieval paths are excluded from the portable run
identity. Model repository references are recorded exactly as supplied; callers
remain responsible for using immutable local snapshots or otherwise pinned model
revisions for claim-bearing execution.

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
