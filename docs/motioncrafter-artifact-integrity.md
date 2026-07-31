# MotionCrafter artifact integrity and resume contract

`prob4d-motioncrafter` publishes prediction bundles through the crash-safe runner
in `prob4d.motioncrafter_safe`. The public route adds an evidence boundary around
each expensive inference call and loads models only through an immutable,
portable model-set contract.

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
- the immutable UNet, geometry/motion VAE, and base-pipeline model-set identity;
- the exact model-loader module hash and size; and
- every completed member's safe relative path, product kind, byte count, and
  SHA-256 digest.

The cache directory and machine-local retrieval paths are excluded from the
portable run identity.

## Immutable model sources

Every public run must identify each of its three model components as either:

- a local directory whose complete regular-file tree is content-addressed; or
- a Hugging Face repository with an exact lowercase 40- or 64-character
  revision.

Mutable branch or tag names and omitted remote revisions fail closed. The three
sources and the loader implementation form one
`prob4d.motioncrafter-model-set.v1` identity. Changing the base Stable Video
Diffusion pipeline therefore invalidates the run and calibration identity just as
changing the MotionCrafter UNet or VAE does.

See [immutable MotionCrafter model sources](motioncrafter-model-sources.md) for
the exact local and remote invocation forms.

## Resume

Resume only the identical recorded run, supplying the same local snapshots or
exact remote revisions used initially:

```bash
prob4d-motioncrafter input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/sequence_name \
  --unet-path TencentARC/MotionCrafter \
  --unet-revision <exact-commit> \
  --vae-path TencentARC/MotionCrafter \
  --vae-revision <exact-commit> \
  --base-pipeline-revision <exact-commit> \
  --resume
```

Before loading MotionCrafter, resume recomputes the input-video hash, checks the
upstream checkout state, reconstructs the model-set identity, recomputes the
run-spec digest, and verifies every journaled member. Valid members are skipped.
Missing, corrupted, path-escaping, model-drifted, or configuration-incompatible
evidence fails closed rather than being silently reused. A completed bundle is
verified and returned without loading the GPU models.

## Verification-only operation

Verify a portable bundle without the MotionCrafter environment or model sources:

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

Verification confirms the model-set identity recorded at execution time. It does
not redownload remote repositories or establish observation competence.

## Dirty upstream checkouts

A dirty MotionCrafter checkout is rejected by default. Exploratory runs may opt
in explicitly:

```bash
prob4d-motioncrafter input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/exploratory \
  --unet-path /snapshots/motioncrafter \
  --vae-path /snapshots/motioncrafter \
  --base-pipeline-path /snapshots/stable-video-diffusion \
  --allow-dirty-upstream
```

The dirty-state digest is then part of the run identity, so a later resume must
match the same checkout state. The model sources remain immutable even for this
explicit dirty-upstream diagnostic.
