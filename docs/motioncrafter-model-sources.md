# Immutable MotionCrafter model sources

The public `prob4d motioncrafter` and `prob4d-motioncrafter` routes require an
immutable identity for every model component used during inference:

1. the MotionCrafter UNet;
2. the geometry/motion VAE; and
3. the base Stable Video Diffusion pipeline.

A mutable repository name such as `TencentARC/MotionCrafter` or a branch name such
as `main` is not sufficient for a claim-bearing run.

## Supported source forms

Each component may be supplied in one of two forms.

### Exact remote revision

Pass a Hugging Face repository and an exact lowercase 40- or 64-character commit
revision:

```bash
prob4d-motioncrafter input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/sequence \
  --unet-path TencentARC/MotionCrafter \
  --unet-revision <exact-commit> \
  --vae-path TencentARC/MotionCrafter \
  --vae-revision <exact-commit> \
  --base-pipeline-path stabilityai/stable-video-diffusion-img2vid-xt \
  --base-pipeline-revision <exact-commit>
```

The exact revision is forwarded to every `from_pretrained` call. Tags, branches,
short revisions, and omitted revisions fail closed.

### Local content-addressed snapshot

Pass a local directory and omit the corresponding revision option:

```bash
prob4d-motioncrafter input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/sequence \
  --unet-path /snapshots/motioncrafter \
  --vae-path /snapshots/motioncrafter \
  --base-pipeline-path /snapshots/stable-video-diffusion
```

The producer recursively hashes every regular file, following file symlinks and
excluding only `.git` metadata. The portable source descriptor records the tree
SHA-256, file count, and total byte count without recording the machine-local
snapshot path.

The local UNet snapshot must contain `unet_determ` or `unet_diff`, according to
`--model-type`. The VAE snapshot must contain `geometry_motion_vae`, and the base
pipeline must contain `model_index.json`.

## Portable model-set identity

The three source descriptors, selected model type, and SHA-256/byte identity of
the executed `prob4d.motioncrafter_models` loader are canonicalized into:

```text
prob4d.motioncrafter-model-set.v1:<sha256>
```

The public run config stores role-specific identities derived from this value
rather than mutable paths. Consequently:

- changing any model revision or local file changes the run-spec digest;
- `--resume` refuses a different model set before loading GPU models;
- the existing MotionCrafter calibration model identifier changes as well,
  preventing covariance calibration from being reused silently across model
  bytes or base-pipeline revisions;
- a moved but byte-identical local snapshot retains the same portable identity.

The full compact model-source manifest is embedded in the run configuration and
therefore in `artifact_integrity.run_spec`.

## Verification boundary

`--verify-only` validates a completed bundle without requiring model access. It
confirms the model-set identity that was bound at execution time, but it does not
redownload or rehash remote model repositories after the fact.

This contract establishes execution provenance, not model competence. A
content-valid model set can still produce inaccurate or poorly calibrated
observations; held-out provider evaluation and the separate Bayesian-PhysTwin
guard remain required.
