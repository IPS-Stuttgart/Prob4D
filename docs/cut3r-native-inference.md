# Native CUT3R support

`prob4d prediction run-cut3r` runs a pinned official CUT3R model on RGB frames or
a bounded video prefix and publishes a verified, provider-neutral prediction
manifest. This is the reusable execution path, separate from the frozen science
runners. The existing depth and direct-point-map import commands remain unchanged.

## Runtime

Use a dedicated Python 3.11 environment with a compatible CUDA PyTorch build,
CUT3R's dependencies, and Prob4D installed. PyTorch 2.5.1 with CUDA 12.1 is a
conservative upstream-era build choice; newer PyTorch can require the build-only
compatibility variant below. Prob4D itself still requires only NumPy;
importing it or requesting CLI help does not load Torch, CUT3R, or a viewer.
Follow the [official installation instructions](https://github.com/CUT3R/CUT3R)
for CUDA/PyTorch and compile native RoPE in the exact source checkout:

```bash
git clone https://github.com/CUT3R/CUT3R.git /path/to/CUT3R
git -C /path/to/CUT3R checkout 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
python -m pip install -r /path/to/CUT3R/requirements.txt
cd /path/to/CUT3R/src/croco/models/curope
python setup.py build_ext --inplace
python -m pip install /path/to/Prob4D
```

Start each execution in a fresh Python process. No `demo.py`/viser import is
needed. A missing native RoPE extension, dirty tracked source, wrong revision,
or incorrect checkpoint digest fails before input staging. The supported revision
is intentionally explicit because the RGB stepping API is upstream-private.

An opt-in `--allow-native-build-compatibility` accepts only the reviewed,
SHA-256-pinned modern-ATen/SM89 variant: `tokens.type()` becomes
`tokens.scalar_type()` in `kernels.cu`, and `setup.py` targets compute/sm 89.
Every other tracked edit is still rejected. The exact accepted file digests are
in `NATIVE_BUILD_COMPATIBILITY_SHA256`; the runtime receipt binds both source and
compiled binary. This option does not edit the checkout or accept arbitrary
locally patched kernels. It is for the separately identified compatible build,
not a claim that its bytes equal the pristine upstream runtime.
To reproduce that exact variant in a fresh checkout, apply
[`runtime/cut3r-modern-aten-sm89.patch`](../runtime/cut3r-modern-aten-sm89.patch)
with `git apply` before building RoPE, then pass the opt-in flag. The patch retains
the original attribution and noncommercial CC BY-NC-SA 4.0 source licensing.
Its historical comment identifies the origin of this already-used build variant;
it confers no authority to execute the named dataset study.

Use trusted official weights only: the upstream loader unpickles a configuration
and evaluates its architecture. The runner verifies the supplied SHA-256 **before**
loading and scopes Torch's legacy-checkpoint compatibility to that load without
editing the checkout. A digest establishes identity, not trust in an arbitrary
download. The official 512 DPT checkpoint used here has SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
The 224 linear architecture is supported by the interface using `--image-size 224`
and its own independently verified checkpoint digest; this does not imply that
both checkpoints have been empirically qualified. Upstream licensing applies to
CUT3R code and weights separately from Prob4D.

## Inference

Frames must have exact six-digit names (`000000.png`, `000001.png`, ...).
Only the requested interval is staged; a missing or malformed future file is never
read. `--frame-stop` is always required and exclusive. A nonzero start resets the
state there, without secretly initializing it from earlier frames.

```bash
CUDA_VISIBLE_DEVICES=0 prob4d prediction run-cut3r \
  --cut3r-checkout /path/to/CUT3R \
  --checkpoint /path/to/cut3r_512_dpt_4_64.pth \
  --checkpoint-sha256 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 \
  --frames /path/to/permitted-frames --frame-start 0 --frame-stop 12 \
  --sequence-id my-sequence --output /path/to/new-output
```

For video, replace `--frames` with `--video /path/to/video.mp4`; FFmpeg must be
installed. No emitted frame at or after the bound reaches CUT3R. Compressed-video
decoders can internally read reference/lookahead frames: protocols requiring
strict decode custody should supply a separately staged permitted image prefix.
The output parent must exist, and the output directory must not already exist.
Use `--frame-extension .jpg` for JPEG sequences. Mixed output resolutions within
a dense window are rejected; no hidden padding or geometry resampling is used.

The runtime encodes one RGB image and updates the official recurrent decoder per
step. It does **not** call CUT3R's ray-query `inference_step`, replay the prefix,
retain all GPU image features, revisit observations, or run global alignment.
State is reset at each invocation. Raw per-frame outputs are written incrementally.

## Products and semantics

```text
new-output/
  direct/{points,depth,conf,camera}/...
  prediction/provider.json
  prediction/payloads/...
  run.json
```

- `points`: original camera-frame XYZ, not depth-reprojected approximations.
- `depth`: the direct map's Z component; `conf`: raw CUT3R source confidence.
- `camera`: camera-to-sequence pose and estimated intrinsics.
- Canonical payload: sequence-local XYZ, valid mask, camera-origin unit rays,
  exact frame indices, and causal lineage through the current frame only.
- Coordinates remain **sequence-local Sim(3)**, not registered metres. A separately
  validated gauge is required before metric BayesianPhysTwin assimilation.
- No material identity, scene flow, deformation mask, or calibrated covariance is
  invented from dense point-map differences or confidence. Dense pixels and views
  remain dependent evidence. Existing covariance/calibration adapters are separate.
- `run.json` binds runtime/source/kernel/checkpoint, inputs, raw outputs, manifest,
  and implementation hashes. For image directories, the legacy importer's
  `input_video_sha256` slot holds the ordered-prefix inventory digest; `run.json`
  explicitly identifies it as `image-prefix`, not an actual video-file hash.

The `prediction` directory appears only after complete inference and successful
manifest verification. A technical failure retains a failure receipt and partial
raw generated outputs, never a successful manifest. Generated staging images are
cleaned on either outcome; source files are not modified. A repeated output path
is rejected rather than overwriting evidence.

## Dataset-free verification

```bash
PYTHONPATH=src python scripts/smoke_cut3r_native.py \
  --cut3r-checkout /path/to/CUT3R \
  --checkpoint /path/to/cut3r_512_dpt_4_64.pth \
  --device cuda:0 --output /path/to/new-synthetic-smoke
```

This generates three synthetic RGB frames, runs the complete export, checks the
manifest, tests reset/prefix closure, and compares streamed XYZ to upstream's
`forward_recurrent` on the same frames. It tests software compatibility only.
Historical quarantines, failed/no-retry studies, protected cohorts, and scientific
promotion decisions are not reopened or superseded by this command.

### Verified software result

The [synthetic smoke summary](../evidence/cut3r-native-software-smoke-v1/summary.json)
records a successful native run of the official 512 DPT checkpoint on an RTX 4090,
Torch 2.11.0+cu126, using the explicitly bound native-build compatibility variant.
All three 384x512 direct point maps exported and the manifest verified. Maximum
absolute XYZ difference was **0** both against official `forward_recurrent` and
against the shorter-prefix replay. Peak allocated GPU memory was 3,691,858,432
bytes (about 3.44 GiB). The three production source modules' hashes in the receipt
were independently compared with the delivered files and matched exactly.

These recorded module hashes identify the initial image-path implementation
at commit `6b92dc8`; the subsequent FFmpeg compatibility repair changes only
video staging/error reporting, not the native RGB recurrence or the image path.
The real video CLI was also verified using a three-frame synthetic FFV1 clip
with an exclusive two-frame bound on FFmpeg 4.4.2. It published exactly two
predictions, whose point-map files were byte-identical to the corresponding
image-path files. This caught and repaired an unsupported `-fps_mode` option;
the command now uses the compatible `-vsync 0` form and retains decoder errors
in failure receipts. The separate video receipt binds the updated provider hash.
The distributed build patch was applied to pristine pinned upstream files, and
both resulting source hashes matched the runtime's expected compatibility hashes.

The software check first encountered a checkpoint-file permission error before
model loading; that failed synthetic workspace was retained. A readable copy of
the same digest-verified official checkpoint was then transferred directly
between hosts for the successful software run. No historical scientific attempt
was retried. Raw synthetic outputs and full runtime receipts remain at
`gpuserver4090:/home/florianpfaff/source-only/cut3r-native-support-v1-synthetic-smoke-readable`.
This result qualifies the RGB execution/export interface for that runtime, not
real-data accuracy, metric scale, uncertainty calibration, or transfer.
