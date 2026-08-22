# CUT3R recurrent-online provider adapter

Prob4D can convert the file layout written by CUT3R's official recurrent
`demo.py` path into one provider-neutral `PredictionProviderManifestV1`. The
adapter does not import CUT3R, Torch, OpenCV, or model checkpoints. CUT3R remains
responsible for inference; Prob4D owns validation, causal lineage, portable
payloads, and downstream readiness gates.

## Admissible execution mode

Claim-bearing use is restricted to a single recurrent online pass:

- recurrent inference in source-frame order;
- `revisit=1`;
- no second pass over earlier views;
- no global alignment or bundle adjustment; and
- no future-frame postprocessing of an earlier output.

The adapter binds this declaration as `execution_mode="recurrent-online"`,
`online_prefix_only=true`, `revisit_count=1`, and `global_alignment=false`. It
cannot prove how an external process was launched. The exact CUT3R revision,
checkpoint digest, input-video digest, and generated source bytes must therefore
be retained with the run evidence. Outputs from revisiting or `demo_ga.py` are
not admissible through this route.

## Expected source layout

The source root must contain the ordinary files written by the official demo:

```text
depth/000000.npy
conf/000000.npy
camera/000000.npz
...
```

Frame stems must be six-digit, contiguous, and identical across all three
directories. Every camera archive must contain exactly:

```text
pose        4 x 4 camera-to-common-frame rigid transform
intrinsics  3 x 3 pinhole intrinsics
```

For depth `z` at pixel `(u, v)`, the adapter backprojects the pixel with the
stored intrinsics and applies the camera-to-common-frame pose. Invalid depth,
non-finite confidence, nonpositive depth, and confidence below the frozen
threshold are excluded from the canonical support mask. The confidence threshold
is a support rule, not calibrated source reliability. The exact confidence files
remain part of the source-bundle identity and must be retained for any later
source-only reliability or heteroscedastic-covariance study.

## Camera-ray and depth semantics

A point in the CUT3R common frame is

```text
p_common = R_camera_to_common (z K^-1 [u, v, 1]) + t_camera_to_common.
```

The corresponding viewing direction is not generally
`p_common / ||p_common||`, because the common-frame origin is arbitrary and the
camera translation can be nonzero. The adapter therefore exports the true unit
ray

```text
r_common =
    normalize(R_camera_to_common K^-1 [u, v, 1]).
```

Camera translation changes the common-frame point but not this direction. The
portable prediction window records these vectors as `ray_directions`, the
provider payload declares `has_ray_directions=true`, and the manifest uses the
provider-neutral ray semantic
`camera-ray-unit-vector`. Its metadata binds the stricter frame declaration

```text
camera-origin-unit-rays-in-sequence-local-frame-v1
```

so downstream code cannot reinterpret the vectors as common-origin directions
or replace them with a world-origin fallback.

Depth-conditioned covariance must likewise use camera-relative range rather than
`||p_common||`. `prob4d.cut3r_camera_geometry` recovers one camera centre per
frame from the common-frame points and rays and supplies
`CameraRelativeDepthDisagreementModel`. Its range and anisotropic covariance are
invariant to a rigid translation of the complete common frame. Recovery fails
closed when rays are absent, geometrically inconsistent, or too degenerate to
identify a camera centre.

```python
from prob4d.cut3r_camera_geometry import (
    CameraRelativeDepthDisagreementModel,
    recover_camera_relative_geometry,
)
from prob4d.data import PredictionWindow

window = PredictionWindow.from_npz(
    "outputs/provider/payloads/sequence-cut3r-online.npz",
    dense_storage_dtype="float32",
)
geometry = recover_camera_relative_geometry(window)
model = CameraRelativeDepthDisagreementModel()
conditional_covariance = model.predict(window)
```

This model is an additive source-side correction. It does not by itself promote
CUT3R, replace the ordered provider-readiness gates, or establish calibrated
target uncertainty.

## Import

```bash
prob4d prediction import-cut3r-online \
  outputs/cut3r/sequence-a \
  outputs/provider/sequence-a.json \
  --sequence-id sequence-a \
  --cut3r-revision <exact-40-or-64-character-revision> \
  --checkpoint-sha256 <checkpoint-sha256> \
  --input-video-sha256 <input-video-sha256> \
  --input-video-byte-count <bytes> \
  --frame-start 0 \
  --confidence-threshold 1.5
```

The importer:

1. rejects symlinks, missing or mismatched frames, malformed arrays, non-rigid
   poses, singular or invalid intrinsics, and empty selected support;
2. hashes every depth, confidence, and camera member while checking that it does
   not change during reading;
3. rechecks the complete source tree after canonical loading;
4. writes one immutable versioned `PredictionWindow` payload containing the
   common-frame point map, support mask, and true common-frame camera rays;
5. records frame `i` as depending on the complete source prefix ending at
   `i + 1` exclusively;
6. binds the CUT3R code revision, checkpoint, input video, source bundle, the
   complete adapter implementation-set digest, threshold, ray semantics, and
   output payload into content identities; and
7. immediately reopens and verifies the published provider manifest.

The canonical coordinate declaration is `sequence-local-sim3`. Even when an
upstream model describes its output as metric-scale, Prob4D does not promote that
claim without an independently calibrated metric anchor.

## Bounded-memory canonicalization

The adapter reads depth and confidence members through read-only NumPy memory
maps, reconstructs one frame at a time, and writes the canonical point map,
camera rays, and support mask into temporary NPY memory maps. It does not retain
a Python list of every reconstructed frame, perform a sequence-wide dense
`stack`, or ask the ordinary `PredictionWindow` constructor to copy the complete
sequence before NPZ publication. The existing portable NPZ schema already
supports optional ray directions, so no prediction-window schema migration is
required.

Imports fail closed before large allocations when any configured budget is
exceeded. The defaults are:

- `--max-frames 4096`;
- `--max-height 4096`;
- `--max-width 4096`;
- `--max-source-bytes 137438953472` (128 GiB); and
- `--max-dense-bytes 137438953472` (128 GiB of uncompressed frame indices,
  point-map values, camera-ray values, and support-mask values).

Use smaller values in automated ingestion environments. Raising a limit only
permits a larger import; it does not alter the reconstructed values, confidence
threshold, causal lineage, statistical dependence declarations, or scientific
readiness decision. The manifest separately records the historical
point/mask/index byte count, the camera-ray byte count, and their total.

## Scientific progression

A valid import proves byte-level interoperability, declared causal lineage, and
correct camera-origin ray geometry only. The next order is:

1. freeze and run provider support feasibility before opening source residuals;
2. evaluate source mean quality and identity competence by complete physical
   object or acquisition session;
3. localize gauge/dependence and conditional point-covariance failures;
4. use camera-relative range and retained source confidence only if the ordered
   source gates authorize uncertainty or reliability development; and
5. permit one held-out target evaluation only after the complete fresh-provider
   readiness decision passes.

BayesianPhysTwin still owns the baseline-relative physical update guard and exact
fallback. Causal4D consumes only the selected BayesianPhysTwin belief, never raw
CUT3R or Prob4D output.
