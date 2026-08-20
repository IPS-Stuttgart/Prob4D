# CUT3R depth-reprojected recurrent-online compatibility adapter

> **Representation boundary.** This route imports CUT3R's historical
> `depth/conf/camera` demo output and reconstructs XYZ from depth and fitted
> intrinsics. It remains supported for frozen evidence and compatibility. New
> source qualification should preserve CUT3R's original direct
> `pts3d_in_self_view` output through the
> [direct point-map route](cut3r-direct-pointmap.md) and complete its target-free
> geometry-fidelity and causal-prefix-closure audit before source residuals are
> opened.

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
is a support rule, not calibrated source reliability.

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
4. writes one immutable versioned `PredictionWindow` payload;
5. records frame `i` as depending on the complete source prefix ending at
   `i + 1` exclusively;
6. binds the CUT3R code revision, checkpoint, input video, source bundle, the
   complete adapter implementation-set digest, threshold, and output payload into
   content identities; and
7. immediately reopens and verifies the published provider manifest.

The canonical coordinate declaration is `sequence-local-sim3`. Even when an
upstream model describes its output as metric-scale, Prob4D does not promote that
claim without an independently calibrated metric anchor.

## Bounded-memory canonicalization

The adapter reads depth and confidence members through read-only NumPy memory
maps, reconstructs one frame at a time, and writes the canonical point map and
support mask into temporary NPY memory maps. It no longer retains a Python list
of every reconstructed frame, performs a sequence-wide dense `stack`, or asks the
ordinary `PredictionWindow` constructor to copy the complete sequence before NPZ
publication. The portable NPZ schema and provider-manifest semantics are
unchanged.

Imports fail closed before large allocations when any configured budget is
exceeded. The defaults are:

- `--max-frames 4096`;
- `--max-height 4096`;
- `--max-width 4096`;
- `--max-source-bytes 137438953472` (128 GiB); and
- `--max-dense-bytes 137438953472` (128 GiB of uncompressed frame indices,
  point-map values, and support-mask values).

Use smaller values in automated ingestion environments. Raising a limit only
permits a larger import; it does not alter the reconstructed values, confidence
threshold, causal lineage, statistical dependence declarations, or scientific
readiness decision. The manifest records the actual source-member and dense-array
byte counts together with `canonicalization_backend="frame-streamed-npy-memmap-v1"`.

## Scientific progression

A valid import proves byte-level interoperability and declared causal lineage
only. The next order is:

1. complete the direct-point-map representation and causal-prefix preflight;
2. freeze and run provider support feasibility before opening source residuals;
3. evaluate source mean quality and identity competence by complete physical
   object or acquisition session;
4. localize gauge/dependence and conditional point-covariance failures;
5. fit reliability or uncertainty only when the corresponding source gate
   authorizes it; and
6. permit one held-out target evaluation only after the complete fresh-provider
   readiness decision passes.

BayesianPhysTwin still owns the baseline-relative physical update guard and exact
fallback. Causal4D consumes only the selected BayesianPhysTwin belief, never raw
CUT3R or Prob4D output.
