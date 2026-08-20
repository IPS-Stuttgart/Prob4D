# CUT3R direct point-map import and fidelity closure

## Why this route exists

CUT3R's recurrent demo predicts `pts3d_in_self_view`, a dense three-dimensional
point map in the current camera/self-view coordinates. The historical CUT3R demo
writes only its third coordinate as `depth`, estimates pinhole intrinsics from the
predicted point map, and later reconstructs XYZ from that depth and those fitted
intrinsics.

That reconstruction is not guaranteed to reproduce the original three-channel
prediction. A deterministic lateral distortion introduced at this boundary could
otherwise be misdiagnosed downstream as gauge error, point uncertainty, material
identity failure, or coherent visual bias.

Prob4D therefore provides two deliberately separate routes:

- `prob4d prediction import-cut3r-direct` preserves the original direct XYZ point
  map and transforms it only by the matching CUT3R camera pose;
- `prob4d prediction import-cut3r-online` retains the historical
  depth-plus-intrinsics reprojection as an explicit compatibility route.

The compatibility route remains available for frozen evidence. New CUT3R source
qualification should use the direct route unless a target-free fidelity report
classifies depth reprojection as equivalent under a threshold frozen before
source outcomes are read.

## External CUT3R output layout

Prob4D does not import or execute CUT3R. A thin external wrapper should retain the
ordinary recurrent-online output and add the direct point map before converting
tensors to a visualization-only representation:

```python
points = pts3ds_self_tosave[frame_id].detach().cpu().numpy()
np.save(output_dir / "points" / f"{frame_id:06d}.npy", points)
```

The direct importer expects:

```text
points/000000.npy   # H x W x 3, pts3d_in_self_view
conf/000000.npy     # H x W, original CUT3R confidence
camera/000000.npz   # exact pose and intrinsics members
...
```

Frame stems must be six-digit, contiguous, and identical across all three
directories. The importer rejects symbolic links, malformed grids, non-rigid
poses, invalid intrinsics, source mutation during reading, resource-limit
violations, and empty selected support.

Confidence remains an uncalibrated provider score. The frozen threshold determines
support only; it is not converted into Bayesian reliability or an association
probability. Every original confidence member is bound into the direct provider
source identity.

## Import direct XYZ

```bash
prob4d prediction import-cut3r-direct \
  outputs/cut3r/raw/group-a/direct \
  outputs/cut3r/provider/group-a/direct.json \
  --sequence-id group-a-direct \
  --cut3r-revision <exact-revision> \
  --checkpoint-sha256 <checkpoint-sha256> \
  --input-video-sha256 <input-video-sha256> \
  --input-video-byte-count <input-video-bytes> \
  --frame-start 0 \
  --confidence-threshold <frozen-threshold>
```

For camera-frame point `p` and the matching camera-to-common-frame pose `T`, the
canonical payload stores

```text
y = T p.
```

It does not replace `p` with `z K^-1 [u, v, 1]'`. The manifest records
`geometry_source=pts3d-in-self-view-direct-v1`,
`direct_pointmap_preserved=true`, and `depth_reprojection_used=false`.

## Target-free geometry and causal-prefix audit

For the audit, retain the complete four-directory layout:

```text
points/  depth/  conf/  camera/
```

Run CUT3R once on an exact prefix and once on a longer sequence whose first
frames and execution settings are identical. Then build the report:

```bash
prob4d prediction cut3r-fidelity build \
  outputs/cut3r/raw/group-a/prefix \
  outputs/cut3r/raw/group-a/extended \
  outputs/cut3r/audit/group-a.json \
  --prefix-frame-count <count> \
  --confidence-threshold <frozen-threshold> \
  --maximum-rms-error-m <frozen-equivalence-margin> \
  --maximum-frame-p95-error-m <frozen-equivalence-margin> \
  --require-direct-ready
```

Replay it from the retained bytes with:

```bash
prob4d prediction cut3r-fidelity verify \
  outputs/cut3r/audit/group-a.json \
  outputs/cut3r/raw/group-a/prefix \
  outputs/cut3r/raw/group-a/extended \
  --require-direct-ready
```

The report contains two distinct decisions:

1. **Geometry classification.** It compares direct XYZ with the historical
   depth-reprojected representation on common support. A result of
   `direct-pointmap-required` is not a provider failure; it means the compatibility
   representation is not equivalent under the frozen margin.
2. **Causal-prefix closure.** It verifies that direct points, depth, confidence,
   cameras, finite-value patterns, and selected support in the prefix agree with
   the already-emitted part of the longer run. Failure blocks claim-bearing use of
   that execution mode.

The report retains the exact path, SHA-256, and byte count of every prefix and
extended source member, together with thresholds, tolerances, per-frame metrics,
and its own content identity. It opens no truth, physical innovation, target
outcome, or BayesianPhysTwin decision.

## Scientific order

Before source residuals or target outcomes are opened:

1. freeze CUT3R revision, checkpoint, argument vector, input bytes, causal mode,
   confidence rule, and complete object/session roster;
2. run and retain the direct-point-map fidelity and causal-prefix report;
3. import the direct point maps and attest every execution;
4. evaluate support, source means, and identity competence;
5. only then evaluate gauge/dependence, nonlinear closure, conditional covariance,
   and physical-query relevance; and
6. authorize one target evaluation only after the complete target-free readiness
   decision passes.

A valid artifact or passing representation audit is infrastructure evidence. It
does not establish real-provider competence, calibrated uncertainty,
BayesianPhysTwin physical benefit, Causal4D intervention benefit, deployment
safety, or state of the art.
