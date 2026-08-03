# Tiled dense fusion

Prob4D retains dense fused means and marginal covariance because they are part of
its reconstruction and evaluation interfaces. It does not need to retain one
fully transformed dense covariance field for every contributing MotionCrafter
window at the same time.

## Execution model

`fuse_windows` processes each output frame and validity-mask pattern in bounded
spatial tiles. The default tile contains 16,384 pixels and can be changed with
`fusion_tile_size` for engineering benchmarks.

For each contributing window, the implementation materializes only the selected
rows of:

- the point map or scene-flow field transformed into the global gauge;
- the ray-parallel/lateral conditional covariance expanded into dense `3 x 3`
  matrices; and
- the optional gauge-induced covariance propagated through the local `Sim(3)`
  Jacobian.

The complete fused output remains dense and immutable. Tiling bounds
contributor-sized temporaries; it does not change output artifact schemas or
provider-v1/provider-v2 observation semantics.

## Covariance-intersection semantics

Prob4D's production dense covariance-intersection mode uses one weight vector for
all pixels with the same contributor mask in one frame. Tiled application must
not accidentally optimize a different weight in every tile.

The implementation therefore:

1. selects the same deterministic, evenly spaced sample of at most 4,096 rows as
   the previous dense path;
2. optimizes one pairwise or generalized-CI weight vector on that complete
   pattern sample; and
3. reuses the frozen weights while applying CI to each spatial tile.

Uniform and independent-precision fusion are point-separable and are evaluated
in the same bounded tile loop. Regression tests compare large and one-pixel
tiles for all three methods and verify that structured covariance is never
expanded for a complete 5,000-row contributor field.

## Benchmark

Run the installed package benchmark with:

```bash
python -m prob4d.fusion_memory_benchmark \
  --height 320 \
  --width 640 \
  --contributors 3 \
  --method covariance_intersection \
  --fusion-tile-size 16384 \
  --output-json outputs/dense-fusion-memory.json
```

The benchmark records the exact repository revision, Python and NumPy versions,
platform, configuration, retained-array accounting, process peak RSS, timing,
and a deterministic digest of the fused arrays. Peak RSS includes construction
of the synthetic inputs and should be compared only between otherwise matched
fresh processes.

A matched local synthetic run compared commit
`9496a36b9fe9f80a13396e0c65ce06841636a249` with the tiled implementation at one
`320 x 640` frame, three contributors, conditional structured covariance, gauge
covariance, and frame-global covariance intersection:

| Measurement | Eager baseline | Tiled candidate | Change |
| --- | ---: | ---: | ---: |
| Peak RSS | 579,784 KiB | 219,132 KiB | -62.2% |
| Fusion time | 3.718 s | 3.083 s | -17.1% |
| Point sum | 819,319.9940782051 | 819,319.9940782051 | exact match |
| Covariance-trace sum | 12,191.715942554092 | 12,191.715942554092 | exact match |
| Contributor-count sum | 614,400 | 614,400 | exact match |

The run used NumPy 2.3.5 on Linux 6.12.13 x86-64 with an Intel Xeon Platinum
8370C. These numbers are engineering evidence for that process and host, not a
runtime guarantee or a reconstruction-accuracy, uncertainty-calibration,
Bayesian-PhysTwin, or Causal4D result.

## Remaining work

Tiling does not eliminate the dense fused output covariance. Issue #50 still
tracks optional memory-mapped prediction loading, end-to-end production-host
profiling, export-stage memory, and hardware-recorded benchmarks at the full
25-frame setting. Those changes must remain explicit execution modes and must
not alter frozen provider artifacts silently.
