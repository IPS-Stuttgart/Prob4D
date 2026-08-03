# Bounded-memory provider evaluation

Prob4D retains dense fused point and covariance fields because they are part of
its public reconstruction and provider-evaluation contracts. Evaluation does not
need to duplicate those complete fields or materialize a second transformed
`FusedSequence` for every alignment mode.

## Execution model

`evaluate_sequence` processes one spatial chunk at a time. The default chunk
contains 65,536 pixels and can be changed explicitly through
`evaluation_chunk_size` or the provider command's `--evaluation-chunk-size`
option.

The evaluator uses two bounded passes over common support:

1. accumulate metric-frame error and, when requested, sufficient statistics for
   one global isotropic scale and translation;
2. accumulate aligned accuracy, covariance, flow, and framewise diagnostics.

Seam error is accumulated in a separate bounded pass over the registered
boundary pairs. Prefix and oracle alignment modes pass their fitted transform to
this evaluator rather than constructing transformed copies of the full point,
flow, and covariance fields.

Covariance inversion, log-determinant, Mahalanobis distance, and coverage are
computed only for the active rows in one chunk. Three scalar vectors remain
sequence-sized because exact median, uncertainty-error rank correlation,
tie-aware selective risk, and risk-coverage area require global ordering:

- squared Mahalanobis distance;
- relative point error;
- relative covariance-trace score.

For `N` evaluated points, these retained diagnostics require at most `24N` bytes
in `float64`, excluding Python list/array headers. Dense covariance temporaries
are bounded by the selected chunk size.

## Semantics

Chunking is an execution setting, not an estimator or artifact setting:

- point, flow, covariance, support, and alignment meanings are unchanged;
- metric, prefix-aligned, and oracle-aligned results use the same registered
  support as before;
- covariance scales by the square of the fitted point scale;
- scene flow scales by the fitted point scale but is not translated;
- no provider-v1, provider-v2, factor-bundle, or fused-prediction schema changes;
- no artifact identity changes are introduced.

The provider report records `evaluation_chunk_size` so resource-sensitive runs
remain auditable. Results may differ from the former eager implementation by
floating-point summation roundoff only. Regression tests compare small and large
chunks and compare execution-time transforms with the historical materialized
transform construction.

## Benchmark

Run the deterministic process-level benchmark with:

```bash
python -m prob4d.evaluation_memory_benchmark \
  --frames 3 \
  --height 240 \
  --width 320 \
  --evaluation-chunk-size 65536 \
  --output-json outputs/evaluation-memory.json
```

The benchmark records Python, NumPy, platform, configuration, retained-array
accounting, peak process RSS, timing, selected metric values, and a deterministic
output digest. Peak RSS includes synthetic input construction and should only be
compared between otherwise matched fresh processes.

A matched local run compared the source archive built from
`cf76ab5a250027ffd59cf085a46bf5d63e0bd551` with the bounded evaluator. Both
processes used Python 3.13.5, NumPy 2.3.5, Linux 6.12.13 x86-64, three
`240 x 320` frames, no flow, and the same deterministic seed:

| Measurement | Eager baseline | Bounded evaluator | Change |
| --- | ---: | ---: | ---: |
| Peak RSS | 327.45 MiB | 187.47 MiB | -42.75% |
| Evaluation time | 9.654 s | 8.050 s | -16.62% |
| Maximum absolute metric difference | — | `3.66e-13` | roundoff only |

This is synthetic engineering evidence for one host and process configuration,
not a runtime guarantee or a reconstruction-accuracy, uncertainty-calibration,
Bayesian-PhysTwin, or Causal4D result.

## Remaining memory work

The evaluator still requires the dense immutable prediction and truth inputs,
and exact ranking diagnostics still retain three scalar values per evaluated
point. Issue #50 continues to track memory-mapped prediction loading,
export-stage streaming, and production-host profiling at the full
`25 x 320 x 640` setting. Any such mode must remain explicit and must not alter
frozen provider semantics silently.
