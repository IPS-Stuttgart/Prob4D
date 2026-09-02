# DOT R11–R20 query-message real-source result

## Outcome

Registered decision: **`source-real-overlap-positive`**.

This is a valid positive decision under the frozen source gate, but the empirical effect is **modest and non-confirmatory**. Equal-weight query-space covariance intersection has the lowest aggregate point RMSE, yet its paired complete-sequence confidence intervals overlap zero, the physical fallback has the best aggregate Gaussian NLL, and every registered method abstains at the 0.9 execution threshold. The result therefore supports implementation feasibility and a weak real-source dependence signal, not an ICRA-level held-out performance or decision-value claim.

## Registered design

- Public DOT V29 rope source cohort R11–R20; ten complete sequences and 49 common material-marker queries.
- Query: marker displacement from frame 3 to frame 4, normalized by frame-3 rope span.
- Window A alignment: frames 1–3; window B alignment: frames 5–7; continuous alignment: frames 1–3.
- Biases, marginal covariances, and A/B cross covariance are fitted leave-one-complete-sequence-out from the other nine sequences with equal sequence weight.
- The sealed CUT3R artifact supplies point maps and confidence but no covariance. Reported uncertainty is a source-fitted empirical error model, not native CUT3R calibration.
- R18 is the sole rank-6 sequence; the remaining nine are rank 7. The earlier fixed-rank result remains terminal negative and is not reinterpreted.

## Equal-sequence aggregate

| Method | RMSE/coord [% span] | NLL/dim | 90% cov. [%] | nNEES | Raw harmful updates [%] | Execute [%] | Deployed RMSE/coord [% span] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Physical fallback | 0.230164 | -4.354713 | 80.278 | 1.483689 | 0.000 | 0.000 | 0.230164 |
| Window A only | 0.230353 | -4.344236 | 77.778 | 1.513966 | 39.802 | 0.000 | 0.230164 |
| Window B only | 0.232532 | -4.335016 | 80.278 | 1.529311 | 57.437 | 0.000 | 0.230164 |
| Continuous CUT3R | 0.230353 | -4.344236 | 77.778 | 1.513966 | 39.802 | 0.000 | 0.230164 |
| Dense joint correlated | 0.230934 | -4.341507 | 77.778 | 1.528348 | 48.651 | 0.000 | 0.230164 |
| Equal-weight query CI | 0.229280 | -4.350357 | 80.278 | 1.500182 | 44.722 | 0.000 | 0.230164 |
| Log-det query CI | 0.230353 | -4.344236 | 77.778 | 1.513966 | 39.802 | 0.000 | 0.230164 |
| Naive independent sum | 0.230041 | -4.341000 | 80.278 | 1.526566 | 51.151 | 0.000 | 0.230164 |
| Diagonal joint covariance | 0.229693 | -4.350014 | 77.778 | 1.502938 | 50.278 | 0.000 | 0.230164 |

Equal-weight query CI changes aggregate RMSE from `0.230164%` to `0.229280%` of rope span per coordinate, a relative reduction of `0.384%`. However, its NLL is `0.004356` nats/dimension worse than fallback. Relative to naive independent addition, CI improves NLL by `0.009357` nats/dimension and RMSE by `0.000761` percentage points, but neither difference is resolved at the complete-sequence level.

All methods have `0%` execution at the registered posterior-improvement probability threshold. Consequently, deployed decision loss and harmful-execution rates are identical to the exact physical fallback. The raw CI candidate is worse than fallback on `44.722%` of marker queries, illustrating why the guard abstains.

## Paired complete-sequence uncertainty

| Difference (first minus second) | Endpoint | Estimate | 95% bootstrap interval | W/T/L |
|---|---|---:|---:|---:|
| `two_window_query_ci_equal-minus-naive_independent_message_sum` | NLL/dim | -0.009357 | [-0.028702, 0.006299] | 5/0/5 |
| `two_window_query_ci_equal-minus-naive_independent_message_sum` | RMSE/coord [% span] | -0.000761 | [-0.003683, 0.002789] | 7/0/3 |
| `two_window_query_ci_equal-minus-physical_fallback` | NLL/dim | 0.004356 | [-0.010585, 0.022371] | 7/0/3 |
| `two_window_query_ci_equal-minus-physical_fallback` | RMSE/coord [% span] | -0.000884 | [-0.004460, 0.002144] | 5/0/5 |
| `two_window_query_ci_equal-minus-window_a_only` | NLL/dim | -0.006120 | [-0.042767, 0.026138] | 5/0/5 |
| `two_window_query_ci_equal-minus-window_a_only` | RMSE/coord [% span] | -0.001073 | [-0.005072, 0.002941] | 5/0/5 |
| `two_window_query_ci_equal-minus-window_b_only` | NLL/dim | -0.015341 | [-0.042281, 0.003505] | 7/0/3 |
| `two_window_query_ci_equal-minus-window_b_only` | RMSE/coord [% span] | -0.003252 | [-0.007822, 0.000673] | 6/0/4 |

## Algebraic and custody checks

- Maximum single-message mean parity error: `4.337e-19`.
- Maximum single-message covariance parity error: `1.694e-21`.
- Duplicate-message mean and covariance errors: `0.0` and `0.0`.
- Both scientific executions produced byte-identical retained files.
- CUT3R inference was not rerun; the exact sealed provider artifact was reused.
- No raw provider arrays, images, or DOT archives are retained in this evidence directory.
- R21–R30 and R31–R70 were not opened; BayesianPhysTwin and Causal4D were not executed.

## Provenance

- Workflow run: `33589397896`.
- Evaluated revision: `400f866794821ee84565ee094f2313bff422fd0d`.
- Result ID: `f9bcd64e15a9ca77c333bda2c8c65aa0a7805d598104dc2affa32be2ef0c7c86`.
- Result SHA-256: `e3749c630b53ca7a7054bc6d1fe593a1d07cd4cf808cdcb58e50082030c42f46`.
- Protocol ID: `77f23d6e4a77e3d4ea579fc86dab54cb485264e4b649f5621258f861b42bb70d`.
- Request ID: `b20f2e65d75301b66ca82386adb47fe6085da8866a01ce2d616fad7a500e0ff2`.
- Artifact ID: `9831306394`.
- Artifact digest: `sha256:b4e9b34128d4a3716c8ee9adaf6d9d06b8d4315d9e6be2b8cd0e5e8b1fa0334c`.

## Scientific disposition

Do **not** open R21–R30 on the strength of this result. The source gate's Boolean `positive` label reflects noninferiority checks, not a practically or statistically decisive gain. The honest paper use is a secondary real-provider feasibility result: query messages remain exact across rank-6/rank-7 sequences and CI weakly improves the source point estimate relative to fallback/naive fusion, while the registered decision guard correctly finds no action-worthy evidence. A flagship result still requires a fresh, frozen cohort or task with larger motion, meaningful nonzero decisions, and proper-score or harm-control separation at the independent-sequence level.
