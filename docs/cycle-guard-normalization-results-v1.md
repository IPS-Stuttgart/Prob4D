# Uncertainty-normalized cycle-guard results v1

This document records the preregistered controlled synthetic experiment
`prob4d-cycle-guard-normalization-v1`. It evaluates a source-only uncertainty
normalization for Prob4D's causal gauge-graph cycle guard. It is not evidence of
held-out physical-object provider competence, BayesianPhysTwin acceptance or
physical-prediction benefit, harmful accepted-update control, or Causal4D
intervention benefit.

## Experiment

The experiment used 48 balanced clean calibration trials and 128 independent
target trials in each of six registered scenarios, for 768 target trials total.
Calibration and target seed ranges were disjoint from one another and from the
earlier gauge-graph pilot and replication.

Both guards used a threshold frozen before target evaluation at the higher
empirical 0.95 calibration quantile:

| Guard | Frozen threshold |
| --- | ---: |
| raw representative displacement | `0.020558842225246163` |
| uncertainty-normalized score | `0.7605970627575223` |

The normalized score propagates each direct and path-edge marginal covariance
through analytic `Sim(3)` composition and representative-point Jacobians. For
each point it sums the three root expected squared-displacement contributions.
That sum is a Minkowski upper bound under arbitrary cross-edge correlation. The
resulting dimensionless statistic is empirically calibrated; it is not interpreted
as chi-square.

Every fallback was required to equal the analytic production spanning tree
exactly, including transform vectors and joint covariance arrays.

## Preregistered primary decision

The candidate had to pass every registered condition:

| Criterion | Result |
| --- | --- |
| strong-outlier detection at least 0.95 | PASS |
| strong-outlier detection no more than 0.05 below raw | PASS |
| mild-outlier detection at least 0.90 | PASS |
| mild-outlier detection no more than 0.05 below raw | PASS |
| worst clean false-fallback rate at most 0.10 | **FAIL** |
| worst clean false-fallback rate at most half raw | PASS |

**Overall preregistered decision: FAIL.**

This is an informative partial negative result, not a failed execution. The
quality checks, 13 focused regressions, 48 calibration trials, 768 target trials,
protocol checks, exact-fallback assertions, and evidence export all completed.

## Detection and false fallback

The injected edges were deliberately hard: one direct skip edge was biased while
its reported covariance and overlap residual were left unchanged.

| Quantity | Raw guard | Uncertainty-normalized guard |
| --- | ---: | ---: |
| all injected edges detected | 89/89 | 89/89 |
| mild injected edges detected | 34/34 | 34/34 |
| strong injected edges detected | 55/55 | 55/55 |
| independent-clean false fallback | 38/128 (29.7%) | 12/128 (9.4%) |
| correlated-clean false fallback | 0/128 | 17/128 (13.3%) |
| highly-correlated-clean false fallback | 0/128 | 10/128 (7.8%) |
| worst clean false fallback | 29.7% | 13.3% |

Normalization therefore preserved 100% detection and reduced the worst clean
false-fallback rate by 55.3% relative to the raw guard. It nevertheless missed
the absolute preregistered 10% limit by 3.3 percentage points. The error pattern
also shifted: independent-clean behavior improved substantially, while the
correlated-clean regime became the new worst group.

## Secondary estimator effects

Endpoint differences below are `candidate - production tree`; negative values
favor the candidate. Intervals are deterministic paired trial-bootstrap 95%
intervals.

### Clean scenarios

| Scenario | Method | Endpoint | Paired delta vs tree | Coverage 95% | Normalized NEES | Fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| independent clean | tree | 0.020885 | 0 | 0.995 | 0.708 | — |
| independent clean | full graph | 0.020901 | +0.000015 `[-0.000556, +0.000643]` | 0.992 | 0.716 | — |
| independent clean | raw guard | 0.020941 | +0.000055 `[-0.000391, +0.000508]` | 0.992 | 0.715 | 29.7% |
| independent clean | normalized guard | 0.020970 | +0.000084 `[-0.000471, +0.000614]` | 0.992 | 0.716 | 9.4% |
| correlated clean | tree | 0.009853 | 0 | 0.997 | 0.570 | — |
| correlated clean | full graph | 0.009835 | -0.000018 `[-0.000317, +0.000291]` | 0.997 | 0.575 | — |
| correlated clean | raw guard | 0.009835 | -0.000018 `[-0.000307, +0.000303]` | 0.997 | 0.575 | 0% |
| correlated clean | normalized guard | 0.009868 | +0.000014 `[-0.000264, +0.000310]` | 0.997 | 0.576 | 13.3% |
| highly correlated clean | tree | 0.004643 | 0 | 1.000 | 0.396 | — |
| highly correlated clean | full graph | 0.004643 | +0.000000 `[-0.000129, +0.000137]` | 1.000 | 0.404 | — |
| highly correlated clean | raw guard | 0.004643 | +0.000000 `[-0.000127, +0.000138]` | 1.000 | 0.404 | 0% |
| highly correlated clean | normalized guard | 0.004645 | +0.000002 `[-0.000122, +0.000123]` | 1.000 | 0.405 | 7.8% |

The normalized guard did not create a material clean endpoint, coverage, or NEES
regression. Its main deficiency is admission efficiency, not the posterior
returned after admission.

### Outlier scenarios

| Scenario | Method | Endpoint | Paired delta vs tree | Coverage 95% | Normalized NEES | Fallback/detection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| correlated mild | tree | 0.009798 | 0 | 1.000 | 0.556 | — |
| correlated mild | full graph | 0.015001 | +0.005203 `[+0.002059, +0.009325]` | 0.889 | 4.064 | — |
| correlated mild | raw guard | 0.009948 | +0.000151 `[-0.000088, +0.000403]` | 1.000 | 0.558 | 34/34 detected |
| correlated mild | normalized guard | 0.009968 | +0.000170 `[-0.000062, +0.000406]` | 1.000 | 0.560 | 34/34 detected |
| correlated strong | tree | 0.010396 | 0 | 1.000 | 0.577 | — |
| correlated strong | full graph | 0.035522 | +0.025127 `[+0.010546, +0.043280]` | 0.866 | 45.314 | — |
| correlated strong | raw guard | 0.010219 | -0.000177 `[-0.000464, +0.000106]` | 0.999 | 0.582 | 26/26 detected |
| correlated strong | normalized guard | 0.010201 | -0.000194 `[-0.000486, +0.000086]` | 0.999 | 0.581 | 26/26 detected |
| highly correlated strong | tree | 0.005059 | 0 | 0.999 | 0.433 | — |
| highly correlated strong | full graph | 0.025594 | +0.020535 `[+0.010897, +0.032678]` | 0.827 | 160.325 | — |
| highly correlated strong | raw guard | 0.004927 | -0.000132 `[-0.000247, -0.000011]` | 0.999 | 0.435 | 29/29 detected |
| highly correlated strong | normalized guard | 0.004914 | -0.000145 `[-0.000267, -0.000023]` | 0.999 | 0.433 | 29/29 detected |

The result independently confirms that an unguarded multi-edge graph remains
unsafe under one apparently precise biased edge. Both guards prevented that
failure and retained conservative coverage.

## Why the normalized guard missed its gate

Normalization largely removed the mean scale disparity between correlation
regimes. In clean target trials, mean maximum normalized scores were approximately
0.590, 0.587, and 0.560 for correlations 0.00, 0.75, and 0.95. The frozen
threshold was 0.761.

The remaining failure is a tail-calibration problem rather than the original
raw-unit mismatch. With only 48 pooled calibration trials, the group-specific
clean target tails exceeded the frozen threshold at rates of 9.4%, 13.3%, and
7.8%. A future method must control the worst registered source group rather than
only the pooled clean mixture.

## Method decision

1. Keep the production spanning tree as the claim-bearing default.
2. Keep the full-joint graph and both guards experimental.
3. Do not promote this normalized guard because the preregistered overall gate
   failed.
4. Preserve the normalization machinery: it retained all injected-edge detection
   and more than halved worst-clean false fallback.
5. Test a new, independently seeded protocol using source-only finite-sample
   group-robust or conformal threshold calibration. Any source stratum must be
   declared before target access and derivable from edge/cycle uncertainty
   metadata, not from truth or downstream innovation.
6. Physical-object and BayesianPhysTwin promotion gates remain unexecuted.

## Evidence identity

- report ID: `d51e4cd30115d774cd59f09fff03fe920f56ff4ecf4cacfee974822a5a88d162`
- source revision: `0b6e67757ce141711036103fcefda8273ef5d981`
- workflow run: `30887486572`
- artifact ID: `8883841829`
- artifact ZIP SHA-256: `8d8015e1bfb8aedca3f3609f5a8c2ba7674d2adaaf4d29cba400d97845430e1c`
- raw trial CSV SHA-256: `c1684a801cae7d37729b9d1b80f2689f9d1c0ef89bddd765ecc45218dfdd7c65`
- protocol SHA-256: `8da995c4a0215908b67443da8110f6de239189ab3eabffc6af2fb713f9f3513e`

The downloaded artifact was rechecked locally. Every member listed in its
`SHA256SUMS` file matched; the raw-trial CSV hash is recorded above because the
original evidence writer's glob did not include the `_trials.csv` filename. The
writer is corrected in the implementation branch.

## Claim boundary

Controlled synthetic source-cycle admission only. No held-out physical-object
provider, BayesianPhysTwin, harmful-update, or Causal4D benefit is claimed.
