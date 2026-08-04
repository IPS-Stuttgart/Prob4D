# Gauge-graph science results v1

This document records a controlled synthetic study of Prob4D's causal gauge
estimators. It is evidence about estimator mechanics under the repository's
correlated overlapping-window generator. It is not evidence of held-out
physical-object provider competence, BayesianPhysTwin acceptance or physical
prediction benefit, harmful accepted-update control, or Causal4D intervention
benefit.

## Frozen studies

Two calibration/target-separated studies were executed through draft PR #75.
Both used the same six scenarios, geometry, estimator implementations, 0.95
higher-quantile cycle threshold rule, primary endpoint, and outlier semantics.
The replication changed only seed ranges and trial counts.

| Study | Clean calibration trials | Target trials per scenario | Total target trials | Report ID |
| --- | ---: | ---: | ---: | --- |
| pilot | 12 | 16 | 96 | `f4159a193a26a6d2bee5a1aabb4303a569f2ba67e4b3d78b8df55a0fda0f689b` |
| independent replication | 48 | 128 | 768 | `a4c9a5294dbb0f6c12040494900c14f39f4c8e42a7499fc2fc5db12f4d789460` |

The replication ran from source revision
`79e590157ce2514369f395130e3938dc755628d1`, with Python 3.12.13 and NumPy
2.5.1. Its GitHub Actions artifact is ID `8882704456`, has archive digest
`sha256:890664f63ca872b156fc44b9687924cf577c28d7ecc03064aa2df1316fd903a3`,
and contains the frozen protocol, full JSON report, aggregate CSV, raw trial CSV,
Markdown summary, and verified member checksums.

## Replication design

The source-only guard threshold was calibrated before target evaluation from 48
clean trials, balanced across shared-noise correlations 0.00, 0.75, and 0.95.
The frozen threshold was
`0.02011976247247262` representative-displacement units. Target evaluation used
128 independent seeds in each of six regimes:

- independent, correlated, and highly correlated clean windows;
- correlated windows with mild overconfident skip-edge outliers;
- correlated windows with strong overconfident skip-edge outliers; and
- highly correlated windows with strong overconfident skip-edge outliers.

An injected outlier perturbed one direct skip edge that had an available directed
two-edge path, while leaving the reported edge covariance and overlap residual
unchanged. This tests cycle consistency against an apparently precise but biased
constraint rather than against a trivially high-residual edge.

The compared methods were the production single-parent joint spanning tree,
sequential marginal covariance intersection, the experimental full-joint graph,
and the full-joint graph protected by the source-only cycle gate with exact
whole-case tree fallback.

## Primary endpoint

The primary endpoint was the representative-axis endpoint displacement. Every
reported delta is `candidate - tree`; negative values favor the candidate.
Intervals are deterministic paired trial-bootstrap 95% intervals.

### Clean regimes

| Scenario | Tree | Marginal CI delta | Full-joint graph delta | Guarded graph delta | Guard fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent clean | 0.021621 | +0.001276 `[-0.000260, +0.002847]` | −0.000438 `[−0.001066, +0.000214]` | −0.000304 `[−0.000806, +0.000177]` | 41/128 |
| correlated clean | 0.010250 | +0.000293 `[−0.000532, +0.001135]` | **−0.000320 `[−0.000601, −0.000037]`** | **−0.000320 `[−0.000607, −0.000031]`** | 0/128 |
| highly correlated clean | 0.004694 | +0.000313 `[−0.000035, +0.000670]` | **−0.000192 `[−0.000323, −0.000057]`** | **−0.000192 `[−0.000325, −0.000065]`** | 0/128 |

The full-joint graph therefore reduced mean endpoint displacement by about 3.1%
in the correlated-clean regime and 4.1% in the highly-correlated-clean regime,
with paired bootstrap intervals excluding zero. The independent-clean estimate
favored the graph by about 2.0%, but its interval crossed zero. Sequential
marginal CI showed no clean advantage and had a larger mean endpoint error in all
three clean regimes.

The clean graph results did not buy accuracy through visible undercoverage:
full-joint graph 95% coverage was 0.984, 0.996, and 1.000 in the independent,
correlated, and highly correlated regimes, respectively. Mean normalized NEES
was 0.765, 0.559, and 0.436. These values indicate conservative rather than
aggressively narrow reported gauge uncertainty in this synthetic study.

The graph's mean effective edge count was only approximately 1.20 across
scenarios. Covariance intersection therefore assigned most effective information
to one edge while retaining a smaller contribution from additional edges. This
is consistent with a modest clean gain rather than a large graph-optimization
gain.

### Overconfident inconsistent edges

| Scenario | Injected edges | Tree | Marginal CI | Full-joint graph | Guarded graph | Guard detections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| correlated mild | 27 | 0.011081 | 0.017461 | 0.013737 | 0.010905 | 27/27 |
| correlated strong | 35 | 0.010108 | 0.039177 | 0.041624 | 0.010292 | 35/35 |
| highly correlated strong | 37 | 0.004636 | 0.034268 | 0.036355 | 0.004564 | 37/37 |

On only the injected trials, the failure was substantially larger:

| Scenario | Tree | Marginal CI | Full-joint graph | Guarded graph |
| --- | ---: | ---: | ---: | ---: |
| correlated mild, injected only | 0.011020 | 0.038807 | 0.024443 | 0.011020 |
| correlated strong, injected only | 0.010393 | 0.111114 | 0.124976 | 0.010393 |
| highly correlated strong, injected only | 0.004288 | 0.105723 | 0.114268 | 0.004288 |

The full-joint graph was about 4.1 times the tree endpoint error in the aggregate
correlated-strong regime and 7.8 times the tree error in the aggregate
highly-correlated-strong regime. Its aggregate normalized NEES rose to 69.1 and
284.4, with 95% coverage falling to 0.822 and 0.796. On injected trials alone,
normalized NEES reached 251 and 983. A single precise-looking biased edge can
therefore dominate the clean graph benefit.

Sequential marginal CI did not solve this problem. Its aggregate normalized NEES
was 187 and 292 in the two strong-outlier regimes; on injected trials it reached
682 and 1009. It should not be promoted as the safer multi-edge alternative.

The guarded graph detected all 99 injected edges and, because fallback is exact,
reproduced the production tree on every injected trial. Its aggregate coverage
remained 1.000, 1.000, and 0.999 in the mild, correlated-strong, and
highly-correlated-strong regimes. There were no false fallbacks among the 285
non-injected trials in those three mixed regimes.

## Guard limitation

The current guard thresholds raw representative displacement. That statistic is
not comparable across noise regimes. In clean calibration data, the mean maximum
cycle displacement was:

| Shared-noise correlation | Mean calibration maximum | Maximum calibration value |
| ---: | ---: | ---: |
| 0.00 | 0.017833 | 0.046693 |
| 0.75 | 0.008410 | 0.014267 |
| 0.95 | 0.003871 | 0.005966 |

Independent edge noise produces much larger benign cycle discrepancies than
shared correlated noise. A single global absolute threshold consequently caused
41 false fallbacks among 128 independent-clean target trials, a 32.0% rate,
while causing zero false fallbacks in 256 correlated/highly-correlated clean
trials. The guard is therefore useful as a fail-safe mechanism but is not ready
for production selection in its current unnormalized form.

## Provisional method decision

1. Keep the single-parent joint spanning tree as the production and claim-bearing
   default.
2. Retain the full-joint graph as an experimental method. Its small replicated
   clean benefit justifies physical held-out evaluation, but not promotion.
3. Retain exact whole-case tree fallback. The synthetic outlier evidence strongly
   supports guarding any multi-edge graph.
4. Do not promote sequential marginal CI; it showed no clean advantage and severe
   vulnerability to precise-looking biased edges.
5. Make source-side uncertainty-normalized cycle consistency the next method
   experiment. The target is to preserve the observed 99/99 outlier detection
   while materially reducing the 41/128 worst-clean false-fallback count without
   using target truth or downstream physical innovations.
6. After that guard is frozen, apply the existing issue #49 acceptance criteria on
   held-out physical objects or independent sessions, including point/seam/drift,
   coverage, covariance width, and a separately sealed BayesianPhysTwin gate.

## Claim boundary

These findings are controlled synthetic estimator evidence. They do not establish
that Prob4D observations improve a physical twin, that BayesianPhysTwin will
accept an update, that accepted updates are harmless, or that Causal4D
intervention forecasts improve. Issue #49 should remain open until its held-out
physical-object and downstream acceptance criteria are executed.
