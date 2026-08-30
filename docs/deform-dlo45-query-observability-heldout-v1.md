# DEFORM DLO4/DLO5 held-out query-observability result

Status: **registered held-out public-real-geometry result; all registered criteria passed**.

This note records the immutable result of the source-frozen query-aware
observability experiment. It does not authorize post-open calibration, threshold
changes, support changes, or reinterpretation as learned-provider competence.

## Scientific question

Can a rank-deficient `Sim(3)` factor be used for a downstream point query when
the query is insensitive to the unresolved gauge direction, while returning the
complete physical fallback exactly for a query that remains sensitive to that
direction?

The experiment uses real DLO4 and DLO5 trajectories as held-out geometries. A
known controlled `Sim(3)` transform and controlled 2 mm correspondence noise
provide auditable gauge ground truth. The visual-provider question is therefore
not tested here.

## Information order

The complete official DEFORM checkout was fixed at upstream revision
`b73b8b8ecc033caefa693fab7898741d4e6dbeff`.

Before any DLO4/DLO5 evaluation trajectory was opened, two source-only studies
were completed on all 112 training trajectories:

- sliding-segment geometry result:
  `04f0df72492e97de2b16b7db57da707c97a37c1fcc545f6e6853f60498fe58a9`;
- source query-gate result:
  `a3b48a522e509e53935cc42c9f1cd293cd5f7753057979c99c743a10d74c14e2`;
- source manifest SHA-256:
  `a715b4544a8395c3d8770a0b2eb4efd41d78e06e4a31969abc09beb49aea9bba`;
- fixed support: four consecutive DLO vertices;
- fixed effective-rank threshold: `0.01`;
- fixed query gate: minimum direct-observability fraction `0.90`;
- source selection rule: retain at least 99% of segment-centroid queries while
  admitting at most 10% of off-axis probes.

The source gate was fitted on 3,668 rank-six training cases. Threshold `0.90`
admitted 100% of centroid queries and 0% of off-axis probes, and was selected as
the smallest qualifying candidate.

## Held-out execution identity

- Prob4D execution revision:
  `401e3125b968fd0612b999cded8f06d182ef0a52`;
- GitHub Actions run: `33330222025`;
- job: `99307352307`;
- retained artifact: `9737419711`;
- artifact SHA-256:
  `0dd725d93e1357c1419b4cbbfbc0234fb379d17c4938547814a55e59ebc58954`;
- result ID:
  `1ac8cd083b39877888ea0eb2f4b9400ca89eda09436f25f5f0a6f43b154b1007`;
- evaluation manifest SHA-256:
  `f76200cb8259ace7054898412c91a60a9538b64066479d5c7809ce296f50c84b`.

All 28 official DLO4/DLO5 evaluation files contributed as independent groups.
The evaluation contained 1,913 successfully fitted rank-six cases: 886 from
DLO4 and 1,027 from DLO5. Six preregistered candidate fits ended in
`AlignmentNonConvergenceError`; one geometry-preselected case fitted as rank
seven and was excluded by the frozen rank-six rule.

## Comparison arms

The same held-out cases were scored using:

1. the complete physical fallback;
2. full-rank-only admission, which returns fallback for these rank-six factors;
3. unconditional observable-subspace fusion;
4. source-frozen query-aware fusion with exact fallback; and
5. an intentionally invalid full-rank completion that inserts precision into
   the factor nullspace.

The statistical unit is one official evaluation trajectory file. Values below
are equal-file means; confidence intervals are file-level percentile-bootstrap
95% intervals with 5,000 registered replicates.

## Segment-centroid query

The centroid query was admitted in every held-out case because its direct
observability fraction was effectively one: median `1.000000`, minimum
`0.999690`.

| Method | RMSE [mm] | Gaussian NLL | 90% coverage | normalized NEES | harmful accepted updates |
|---|---:|---:|---:|---:|---:|
| Physical fallback | 15.479 | -8.294 | 0.903 | 1.025 | 0.000 |
| Observable subspace, unconditional | 0.980 | -15.553 | 0.741 | 2.020 | 0.000 |
| Query-aware | **0.980** | **-15.553** | 0.741 | 2.020 | **0.000** |
| Invalid full-rank completion | 0.979 | -15.492 | 0.742 | 2.064 | 0.000 |

Query-aware fusion reduced equal-file RMSE by `14.499 mm`, with 95% interval
`[14.184, 14.807] mm`, corresponding to a `93.67%` reduction relative to
fallback. It improved Gaussian NLL by `7.259`, with 95% interval
`[7.039, 7.464]`, and won on RMSE and NLL in all `28/28` trajectory groups.

The important limitation is calibration: nominal-90% coverage was only
`74.15%` with 95% interval `[71.88%, 76.20%]`, and normalized NEES was `2.020`
with 95% interval `[1.873, 2.170]`. The held-out result therefore supports
query-selective mean and proper-score value under the frozen controlled model,
but it does **not** establish calibrated accepted-query covariance.

## Off-axis probe query

The off-axis probe retained sensitivity to the unresolved axial gauge. Its
direct-observability fraction had median `0.833139`, range
`[0.822651, 0.847924]`; every case was below the frozen `0.90` threshold.

| Method | RMSE [mm] | Gaussian NLL | 90% coverage | normalized NEES | harmful accepted updates |
|---|---:|---:|---:|---:|---:|
| Physical fallback | 22.773 | -7.219 | 0.884 | 1.025 | 0.000 |
| Observable subspace, unconditional | 11.771 | -0.531 | 0.555 | 9.441 | 0.084 |
| Query-aware exact fallback | **22.773** | **-7.219** | **0.884** | **1.025** | **0.000** |
| Invalid full-rank completion | 11.392 | 205.635 | 0.108 | 148.791 | 0.127 |

Unconditional partial-factor use reduced point RMSE, but worsened NLL relative
to fallback by `6.689`, with the paired 95% interval for NLL improvement equal
to `[-9.573, -4.370]`. It was harmful relative to fallback in `8.35%` of cases,
with equal-file 95% interval `[7.32%, 9.43%]`.

The invalid full-rank completion looked favorable under point RMSE while being
catastrophically overconfident: Gaussian NLL `205.635`, coverage `10.79%`,
normalized NEES `148.791`, and harmful-update fraction `12.65%`. This is the
failure mode that centerline or point-error-only evaluation conceals.

The query-aware method rejected all off-axis probes and reproduced the complete
fallback exactly in all rejected cases. It therefore incurred no harmful
accepted off-axis updates and preserved the fallback's uncertainty behavior.

## Registered decision

All preregistered criteria passed:

- all 28 official evaluation groups contributed;
- at least 1,000 rank-six cases were scored;
- centroid acceptance was at least 99%;
- off-axis rejection was at least 90%;
- every rejected query-aware update reproduced exact fallback;
- centroid RMSE and Gaussian NLL beat fallback; and
- harmful accepted off-axis updates were at most 1%.

## Supported contribution

The result supports the following bounded claim:

> On held-out real DLO4/DLO5 geometries under a source-frozen controlled-gauge
> protocol, query-aware observability admits rank-deficient factors for a
> gauge-insensitive physical query and rejects them for a gauge-sensitive query.
> This obtains large centroid-query gains while preventing the harmful and
> overconfident off-axis updates produced by unconditional or artificially
> full-rank fusion.

It does not establish a learned visual provider, calibrated accepted-query
covariance, arbitrary DLO generalization, end-to-end BayesianPhysTwin benefit,
Causal4D intervention benefit, deployment safety, or state of the art.

## Frozen next step

No DLO4/DLO5 evaluation-side retuning is permitted. A stronger next experiment
must use a separately preregistered real observation provider and fresh
source/target scope, for example marker-free DOT rope video with marker-assisted
3-D evaluation or a separately qualified persistent-point provider. The current
DLO4/DLO5 result remains immutable regardless of that later outcome.
