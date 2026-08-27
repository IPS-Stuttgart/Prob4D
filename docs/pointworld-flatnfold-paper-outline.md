# Paper section outline enabled by the fresh-provider study

## Proposed section title

**Guarded recursive beliefs for an action-conditioned 3-D world model**

## Research question

Can Prob4D convert overlapping PointWorld forecasts into a better calibrated and
more useful recursive belief on garment-disjoint real manipulation sequences,
and does that belief improve one frozen BayesianPhysTwin query without harmful
accepted updates?

## Required table blocks

### Provider competence

- persistence;
- raw disjoint PointWorld;
- latest-window overwrite;
- simple uniform overlap;
- Prob4D production tree; and
- conformal-guarded multi-edge graph, only if source-selected.

Report garment-clustered terminal/trajectory error, seam/drift, proper score,
50/90/95 percent coverage, normalized NEES, covariance width, support, and exact
fallback.

### Downstream query

- unchanged physical fallback;
- BayesianPhysTwin plus raw PointWorld; and
- BayesianPhysTwin plus the source-selected Prob4D belief.

Report garment-clustered query error and proper score, accepted/rejected/fallback
counts, accepted-update coverage and width, harmful accepted updates, and worst
accepted regret.

## Required figures

1. Sparse representation: context scene points and action-conditioned PointWorld
   trajectories, with no image-grid rasterization.
2. Recursive belief: overlapping forecast windows, within-window identities,
   cross-window association, dependence-aware fusion, and exact fallback.
3. Calibration: reliability or coverage versus horizon and garment, including
   covariance width.
4. Downstream value: paired garment-level query deltas with accepted/fallback
   status.

## Claim rule

The section is positive only when provider support and competence pass, Prob4D
adds a resolved benefit or safety improvement over the simple comparator, and
the guarded BayesianPhysTwin query improves without exceeding the frozen harmful
update limit. Otherwise the paper reports the localized negative layer and keeps
all stronger claims out.
