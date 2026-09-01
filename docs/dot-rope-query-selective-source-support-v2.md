# DOT R11-R20 query-selective source/support qualification v2

## Purpose

The frozen R04-R10 CUT3R confirmation ended with the scientific decision
`heldout-support-negative`: the learned provider was valid, but the registered
metric-fit support rule did not retain every held-out sequence. This protocol
therefore does **not** repair or reinterpret R04-R10. It starts a new,
prospective information split:

- R11-R20: source/support qualification only;
- R21-R30: future one-shot held-out confirmation;
- R31-R70: untouched reserve.

The old R11-R30 `v1` protocol required an R04-R10 strong-positive result and is
therefore terminal after the observed support-negative outcome. `v2` is a new
protocol rather than a weakened prerequisite.

## Source stage

1. Run the exact frozen CUT3R revision and checkpoint on ordinary R11-R20
   normal-view images only.
2. Seal the provider bundle before any R11-R20 marker payload is opened.
3. Open matching R11-R20 2-D/3-D marker files only for support accounting and
   the already-registered rank-deficient factor construction.
4. Evaluate a finite support-design grid consisting of three metric-fit frame
   profiles and three overlap groups.
5. Select exactly one support geometry by this lexicographic objective:
   maximize supported source sequences; maximize rank-six sequences; maximize
   the worst normalized support margin; maximize the worst nonzero observable
   information condition ratio; minimize selected frame count; then use the
   lexicographically smallest candidate ID.

No source reconstruction RMSE, NLL, coverage, proper score, or downstream
outcome enters this selection. A candidate is supported only if it has enough
metric-fit and overlap markers **and** the frozen `0.01` rank threshold yields
the intended rank-six observable factor. Promotion requires at least 9 of the
10 source sequences.

## Frozen downstream quantities

The factor rank threshold, query gate, local prior, comparison methods,
centerline/off-axis query definitions, invalid-nullspace negative control,
exact-fallback rule, and confirmation statistics are frozen in the source
protocol before R11-R20 is opened. Source data may choose only the support
geometry.

## Promotion semantics

`source-support-qualified` permits creation of a **separate** R21-R30
confirmation protocol/request that binds the source result ID and selected
support rule. It does not itself authorize any R21-R30 access.

`source-support-negative` terminates this provider version without opening
R21-R70.

## Nonclaims

This stage is source-only feasibility evidence. It does not establish
held-out benefit, fully marker-free factor construction, covariance
calibration, BayesianPhysTwin or Causal4D benefit, deployment safety, arbitrary
DLO transfer, or state of the art.
