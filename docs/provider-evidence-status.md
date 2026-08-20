# Prob4D provider evidence status

This page is a navigation aid for the public repository. It separates implemented
capabilities from the evidence required to promote a real 4-D observation feeder.
Exact revisions, cohort rosters, target-access decisions, and generated results
remain bound inside their owning immutable artifacts rather than this mutable
summary.

## Current scientific priority

The current repository-local priority is the frozen
[CUT3R source qualification](cut3r-qualification-runbook.md), coordinated with
[issue #49](https://github.com/IPS-Stuttgart/Prob4D/issues/49). The
[scientific kernel](scientific-kernel.md) gives the compact model and ownership
boundary.

Until that execution localizes a failure, new provider adapters, point-covariance
families, calibration scores, fusion heuristics, and target-side guards are out
of scope. A source-retained negative is a complete result; it must not be repaired
by retuning the same opened groups.

| Boundary | Current public status | Required next action |
| --- | --- | --- |
| Correlation-aware gauge mechanism | Implemented and supported by controlled mechanism studies | Retain the production causal gauge tree unless a source-localized failure identifies a concrete defect |
| Real MotionCrafter feeder | Not promoted | Do not retune on the already-open diagnostic cohort; a materially new provider or protocol requires a fresh source design |
| CUT3R recurrent-online feeder | Byte-level adapter, causal lineage, bounded import, and a frozen native-versus-Prob4D comparison are implemented | Execute support, source-mean, identity, gauge/dependence, linearization, covariance, and query-value qualification on complete source objects/sessions |
| Cross-window material identity | Append-only hypotheses and uncertainty marginalization are implemented | Do not add another identity method before the CUT3R source execution localizes identity as the first failed boundary |
| Joint `Sim(3)` linearization | Analytic first-order propagation is implemented | Run the [linearization-closure diagnostic](gauge-linearization-closure.md) before assigning remaining failure to point covariance |
| Richer point covariance | Not automatically authorized | Require the existing source-localization decision `point-covariance-localized` after support, means, identities, gauge/dependence, and closure are adequate |
| BayesianPhysTwin physical-query value | Controlled mechanism evidence exists; fresh real-provider benefit remains unconfirmed | Run one frozen independent object/session target evaluation only after complete target-free readiness passes |
| Causal4D intervention value attributable to Prob4D | Not established | Keep the one-way path through an accepted BayesianPhysTwin belief and evaluate a separately registered downstream query |
| Immutable release and branch governance | Tracked separately from scientific evidence | Complete protected-branch, independent-review, historical-tag, and released-wheel requirements in issue #243 |

## Executable decision order

The provider should advance only through the ordered target-free gates described
in [provider readiness localization](provider-readiness-localization.md):

```text
support
  -> source means
  -> identity and reliability
  -> gauge/dependence
  -> linearization closure
  -> conditional point covariance
  -> physical-query relevance
  -> one frozen target evaluation
```

A failure at one boundary redirects or stops that provider version. Downstream
performance must not rescue an upstream negative, and a valid negative result
must not be retuned on the same opened target cohort.

## Related entry points

- [Scientific kernel](scientific-kernel.md)
- [CUT3R source qualification runbook](cut3r-qualification-runbook.md)
- [CUT3R recurrent-online provider](cut3r-online-provider.md)
- [CUT3R native-versus-Prob4D comparison](cut3r-comparison.md)
- [Provider readiness localization](provider-readiness-localization.md)
- [Held-out provider promotion](heldout-provider-promotion.md)
- [Material-identity stream](material-identity-stream.md)
- [Material-identity marginalization](material-identity-marginalization.md)
- [Query-space covariance relevance](query-covariance-relevance.md)

## Claim boundary

This page summarizes repository state and decision ownership only. It is not a
claim-bearing evidence artifact and does not establish provider accuracy,
uncertainty calibration, physical-query benefit, intervention benefit,
deployment safety, or state of the art.
