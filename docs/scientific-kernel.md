# Prob4D scientific kernel

This page is the shortest claim-safe description of Prob4D's statistical model,
recursive uncertainty propagation, and ownership boundary. Detailed schemas,
commands, and experiment protocols remain in the linked documents.

## 1. Causal overlapping-window observation model

Let \(w=1,\ldots,K\) index independently decoded causal prediction windows. A
local point \(p_{wi}\in\mathbb{R}^3\) has conditional covariance
\(R^{\mathrm{local}}_{wi}\). Window \(w\) has an uncertain global
\(Sim(3)\) gauge \(g_w\in\mathbb{R}^7\). Around the estimated gauge
\(\bar g_w\),

```text
y_wi = Sim3(g_w) p_wi + epsilon_wi

delta y_wi
  approximately A_wi delta p_wi + J_wi delta g_w,
```

where \(A_{wi}\) maps local point perturbations into the world frame and
\(J_{wi}\) is the point-to-gauge Jacobian.

After stacking all admitted rows,

```text
Sigma_y = R_cond + J_g Sigma_g J_g^T.
```

Here:

- `R_cond` is the block-diagonal conditional world-point covariance;
- `Sigma_g` is the ordered joint covariance of all retained gauges; and
- `J_g` is sparse because each observation row depends on its own window gauge.

The off-diagonal blocks of `Sigma_g` matter. A shared metric anchor and recursive
relative-gauge propagation induce covariance between observations from different
windows. Replacing `Sigma_g` by independent marginal blocks preserves individual
row variances but destroys that cross-row dependence.

Prob4D can export the shared term through a low-rank factor `U` satisfying

```text
U U^T approximately J_g Sigma_g J_g^T.
```

Rank reduction is accepted only under the declared retained-covariance criterion;
otherwise the full factor or exact fallback is retained.

## 2. Recursive gauge estimation

The production gauge path starts from an independently fitted metric anchor and
propagates relative `Sim(3)` alignments in causal source order. The default
sequential tree avoids treating redundant edges as independent measurements.
Composition Jacobians propagate both marginal and cross-window covariance, so a
new window remains correlated with all ancestors that share anchor or transition
uncertainty.

When contributor cross-correlation is unknown, covariance intersection provides
a conservative fusion rule. It can protect consistency but does not guarantee a
better realized mean. Additional graph edges, normalized cycle guards, and
analytic propagation remain admissible only under their registered source-side
guards and closure tests.

First-order propagation is not assumed valid merely because the algebra is
implemented. The joint `Sim(3)` linearization-closure diagnostic compares the
analytic result with a deterministic nonlinear reference before a remaining
failure can be assigned to conditional point covariance.

## 3. Two valid downstream covariance representations

A consumer must choose exactly one of the following representations.

### Explicit gauge nuisance variables

Use:

```text
conditional point covariance
+ point-to-gauge Jacobian
+ full joint gauge prior covariance.
```

This is the richer `ObservationFactorBundle` path. It lets BayesianPhysTwin keep
gauge errors as explicit nuisance variables.

### Marginalized portable observation

Use:

```text
conditional point covariance
+ shared low-rank gauge factor.
```

This is the portable `ObservationBeliefV1` path.

Do not combine marginal point covariance with explicit gauge variables, and do
not add the shared factor twice. Either mistake double-counts the same gauge
uncertainty.

For repeated Gaussian scoring or inference, Prob4D supplies structured covariance
actions, inverse actions, precision quadratics, log determinants, and negative log
likelihoods without materializing the dense `3N x 3N` covariance.

## 4. Reliability and dependence are separate quantities

The observation contracts deliberately keep these values distinct:

- `association_probability`: support for the named point or material identity;
- `prior_reliability`: source-only evidence that the row is nominal;
- `prior_nominal_probability`: the nominal-component prior for a dependence
  group; and
- `composite_weight`: the amount of independent likelihood mass assigned to that
  dependent group.

A downstream physical innovation may influence BayesianPhysTwin's likelihood or
guard, but it must not be fed back into Prob4D's prior reliability. Rows, pixels,
frames, tracks, and views from one object or acquisition session remain nested
observations, not independent calibration replicates.

## 5. Causal and repository ownership boundary

Every claim-bearing row satisfies the exclusive cutoff

```text
source_frame < causal_frame_stop.
```

Appending future windows must not change an already valid prefix artifact.

The one-way ownership chain is:

```text
4-D provider
    -> Prob4D uncertain observation
    -> BayesianPhysTwin candidate belief and exact accept/fallback decision
    -> Causal4D factual abduction, intervention, and held-out prediction.
```

Prob4D owns provider-side gauges, covariance, source reliability, causal lineage,
and portable artifacts. It does not decide whether a physical-state update is
accepted. BayesianPhysTwin owns physical-query identifiability, the
baseline-relative guard, and exact complete-belief fallback. Causal4D consumes
only the selected BayesianPhysTwin belief; raw Prob4D output cannot rescue a
failed upstream gate.

## 6. Evidence-first progression

A real provider advances only through this order:

```text
support feasibility
  -> source mean quality
  -> identity and reliability
  -> gauge and dependence
  -> linearization closure
  -> conditional point covariance
  -> physical-query relevance
  -> one frozen target evaluation.
```

A failure stops or redirects that exact provider version. In particular:

- bad or drifting means do not authorize a richer covariance model;
- broken identities do not authorize a richer covariance model;
- failed gauge/dependence or linearization closure redirects the gauge model;
- only `point-covariance-localized` authorizes point-uncertainty development; and
- only `ready-for-one-target-evaluation` authorizes opening the exact bound target
  roster once.

The current executable real-provider priority is the frozen CUT3R source
qualification. See [the runbook](cut3r-source-qualification.md) and
[issue #49](https://github.com/IPS-Stuttgart/Prob4D/issues/49).

## 7. Minimal validated handoff

```python
from prob4d.api.v2 import load_claim_bearing_observation_belief

observation = load_claim_bearing_observation_belief(
    "outputs/sequence/observation_belief.npz"
)
```

Loading revalidates the closed schema, causal cutoff, covariance structure,
identities, provenance, and content address. It establishes artifact validity,
not provider accuracy or physical benefit.

## Claim boundary

Green CI, immutable artifacts, a mature adapter, controlled synthetic success, or
good marginal coverage is not by itself real-provider competence. Provider
accuracy, dependence calibration, BayesianPhysTwin physical-query value, and
Causal4D intervention value are separate conjunctive claims.

Further detail:

- [Architecture and repository boundary](architecture.md)
- [Unfused observation-factor bundle](observation-factor-bundle.md)
- [Observation-belief export](observation-belief-export.md)
- [Structured observation covariance queries](observation-covariance-queries.md)
- [Joint covariance diagnostics](joint-covariance-diagnostics.md)
- [Provider readiness localization](provider-readiness-localization.md)
- [What uncertainty can improve](theoretical-benefit.md)
