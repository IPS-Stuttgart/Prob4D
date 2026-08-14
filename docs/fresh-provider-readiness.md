# Fresh-provider readiness and failure localization

`FreshProviderReadinessDecisionV1` implements the pre-target orchestration needed
before one frozen real-provider evaluation. It composes existing support,
calibration, identity, covariance, query, and fallback evidence without changing
the stable provider-v2 or held-out promotion contracts. The canonical command is
`prob4d experiment fresh-provider-readiness`; the Python implementation remains
in `prob4d.fresh_provider_readiness`.

## Cohort lock

`FreshProviderCohortLockV1` binds:

- exact Prob4D and provider repositories and revisions;
- model-set, loader, cohort, and held-out promotion-lock identities;
- the BayesianPhysTwin-owned query definition and exact fallback identity;
- pairwise-disjoint development, calibration, target, and optional confirmation
  object/session rosters; and
- explicit declarations that target and confirmation payloads and outcomes are
  still closed.

Every roster is sorted, unique, and disjoint. A lock cannot be constructed after
protected data have been opened.

## Prospective gate order

A readiness request contains exactly seven gates in this order:

1. `support-feasibility`;
2. `source-mean`;
3. `identity-reliability`;
4. `gauge-dependence`;
5. `point-covariance`;
6. `query-relevance`; and
7. `exact-fallback`.

Each gate is `pass`, `fail`, `technical-failure`, or `not-evaluated`. After the
first terminal result, every later gate must remain `not-evaluated`. Conversely,
an unevaluated gate after all earlier gates passed is invalid rather than being
silently interpreted as a negative result.

## Terminal decisions

| First terminal gate | Classification | Authorized next step |
| --- | --- | --- |
| support feasibility | `support-negative` | stop before predictions or residuals |
| source mean | `source-mean-negative` | stop; do not fit richer covariance |
| identity/reliability | `identity-or-association-negative` | improve source-only identities |
| gauge/dependence | `gauge-or-dependence-negative` | localize shared uncertainty |
| point covariance | `point-covariance-localized` | source-only uncertainty development |
| query relevance | `query-irrelevant-or-nonidentifiable` | exact physical fallback |
| exact fallback | `technical-failure` | repair under a new reviewed execution |
| all gates pass | `ready-for-one-target-evaluation` | one bound target evaluation |

Any explicit technical failure produces `technical-failure` at its exact stage.

Only `point-covariance-localized` sets
`authorize_point_uncertainty_development=true`. This authorization is possible
only after source means, identities, and gauge/dependence have passed.

Only `ready-for-one-target-evaluation` sets
`authorize_target_evaluation=true` and `target_evaluation_budget=1`.
`FreshProviderTargetAuthorizationV1` binds that one-shot budget to the exact
target roster and still requires the target to be unopened at authorization.

## Python example

```python
from prob4d.fresh_provider_readiness import (
    FreshProviderCohortLockV1,
    FreshProviderReadinessRequestV1,
    ReadinessGateV1,
    authorize_fresh_provider_target,
    evaluate_fresh_provider_readiness,
)

lock = FreshProviderCohortLockV1(
    protocol_id="fresh-provider-v1",
    source_repository="IPS-Stuttgart/Prob4D",
    source_revision=prob4d_revision,
    provider_repository=provider_repository,
    provider_revision=provider_revision,
    model_set_id=model_set_id,
    loader_id=loader_id,
    cohort_binding_id=cohort_binding_id,
    promotion_lock_id=promotion_lock_id,
    query_definition_id=query_definition_id,
    fallback_identity_id=fallback_identity_id,
    development_group_ids=development_ids,
    calibration_group_ids=calibration_ids,
    target_group_ids=target_ids,
)

request = FreshProviderReadinessRequestV1(
    cohort_lock=lock,
    gates=(
        support_gate,
        source_mean_gate,
        identity_gate,
        gauge_gate,
        point_covariance_gate,
        query_gate,
        fallback_gate,
    ),
)
decision = evaluate_fresh_provider_readiness(request)

if decision.authorize_target_evaluation:
    authorization = authorize_fresh_provider_target(decision)
```

The support adapter accepts a validated
`ProviderSupportFeasibilityV1`-like result. The source-competence adapter accepts
`SourceProviderCompetenceReportV1` and preserves the rule that identity is not
evaluated after a source-mean failure.

## Command-line replay

The grouped route preserves the existing subcommands and does not add a second
console-script alias:

```bash
prob4d experiment fresh-provider-readiness evaluate \
  --request readiness-request.json \
  --output readiness-decision.json

prob4d experiment fresh-provider-readiness verify-decision \
  --artifact readiness-decision.json

prob4d experiment fresh-provider-readiness authorize-target \
  --decision readiness-decision.json \
  --output target-authorization.json

prob4d experiment fresh-provider-readiness verify-authorization \
  --artifact target-authorization.json
```

A valid non-ready decision returns exit status 2 from `evaluate`; a target-ready
decision returns 0. Invalid or tampered artifacts fail validation rather than
being converted into scientific negatives.

## Repository boundary

Prob4D owns provider-side support, source competence, covariance diagnostics,
and this readiness composition. BayesianPhysTwin owns the physical query,
practical-equivalence margins, update admission, and exact fallback. Causal4D
consumes only the BayesianPhysTwin belief selected after that admission.

A positive readiness authorization is not empirical evidence. The later target
result must still pass the existing held-out provider-competence and guarded
physical-query gates and must retain a valid negative result without retuning the
same target cohort.
