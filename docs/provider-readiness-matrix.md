# Prospective provider-readiness matrix

`prob4d.provider_readiness_matrix` compares several complete
`FreshProviderReadinessDecisionV1` artifacts without opening protected targets or
selecting a provider after its source result is known. The workflow has two
separate content-addressed phases:

```text
source-outcome-blind matrix lock
    -> provider-specific source-only readiness decisions
    -> exact matrix replay and at-most-one target authorization
```

The matrix does not replace the seven ordered readiness gates. It composes their
terminal decisions under one predeclared provider set and comparison policy.

## Phase 1: freeze before source execution

Create a matrix-lock specification from the example and replace every placeholder
identity:

```bash
cp docs/examples/provider-readiness-matrix-lock-spec.json matrix-lock-spec.json
prob4d prediction readiness-matrix freeze \
  --spec matrix-lock-spec.json \
  --output matrix-lock.json
```

The lock freezes:

- exact Prob4D source revision;
- development, calibration, target, and optional confirmation object/session
  rosters;
- common cohort, BayesianPhysTwin query, and exact-fallback identities;
- the complete finite comparison policy;
- every provider repository, revision, model set, loader, and promotion lock;
- adapter identity and passed conformance artifact for every provider;
- a unique provider priority order; and
- exactly one matrix-level target-evaluation budget.

The lock rejects any declaration that source payloads, source outcomes, target
payloads, target outcomes, or confirmation payloads have already been opened.
Provider priority is therefore fixed before source results can reveal which route
looks favorable.

Verify it with:

```bash
prob4d prediction readiness-matrix verify-lock \
  --artifact matrix-lock.json
```

## Bind provider-specific readiness requests

After freezing the lock, add the required metadata to each
`FreshProviderReadinessRequestV1` before its source-only gates are evaluated:

```python
from prob4d.provider_readiness_matrix import (
    load_provider_readiness_matrix_lock,
    readiness_matrix_provider_metadata,
)

lock = load_provider_readiness_matrix_lock("matrix-lock.json")
metadata = readiness_matrix_provider_metadata(lock, "motion-provider")

request = FreshProviderReadinessRequestV1(
    cohort_lock=provider_cohort_lock,
    gates=ordered_source_gates,
    metadata={
        **metadata,
        "execution_stage": "source-only",
    },
)
```

The returned metadata binds:

- the exact matrix lock;
- the exact comparison policy;
- the provider adapter identity; and
- the adapter conformance artifact.

The provider-specific cohort lock must also match the matrix lock exactly for
Prob4D source revision, provider identity, promotion lock, cohort, query,
fallback, and every statistical-unit roster.

## Phase 2: compose source decisions

Create a decision specification that references the immutable matrix lock and
one readiness decision per frozen provider:

```bash
cp docs/examples/provider-readiness-matrix-decision-spec.json \
  matrix-decisions.json

prob4d prediction readiness-matrix evaluate \
  --spec matrix-decisions.json \
  --request-output provider-matrix-request.json \
  --output provider-matrix-decision.json
```

The builder snapshots and hashes every input file, embeds every complete readiness
decision, and rejects:

- missing, additional, or duplicated providers;
- changed provider priorities;
- changed provider/model/loader identities;
- a different cohort, query, fallback, or statistical-unit roster;
- a missing or changed matrix-lock binding;
- a missing or changed comparison-policy binding; and
- a missing or changed adapter identity or conformance binding.

The matrix reports every terminal classification and the providers for which
point-uncertainty development is specifically authorized. When more than one
provider passes all source gates, only the first provider in the frozen priority
order is selected. Other source-ready providers remain visible but receive no
matrix-level target budget.

A valid result with no ready provider is retained and returns exit status `2`.
It is not converted into a technical failure and must not be rescued by changing
the provider roster or priority after source outcomes are known.

Verify the request or decision with:

```bash
prob4d prediction readiness-matrix verify-request \
  --artifact provider-matrix-request.json

prob4d prediction readiness-matrix verify-decision \
  --artifact provider-matrix-decision.json
```

## One-shot target authorization

Only a matrix decision with one selected source-ready provider can produce a
target authorization:

```bash
prob4d prediction readiness-matrix authorize-target \
  --decision provider-matrix-decision.json \
  --output provider-matrix-authorization.json
```

The authorization embeds the matrix decision and the existing
`FreshProviderTargetAuthorizationV1` for the selected provider. It permits one
evaluation of the exact target roster. It does not authorize target-side method,
policy, covariance, guard, denominator, or provider-priority changes.

Verify it with:

```bash
prob4d prediction readiness-matrix verify-authorization \
  --artifact provider-matrix-authorization.json
```

A matrix-bound study should require this wrapper rather than directly authorizing
an unselected provider's standalone readiness decision.

## Comparison policy

The lock stores the complete finite JSON comparison policy and derives
`comparison_policy_id` from its canonical content. A real protocol should bind at
least:

- complete object/session statistical-unit semantics;
- support feasibility and technical-exclusion policy;
- source-mean and identity/reliability policy identities;
- gauge/dependence and point-covariance localization policy identities;
- BayesianPhysTwin query-relevance and fallback policy identities; and
- the exact metrics, margins, and group-level aggregation semantics used by all
  provider routes.

The matrix enforces common policy identity; it does not infer that differently
constructed source evidence is statistically exchangeable.

## Scientific boundary

A selected provider is only ready for one protected evaluation. The later target
result must separately pass provider-competence and guarded BayesianPhysTwin
physical-query criteria. Causal4D may consume only the accepted, content-bound
twin belief. A valid source-negative or target-negative result is complete
scientific evidence and must not be repaired on the same opened cohort.
