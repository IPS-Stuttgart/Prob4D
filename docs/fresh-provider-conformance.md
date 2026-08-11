# Fresh-provider readiness conformance corpus

`prob4d.fresh_provider_conformance` is a deterministic, target-free corpus for
the ordered readiness state machine. It verifies that later refactors preserve
the exact distinction between unsupported providers, poor source means,
identity failures, gauge/dependence failures, point-covariance-localized failures,
query irrelevance, technical failure, and readiness for one target evaluation.

The corpus contains isolated fixtures for:

1. support-negative termination;
2. source-mean-negative termination;
3. identity or association termination;
4. gauge or dependence termination;
5. point-covariance localization;
6. query irrelevance or non-identifiability;
7. exact-fallback contract failure;
8. an explicit pre-fallback technical failure; and
9. a fully passing route.

Every fixture uses one fixed synthetic cohort lock, deterministic evidence IDs,
and the public `FreshProviderReadinessRequestV1` evaluator. Gates after the first
terminal result are explicitly `not-evaluated`. Only the point-covariance fixture
authorizes source-only point-uncertainty development. Only the clean fixture
receives a target budget of one, and its authorization must reproduce the exact
locked target roster.

Run the complete corpus with:

```bash
python -m prob4d.fresh_provider_conformance
```

Use `--compact` for canonical single-line JSON. The command returns zero only
when every packaged fixture matches its expected classification, terminal gate,
authorization flags, target budget, and target-roster binding.

The report is content-addressed and deterministic. It opens no target payload or
outcome and contributes no physical evidence. Passing proves software decision
conformance only; it does not establish provider competence, calibrated
uncertainty, BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or
state of the art.
