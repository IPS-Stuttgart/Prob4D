# Provider-boundary examples

- `prediction-batch-preflight-policy.json` is the strict default scorer-batch policy.
- `provider-terminal-batch-incompatible.json` is an input specification for
  `python -m prob4d.provider_terminal_decision build`.
- `provider-readiness-matrix-lock-spec.json` freezes a finite provider set,
  adapter-conformance identities, common comparison policy, and priority before
  source execution.
- `provider-readiness-matrix-decision-spec.json` binds the frozen lock to one
  source-only readiness decision per provider.

The example specifications intentionally omit derived content addresses. Replace
every `REPLACE_WITH_...` value with the exact identity from the frozen provider
protocol; builders derive and validate the resulting artifact IDs.
