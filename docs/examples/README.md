# Provider-boundary examples

- `prediction-batch-preflight-policy.json` is the strict default scorer-batch policy.
- `provider-terminal-batch-incompatible.json` is an input specification for
  `python -m prob4d.provider_terminal_decision build`.

The example terminal specification intentionally contains no `artifact_id`; the
builder derives it from the canonical content. Replace every placeholder identity
with the exact content identity from the frozen provider protocol.
