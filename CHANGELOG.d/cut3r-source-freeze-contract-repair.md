### Fixed

- Rotate the content-addressed CUT3R source-freeze v2 request identity after the
  issue binding was added, and propagate that identity through the exact-byte
  publication request and artifact prefix.
- Restore the publication request's strict schema by removing an unsupported
  duplicate issue field.
- Register the automatic v2 source-freeze workflow as a reviewed self-hosted
  workflow and assert its exact-main, read-only, no-secret execution boundary.

### Scientific boundary

This repair changes evidence identity and workflow validation only. It does not
change the source cohort, CUT3R inputs, comparison arms, target-access boundary,
BayesianPhysTwin or Causal4D methods, or any scientific outcome.
