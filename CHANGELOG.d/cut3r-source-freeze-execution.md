### Added

- Add a merged-main, protected self-hosted execution path for the target-closed
  CUT3R Deform360 source freeze.
- Bind the execution to a content-addressed request, exact source-protocol Git
  blob, reviewed wheel, retained provider/checkpoint bytes, and immutable source
  roster.
- Publish a compact issue pointer from a separate hosted job while keeping
  write-capable tokens off the self-hosted runner.
- Consume the builder's canonical `source_freeze_id`, rederive it from the exact
  freeze record, and reject obsolete `artifact_id` aliases or post-seal drift.

### Scientific boundary

This executes source-input support and identity closure only. It does not run
CUT3R, score source outcomes, open confirmation or target payloads, authorize a
BayesianPhysTwin update, or establish provider benefit.
