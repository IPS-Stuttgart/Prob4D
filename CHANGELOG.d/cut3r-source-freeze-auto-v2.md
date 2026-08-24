### Added

- Add a content-addressed merged-main trigger for the retained CUT3R Deform360
  source-input freeze.
- Add a main-only exact historical retry that requires the original 40-character
  merged-main execution SHA and 64-character retained request ID, proves ancestry
  and byte-identical request/protocol binding, and executes the historical code
  rather than current development head.
- Check required repository-variable names on a hosted runner without exposing
  values or retained paths, and fail before scheduling privileged work when the
  configuration is incomplete.
- Bound self-hosted runner acceptance to 20 minutes; publish a target-closed issue
  receipt and cancel the run when no matching runner accepts the job.
- Run the self-hosted stage with a read-only repository token, publish queued and
  terminal issue receipts from hosted jobs, and retain either the registered
  support pass or support-negative decision.
- Add independently testable authorization and variable-readiness helpers plus
  adversarial ancestry, byte-drift, identity, and redaction tests.
- Keep the execution driver path-sanitized, target-closed, and able to create the
  comparison lock only after a source-support pass.

### Scientific boundary

This route changes no CUT3R, Prob4D, BayesianPhysTwin, or Causal4D method and
opens no source outcome or target payload. It authorizes only the already frozen
source-input support and identity stage. Configuration and queue-timeout receipts
carry no scientific evidence.
