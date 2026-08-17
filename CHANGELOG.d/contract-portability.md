### Added

- A consolidated read-only contract-portability workflow that type-checks the
  installed Prob4D wheel from a downstream `prob4d.api.v2` consumer with
  expression-level `Any` rejected and a required invalid-call negative control.
- Focused macOS and Windows validation for atomic no-clobber publication,
  prediction-window serialization, and read-only memory-mapped prediction
  stores.
- Repository policy tests and documentation for the installed typing and
  filesystem portability boundaries.

### Scientific boundary

These checks strengthen installed-package typing and local-filesystem portability
only. They change no estimator, covariance, provider output, cohort, target-access
decision, physical update, exact fallback, empirical result, or scientific claim.
