# Changelog

All notable changes to Prob4D are documented here.

## 0.2.0 — 2026-07-26

### Changed

- Portable Bayesian observation export now defaults to a causal sequential gauge
  tree and carries the full joint cross-window gauge covariance.
- Gauge-root rank reduction is trace-audited and fails closed below a declared
  retained-covariance threshold.
- Fixed-lag covariance export requires an explicit acknowledgement that its
  marginalized boundary covariance is approximate.
- `PredictionWindow` now rejects negative frame identities, deformation masks
  outside valid geometry, non-finite active flow, and invalid active rays.
- Continuous integration covers Python 3.10, 3.12, and 3.14, validates source and
  wheel distributions, installs the wheel in isolation, and smoke-tests every
  command.
- Project metadata and documentation now point to the canonical
  `2026-07-Causal4D-BPT-Paper` project-notes repository.

### Added

- `prob4d-validate-observation` command for installed-artifact validation.
- Joint covariance propagation and cross-window factor regression tests.
