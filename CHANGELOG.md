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
- `PredictionWindow` rejects invalid geometry and defensively copies and freezes
  every NumPy field after validation.
- Continuous integration covers Python 3.10, 3.12, and 3.14, validates wheel and
  source distributions, installs both artifacts in isolation, and smoke-tests
  grouped and legacy commands.
- Project metadata and documentation point to the canonical
  `2026-07-Causal4D-BPT-Paper` project-notes repository.

### Added

- A content-addressed provider manifest and lazy grouped `prob4d` command surface.
- `prob4d.provider_v1` as the versioned Python import boundary for downstream
  development.
- `prob4d-validate-observation` for strict installed-artifact validation.
- Joint covariance propagation, cross-window factor, provider-boundary, and input
  immutability regression tests.
