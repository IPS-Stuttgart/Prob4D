# Changelog

All notable changes to Prob4D are documented here.

## 0.2.1 — 2026-07-26

### Fixed

- Portable observation exports now embed a versioned, machine-readable covariance
  layout instead of relying on legacy seven-factor conventions.
- The metric-anchor record now binds the exact calibration artifact and declares
  its world frame and covariance treatment. Nonzero anchor covariance is carried
  through the shared joint gauge factor; fixed anchors remain a zero-covariance
  special case.
- Bayesian-PhysTwin and Causal4D can independently distinguish the production
  joint-tree layout, the explicitly approximate fixed-lag layout, and legacy
  per-window factors.
- Observation archives and metric-anchor JSON files are written atomically;
  observation archives are strictly reloaded before publication.

### Added

- `prob4d observation create-anchor` and
  `prob4d-create-metric-gauge-anchor` for reproducible anchor construction.
- Provider-contract regression tests for complete anchor metadata, shared-factor
  semantics, rank consistency, tamper rejection, and validated serialization.

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
  `FlorianPfaff/BayesianPhysTwin-Paper` project-notes repository.

### Added

- A content-addressed provider manifest and lazy grouped `prob4d` command surface.
- `prob4d.provider_v1` as the versioned Python import boundary for downstream
  development.
- `prob4d-validate-observation` for strict installed-artifact validation.
- Joint covariance propagation, cross-window factor, provider-boundary, and input
  immutability regression tests.
