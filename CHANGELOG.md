# Changelog

All notable changes to Prob4D are documented here.

## Unreleased

### Added

- `prob4d.provider_v2` as a safe-by-default Python surface with distinct
  exploratory and claim-bearing export functions.
- Strict prediction/calibration compatibility checks covering MotionCrafter
  revision, canonical model settings, resolution, window geometry, covariance
  cluster size, and gauge/point covariance methods before payload loading.
- A context-local canonical covariance-root basis for repeated eigenspaces in
  provider v2, with fail-closed rank boundaries and unchanged provider-v1 defaults.
- Context-local analytic `Sim(3)` composition Jacobians for provider-v2 sequential
  covariance propagation, with fail-closed handling at the SO(3) logarithm branch
  cut and exact provider-v1 finite-difference compatibility.
- Explicit `prob4d observation export-calibrated` and
  `prob4d observation export-exploratory` commands, plus installed legacy-style
  entry points for scripted use.
- Runtime source-revision attestation for provider-v2 export. Claim-bearing runs
  require independent VCS metadata or a clean source checkout; an environment-only
  `PROB4D_RUNTIME_REVISION` assertion is recorded only for exploratory deployments.
- Content-addressed provider-v2 artifact metadata binding the provider manifest,
  export mode, calibration IDs, covariance-root and composition-Jacobian modes, and
  runtime revision evidence.
- Version-selectable provider manifest emission through
  `prob4d provider manifest --api-version {1,2}`.
- Provider-v2 unit, type, import, wheel, and source-distribution coverage.

### Changed

- Fixed-lag gauge smoothing now Schur-marginalizes expired gauges into an
  uncertainty-bearing boundary prior. Portable historical covariance remains an
  explicit block-diagonal reconstruction approximation.
- Documentation now recommends the calibrated provider-v2 command for new
  claim-bearing experiments and labels provider-v1 commands as frozen compatibility
  surfaces.
- CI verifies analytic-versus-finite-difference derivative parity, clean-checkout
  runtime attestation, both provider manifests, and every provider-v2 command from
  installed wheel and source artifacts.

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
