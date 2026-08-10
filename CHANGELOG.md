# Changelog

All notable changes to Prob4D are documented here. Detailed, claim-bearing
release notes and scientific boundaries live under `docs/releases/`.

## Unreleased

No changes yet.

## 0.4.0 — 2026-08-10

### Added

- A versioned `prob4d.api.v1` façade as the supported Python dependency boundary
  for BayesianPhysTwin, Causal4D, and independent downstream providers.
- One runtime-version resolver backed by installed distribution metadata, with an
  explicit `0+unknown` sentinel for an uninstalled source tree.
- An installed source distribution audit that verifies package-only contents,
  installs the archive in an isolated environment, and smoke-tests the public API,
  provider contracts, contract data, and canonical CLI.
- Portable `O(K)` causal gauge-tree prior artifacts and tree-backed sparse
  observation-factor stacks that avoid materializing the complete joint gauge
  covariance.
- Portable and claim-bearing tree-sparse observation artifacts binding provider,
  calibration, runtime-revision, causal-lineage, observation, and sparse-prior
  identities for strict BayesianPhysTwin admission.
- Outcome-blind provider support feasibility over frozen source streams, geometry,
  camera calibration, metric anchors, admission rules, and technical exclusions.
- Spatially stratified causal tracklets and frozen-roster camera-panel support with
  a frame-level likelihood-power budget.
- Replay-complete held-out provider evidence binding calibration selection, exact
  provider-report bytes, sealed target decisions, exact fallback, bootstrap
  settings, and the final Prob4D-to-BayesianPhysTwin promotion report.
- Paired joint-covariance dependence ablations comparing full shared covariance,
  marginal-preserving independence, and conditional-only controls.
- A content-addressed three-repository installed-wheel release capsule binding
  exact Prob4D, BayesianPhysTwin, and Causal4D revisions and wheel digests.
- Bounded-memory dense fusion, provider evaluation, prediction storage, and
  production memory profiling.
- Versioned material-identity, source-reliability, stochastic-seed, and provider
  evaluation contracts.

### Changed

- The canonical grouped `prob4d` registry owns command targets while historical
  executables remain available through documented compatibility wrappers.
- The authoritative Ruff and mypy job uses one pinned quality environment; the
  main test workflow owns repository, release, package, and provider-contract
  checks instead of duplicating quality execution.
- Wheel and source distributions are separated from GitHub workflows, generated
  evidence, CI environments, repository tests, and one-off maintenance scripts.
  Scientific evidence remains in explicit content-addressed capsules.
- Provider manifests name `IPS-Stuttgart/Prob4D` and `prob4d.api.v1` as the
  canonical repository and Python import boundary.
- Provider API v1 remains frozen at observation-factor schema 3, while provider
  API v2 and tree-sparse manifests retain their newer joint-covariance contracts.

### Fixed

- The 0.4.0 source boundary is an ordinary reviewed commit rather than a workflow
  that mutates and pushes its own branch.
- Selection-lock publication, content-addressed persistence, and claim-bearing
  artifact loading fail closed on overwrite, coercion, lineage, and covariance
  inconsistencies.
- Sparse observation and camera-panel contracts preserve likelihood budgets and
  reject lossy indices, omitted cameras, forged support, and malformed geometry.
- Immutable dense and truth contracts no longer retain mutable caller aliases and
  reject non-finite or materially invalid covariance and geometry.

## 0.3.0 — 2026-07-30

### Added

- Provider API v2 with claim-bearing observation loading, schema-4 joint gauge
  covariance, analytic Sim(3) composition Jacobians, runtime source attestation,
  and version-selectable provider manifests.
- Append-only causal observation-factor streams, content-addressed covariance
  calibration, source reliability, causal tracklets, cycle audits, and strict
  prediction/calibration compatibility.
- A PEP 561 `py.typed` marker and installed grouped and compatibility commands.

### Changed

- New claim-bearing exports require complete calibration and preserve shared
  cross-window covariance; provider API v1 remains a frozen compatibility surface.
- Fixed-lag smoothing carries an uncertainty-bearing marginalized boundary prior
  while historical all-window covariance remains an explicit approximation.

## 0.2.0 — 2026-07-26

### Added

- A content-addressed provider manifest, a lazy grouped command surface,
  `prob4d.provider_v1`, strict observation validation, and joint-covariance
  regression tests.

### Changed

- Portable observation export defaults to a causal sequential gauge tree with
  shared gauge covariance and trace-audited rank reduction.
- Prediction windows are immutable, CI validates wheel and source artifacts across
  supported Python versions, and project metadata points to the canonical project
  notes repository.
