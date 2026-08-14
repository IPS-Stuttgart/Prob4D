# Changelog

All notable changes to Prob4D are documented here. Detailed, claim-bearing
release notes and scientific boundaries live under `docs/releases/`.

## Unreleased

### Added

- A bounded exact joint multi-window material-identity posterior that conditions
  source-calibrated local mixtures on a window-unique forest constraint, embeds
  and replays every source mixture, preserves exact local null fallback, and
  exposes grouped build, validation, and downstream marginalization commands.

### Scientific boundary

The joint posterior is source-side identity and consistency infrastructure. It
requires separately calibrated local weights and does not establish association
precision, real-provider competence, BayesianPhysTwin benefit, Causal4D benefit,
physical-state identifiability, deployment safety, or state of the art.

## 0.5.0 — 2026-08-12

### Removed

- All standalone `prob4d-*` console-script aliases and the central
  `prob4d.legacy_cli` compatibility wrapper.
- Legacy executable migration metadata, `prob4d commands migrate`, and its
  dedicated CI workflow.
- `prob4d observation export-v1`, `prob4d.api.v1`, and provider-v1 estimator and
  export entry points.
- The broad lazy package-root export inventory and its large compatibility typing
  stub.

### Changed

- `import prob4d` now exposes only `__version__`; current integrations use
  `prob4d.api.v2` explicitly.
- Provider v2 uses a private shared export core rather than inheriting execution
  from the public provider-v1 module.
- `prob4d.provider_v1` is reduced to an artifact compatibility bridge containing
  historical records, manifests, serializers, validators, and schema-v3 factor
  IO; it exposes no estimator or exporter.
- Provider-manifest CLI output is always API v2, and the canonical command
  registry contains only grouped routes.
- Public API manifest schema version 2 records only the minimal root and current
  v2 façade.
- Source-distribution and installed-artifact checks prove that retired modules and
  executables are absent.

### Scientific boundary

This release changes packaging, command dispatch, and Python compatibility
surfaces. It changes no estimator equations, calibration fit, provider output,
target-access rule, fallback decision, data split, or scientific result. Full
provider-v1 reproduction remains available from the exact Prob4D 0.4.1
wheel/source revision; historical content-addressed artifacts are not rewritten.

## 0.4.1 — 2026-08-12

### Added

- A source-fitted correlation-group robust likelihood with complete
  object/session selection, one contamination state per declared dependence
  group, and exact Gaussian fallback when source support is insufficient.
- An additive analytic `Sim(3)` composition and inversion covariance path with
  branch-cut rejection and factorized solves for declared full-rank alignment
  systems.
- Source-only calibration-transport certificates, source-provider competence
  reports, and query-space covariance relevance and preservation diagnostics.
- A target-closed fresh-provider cohort lock, ordered seven-gate readiness
  decision, deterministic failure localization, one-shot target authorization,
  and a conformance corpus covering every terminal classification.
- A content-addressed public API manifest for the compatibility root and the
  stable `prob4d.api.v1` and `prob4d.api.v2` façades.
- A packaged root typing stub that preserves the historical typed import surface
  under lazy runtime loading.

### Changed

- Importing the broad `prob4d` compatibility root no longer eagerly imports
  calibration, fusion, gauge-graph, observation, storage, and reliability
  implementations. Historical exports load on first access and retain exact
  object identity.
- Source-distribution and release checks now require and exercise the root typing
  stub, public API manifest, current release note, and installed lazy-import
  behavior.
- Public API documentation now distinguishes the lazy compatibility root from the
  supported versioned façades and provides a reproducible manifest workflow.

### Fixed

- Sampled Gaussian fusion and declared full-rank covariance operations use
  validated factorized solves instead of explicit inverses or pseudoinverses.
- Invalid Sintel truth coordinates are prevented from leaking through bilinear
  resampling.
- Fresh-provider readiness cannot conflate failed means, identities,
  gauge/dependence, point covariance, query relevance, or technical fallback
  failures, and target authorization is bound to exactly one unopened roster.

### Scientific boundary

This release adds implementation, interoperability, and prospective protocol
infrastructure. It does not establish fresh real-provider competence, calibrated
target uncertainty, unseen-object BayesianPhysTwin benefit, Causal4D
intervention benefit, deployment safety, or state of the art.

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
- Current project identity and the additive `prob4d.api.v1` façade are published
  separately, while content-addressed provider manifests retain their frozen
  historical repository and import-boundary fields.
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
