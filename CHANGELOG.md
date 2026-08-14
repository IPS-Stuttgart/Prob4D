# Changelog

All notable changes to Prob4D are documented here. Detailed, claim-bearing
release notes and scientific boundaries live under `docs/releases/`.

## Unreleased

### Added

- Canonical grouped routes for outcome-blind provider-support envelopes,
  source-covariance failure localization, causal-prefix admission, and complete
  fresh-provider readiness decisions.
- Registry and import checks that preserve the lifecycle, claim-bearing, and
  runtime boundaries of those provider-readiness commands.

### Fixed

- The architecture guide now matches the Prob4D 0.5 runtime: provider v1 is an
  artifact-only compatibility bridge, provider manifests are v2, and historical
  standalone executables are not installed.
- Release regression checks now reject reintroduction of the stale provider-v1
  execution and legacy-command documentation.

### Scientific boundary

These changes expose already-existing readiness contracts through the canonical
CLI and correct documentation. They change no estimator equation, covariance fit,
provider artifact, target-access decision, physical-update guard, exact fallback,
or scientific result.

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

- Strict provider-v2 attestation rejects runtime-revision metadata that is not
  independently verified.
- Calibration transport and provider readiness artifacts preserve exact source,
  cohort, feature-contract, query, and fallback identities.

### Scientific boundary

This release improves packaging and source-side provider-readiness
infrastructure. It does not establish real-provider competence, fresh-object
physical benefit, calibrated deployment uncertainty, Causal4D intervention
benefit, deployment safety, or state of the art.

## 0.4.0 — 2026-08-11

### Added

- Versioned `prob4d.api.v1` and `prob4d.api.v2` public façades.
- Provider-v2 strict claim-bearing loaders, explicit joint-gauge factor bundles,
  tree-sparse observations, and portable gauge-tree priors.
- Content-addressed selection, admission, evidence-decision, and held-out
  promotion artifacts.
- Reproducible installed-wheel integration across Prob4D, BayesianPhysTwin, and
  Causal4D.

### Changed

- Provider-v2 calibrated export uses analytic `Sim(3)` composition Jacobians and
  canonical covariance roots.
- Claim-bearing exports require exact calibration compatibility, runtime
  revision attestation, causal source lineage, and exact fallback semantics.

### Scientific boundary

The 0.4.0 release establishes software contracts and controlled mechanism
infrastructure. It does not by itself establish real provider competence,
uncertainty calibration on a fresh physical cohort, BayesianPhysTwin benefit,
Causal4D intervention benefit, deployment safety, or state of the art.
