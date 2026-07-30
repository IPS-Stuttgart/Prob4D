# Changelog

All notable changes to Prob4D are documented here.

## Unreleased

## 0.3.0 — 2026-07-30

### Added

- `ObservationFactorStreamV1`, a portable append-only chain of causally disjoint
  schema-v4 factor bundles with bundle/payload checksums, stable observation
  identity digests, contiguous frame intervals, and previous-update hash binding.
- Direct provider-v2 exports of `load_claim_bearing_observation_belief`,
  `validate_claim_bearing_observation_belief`, and
  `ValidatedClaimBearingObservation` so safe export and admission live on one
  versioned public surface.
- A PEP 561 `py.typed` marker for downstream static type checking.
- `load_claim_bearing_observation_belief` and
  `validate_claim_bearing_observation_belief` as explicit admission boundaries for
  calibrated, causal, independently attested observations.
- `ObservationFactorBundle` schema v4 with an ordered joint `7K x 7K` gauge
  covariance, explicit joint-versus-marginal covariance semantics, fail-closed
  marginal-block validation, and conservative schema-v2/v3 migration.
- A public cluster-cross-fitted overlap-disagreement diagnostic that holds out
  frame-by-spatial-tile clusters, refits the relative gauge without them, fails
  closed on unfittable folds, and reports the strictly out-of-fold evaluated
  fraction.
- Equal-group point-uncertainty calibration with canonical group ordering,
  within-group trimming, per-group diagnostics, and row-order-invariant aggregate
  scales so dense sequences cannot dominate solely through sample count.
- A provider-compatible helper that binds the full equal-group report and grouping
  definition into `PointUncertaintyCalibrationV1` content-addressed metadata, with
  explicit pooled-versus-balanced inspection and round-trip validation.
- Prefix-only scene-flow tracklets with persistent within-window point IDs,
  deterministic collision handling, cumulative association probabilities,
  termination diagnostics, and conversion to unfused observation factors.
- A directed gauge-cycle audit that compares direct overlap alignments with
  two-edge paths in representative observation displacement, preserves
  deterministic thresholds, and avoids chi-square claims under unknown edge
  dependence.
- A content-addressed, equal-group logistic source-reliability calibration with a
  source-only feature contract, canonical row ordering, exact label/group
  semantics, probability clipping, immutable metadata, and tamper-checked JSON
  round trips.
- `prob4d.provider_v2` as a safe-by-default Python surface with distinct
  exploratory and claim-bearing export functions.
- Strict prediction/calibration compatibility checks covering MotionCrafter
  revision, canonical model settings, resolution, window geometry, covariance
  cluster size, and gauge/point covariance methods before payload loading.
- A context-local canonical covariance-root basis for repeated eigenspaces in
  provider v2, with fail-closed rank boundaries and unchanged provider-v1
  defaults.
- Context-local analytic `Sim(3)` composition Jacobians for provider-v2 sequential
  covariance propagation, with fail-closed handling at the SO(3) logarithm branch
  cut and exact provider-v1 finite-difference compatibility.
- Explicit `prob4d observation export-calibrated`,
  `prob4d observation export-exploratory`, and
  `prob4d observation export-v1` grouped commands, plus installed legacy-style
  entry points for scripted reproduction.
- Runtime source-revision attestation for provider-v2 export. Claim-bearing runs
  require independent VCS metadata or a clean source checkout; an environment-only
  `PROB4D_RUNTIME_REVISION` assertion is recorded only for exploratory deployments.
- A versioned self-contained provider-attestation schema that embeds the complete
  content-addressed provider manifest, export mode, calibration IDs,
  covariance-root and composition-Jacobian modes, and runtime revision evidence.
- Version-selectable provider manifest emission through
  `prob4d provider manifest --api-version {1,2}`.
- A provider-v2 gauge-backend ablation runner that preserves the seven-row
  reconstruction contract while using the production causal spanning tree,
  analytic composition Jacobians, and joint-covariance marginal adapter.

### Changed

- The grouped `prob4d observation export` route now fails closed with migration
  guidance instead of silently selecting provider v1. The historical standalone
  `prob4d-export-observation-belief` executable remains unchanged.
- Claim-bearing loading now requires complete alignment-level covariance
  calibration, rejects uncalibrated or pointwise-fallback permission, and rejects
  recorded covariance fallback use.
- `ObservationBeliefExportV1.metadata` is recursively immutable after finite-JSON
  normalization. Ordinary `dict`/`list` checks and mutable `copy`/`deepcopy`
  workflows remain supported.
- Stacked unfused factors retain cross-window gauge covariance rather than
  reconstructing a block-diagonal prior from per-gauge marginals. The frozen
  provider-v1 writer remains schema v3 and rejects joint covariance it cannot
  encode.
- Fixed-lag gauge smoothing Schur-marginalizes expired gauges into an
  uncertainty-bearing boundary prior. Portable historical covariance remains an
  explicit block-diagonal reconstruction approximation.
- Documentation recommends calibrated provider v2 for new claim-bearing work and
  labels provider-v1 routes as frozen compatibility surfaces.
- CI verifies analytic-versus-finite-difference derivative parity, provider
  manifest and attestation hashing, clean-checkout runtime provenance, and every
  provider-v2 command from installed wheel and source artifacts.

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
