# Changelog

All notable changes to Prob4D are documented here.

## Unreleased

### Added

- A target-free, content-addressed held-out provider promotion protocol that
  freezes complete object/session splits, source/model/calibration identities,
  seven required comparison roles, BayesianPhysTwin guard settings, bootstrap
  semantics, and decision margins; seals the complete target group-by-arm query
  matrix; composes provider competence with guarded-query gates; and replays the
  final report deterministically through `prob4d experiment heldout-provider`.
- A grouped `prob4d identity` interface for atomic material-identity mixture
  construction from externally source-calibrated weights, strict mixture and
  append-only stream validation, candidate-aligned log-sum-exp likelihood
  marginalization, and law-of-total-covariance Gaussian moment matching.
- An optional source-only directed-cycle gate for the experimental full-joint
  gauge graph, with a preregistered displacement threshold,
  per-multi-edge-child cycle support, complete audit reporting, and exact analytic
  provider-v2 tree fallback.
- Explicit MotionCrafter `legacy-common` and `derived-per-call` stochastic seed
  policies, with a versioned source-bound call schedule recorded in every new
  prediction manifest.
- Fail-closed seed-schedule validation that recomputes effective seeds and
  rejects missing, reordered, inconsistent, duplicated, or colliding derived
  calls.
- Atomic, crash-safe, resumable MotionCrafter prediction production with a
  content-addressed progress journal and member verification before reuse.
- Immutable model-set identities for the MotionCrafter UNet, geometry/motion VAE,
  base pipeline, and executed loader, using exact remote revisions or recursively
  content-addressed local snapshots.
- Versioned fused-prediction covariance semantics and a held-out provider
  evaluator with paired equal-group aggregation and deterministic group
  bootstrap intervals.
- Explicit `float32` or `float64` dense prediction storage, frame-local ray
  access, and deterministic retained-storage accounting for benchmark runs.
- `FusedSequence` as a public immutable dense-output contract with canonical
  dtypes, bounded active covariance validation, and read-only arrays.
- `TruthSequence` as an immutable, canonical, alias-safe truth-input contract with
  strict frame ordering and active point/flow finiteness validation.
- Common-support primary provider evaluation, native-support secondary results,
  support-retention diagnostics, frame-balanced errors, multi-level coverage,
  covariance-width summaries, uncertainty-error ranking, selective risk, and
  worst-group coverage-shortfall reporting. Selective-risk metrics are invariant
  to sample ordering when uncertainty scores tie.
- Bounded spatial-tile dense fusion that preserves structured covariance until a
  representative covariance-intersection sample or active tile is required, plus
  a deterministic process-level memory and timing benchmark.
- Bounded-memory metric, prefix-aligned, and oracle-aligned evaluation with an
  explicit chunk-size execution contract and deterministic process-level memory,
  timing, retained-storage, and numerical-agreement benchmark.
- An opt-in self-hosted full-resolution memory-profile workflow that runs fusion
  and provider evaluation in fresh processes, optionally profiles verified eager
  and mmap prediction loading, binds host and revision identity, and uploads
  checksummed machine-readable evidence.
- Explicit upper-winsorized calibration aggregation identifiers, public quantile
  aliases, and a shared validated aggregation primitive for point and gauge scales.

### Changed

- Claim-bearing prediction/calibration compatibility now retains the historical
  version-1 model identifier for legacy common-seed runs and uses a version-2
  identifier for `derived-per-call` runs, preventing silent calibration reuse
  across stochastic semantics.
- `prob4d-benchmark` forwards and records the selected MotionCrafter seed policy,
  model-set identity, prediction manifest identity, and dense-storage mode.
- Reused benchmark outputs are admitted only after the prediction bundle,
  baseline bytes, fused semantics, revisions, model identities, covariance
  presence, and execution settings validate against the current run.
- Held-out provider evaluation rejects one method label that mixes revisions,
  model sets, seed policies, covariance meanings, gauge estimators, calibration
  states, or dense-storage execution semantics across cases.
- Fusion and fused-artifact loading transfer ownership of private arrays into the
  immutable `FusedSequence` contract without an unnecessary second dense copy.
- Provider-evaluation report schema version 2 keeps the established unprefixed
  metric paths for common-support primary results and adds explicit
  `native_support` and `support` secondary diagnostics.
- Evaluation modes rely on the immutable finite-active point/flow contracts
  directly instead of rebuilding dense sequences solely to sanitize flow masks.
- Dense covariance-intersection weights remain optimized once per complete
  frame/contributor-mask pattern and are reused across bounded application tiles,
  so the tile size changes temporary memory rather than estimator semantics.
- Provider evaluation accumulates active point, covariance, flow, alignment, and
  seam statistics in bounded chunks; prefix and oracle modes apply transforms
  during evaluation instead of constructing complete transformed sequences.
- Provider reports record `evaluation_chunk_size`; changing it affects resource
  use and floating-point summation order, not registered support or metric
  semantics.
- New group-balanced point calibration metadata names its actual upper-winsorized
  operation; the frozen `trim_quantile` field and legacy misnamed aggregation ID
  remain readable without rewriting historical artifact identities.

### Fixed

- Scientific scalar contracts for robust alignment, cross-fitted disagreement,
  and uncertainty calibration reports now reject booleans, integral floats,
  numeric strings, and fractional count truncation. Robust alignment also rejects
  non-positive iteration, Huber, and convergence controls before numerical work.
- `FusedSequence` no longer retains mutable aliases to caller-owned arrays or
  leaves apparent dtype normalization local to validation. Active point and flow
  covariance now fail closed on non-finite, asymmetric, or materially indefinite
  matrices, while tolerated floating-point-scale negative eigenvalues are
  projected to the positive-semidefinite boundary.
- Truth arrays can no longer change after validation through caller-owned aliases,
  and malformed frame indices or non-finite active truth geometry now fail before
  evaluation.
- Paired provider comparisons no longer reward methods for omitting difficult
  frames or pixels from their primary score.

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
