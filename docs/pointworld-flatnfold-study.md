# PointWorld → Prob4D → BayesianPhysTwin on Flat'n'Fold

## Purpose

This branch starts the highest-value remaining Prob4D paper experiment: test whether a provider-agnostic, correlation-aware recursive belief improves a real action-conditioned 3-D world model and one frozen downstream BayesianPhysTwin query.

The intended external provider is PointWorld (`NVlabs/PointWorld`) and the intended independent real cohort is the robot subset of Flat'n'Fold (`lipeng-zhuang521/flat-n-fold`). The experiment is deliberately separated from the terminal CUT3R attempt, the already-opened MotionCrafter/Deform360 cohorts, and the locked Causal4D physical-acquisition protocol.

The present branch is **source qualification only**. No target outcome may be opened or used to choose a representation mapping, checkpoint, split, mask, exclusion, covariance treatment, or downstream query.

The machine-readable lock is `protocols/pointworld-flatnfold-source-qualification-v1.json`.

## Frozen public revisions

The source-qualification protocol binds:

- Prob4D: `d02f057671023ff586ebfc15c904dbb0a60f4425`;
- BayesianPhysTwin: `b5f07d649ac2cd7dc6ca1aceb4004ff4803bddc8`;
- PointWorld: `05484826dfef74cbe278a3974179a5a16705d35d`;
- Flat'n'Fold code: `fa0d3d17ac827e7b5c43ec1ef7c0c38ad5e39340`.

These source revisions are not a completed experiment lock. The exact PointWorld checkpoint bytes, runtime, Flat'n'Fold dataset bytes, garment roster, frame/action schedule, and downstream query still have to be bound before target access.

## Why this experiment

Prob4D already contains the methodological ingredients needed for a paper contribution: joint gauge uncertainty, cross-window covariance, covariance intersection, guarded multi-edge graph fusion, finite-sample source-cycle admission, sparse metric-anchor calibration, provider-neutral manifests, and exact fallback. Another covariance or planner component would add complexity without resolving the main empirical gap.

The missing evidence is a fresh real provider and fresh physical object/session cohort showing that the same recursive belief machinery can wrap a different action-conditioned model and improve a downstream physical query.

PointWorld is a materially different provider from MotionCrafter. Its released model predicts full-scene 3-D point flows from RGB-D observations and robot actions represented as 3-D point flows. Flat'n'Fold provides 887 robot demonstrations over 44 garments in eight categories, with three-camera RGB-D observations, camera calibration, Baxter action information, and point-cloud construction code.

## Critical source-side representation gate

This is the first issue that must be resolved before an adapter is authorized.

Prob4D `PredictionWindow` v2 stores a dense `T × H × W × 3` point map and, when present, a dense scene-flow field on the same grid. The released PointWorld model instead predicts over `N_scene` seeded scene points: its dynamics output is shaped `B × T × N_scene × 3`, with a heteroscedastic log-variance output shaped `B × T × N_scene × 1`.

Therefore we must **not** silently reshape or nearest-neighbor rasterize PointWorld output into a Prob4D image grid. One of the following must be established using source information only:

1. a deterministic mapping from every PointWorld scene point back to a registered RGB-D grid location, preserving identity and without using future/target truth;
2. a separately versioned sparse persistent-point provider contract in Prob4D, with explicit point identities, source lineage, dependence groups, and calibrated uncertainty semantics; or
3. a valid negative result that this PointWorld version is not compatible with the current Prob4D claim.

Target-truth nearest-neighbor mapping, target-tuned interpolation, and post-outcome removal of unsupported garments are prohibited.

## Source-only qualification sequence

Before provider residuals or target outcomes are opened:

1. bind the exact PointWorld checkpoint SHA-256 and runtime environment;
2. inventory the Flat'n'Fold robot data and bind the exact byte manifest;
3. freeze complete physical garment identities as the top-level statistical units;
4. verify camera timestamp coverage plus intrinsic and extrinsic identities;
5. verify that Baxter action poses can be converted to PointWorld robot point-flow actions without target state;
6. verify metric coordinate transformations using released calibration only;
7. verify PointWorld scene-point identity semantics over the ten-frame prediction horizon;
8. resolve the representation gate above;
9. create an outcome-blind `ProviderSupportFeasibilityV1` request with all required streams and geometry support;
10. stop if the support gate fails.

The existing `prob4d.provider_support_feasibility` contract should be reused; no PointWorld-specific support framework is needed.

## Later held-out study, only after qualification passes

Use complete garment identity as the independent unit. Demonstrations/sessions are nested observations; frames, pixels, and points are not independent replicates.

The primary provider comparison should remain small:

1. raw/disjoint PointWorld;
2. decoded uniform overlap fusion;
3. Prob4D sequential full-joint spanning tree;
4. conformal-guarded full-joint graph with exact tree fallback.

Useful diagnostic controls include persistence, latest-window overwrite, naive precision fusion, and the unguarded graph.

Provider endpoints should include long-horizon and terminal point/flow error, overlap seam or drift, proper score, 50/90/95% coverage, normalized NEES, covariance width, support/fallback accounting, and worst-garment performance. Inference should cluster at garment identity.

## BayesianPhysTwin gate

Freeze exactly one downstream physical query before target access. Compare:

1. unchanged physical fallback;
2. BayesianPhysTwin with raw PointWorld observations;
3. BayesianPhysTwin with the selected Prob4D belief.

Report query error/proper score, coverage and width, accepted/rejected/fallback counts, harmful accepted updates, and worst accepted regret. Rejected or unsupported observations must return the exact physical fallback.

Provider competence and downstream value are separate conjunctive claims: neither may rescue failure of the other.

## Causal4D boundary

Do not modify the locked Causal4D physical protocol to enlarge this result. Causal4D can be described as a compatible later consumer, but the PointWorld/Flat'n'Fold study must stand on its own provider and BayesianPhysTwin evidence.

## Completion states

This source-qualification branch should close with exactly one of these outcomes:

- **authorized:** PointWorld action, coordinate, point-identity, representation, and support semantics are frozen without target outcomes, permitting a new held-out protocol version;
- **representation-negative:** the sparse PointWorld output cannot be mapped into a defensible Prob4D observation contract without target-dependent choices;
- **support-negative:** the required Flat'n'Fold streams/geometry do not support the frozen provider version;
- **runtime-negative:** the exact released PointWorld checkpoint cannot execute on the frozen source inventory.

Any negative state is a complete result for this protocol. A later retry requires a new protocol version rather than rewriting this one.
