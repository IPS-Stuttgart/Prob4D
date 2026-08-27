# PointWorld → Prob4D → BayesianPhysTwin on Flat'n'Fold

## Purpose

This is the highest-value remaining Prob4D paper experiment: test whether a
provider-agnostic, correlation-aware recursive belief improves a real
action-conditioned 3-D world model and one frozen downstream BayesianPhysTwin
query.

The intended external provider is PointWorld (`NVlabs/PointWorld`) and the
intended independent real cohort is the robot subset of Flat'n'Fold
(`lipeng-zhuang521/flat-n-fold`). The experiment is deliberately separated from
the terminal CUT3R attempt, the already-opened MotionCrafter/Deform360 cohorts,
and the locked Causal4D physical-acquisition protocol.

The current work remains **source qualification only**. No target outcome may be
opened or used to choose a representation mapping, checkpoint, split, mask,
exclusion, covariance treatment, action conversion, synchronization rule, or
downstream query.

The machine-readable source lock is
`protocols/pointworld-flatnfold-source-qualification-v1.json`. Source-only
representation and static-input findings are retained under
`evidence/pointworld-flatnfold-source-qualification-v1/`.

## Frozen public revisions

The original source-qualification protocol binds:

- Prob4D: `d02f057671023ff586ebfc15c904dbb0a60f4425`;
- BayesianPhysTwin: `b5f07d649ac2cd7dc6ca1aceb4004ff4803bddc8`;
- PointWorld: `05484826dfef74cbe278a3974179a5a16705d35d`;
- Flat'n'Fold code: `fa0d3d17ac827e7b5c43ec1ef7c0c38ad5e39340`.

The persistent-point follow-up was implemented additively on later Prob4D main
without changing the production dense provider, estimator, gauge, fusion, or
BayesianPhysTwin contracts. The exact PointWorld checkpoint bytes, runtime,
normalization-statistics bytes, Flat'n'Fold dataset bytes, garment roster,
frame/action schedule, and downstream query remain unbound.

## Why this experiment

Prob4D already contains the methodological ingredients needed for a paper
contribution: joint gauge uncertainty, cross-window covariance, covariance
intersection, guarded multi-edge graph fusion, finite-sample source-cycle
admission, sparse metric-anchor calibration, provider-neutral manifests, and
exact fallback. Another covariance or planner component would add complexity
without resolving the main empirical gap.

The missing evidence is a fresh real provider and fresh physical object/session
cohort showing that the same recursive belief machinery can wrap a different
action-conditioned model and improve a downstream physical query.

PointWorld is materially different from MotionCrafter. Its released model
predicts full-scene 3-D point flows from RGB-D observations and robot actions
represented as 3-D point flows. Flat'n'Fold reports 887 robot demonstrations over
44 garments in eight categories, with three-camera RGB-D observations, camera
calibration, Baxter action information, and point-cloud construction code.

## Representation gate: resolved at the artifact level

Prob4D `PredictionWindow` v2 stores a dense `T × H × W × 3` image-grid point
map. The released PointWorld model instead preserves `N_scene` seeded scene
points. At the pinned PointWorld revision it:

1. takes the first scene-point frame as `scene_coord0`;
2. predicts `B × T × N_scene × 3` relative displacements;
3. forms absolute positions as `scene_coord0 + displacement`; and
4. emits `B × T × N_scene × 1` log variance.

The source-point axis is therefore persistent within one PointWorld forecast
window. Rasterizing it onto `H × W` would discard native identity and introduce
an unnecessary mapping.

Prob4D now provides the separately versioned experimental artifact
`PersistentPointPredictionWindow`, serialized as
`prob4d.persistent-point-prediction-window-npz` version 1. It retains:

- absolute output-frame indices;
- source point-axis indices;
- deterministic window-scoped point IDs;
- `T × N × 3` absolute positions;
- `T × N` validity;
- explicit position and identity semantics; and
- optional raw PointWorld log variance plus the exact normalization-statistics
  SHA-256 that gives that quantity meaning.

Point IDs deliberately differ across windows. This resolves the representation
shape without silently claiming cross-window material correspondence.

PointWorld's uncertainty head is trained against normalized initial-frame-relative
displacement. Its scalar log variance is retained as raw provider evidence and
is **not** called metric covariance. It must not be passed directly to
`ObservationFactor.local_covariance_m2`.

The representation decision is
`evidence/pointworld-flatnfold-source-qualification-v1/representation-decision.json`.
Its status is
`PERSISTENT_POINT_ARTIFACT_AUTHORIZED_PROVIDER_NOT_YET_QUALIFIED`.

## Static camera and action audit

The pinned source files establish some useful facts but do not yet authorize an
executable PointWorld bridge.

### PointWorld action input

PointWorld consumes sampled robot surface-point positions over the complete
horizon. Its released domain dispatch accepts only `behavior` and `droid`:

- BEHAVIOR requires joint positions, joint names, base pose, and a compatible
  robot sampler;
- DROID requires seven Panda joint positions, gripper positions, and a compatible
  robot sampler.

### Flat'n'Fold action source

The public Flat'n'Fold parser reads per-arm end-effector position, quaternion
orientation, and binary gripper state. The repository also supplies one electric
Baxter-gripper STL mesh. The inspected public source does not establish complete
Baxter joint trajectories or a PointWorld-compatible Baxter robot sampler.

A technically defensible source-only candidate is therefore to sample the frozen
gripper mesh once and transform the same surface points with the two timestamped
end-effector pose streams. That candidate must be labelled a distribution-shifted
**dual-gripper-surface** action representation, not as equivalent to the released
DROID or BEHAVIOR robot representation. Mesh sampling, open/closed geometry,
coordinates, interpolation, arm identity, and source lineage must be frozen
before target access.

### Camera geometry and synchronization

The published calibration text gives intrinsic tuples for front, top, and side
cameras and separate robot-demonstration extrinsics for top and side. The merge
script uses the robot-specific top/side extrinsics and the published front
extrinsic.

Two source limitations remain:

1. the front visualization helper passes intrinsic values inconsistent with the
   published front tuple, so that helper is not an authoritative calibration
   implementation; and
2. the three-camera merge script defines nearest-timestamp logic but comments out
   its use, pairing sorted front/top/side filenames by list index instead.

The dataset bytes must therefore demonstrate one-to-one timestamp association,
a frozen maximum skew, and action-to-camera synchronization. Sorted-index pairing
alone is not accepted.

The complete static result is
`evidence/pointworld-flatnfold-source-qualification-v1/static-source-audit.json`.
Its status is
`SOURCE_STATIC_AUDIT_COMPLETE_DATASET_BYTE_QUALIFICATION_REQUIRED`.

## Remaining source-only qualification sequence

Before provider residuals or target outcomes are opened:

1. bind the exact PointWorld checkpoint SHA-256, runtime, and normalization
   statistics SHA-256;
2. inventory the Flat'n'Fold robot data and bind the exact byte manifest;
3. freeze complete physical garment identities as the top-level statistical
   units;
4. verify three-camera timestamp coverage, one-to-one association, maximum skew,
   and intrinsic/extrinsic identities from the dataset bytes;
5. freeze and execute the dual-gripper-surface action conversion, or retain an
   action-representation-negative result;
6. verify metric coordinate transformations using released calibration only;
7. bind complete visual and action source lineage for every forecast frame;
8. calibrate PointWorld uncertainty and source reliability on source/calibration
   garments only, or retain a calibration-negative result;
9. create an outcome-blind `ProviderSupportFeasibilityV1` request with all
   required streams and geometry support; and
10. stop if the support gate fails.

The existing `prob4d.provider_support_feasibility` contract should be reused; no
PointWorld-specific support framework is needed.

## Later held-out study, only after qualification passes

Use complete garment identity as the independent unit. Demonstrations/sessions
are nested observations; frames and points are not independent replicates.

The primary provider comparison should remain small:

1. raw/disjoint PointWorld;
2. uniform overlap fusion;
3. Prob4D sequential full-joint spanning tree; and
4. conformal-guarded full-joint graph with exact tree fallback.

Useful diagnostic controls include persistence, latest-window overwrite, naive
precision fusion, and the unguarded graph.

Provider endpoints should include long-horizon and terminal point/flow error,
overlap seam or drift, proper score, 50/90/95% coverage, normalized NEES,
covariance width, support/fallback accounting, and worst-garment performance.
Inference should cluster at garment identity.

## BayesianPhysTwin gate

Freeze exactly one downstream physical query before target access. Compare:

1. unchanged physical fallback;
2. BayesianPhysTwin with raw PointWorld observations; and
3. BayesianPhysTwin with the selected Prob4D belief.

Report query error/proper score, coverage and width,
accepted/rejected/unsupported/fallback counts, harmful accepted updates, and
worst accepted regret. Rejected or unsupported observations must return the exact
physical fallback.

Provider competence and downstream value are separate conjunctive claims:
neither may rescue failure of the other.

## Causal4D boundary

Do not modify the locked Causal4D physical protocol to enlarge this result.
Causal4D can be described as a compatible later consumer, but the
PointWorld/Flat'n'Fold study must stand on its own provider and BayesianPhysTwin
evidence.

## Completion states

Source qualification closes with exactly one of these outcomes:

- **authorized:** PointWorld action, coordinate, point-identity, representation,
  calibration, lineage, and support semantics are frozen without target outcomes;
- **action-negative:** the Flat'n'Fold Baxter records cannot be converted into a
  defensible PointWorld robot point-flow input;
- **support-negative:** the required Flat'n'Fold streams, synchronization, or
  geometry do not support the frozen provider version;
- **runtime-negative:** the exact released PointWorld checkpoint cannot execute
  on the frozen source inventory; or
- **calibration-negative:** the provider runs but its source-only uncertainty or
  reliability treatment does not pass the frozen calibration gate.

Any negative state is a complete result for this protocol. A later retry requires
a new protocol version rather than rewriting this one.
