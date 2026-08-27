# PointWorld--Flat'n'Fold: next execution order

The branch now contains the sparse representation and support-feasibility
contracts. The next work must use real source bytes in this order; target
garments remain closed throughout.

## 1. Freeze external bytes

Record and checksum:

- PointWorld checkout at the protocol revision;
- the selected released PointWorld checkpoint;
- DINOv3 and every other model file loaded by the runtime;
- the exact Python/CUDA environment;
- Flat'n'Fold archive members used for the inventory/source cohort;
- the three camera calibration files;
- Baxter-base pose/action files; and
- any gripper geometry used to construct robot point flow.

Derive one loader ID from the complete loader configuration and one model-set ID
from all executed model bytes. Do not equate a checkpoint filename with either
identity.

## 2. Freeze complete garment inventory

Copy
`examples/pointworld-flatnfold-support-inventory.example.json` to a new output
directory and replace the synthetic identities with the complete inventory/source
roster. Every demonstration must retain all three cameras, one action digest, and
one causal frame schedule.

Generate and retain:

```bash
python -m prob4d.pointworld_flatnfold_support evaluate \
  source-support-inventory.json \
  provider-support-request.json \
  provider-support-result.json
```

A support-negative result stops this PointWorld version before inference.

## 3. Execute one source-only representation smoke test

For one predeclared source garment and demonstration:

1. transform RGB-D scene points and Baxter/gripper action geometry into the
   declared metric Baxter-base frame;
2. run the frozen PointWorld checkpoint once;
3. export exact context points, displacement-from-context, support masks, native
   log variance, frame IDs, and bound identities with
   `write_pointworld_sparse_source_export`;
4. convert with `python -m prob4d.pointworld_sparse_adapter convert`;
5. verify the canonical content ID; and
6. retain visualization only as a diagnostic, never as a target-driven mask or
   correspondence editor.

The smoke test passes representation only when the context displacement is exact
zero, context validity equals source support, inactive points never reappear, and
all coordinates remain finite. Failure is retained without substituting a dense
rasterization.

## 4. Localize the remaining scientific problem

After representation succeeds, use source garments to determine whether the
next limitation is:

- PointWorld mean/identity quality;
- action or coordinate conversion;
- native uncertainty calibration;
- cross-horizon/cross-window dependence;
- cross-window point association; or
- downstream BayesianPhysTwin query identifiability.

Only the localized failure may authorize a new method. In particular, do not add
an uncertainty model when source means or action conversion have failed.

## 5. Freeze the held-out experiment

Only after source support, representation, mean quality, covariance calibration,
and downstream relevance are positive may a new protocol bind:

- garment-disjoint source, calibration, and target rosters;
- raw PointWorld, simple overlap, Prob4D tree, and guarded graph arms;
- one BayesianPhysTwin query and exact physical fallback;
- garment-clustered intervals;
- coverage, proper-score, covariance-width, and harmful-update gates; and
- one target execution with no retuning.

The frozen Causal4D physical experiment remains independent and is not modified
by this route.
