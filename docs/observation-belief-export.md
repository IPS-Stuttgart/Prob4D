# Source-causal ObservationBeliefV1 export

Prob4D can export independently decoded MotionCrafter windows through the
content-addressed `phys4d.observation_belief` interface consumed by
Bayesian-PhysTwin and independently validated by Causal4D.

## Command

```bash
prob4d-export-observation-belief \
  outputs/sequence/predictions.json \
  outputs/sequence/observation_belief.npz \
  --case-id sequence \
  --causal-frame-stop 134 \
  --pixel-stride 4 \
  --gauge-mode fixed_lag \
  --source-revision <exact-prob4d-commit> \
  --reliability-calibration-id <sha256>
```

`--causal-frame-stop` is exclusive. An observation row is not admitted merely
because its output frame precedes the cutoff. MotionCrafter generates a window
from all RGB frames in that source interval, so the exporter first rejects every
window with

```text
window.stop_frame > causal_frame_stop.
```

Gauge estimation, overlap disagreement, uncertainty construction, and row
sampling are then rerun using only the admitted windows. A crossing window
cannot contribute indirectly through alignment or calibration. If no complete
source window remains, export fails closed.

## Reliability boundary

The artifact stores four distinct concepts:

- `association_probability`: support for the point/material association;
- `prior_reliability`: residual-independent feeder evidence;
- `group_prior_nominal_probability`: prior probability of the nominal robust
  component;
- `group_composite_weight`: the power used to cap correlated evidence.

A production export requires a source-locked reliability-calibration content
ID. `--allow-uncalibrated-reliability` is available only for an explicitly
labelled diagnostic; the artifact records that status. The downstream physical
innovation is never used to create prior reliability.

## Structured covariance

Every retained point has an anisotropic local `3 x 3` covariance. Each admitted
window also contributes one shared rank-seven factor obtained by linearizing
its uncertain `Sim(3)` gauge:

```text
U_i = J_i L_k,  Sigma_gauge,k = L_k L_k^T.
```

This preserves coherent cross-point gauge uncertainty. A consumer that keeps
the gauge explicit must use the conditional local covariance and the shared
factor exactly once.

## Provenance

The exported content address covers the descriptor and all arrays. Its source
artifact digest additionally covers `predictions.json` and every admitted NPZ
window payload. Metadata records:

- admitted source-window identities and half-open frame bounds;
- future-dependent windows that were rejected;
- the maximum source frame read and exclusive cutoff;
- exact Prob4D and MotionCrafter revisions;
- gauge mode, lag, pixel stride, effective-sample cap, and calibration ID;
- the manifest temporal-lineage contract.

Latent overlap blends and all-frame reconstruction controls are not used by
this predictive exporter.
