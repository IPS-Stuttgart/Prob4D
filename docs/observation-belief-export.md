# ObservationBeliefV1 export

Prob4D can export independently decoded overlapping MotionCrafter windows through
the versioned `phys4d.observation_belief` interface consumed by
Bayesian-PhysTwin. The archive is a content-addressed, non-pickled NPZ and does
not require Bayesian-PhysTwin to be installed.

## Command

```bash
prob4d-export-observation-belief \
  outputs/sequence/predictions.json \
  outputs/sequence/observation_belief.npz \
  --case-id sequence \
  --causal-frame-stop 134 \
  --pixel-stride 4 \
  --gauge-mode fixed_lag
```

`--causal-frame-stop` is exclusive. The exporter reads and emits only absolute
source frames strictly before that value. A latent-overlap baseline is not used,
because it can blend RGB frames across a predictive boundary. The output uses
the independently decoded `overlap_windows` entries in `predictions.json`.

## Uncertainty representation

Every retained point has a local anisotropic `3 x 3` covariance from the
along-ray/lateral depth-disagreement model. In addition, each window contributes
one shared rank-seven factor obtained by linearizing its uncertain global
`Sim(3)` gauge:

```text
U_i = J_i L_k,   Sigma_gauge,k = L_k L_k^T.
```

Points from the same window share the same factor-group identity. Consequently,
Bayesian-PhysTwin can recover coherent cross-point gauge covariance instead of
counting the propagated marginal `3 x 3` terms as independent noise.

The current exporter represents different window gauges as separate latent
factor groups. Dependence that remains between windows is handled
conservatively by the effective-group composite weight rather than assumed
independent.

## Reliability and correlation boundary

Prior reliability is computed from decoded-window overlap disagreement and the
feeder uncertainty model. It is independent of any downstream physical
innovation. Association probability is a separate field and is one for the
dense pixel identity supplied by MotionCrafter.

Rows with the same absolute source frame form one effective likelihood group.
The group weight caps the information by `--effective-samples-per-group` and by
the number of unique pixels, so repeated overlap-window predictions cannot
silently multiply the evidence.

## Provenance

The descriptor records:

- the Prob4D source revision;
- the SHA-256 digest of `predictions.json`;
- the upstream MotionCrafter revision from the manifest;
- the exact causal frame boundary, gauge mode, pixel stride, and effective
  sample cap;
- a content digest covering the descriptor, every array name, dtype, shape,
  and byte payload.

Use `--source-revision` when running outside a Git checkout so the provider
revision is explicit rather than reported as `unknown`.
