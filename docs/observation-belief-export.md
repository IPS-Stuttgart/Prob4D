# Causally sealed observation-belief export

Prob4D can expose independently decoded MotionCrafter windows through the
provider-neutral `phys4d.observation_belief` version-1 artifact consumed by
Bayesian-PhysTwin and validated independently by Causal4D.

The export is deliberately distinct from the reconstruction products. A window
is admissible only when its complete independently decoded source interval lies
before an exclusive causal cutoff. The exporter reads manifest metadata first,
opens only admitted payloads, and then recomputes alignment, gauge estimation,
overlap disagreement, uncertainty, and prior reliability on that admitted
prefix. It never estimates on the full sequence and slices the final rows.

## Fixed metric gauge anchor

MotionCrafter points have an unresolved global `Sim(3)` gauge. An artifact whose
coordinates and covariance are labelled in metres therefore requires an
independent metric calibration for the first retained overlap window.

The portable version-1 artifact uses one seven-dimensional coherent factor per
window. A nonzero global-anchor covariance would require an additional latent
factor shared by every window and would make the compact factor rank grow with
the number of windows. Consequently, the portable exporter accepts only a
**fixed** metric anchor. Use `ObservationFactorBundle` instead when the global
metric anchor must remain uncertain.

Create a content-addressed fixed anchor after fitting a prefix-only calibration:

```bash
prob4d-create-metric-gauge-anchor \
  outputs/sequence/predictions.json \
  outputs/sequence/metric_gauge_anchor.json \
  --case-id sequence \
  --world-frame-id phystwin-world \
  --reference-window-id window_0000 \
  --calibration-artifact outputs/sequence/prefix_registration.json \
  --sim3-vector LOG_SCALE RX RY RZ TX TY TZ
```

The anchor binds the exact independently decoded reference-window payload and
the calibration artifact by SHA-256. It does not hash or open later prediction
windows. The calibration must use only information authorized before the causal
cutoff. Simulated benchmark truth may be used only for an explicitly labelled
sensor-assisted ablation, not for a monocular claim.

## Export command

```bash
prob4d-export-observation-belief \
  outputs/sequence/predictions.json \
  outputs/sequence/observation_belief.npz \
  --case-id sequence \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/sequence/metric_gauge_anchor.json \
  --pixel-stride 4 \
  --summary-json outputs/sequence/observation_belief_summary.json
```

`--causal-frame-stop` is exclusive. An overlap-window manifest entry is admitted
only when its declared stop is no later than the cutoff and its payload contains
the exact absolute frame IDs implied by the declared bounds and frame stride.
Unknown lineage schemas, path traversal, inconsistent frame IDs, non-prefix
window selections, anchor/payload mismatches, and uncertain global anchors fail
closed.

When Prob4D is run outside its Git checkout, pass `--source-revision` explicitly
so the artifact retains an exact producer revision.

## Artifact semantics

The archive contains metric 3-D means, full conditional `3 x 3` covariance,
absolute frame/entity/view/window identities, separate association probability
and prior reliability, effective correlation groups, and composite-likelihood
weights. For row `i` assigned to window gauge `k`, the shared low-rank factor is

```text
U_i = J_i L_k,       Sigma_gauge,k = L_k L_k^T.
```

A consumer that keeps these gauge terms as explicit nuisance variables must use
`local_covariance_m2` and must not add `U_i U_i^T` again. Association probability
records support for the decoded pixel identity. Prior reliability is derived
from overlap disagreement without reading the downstream physical innovation.

The current compact export retains one marginal relative-gauge factor per
window. Remaining cross-window dependence is not asserted away; frame-level
composite-likelihood weights cap repeated evidence. A future joint sparse gauge
posterior can replace this approximation without changing the causal source
selection rule.

## Content addressing and append invariance

The descriptor, every array name, dtype, shape, and byte payload are covered by
the observation artifact ID. The causal source digest covers only admitted
payload hashes, admitted source bounds, stable prediction-affecting settings,
MotionCrafter revision, temporal-lineage declaration, and the metric-anchor ID.

Appending post-cutoff windows therefore cannot change a previously valid prefix
artifact. The operational summary may report how many future or crossing
manifest entries were skipped, but that post-cutoff bookkeeping is intentionally
excluded from the artifact content address. Future or crossing prediction
payloads are not opened.

## Cross-repository checks

Prob4D, Bayesian-PhysTwin, and Causal4D share a golden contract fixture. The same
artifact must have the same content address in all three repositories.
Bayesian-PhysTwin consumes the low-rank gauge factors as nuisance parameters,
while Causal4D can bind the resulting selected twin belief to the exact
observation artifact without importing Prob4D.
