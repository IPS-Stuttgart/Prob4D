# Causally sealed observation-belief export

Prob4D exposes independently decoded MotionCrafter windows through the
provider-neutral `phys4d.observation_belief` version-1 container consumed by
Bayesian-PhysTwin and independently validated by Causal4D. Prob4D-specific
statistical semantics are versioned separately inside the descriptor metadata;
the current provider contract is version 2.

A row is admissible only when its complete independently decoded source window
lies before the exclusive causal cutoff. The exporter reads manifest metadata
first, opens only admitted payloads, and then recomputes alignment, gauge
estimation, overlap disagreement, uncertainty, and prior reliability on that
prefix. It never estimates on a full sequence and slices the final rows.

## Metric gauge anchor

MotionCrafter points have an unresolved global `Sim(3)` gauge. Coordinates and
covariance can be labelled in metres only after binding an independent metric
calibration to the exact first retained prediction payload.

Create that content-addressed anchor with:

```bash
prob4d observation create-anchor \
  outputs/sequence/predictions.json \
  outputs/sequence/metric_gauge_anchor.json \
  --case-id sequence \
  --world-frame-id phystwin-world \
  --reference-window-id window_0000 \
  --calibration-artifact outputs/sequence/prefix_registration.json \
  --sim3-vector LOG_SCALE RX RY RZ TX TY TZ
```

The default covariance is zero and is declared as
`fixed_external_calibration`. To preserve a calibrated uncertain anchor, pass a
`7 x 7` `.npy` matrix or an `.npz` containing only `covariance`:

```bash
  --covariance outputs/sequence/prefix_registration_covariance.npy
```

A nonzero anchor covariance is declared as
`propagated_joint_gauge_covariance` and is included in the shared joint factor.
The anchor records:

- its schema and content address;
- case, reference window, and world frame;
- the exact reference-window payload SHA-256;
- the exact calibration-artifact SHA-256; and
- the covariance treatment.

Legacy in-process anchors without a calibration digest can still be loaded for
old reconstruction code, but the portable exporter rejects them. Recreate such
anchors with the command above.

The calibration and covariance must use only information authorized before the
causal cutoff. Simulated benchmark truth may be used only for an explicitly
labelled sensor-assisted ablation, not for a monocular claim.

## Export command

```bash
prob4d observation export \
  outputs/sequence/predictions.json \
  outputs/sequence/observation_belief.npz \
  --case-id sequence \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/sequence/metric_gauge_anchor.json \
  --pixel-stride 4 \
  --max-gauge-rank 64 \
  --minimum-retained-gauge-trace 0.999 \
  --source-revision <full-prob4d-commit> \
  --summary-json outputs/sequence/observation_belief_summary.json

prob4d-validate-observation outputs/sequence/observation_belief.npz
```

`--causal-frame-stop` is exclusive. A window is admitted only when its declared
stop is at most the cutoff and its payload contains exactly the absolute frame
IDs implied by the manifest. Unknown lineage schemas, path traversal,
inconsistent frame IDs, non-prefix selections, anchor/payload mismatches, and
missing calibration provenance fail closed. The source revision must be an exact
40- or 64-character commit.

Metric-anchor JSON and observation NPZ files are written through temporary files.
The NPZ is strictly reloaded, its content address is recomputed, and only then is
it atomically moved to its requested path.

## Provider contract version 2

The generic `ObservationBeliefV1` container deliberately does not prescribe one
gauge representation. Prob4D therefore records explicit provider fields:

```json
{
  "prob4d_observation_contract_version": 2,
  "covariance_layout": "joint_sim3_tree_root_v1",
  "factor_group_semantics": "single_shared_standard_normal_latent",
  "metric_anchor_covariance_included_in_joint_factor": true
}
```

Consumers must dispatch on these fields. They must not infer statistical meaning
from a rank of seven or from factor names alone.

### Production joint-tree layout

The default `--gauge-mode sequential` chooses one causal spanning-tree parent for
every retained window. It propagates metric-anchor and selected relative-alignment
uncertainty into

```text
Sigma_g in R^(7K x 7K),
```

including cross-window covariance. A deterministic eigendecomposition produces a
shared root `L`. For row `i` from window `k`,

```text
U_i = J_i L_k,
Sigma_g = L L^T,
L_k = rows 7k : 7(k + 1) of L.
```

Factor names are `joint_gauge_latent_0000`, `...`, up to the exported rank, and
all rows use factor group zero. Bayesian-PhysTwin therefore creates one set of
standard-normal nuisance parameters, not one independent copy per window.

The export fails when the rank cap retains less covariance trace than
`--minimum-retained-gauge-trace`. Redundant alignment edges are reported but not
fused, avoiding an untested independence assumption between dense constraints
from the same backbone.

### Approximate fixed-lag layout

`--gauge-mode fixed_lag` remains available only with
`--allow-approximate-fixed-lag-covariance`. It is declared as

```text
approximate_fixed_lag_block_diagonal_sim3_root_v1
```

and requires a fixed metric anchor. Its current covariance treats retired gauges
as exact posterior means and contains block-diagonal marginals rather than
cross-window covariance. Consumers admit it only when the approximation flag,
posterior model, rank, and factor grouping all agree. It is a labelled
reconstruction ablation, not the production Bayesian uncertainty claim.

### Legacy layout

Consumers continue to admit previously frozen artifacts with exactly seven
`gauge_latent_0` through `gauge_latent_6` factors and
`factor_group_ids == window_indices`. Those artifacts must use a fixed external
metric anchor. New Prob4D exports do not use this representation.

## Remaining observation semantics

`local_covariance_m2` is conditional covariance. A consumer that retains the
low-rank factor as explicit nuisance parameters must not add `U_i U_i^T` to that
local covariance again.

Association probability is diagnostic support for the decoded pixel identity;
it is not a MotionCrafter-to-physical-node association probability. Prior
reliability is derived from overlap disagreement without reading the downstream
physical innovation. The group-level nominal prior remains neutral at `1.0`;
`group_composite_weight` separately caps dense duplicate information.

The descriptor, array names, dtypes, shapes, and bytes are covered by the
artifact ID. The source digest covers only admitted payload hashes and stable
prediction-affecting provenance. Appending post-cutoff windows therefore cannot
change an already valid prefix artifact, and future payloads are never opened.

## Cross-repository validation

Prob4D tests complete anchor metadata, rank and factor-group consistency,
serialization round trips, and tamper rejection. Bayesian-PhysTwin and Causal4D
independently validate the same provider contract without importing Prob4D.
Both consumers retain legacy-layout support and fail closed on ambiguous or
partially migrated joint artifacts.
