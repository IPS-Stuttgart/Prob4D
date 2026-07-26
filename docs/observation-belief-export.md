# Causally sealed observation-belief export

Prob4D exposes independently decoded MotionCrafter windows through the
provider-neutral `phys4d.observation_belief` version-1 container consumed by
Bayesian-PhysTwin and validated independently by Causal4D. Prob4D-specific
statistical semantics are versioned separately; the strict causal stream contract
is version 2.

The export is deliberately distinct from reconstruction products. A row is
admissible only when the entire independently decoded source window lies before
an exclusive causal cutoff. The exporter reads the JSON manifest to decide which
payloads are admissible, opens only those payloads, and then recomputes alignment,
gauge estimation, overlap disagreement, uncertainty, and prior reliability on
that admitted prefix. It never estimates on a full sequence and then slices the
final rows.

## Metric gauge anchor

MotionCrafter points have an unresolved global `Sim(3)` gauge. An artifact whose
coordinates and covariance are labelled in metres therefore requires an
independent metric prior for the first retained overlap window. The anchor is a
content-addressed JSON document:

```python
from prob4d.provider_v1 import MetricGaugeAnchor, save_metric_gauge_anchor

anchor = MetricGaugeAnchor(
    window_id="window_0000",
    global_from_local=registration.transform,
    covariance=registration.covariance,
    coordinate_frame="phystwin-world",
    source_kind="prefix-only RGB-D registration",
    source_artifact_sha256=reference_prediction_payload_sha256,
    metadata={
        "calibration_split": "source-only",
        "calibration_artifact_sha256": calibration_artifact_sha256,
    },
)
save_metric_gauge_anchor("outputs/metric_gauge_anchor.json", anchor)
```

`source_artifact_sha256` must identify the exact first admitted prediction
payload. `calibration_artifact_sha256` must identify the exact external
registration or calibration artifact from which the transform and covariance
were obtained. Strict stream-v2 export rejects anchors without either binding.

The registration and its covariance must use only information authorized before
the causal cutoff. Simulated benchmark truth may be used only for an explicitly
labelled sensor-assisted ablation, not for a monocular claim.

A zero `7 x 7` anchor covariance is declared as
`fixed_external_calibration`. A nonzero covariance is declared as
`propagated_external_prior` and is included in the shared joint gauge factor.
The embedded anchor record states the case, world frame, exact source and
calibration digests, and covariance treatment so consumers do not infer these
semantics from factor names.

## Command

```bash
prob4d observation export \
  outputs/sequence/predictions.json \
  outputs/sequence/observation_belief.npz \
  --case-id sequence \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/metric_gauge_anchor.json \
  --pixel-stride 4 \
  --max-gauge-rank 64 \
  --minimum-retained-gauge-trace 0.999 \
  --source-revision <full-prob4d-commit> \
  --summary-json outputs/sequence/observation_belief_summary.json

prob4d-validate-observation outputs/sequence/observation_belief.npz
```

`--causal-frame-stop` is exclusive. An overlap-window manifest entry is admitted
only when its declared stop is at most the cutoff and its payload contains the
exact absolute frame IDs implied by the declared bounds and frame stride.
Unknown lineage schemas, path traversal, inconsistent frame IDs, non-prefix
window selections, an anchor for the wrong first payload, and incomplete
calibration provenance fail closed. The exporter records an exact 40- or
64-character Prob4D commit. It also fails closed when that revision cannot be
obtained from the checkout and is not provided explicitly.

The strict command first writes a hidden temporary archive, reloads it through
the strict validator, verifies the content address, flushes it, and atomically
replaces the requested output. The summary JSON uses the same temporary-file and
atomic-replace discipline.

## Joint gauge posterior

The production default is `--gauge-mode sequential`. Prob4D chooses one causal
spanning-tree parent for every retained window, preferring more correspondences,
then lower residual RMS, then the earlier reference window. It propagates the
metric-anchor covariance and every selected relative-alignment covariance into
one joint matrix

```text
Sigma_g in R^(7K x 7K),
```

including the cross-covariance between windows. Redundant alignment edges are
reported but are not fused into the production tree, which avoids silently
assuming independence between dense shared-backbone constraints.

A deterministic eigendecomposition produces a shared covariance root. The
export fails when `--max-gauge-rank` would retain less than
`--minimum-retained-gauge-trace` of the joint covariance trace. Rank reduction is
therefore explicit and auditable rather than a silent memory optimization.

The legacy `--gauge-mode fixed_lag` path remains available only with
`--allow-approximate-fixed-lag-covariance`. Its current covariance treats gauges
outside the active lag as exact posterior means and exports only block-diagonal
marginals. It is suitable for a labelled reconstruction ablation, not for the
strict stream-v2 Bayesian uncertainty claim. A future fixed-lag implementation
must carry a marginalized boundary information prior before this acknowledgement
can be removed.

## Artifact semantics

The archive contains metric 3-D means, full conditional `3 x 3` covariance,
absolute frame/entity/view/window identities, separate association probability
and prior reliability, effective correlation groups, and composite-likelihood
weights. For row `i` from window `k`, the shared low-rank factor is

```text
U_i = J_i L_k,
Sigma_g = L L^T,
L_k = rows 7k : 7(k + 1) of L.
```

Every row uses one common factor group because the latent vector is joint across
all windows. This preserves both per-window gauge marginal covariance and
cross-window covariance. A consumer that keeps the gauge terms as explicit
nuisance variables must use the local conditional covariance and must not add
`U_i U_i^T` to it again.

Association probability is diagnostic support for the decoded pixel identity;
it is not a MotionCrafter-to-physical-node association probability. Prior
reliability is derived from overlap disagreement without reading the downstream
physical innovation.

The exporter has no independently calibrated group-level nominal/outlier prior.
It therefore writes the neutral value `1.0` for
`group_prior_nominal_probability`; overlap reliability is not applied a second
time. `group_composite_weight` separately caps dense duplicate information.

The descriptor, all array names, dtypes, shapes, and bytes are covered by the
artifact ID. The source digest covers only admitted payload hashes and stable
prediction-affecting provenance. Consequently, appending post-cutoff windows to
the manifest cannot change an already valid prefix artifact. The operational
summary may report how many future or crossing entries were skipped, but that
post-cutoff bookkeeping is intentionally excluded from the artifact content
address.

## Cross-repository checks

Prob4D, Bayesian-PhysTwin, and Causal4D share a golden contract fixture. The same
artifact must have the same content address in all three repositories.
Bayesian-PhysTwin consumes the shared low-rank gauge factor as explicit nuisance
parameters, keeps association probability separate from reliability, and uses
the group prior and composite weight as distinct inputs. Causal4D independently
checks the version, rank, parent lineage, exact anchor/calibration bindings, and
covariance treatment before it binds the resulting twin belief to the
observation artifact.
