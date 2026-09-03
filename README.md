# Prob4D

Probabilistic recursive fusion for long-horizon 4D reconstruction from
overlapping prediction windows.

Prob4D models every window's global gauge as an uncertain `Sim(3)` state,
calibrates conditional and shared covariance, preserves causal source lineage,
and exports portable observations for BayesianPhysTwin. Causal4D consumes the
resulting BayesianPhysTwin belief rather than raw Prob4D artifacts.

## Provider evidence status

<!-- provider-evidence-status:begin -->
Snapshot: **2026-08-26**. Adapter maturity and scientific evidence are separate.
The authoritative cross-repository claim and data-access status remains in
[`FlorianPfaff/BayesianPhysTwin-Paper`](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper).

| Provider route | Implementation state | Current evidence boundary |
| --- | --- | --- |
| Controlled synthetic provider | Stable mechanism study | Positive controlled evidence only; not real-provider competence |
| MotionCrafter | Mature producer and causal-window path | Frozen retrospective real-transfer negative; the opened cohort is not a promotion candidate |
| Deform360 official-Hub route | Executed under a sealed support protocol | Terminal source-support negative; confirmation remains closed |
| CUT3R recurrent-online | Causal adapter and direct-point-map route; one registered development smoke consumed | The exact smoke loaded the pinned model and checkpoint, decoded 58 causal frames, entered the provider forward call, and ended after a CUDA device assertion; no prediction or source shard was authorized, so competence remains unestablished |
| VGGT | Baseline/export adapter ready | Diagnostic baseline only |

This table is generated from
[`evidence/provider_status_snapshot_v1.json`](evidence/provider_status_snapshot_v1.json),
which pins the repository provider-status contract by Git blob identity.
Regenerate it with `python scripts/render_provider_status.py --write`; CI
checks that the contract, snapshot, and README stay synchronized.
<!-- provider-evidence-status:end -->

A production-quality artifact path is not evidence of production-quality
perception. Every real provider must still pass support, mean, identity,
gauge/dependence, covariance, query-relevance, and exact-fallback gates.

## Install and test

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests scripts/ci
```

The runtime package depends only on NumPy. MotionCrafter, VGGT, Torch, Diffusers,
and model checkpoints remain optional external dependencies.

## Supported interfaces

The optional [native CUT3R RGB runner](docs/cut3r-native-inference.md) adds streaming
state updates and verified direct-point-map export via `prob4d prediction run-cut3r`.
Its software support is separate from the frozen provider-evidence snapshot above.

Prob4D 0.5 has one command and one current Python façade:

```text
prob4d
prob4d.api.v2
```

```python
from prob4d.api.v2 import (
    Sim3,
    load_claim_bearing_observation_belief,
)
```

The package root deliberately exposes only `__version__`:

```python
import prob4d

print(prob4d.__version__)
```

Use `prob4d commands list` and `prob4d commands describe ...` to inspect the
canonical grouped command registry.

## Artifact inspection

Explain an observation, sparse gauge prior, tree-sparse factor artifact,
prediction store, or strict JSON record without guessing that an unknown schema
is valid:

```bash
prob4d artifact explain outputs/sequence_name/observation_belief.npz
prob4d artifact explain outputs/sequence_name/observation_belief.npz --json
prob4d artifact explain artifact.json --require-strict
```

Known schemas use their existing strict content-addressed loader. Unknown JSON
and NPZ containers are labelled `structural-only`; `--require-strict` rejects
them. See [artifact explanation](docs/artifact-explain.md).

## Quick benchmark

```bash
prob4d ablate synthetic \
  --output-dir outputs/synthetic \
  --seed 7
```

The ablation runner evaluates disjoint windows, latent-space overlap blending,
decoded `Sim(3)` alignment, precision fusion, covariance intersection,
fixed-lag gauge smoothing, and sparse metric anchors.

## MotionCrafter producer and historical evaluation path

Create the optional GPU environment:

```bash
scripts/bootstrap_motioncrafter_env.sh \
  ../MotionCrafter \
  ../prob4d-motioncrafter-venv
```

Generate disjoint, latent-overlap, and independently decoded overlapping
products in one model-loading session:

```bash
CUDA_VISIBLE_DEVICES=0 ../prob4d-motioncrafter-venv/bin/prob4d motioncrafter \
  input.mp4 \
  --upstream-root ../MotionCrafter \
  --output-dir outputs/sequence_name \
  --model-type determ \
  --height 320 --width 640 \
  --window-size 25 --overlap 8 \
  --cache-dir /path/to/huggingface-cache
```

The prediction manifest records absolute source frames, internal window
lineage, model identity, seed policy, and content hashes.

## Claim-bearing observation export

A claim-bearing provider-v2 export requires an independently fitted metric
anchor and covariance calibrations:

```bash
prob4d observation export-calibrated \
  outputs/sequence_name/predictions.json \
  outputs/sequence_name/observation_belief.npz \
  --case-id sequence_name \
  --causal-frame-stop 134 \
  --metric-gauge-anchor outputs/sequence_name/metric_gauge_anchor.json \
  --gauge-covariance-calibration outputs/calibration/gauge.json \
  --point-uncertainty-calibration outputs/calibration/point.json \
  --source-revision "$(git rev-parse HEAD)" \
  --summary-json outputs/sequence_name/observation_belief_summary.json

prob4d observation validate \
  outputs/sequence_name/observation_belief.npz
```

Load claim-bearing observations through the stable façade:

```python
from prob4d.api.v2 import load_claim_bearing_observation_belief

validated = load_claim_bearing_observation_belief(
    "outputs/sequence_name/observation_belief.npz"
)
```

Use `prob4d observation export-exploratory` only for explicitly labelled
uncalibrated, pointwise-fallback, alternate-root, or fixed-lag controls. The bare
`prob4d observation export` route prints guidance and runs no exporter.

The default sequential gauge tree propagates the full joint cross-window gauge
covariance from the metric anchor. Rank reduction is admitted only when its
retained covariance trace passes the declared threshold. See
[provider API v2](docs/provider-v2.md),
[observation export](docs/observation-belief-export.md), and
[compatibility boundaries](docs/compatibility.md).

## Provider readiness and failure localization

The target-free provider workflow keeps support, source means, identities,
gauge/dependence, conditional point covariance, downstream query relevance, and
exact fallback as ordered gates:

```bash
prob4d diagnostic provider-support-envelope --help
prob4d diagnostic source-covariance-localization --help
prob4d diagnostic target-free-rehearsal --help
python -m prob4d.gauge_linearization_closure --help
prob4d provider prefix-admission --help
prob4d experiment fresh-provider-readiness --help
```

The [provider evidence status](docs/provider-evidence-status.md) separates
implemented capability, controlled mechanism evidence, real-provider boundaries,
and the next authorized experiment. Before assigning a remaining failure to
conditional point covariance, use the
[joint `Sim(3)` linearization-closure diagnostic](docs/gauge-linearization-closure.md)
to compare analytic propagation with a deterministic nonlinear reference.

Only an explicit `point-covariance-localized` result authorizes source-only
point-uncertainty development. Only a complete
`ready-for-one-target-evaluation` decision authorizes one evaluation of the exact
bound unopened target roster. Before a one-shot source or target execution, run
the [target-free observation rehearsal](docs/target-free-rehearsal.md). It seals a
normative positive control, verifies it through the official loader and a separate
implementation that imports no Prob4D code, and proves that registered adversarial
controls fail closed without opening outcomes. See
[provider readiness localization](docs/provider-readiness-localization.md) and
[fresh-provider readiness](docs/fresh-provider-readiness.md).

## Recursive factors and provider promotion

Provider v2 includes append-only `ObservationFactorStreamV1`, schema-v4
explicit-gauge factor bundles, tree-sparse artifacts, strict runtime attestation,
and exact fallback semantics.

Freeze and replay the held-out promotion protocol with:

```bash
prob4d experiment heldout-provider freeze \
  protocol.json \
  --output promotion-lock.json

prob4d experiment heldout-provider run \
  promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results query-results.raw.json \
  --output-dir outputs/promotion

prob4d experiment heldout-provider verify \
  promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results outputs/promotion/query_results.sealed.json \
  --report outputs/promotion/promotion_report.json
```

The target-free lock binds complete source and target rosters, provider/model
identity, calibration, fallback, bootstrap settings, and decision margins.
Passing infrastructure checks is not scientific promotion.

## Additional grouped routes

```bash
prob4d artifact explain --help
prob4d diagnostic provider-support-envelope --help
prob4d diagnostic source-covariance-localization --help
prob4d evaluate provider --help
prob4d experiment fresh-provider-readiness --help
prob4d identity --help
prob4d provider prefix-admission --help
prob4d provider target-admit --help
prob4d provider target-verify --help
prob4d sintel uncertainty --help
prob4d phystwin evaluate --help
prob4d phystwin uncertainty --help
prob4d phystwin state --help
prob4d storage benchmark --help
prob4d storage materialize --help
prob4d storage validate --help
prob4d vggt baseline --help
```

## Prob4D 0.5 cleanup boundary

Prob4D 0.5 removes:

- all standalone `prob4d-*` executables;
- `prob4d commands migrate` and legacy alias metadata;
- `prob4d.api.v1`;
- provider-v1 execution/export entry points; and
- the broad lazy package-root façade.

Full provider-v1 reproduction belongs to the exact Prob4D 0.4.1 wheel or tagged
source revision. The 0.5 package retains only a narrow `prob4d.provider_v1`
artifact-compatibility bridge for immutable schema records, manifests, and
serialization needed by frozen evidence and the three-repository contract
corpus. It does not expose a provider-v1 estimator or exporter.

Historical content-addressed artifacts may correctly retain old repository
owner strings and schema identifiers. Do not rewrite their provenance.

See [the 0.5.0 release boundary](docs/releases/0.5.0.md),
[public API](docs/public-api.md), and
[distribution boundaries](docs/distribution-boundaries.md).

## Repository and evidence boundary

This repository owns code, tests, protocols, and portable contracts. Videos,
weights, predictions, and generated evidence are gitignored. Paper-facing tables,
figures, frozen run manifests, and exact revision bundles belong in
[`FlorianPfaff/BayesianPhysTwin-Paper`](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper).

Claim-bearing runs must bind exact Prob4D, BayesianPhysTwin, and Causal4D
revisions, wheel hashes, provider manifests, contract identities, input/output
digests, and protocol identifiers. Compatibility and conformance are
infrastructure evidence; they do not establish accuracy, calibration, physical
benefit, intervention benefit, deployment safety, or state of the art.
