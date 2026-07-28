# Provider-v2 gauge-backend ablation

`prob4d-ablate real` is retained as a frozen reproduction surface. Its sequential
initialization combines every available earlier-window candidate with covariance
intersection. Claim-bearing provider-v2 export instead selects one deterministic
causal spanning tree and propagates its full joint cross-window `Sim(3)` covariance
with analytic composition Jacobians.

Use the provider-aligned runner when an experiment is intended to diagnose the
same gauge backend that feeds a provider-v2 observation export:

```bash
prob4d-ablate-provider-v2-gauge \
  --predictions outputs/test/predictions.json \
  --truth data/test_truth.npz \
  --calibration-predictions outputs/calibration/predictions.json \
  --calibration-truth data/calibration_truth.npz \
  --metric-anchor-every 2 \
  --output-dir outputs/provider-v2-gauge-ablation
```

The command preserves the existing seven-row reconstruction contract. The two
upstream MotionCrafter baselines are unchanged. Prob4D rows use:

- posterior mode `sequential_joint_spanning_tree_v1`;
- analytic `Sim(3)` composition Jacobians;
- the full joint tree covariance before adapting its marginal blocks to the dense
  fusion interface; and
- the existing fixed-lag and simulated sparse-anchor rows initialized from those
  provider-aligned marginals.

The output metadata records the backend mode, provider API version, Jacobian mode,
the availability of the joint cross-window covariance, and the explicit
`per_window_marginals` adapter used by dense fusion. Tests compare the adapter
against a direct call to the same tree estimator used by provider v2.

## Claim boundary

This is a **gauge-backend parity benchmark**, not by itself a claim-bearing
provider-v2 observation export. It reuses the historical point-uncertainty
calibration and simulated metric-anchor protocol. A claim-bearing downstream run
still requires independently fitted content-addressed gauge and point covariance
calibrations, strict prediction/calibration compatibility, causal source sealing,
runtime revision attestation, and the calibrated provider-v2 export entry point.
