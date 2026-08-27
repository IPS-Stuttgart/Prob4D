# Joint material-identity controlled study v1

This directory retains the deterministic controlled-mechanism result for exact
one-to-one material-identity marginalization.

- report ID: `9dd90d444f82a3c90a4b3891a5d696eb053d3416e8d98936d0af6e6810af46f4`
- 32,768 trials in each of three ambiguity regimes
- 128 groups per regime and 5,000 paired group-bootstrap resamples
- operator/enumeration maximum absolute discrepancy: `8.88e-16`
- all three independent-soft minus joint NLL intervals are strictly positive

Rebuild and verify:

```bash
python -m prob4d.joint_identity_controlled_study build \
  --output evidence/joint-material-identity-controlled-v1/summary.json \
  --overwrite
python -m prob4d.joint_identity_controlled_study verify \
  evidence/joint-material-identity-controlled-v1/summary.json
```

The data are generated from the one-to-one model by construction. This is
controlled synthetic mechanism evidence only; it establishes no real-provider,
BayesianPhysTwin, Causal4D, deployment, or state-of-the-art claim.
